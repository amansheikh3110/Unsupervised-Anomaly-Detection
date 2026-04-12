"""
I-JEPA Anomaly Detector — Phase 3/4

Uses I-JEPA patch features for high-accuracy anomaly detection via k-NN.

Core idea (PatchCore-style with I-JEPA features):
  - Extract I-JEPA patch features for all NORMAL training images.
  - Build a compact coreset (random subsampling of patch vectors).
  - At test time: extract patch features → find k-NN in coreset.
  - Anomaly score = max patch-level k-NN distance over all patches.

Why this works well:
  - I-JEPA produces SEMANTIC features (not raw pixels), so normal/anomaly
    patches are far apart in embedding space.
  - Patch-level (local) scoring detects both global and localised defects.
  - k-NN requires no further training — just a memory bank lookup.
  - This approach mirrors PatchCore which achieves 99.1% AUC on MVTec
    using ResNet features; I-JEPA features trained on the data should
    perform comparably or better.

Usage:
    # Build detector
    detector = IJEPAAnomalyDetector(k=9, coreset_ratio=0.25)
    detector.fit(model, normal_loader, device)
    detector.save("checkpoints/ijepa_detector_leather.pkl")

    # Score a test image
    score, patch_scores = detector.score(model, image_tensor, device)

    # Load saved detector
    detector2 = IJEPAAnomalyDetector.load("checkpoints/ijepa_detector_leather.pkl")
"""

import os
import pickle
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.model_ijepa import IJEPA
from src.datasets import eval_transform
from src.utils import get_device


class IJEPAAnomalyDetector:
    """
    Patch-level k-NN anomaly detector on top of I-JEPA features.

    Workflow:
      1. fit()   — extract patch features from normal images, build k-NN index
      2. score() — score a single image: max(patch k-NN distances)
      3. evaluate() — compute ROC-AUC on a labeled test set

    Args:
        k:              Number of nearest neighbours for scoring (default: 9)
        coreset_ratio:  Fraction of normal patches to keep in the memory bank.
                        0.25 = 25% subsampling (good speed/accuracy trade-off).
                        1.0  = keep all patches (most accurate, slower).
        layer_avg:      Whether to average features across multiple encoder layers
                        (currently uses only the final layer — set True for future use).
        normalize_feats: Whether to L2-normalize features before k-NN (recommended).
        batch_size:     Batch size for feature extraction (no gradients needed).
    """

    def __init__(
        self,
        k: int = 9,
        coreset_ratio: float = 0.25,
        normalize_feats: bool = True,
        batch_size: int = 32,
    ):
        self.k = k
        self.coreset_ratio = coreset_ratio
        self.normalize_feats = normalize_feats
        self.batch_size = batch_size

        self.memory_bank: Optional[np.ndarray] = None  # [N_normal_patches, D]
        self.knn_index: Optional[NearestNeighbors] = None
        self.category: Optional[str] = None
        self.n_patches: int = 196  # 14×14 for 224×224 / patch16
        self.embed_dim: int = 384

    # -------------------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _extract_patch_features(
        self,
        model: IJEPA,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> np.ndarray:
        """
        Extract all patch-level features from a data loader.

        Returns:
            features: [N_images * num_patches, embed_dim]  float32 numpy array
        """
        model.context_encoder.eval()
        all_features = []

        for batch in tqdm(loader, desc="  Extracting features", leave=False):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device, non_blocking=True)

            # Extract ALL 196 patch features (no masking at inference)
            patch_features, _ = model.extract_features(images)
            # patch_features: [B, 196, D]

            B, N, D = patch_features.shape
            # Flatten to [B*N, D] and move to CPU
            feats = patch_features.reshape(B * N, D).cpu().float().numpy()
            all_features.append(feats)

        return np.concatenate(all_features, axis=0)  # [N_total_patches, D]

    # -------------------------------------------------------------------------
    # Coreset subsampling
    # -------------------------------------------------------------------------

    @staticmethod
    def _random_coreset(features: np.ndarray, ratio: float) -> np.ndarray:
        """
        Random subsampling coreset (fast; greedy coreset can be added later).

        Args:
            features: [N, D]
            ratio:    fraction to keep

        Returns:
            subset: [M, D]  where M = max(1, int(N * ratio))
        """
        N = features.shape[0]
        M = max(1, int(N * ratio))
        if M >= N:
            return features
        idx = np.random.choice(N, M, replace=False)
        return features[idx]

    # -------------------------------------------------------------------------
    # Fit (build memory bank)
    # -------------------------------------------------------------------------

    def fit(
        self,
        model: IJEPA,
        normal_loader: torch.utils.data.DataLoader,
        device: torch.device,
        category: str = "unknown",
    ):
        """
        Build the normal feature memory bank from training (normal) images.

        Args:
            model:         Trained I-JEPA model
            normal_loader: DataLoader over normal training images
            device:        Device for feature extraction
            category:      Category name (stored for reference)
        """
        print(f"  Building memory bank for '{category}'...")
        self.category = category
        self.embed_dim = model.encoder_embed_dim

        # 1. Extract all patch features from normal training images
        features = self._extract_patch_features(model, normal_loader, device)
        # features: [N_images * 196, D]

        print(f"  Raw features: {features.shape}  ({features.nbytes/1e6:.1f} MB)")

        # 2. L2-normalize features
        if self.normalize_feats:
            features = normalize(features, norm="l2", axis=1)

        # 3. Coreset subsampling
        if self.coreset_ratio < 1.0:
            features = self._random_coreset(features, self.coreset_ratio)
            print(f"  Coreset ({self.coreset_ratio*100:.0f}%): {features.shape}  "
                  f"({features.nbytes/1e6:.1f} MB)")

        self.memory_bank = features

        # 4. Build k-NN index (CPU, sklearn)
        # Use cosine metric since features are L2-normalized
        # (cosine distance = 1 - cosine_similarity; with L2-norm, equivalent to L2)
        metric = "euclidean" if self.normalize_feats else "euclidean"
        self.knn_index = NearestNeighbors(
            n_neighbors=self.k,
            algorithm="ball_tree",   # efficient for moderate-dim spaces
            metric=metric,
            n_jobs=-1,               # use all CPU cores
        )
        self.knn_index.fit(features)

        print(f"  k-NN index built. k={self.k}, metric={metric}, "
              f"memory bank: {features.shape[0]} patches.")

    # -------------------------------------------------------------------------
    # Score a single image
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        model: IJEPA,
        image_tensor: torch.Tensor,
        device: torch.device,
    ) -> Tuple[float, np.ndarray]:
        """
        Score one image.

        Args:
            model:        Trained I-JEPA model
            image_tensor: [1, C, H, W] preprocessed image tensor
            device:       Device for feature extraction

        Returns:
            anomaly_score: float — image-level score (higher = more anomalous)
            patch_scores:  [num_patches] numpy array — per-patch anomaly scores
                           (useful for heatmap visualisation)
        """
        assert self.knn_index is not None, "Call fit() before score()."

        model.context_encoder.eval()
        image_tensor = image_tensor.to(device)

        # Extract patch features
        patch_features, cls_feat = model.extract_features(image_tensor)
        # patch_features: [1, 196, D]

        # Reshape to [196, D]
        feats = patch_features.squeeze(0).cpu().float().numpy()  # [196, D]

        if self.normalize_feats:
            feats = normalize(feats, norm="l2", axis=1)

        # k-NN distances for each patch
        distances, _ = self.knn_index.kneighbors(feats)  # [196, k]

        # Per-patch score: mean distance to k nearest neighbours
        patch_scores = distances.mean(axis=1)  # [196]

        # Image-level score: maximum patch score
        # (any localised defect will produce a high-scoring patch)
        anomaly_score = float(patch_scores.max())

        return anomaly_score, patch_scores

    # -------------------------------------------------------------------------
    # Patch score heatmap (for visualisation)
    # -------------------------------------------------------------------------

    def patch_scores_to_heatmap(
        self,
        patch_scores: np.ndarray,
        img_size: int = 224,
        patch_size: int = 16,
    ) -> np.ndarray:
        """
        Convert per-patch scores to a spatial heatmap.

        Args:
            patch_scores: [num_patches]  (196 for 14×14 grid)
            img_size:     Original image size (default: 224)
            patch_size:   Patch size (default: 16)

        Returns:
            heatmap: [img_size, img_size]  float32  (bilinear-upsampled)
        """
        grid_size = img_size // patch_size
        assert len(patch_scores) == grid_size * grid_size, (
            f"Expected {grid_size**2} patch scores, got {len(patch_scores)}"
        )

        # Reshape to 2D grid
        heatmap_small = patch_scores.reshape(grid_size, grid_size).astype(np.float32)

        # Upsample to full image size using torch
        hmap_t = torch.tensor(heatmap_small).unsqueeze(0).unsqueeze(0)  # [1,1,14,14]
        hmap_up = F.interpolate(
            hmap_t, size=(img_size, img_size), mode="bilinear", align_corners=False
        )
        return hmap_up.squeeze().numpy()  # [224, 224]

    # -------------------------------------------------------------------------
    # Evaluate on labelled test set
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        model: IJEPA,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for an entire labelled test set.

        Args:
            model:       Trained I-JEPA model
            test_loader: DataLoader with (image, label) pairs; label 0=normal, 1=anomaly
            device:      Device for feature extraction

        Returns:
            scores: [N_test]  anomaly scores (higher → more anomalous)
            labels: [N_test]  ground-truth labels (0 or 1)
        """
        assert self.knn_index is not None, "Call fit() before evaluate()."

        model.context_encoder.eval()
        all_scores = []
        all_labels = []

        for batch in tqdm(test_loader, desc="  Evaluating", leave=False):
            if isinstance(batch, (list, tuple)):
                images, lbls = batch[0], batch[1]
            else:
                raise ValueError("Test loader must return (image, label) pairs.")
            images = images.to(device, non_blocking=True)

            # Extract patch features
            patch_features, _ = model.extract_features(images)
            # patch_features: [B, 196, D]

            B, N, D = patch_features.shape
            feats_np = patch_features.reshape(B * N, D).cpu().float().numpy()

            if self.normalize_feats:
                feats_np = normalize(feats_np, norm="l2", axis=1)

            # k-NN distances: [B*N, k]
            distances, _ = self.knn_index.kneighbors(feats_np)
            patch_dists = distances.mean(axis=1).reshape(B, N)  # [B, N]

            # Image scores: max over patches
            image_scores = patch_dists.max(axis=1)  # [B]

            all_scores.append(image_scores)
            all_labels.append(lbls.numpy() if hasattr(lbls, "numpy") else np.array(lbls))

        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return scores, labels

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    def save(self, path: str):
        """Persist memory bank and k-NN index to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "memory_bank": self.memory_bank,
                    "knn_index": self.knn_index,
                    "k": self.k,
                    "coreset_ratio": self.coreset_ratio,
                    "normalize_feats": self.normalize_feats,
                    "category": self.category,
                    "embed_dim": self.embed_dim,
                    "n_patches": self.n_patches,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        sz = os.path.getsize(path) / 1e6
        print(f"  Detector saved → {path}  ({sz:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "IJEPAAnomalyDetector":
        """Load a previously saved detector."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        det = cls(
            k=data["k"],
            coreset_ratio=data["coreset_ratio"],
            normalize_feats=data["normalize_feats"],
        )
        det.memory_bank = data["memory_bank"]
        det.knn_index = data["knn_index"]
        det.category = data["category"]
        det.embed_dim = data.get("embed_dim", 384)
        det.n_patches = data.get("n_patches", 196)
        return det


# =============================================================================
# Convenience function: build per-category detectors
# =============================================================================

def build_detector_for_category(
    model: IJEPA,
    category: str,
    data_root: str,
    save_dir: str,
    device: torch.device,
    k: int = 9,
    coreset_ratio: float = 0.25,
    batch_size: int = 32,
    num_workers: int = 0,
) -> IJEPAAnomalyDetector:
    """
    Build and save an anomaly detector for one MVTec category.

    Returns the fitted IJEPAAnomalyDetector.
    """
    from src.datasets import get_dataloader

    normal_loader = get_dataloader(
        root_dir=data_root,
        category=category,
        split="train",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        return_label=False,
        transform=eval_transform,  # No augmentation for feature extraction
    )

    detector = IJEPAAnomalyDetector(
        k=k,
        coreset_ratio=coreset_ratio,
        normalize_feats=True,
        batch_size=batch_size,
    )
    detector.fit(model, normal_loader, device, category=category)

    # Save to checkpoints dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ijepa_detector_{category}.pkl")
    detector.save(save_path)

    return detector


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.model_ijepa import get_ijepa_small

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_ijepa_small().to(device)

    # Simulate a memory bank and a test image
    det = IJEPAAnomalyDetector(k=5, coreset_ratio=1.0)

    # Fake normal features
    fake_normal = np.random.randn(200 * 196, 384).astype(np.float32)
    fake_normal = normalize(fake_normal, norm="l2", axis=1)
    det.memory_bank = fake_normal
    det.category = "test"
    det.knn_index = NearestNeighbors(n_neighbors=5, algorithm="ball_tree", metric="euclidean")
    det.knn_index.fit(fake_normal)

    # Fake test image
    img = torch.randn(1, 3, 224, 224, device=device)
    score, patch_sc = det.score(model, img, device)
    heatmap = det.patch_scores_to_heatmap(patch_sc)
    print(f"Anomaly score: {score:.4f}")
    print(f"Patch scores:  min={patch_sc.min():.4f}  max={patch_sc.max():.4f}")
    print(f"Heatmap shape: {heatmap.shape}")
    print("IJEPAAnomalyDetector test PASSED.")
