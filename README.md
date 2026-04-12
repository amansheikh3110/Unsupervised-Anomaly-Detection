# Unsupervised Anomaly Detection Using Deep Representation Learning
### I-JEPA-Based Industry-Grade AI | Group A-5 | Semester VI | 2025-26
**Shri Ramdeobaba College of Engineering & Management, Nagpur**
**Department of Computer Science & Engineering**
**Members:** Aman Sheikh, Aadil Pathan, Aryan Jaiswal | **Guide:** Dr. P. Sonsare

---

## What This Project Does

This project builds an AI system that detects defects in manufactured products **without ever being shown a defective image during training**. It learns only from normal (good) images, then flags anything that looks abnormal.

We implemented **two approaches** and compare them:

| | Phase 2 — Autoencoder (Baseline) | Phase 3 — I-JEPA (Our Method) |
|---|---|---|
| **Idea** | Reconstruct the image; high error = defect | Predict patch representations; high distance = defect |
| **Signal** | Pixel-level reconstruction error (MSE) | Semantic feature distance (k-NN in latent space) |
| **Strength** | Simple, fast to train | Semantic, more accurate |
| **Anomaly Score** | MSE(input, reconstruction) | max patch k-NN distance to memory bank |

**Dataset:** MVTec Anomaly Detection — 15 industrial categories, train on `good` only, test on `good + defects`.

---

## System Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| OS | Windows 10 / Linux | Windows 11 |
| Python | 3.10+ | 3.13.5 |
| RAM | 8 GB | 16 GB |
| GPU | NVIDIA (CUDA 11.7+) | GTX 1650 Ti (4GB) |
| Disk | 15 GB | ~20 GB |

---

## Environment Setup

### Step 1 — Install Python

Download Python 3.10+ from [python.org](https://www.python.org/downloads/).
During install: check **"Add Python to PATH"**.

```bash
python --version
# Expected: Python 3.10+
```

### Step 2 — Clone the Repository

```bash
git clone <repository-url>
cd 6th_Sem_Project
```

### Step 3 — Create Virtual Environment

```bash
python -m venv ijepa_env

# Activate — Windows CMD
ijepa_env\Scripts\activate.bat

# Activate — Windows PowerShell
.\ijepa_env\Scripts\Activate.ps1

# Activate — Linux / Mac
source ijepa_env/bin/activate
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually, for NVIDIA GPU with CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm opencv-python pillow flask werkzeug
```

### Step 5 — Verify Setup

```bash
python src.py
```

Expected output:
```
PyTorch version: 2.7.1+cu118
CUDA available: True
GPU: NVIDIA GeForce GTX 1650
GPU Tensor Test: SUCCESS
All systems ready for I-JEPA project!
```

To monitor GPU utilization during training, open a second terminal and run:
```bash
nvidia-smi -l 1
```

---

## Project Structure

```
6th_Sem_Project/
│
├── data/                          # MVTec AD dataset (15 categories)
│   ├── bottle/
│   │   ├── train/good/            # Normal training images ONLY
│   │   └── test/                  # good/ + defect subfolders
│   ├── leather/
│   ├── hazelnut/
│   └── ... (15 categories total)
│
├── src/                           # All model and pipeline code
│   ├── datasets.py                # MVTec data loader, transforms, patchify
│   ├── utils.py                   # Seed, device, visualization helpers
│   ├── model_autoencoder.py       # Phase 2 — Convolutional Autoencoder
│   ├── train_baseline.py          # Phase 2 — Autoencoder training script
│   ├── anomaly_eval.py            # Phase 2 — Evaluation (ROC-AUC etc.)
│   ├── heatmap_utils.py           # Phase 2 — Reconstruction error heatmaps
│   ├── model_ijepa.py             # Phase 3 — I-JEPA ViT model
│   ├── masks.py                   # Phase 3 — Block masking strategy
│   ├── train_ijepa.py             # Phase 3 — I-JEPA self-supervised training
│   └── ijepa_anomaly_detector.py  # Phase 3 — k-NN anomaly detector
│
├── checkpoints/                   # Saved model weights
│   ├── autoencoder_leather.pth    # Phase 2 checkpoint
│   ├── ijepa_small_hazelnut.pth   # Phase 3 I-JEPA encoder checkpoint
│   └── ijepa_detector_hazelnut.pkl  # Phase 3 k-NN memory bank
│
├── results/
│   ├── baseline/<category>/       # Phase 2 metrics (ROC-AUC, F1, etc.)
│   ├── ijepa/<category>/          # Phase 3 metrics
│   └── heatmaps/                  # Generated heatmap images (PNG + PDF)
│
├── run_baseline.py                # Phase 2 — one-command train + evaluate
├── run_ijepa.py                   # Phase 3 — one-command train + evaluate
├── check_image.py                 # Phase 2 — single image check (autoencoder)
├── check_image_ijepa.py           # Phase 3 — single image check (I-JEPA)
├── app.py                         # Flask web UI
├── backend.py                     # Web UI backend logic (auto-picks best model)
├── src.py                         # Environment verification script
├── requirements.txt
└── README.md
```

---

## Phase 2 — Autoencoder Baseline

The baseline uses a convolutional autoencoder trained on normal images only.
**Anomaly score = MSE reconstruction error.** High error means likely defect.

### Train

```bash
# Train on one category (50 epochs recommended for baseline)
python run_baseline.py --category leather --epochs 50

# Other available categories:
# bottle, cable, capsule, carpet, grid, hazelnut, leather,
# metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
```

The script trains the autoencoder, then automatically evaluates it.
Checkpoint saved to: `checkpoints/autoencoder_<category>.pth`
Results saved to: `results/baseline/<category>/metrics_baseline.json`

### Evaluate Only (existing checkpoint)

```bash
python -m src.anomaly_eval --category leather --checkpoint checkpoints/autoencoder_leather.pth
```

### Check a Single Image

```bash
# Check if an image is normal or defective
python check_image.py data/leather/test/cut/002.png --category leather

# Check a normal image
python check_image.py data/leather/train/good/001.png --category leather
```

Output: anomaly score, threshold, result (Normal / Anomaly), heatmap saved to `results/heatmaps/`.

### Metrics Output

```
ROC-AUC:              0.7812
Average Precision:    0.8543
F1 (best threshold):  0.7200
Confusion matrix:     TN=28  FP=4  FN=18  TP=74
```

---

## Phase 3 — I-JEPA (Our Core Method)

I-JEPA (Image-based Joint-Embedding Predictive Architecture) trains a Vision Transformer
to predict masked patch representations from visible context patches — in latent space,
not pixel space. This produces rich semantic features.

**Architecture:**
- Context Encoder — ViT-Small/16 (384-dim, 12 layers, 21.7M params) — trained with gradient
- Target Encoder — EMA copy of context encoder (momentum 0.996 → 1.0) — no gradient
- Predictor — narrow ViT (192-dim, 6 layers) — predicts target patch representations

**Anomaly Detection (after training):**
- Extract all 196 patch features per image (14×14 grid, no masking)
- Build a k-NN memory bank from all normal training images
- Score = max patch-level distance to k nearest normal patches (k=9)
- High score = anomaly

### Step 1 — Train I-JEPA

```bash
# Single category (recommended: 100 epochs, takes ~2-4 hours on GTX 1650)
python run_ijepa.py --categories hazelnut --epochs 100

# Multiple categories trained together (one shared model — more efficient)
python run_ijepa.py --categories bottle cable capsule carpet grid hazelnut leather metal_nut pill screw --epochs 100

# Use tiny model if you get CUDA out of memory
python run_ijepa.py --categories hazelnut --epochs 100 --model_size tiny

# Use smaller batch size if needed
python run_ijepa.py --categories hazelnut --epochs 100 --batch_size 8
```

Checkpoint saved to: `checkpoints/ijepa_small_<category>.pth`

After training finishes, the script automatically:
1. Builds the k-NN memory bank for each category
2. Evaluates on the test set
3. Prints ROC-AUC, AP, F1 for each category

### Step 2 — Evaluate Only (skip retraining)

If training is already done and you only want metrics:

```bash
python run_ijepa.py --categories hazelnut --no_train
```

### Step 3 — Check a Single Image

```bash
# Check any image for defects using I-JEPA
python check_image_ijepa.py data/hazelnut/test/crack/001.png --category hazelnut

# Check a good image
python check_image_ijepa.py data/hazelnut/test/good/001.png --category hazelnut
```

Output: anomaly score, threshold, result, heatmap saved to `results/heatmaps/`.

### Training Notes

- **Loss fluctuation is normal.** I-JEPA uses random block masking every epoch, so loss can go up and down between epochs. Watch the 20-30 epoch trend, not individual values.
- **50 epochs** is sufficient for a good model; 100 gives better accuracy.
- **Task Manager GPU:** Your laptop has two GPUs — Intel (GPU 0) and NVIDIA (GPU 1). Task Manager shows Intel by default. Check GPU 1 or run `nvidia-smi -l 1` in a separate terminal to confirm NVIDIA is being used.
- **Do not run multiple categories in separate terminals simultaneously** — the GPU only has 4GB VRAM and will crash. Use one command with all categories listed.
- **Can you stop mid-training?** Yes. Checkpoints are saved whenever loss improves. You can stop at any epoch with Ctrl+C and use what was saved.

### I-JEPA Metrics Output

```
Category        ROC-AUC        AP        F1     vs Baseline
hazelnut          0.9412    0.9156    0.8734      +0.1600
leather           0.9601    0.9402    0.8891      +0.1789
...
MEAN              0.9350
```

---

## Web UI (Demo)

Interactive browser-based demo. Automatically uses I-JEPA if trained; falls back to autoencoder.

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

**Features:**
- **Tester tab:** Drag and drop any image, select the category, click Run Detection. Shows anomaly score, Normal/Anomaly badge, heatmap, ROC-AUC, F1.
- **Dataset tab:** Browse all MVTec images by category and split. Click any image to run detection on it.
- Heatmaps show which patches/regions the model flagged as anomalous.

---

## Package Versions

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.7.1+cu118 | Deep learning |
| torchvision | 0.22.1+cu118 | Vision utilities |
| numpy | 2.3.5 | Numerical computing |
| pandas | 3.0.0 | Data manipulation |
| matplotlib | 3.10.8 | Plotting |
| seaborn | 0.13.2 | Visualization |
| scikit-learn | 1.8.0 | k-NN, metrics |
| scipy | 1.17.0 | Scientific computing |
| opencv-python | 4.13.0 | Image processing |
| pillow | 12.0.0 | Image loading |
| flask | 3.0.0 | Web UI |
| tqdm | 4.67.2 | Progress bars |

---

## Quick Reference — All Commands

```bash
# --- Setup ---
python -m venv ijepa_env
ijepa_env\Scripts\activate.bat       # Windows CMD
python src.py                         # verify GPU + packages

# --- Phase 2: Autoencoder Baseline ---
python run_baseline.py --category leather --epochs 50
python check_image.py data/leather/test/cut/002.png --category leather

# --- Phase 3: I-JEPA ---
# Train (single category)
python run_ijepa.py --categories hazelnut --epochs 100

# Train (multiple categories at once — recommended)
python run_ijepa.py --categories bottle cable capsule carpet grid hazelnut leather metal_nut pill screw --epochs 100

# Evaluate only (no retraining)
python run_ijepa.py --categories hazelnut --no_train

# Single image check
python check_image_ijepa.py data/hazelnut/test/crack/001.png --category hazelnut

# --- Web UI ---
python app.py                         # open http://127.0.0.1:5000

# --- Monitor GPU ---
nvidia-smi -l 1                       # separate terminal, updates every second
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch._strobelight'`**
Corrupted venv. Delete `ijepa_env/` and recreate it.

**`CUDA out of memory`**
Reduce batch size: `--batch_size 8`, or use a smaller model: `--model_size tiny`.

**Task Manager shows 0% GPU**
You are looking at the Intel GPU (GPU 0). Check GPU 1 in Task Manager, or run `nvidia-smi -l 1` to confirm NVIDIA usage.

**`No checkpoint for category`**
You need to train first: `python run_ijepa.py --categories <category> --epochs 100`

**`No k-NN detector found`**
Training completed but evaluation did not run. Run: `python run_ijepa.py --categories <category> --no_train`

**Loss keeps fluctuating during I-JEPA training**
This is expected. I-JEPA uses random masking every epoch, so loss varies. Watch the multi-epoch trend, not individual values.

---

## References

- Assran et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- Bergmann et al. (2019). *The MVTec Anomaly Detection Dataset.* IJCV 2021.
- Roth et al. (2022). *Towards Total Recall in Industrial Anomaly Detection (PatchCore).* CVPR 2022.