"""
I-JEPA Anomaly Detection Web App
Run: python app_ijepa.py
Open: http://127.0.0.1:5001
"""

import os, uuid, json
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATA_ROOT        = "data"
CHECKPOINT_DIR   = "checkpoints"
METRICS_IJEPA    = "results/ijepa"
HEATMAP_DIR      = "results/heatmaps"
ALLOWED_EXT      = {"png", "jpg", "jpeg"}

CATEGORIES = [
    "bottle","cable","capsule","carpet","grid",
    "hazelnut","leather","metal_nut","pill","screw",
    "tile","toothbrush","transistor","wood","zipper"
]


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXT


def is_trained(category):
    det = os.path.join(CHECKPOINT_DIR, f"ijepa_detector_{category}.pkl")
    return os.path.isfile(det)


def get_metrics(category):
    p = os.path.join(METRICS_IJEPA, category, "metrics_ijepa.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "ijepa.html")


@app.route("/api/categories")
def api_categories():
    out = []
    for cat in CATEGORIES:
        m = get_metrics(cat) if is_trained(cat) else {}
        out.append({
            "id":      cat,
            "name":    cat.replace("_", " ").title(),
            "trained": is_trained(cat),
            "roc_auc": round(m.get("roc_auc", 0), 4) if m else None,
            "f1":      round(m.get("f1_at_best_threshold", 0), 4) if m else None,
        })
    return jsonify(out)


@app.route("/api/images")
def api_images():
    """
    Returns structure:
      { "good": [path,...], "defect_types": {"crack":[path,...], ...} }
    """
    category = request.args.get("category", "hazelnut")
    base     = os.path.join(DATA_ROOT, category)

    # Good images (test/good only — these are the test-split normal images)
    good = []
    folder = os.path.join(base, "test", "good")
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                good.append(
                    folder.replace("\\", "/").lstrip("./") + "/" + f
                )

    # Defect images
    defect_types = {}
    test_base = os.path.join(base, "test")
    if os.path.isdir(test_base):
        for dt in sorted(os.listdir(test_base)):
            if dt == "good":
                continue
            sub = os.path.join(test_base, dt)
            if os.path.isdir(sub):
                imgs = [
                    sub.replace("\\","/").lstrip("./") + "/" + f
                    for f in sorted(os.listdir(sub))
                    if f.lower().endswith((".png",".jpg",".jpeg"))
                ]
                if imgs:
                    defect_types[dt] = imgs

    return jsonify({"good": good, "defect_types": defect_types})


@app.route("/api/image")
def api_image():
    path = request.args.get("path", "")
    path = path.lstrip("/").replace("\\", "/")
    if ".." in path or not path.startswith("data/"):
        return "Forbidden", 403
    full = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
    if not os.path.isfile(full):
        return "Not Found", 404
    return send_from_directory(os.path.dirname(full), os.path.basename(full))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Analyze an image from the dataset (path provided)."""
    data     = request.get_json()
    img_path = data.get("path", "")
    category = data.get("category", "")

    if not img_path or not category:
        return jsonify({"error": "path and category required"}), 400

    img_path = img_path.lstrip("/").replace("\\", "/")
    if ".." in img_path or not img_path.startswith("data/"):
        return jsonify({"error": "Invalid path"}), 403

    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), img_path)
    if not os.path.isfile(full_path):
        return jsonify({"error": "Image not found"}), 404

    try:
        result = _run_detection(full_path, category)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload a custom image for analysis."""
    category = request.form.get("category", "hazelnut")
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files["image"]
    if not allowed(file.filename):
        return jsonify({"error": "Allowed: png, jpg, jpeg"}), 400

    ext  = file.filename.rsplit(".", 1)[-1].lower()
    name = f"upload_{uuid.uuid4().hex[:12]}.{ext}"
    save = os.path.join(UPLOAD_FOLDER, name)
    file.save(save)

    try:
        result = _run_detection(save, category)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.isfile(save):
            os.remove(save)

    return jsonify(result)


@app.route("/heatmaps/<path:filename>")
def serve_heatmap(filename):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), HEATMAP_DIR)
    return send_from_directory(folder, filename)


# ── Detection helper ───────────────────────────────────────────────────────────

def _run_detection(image_path, category):
    """Run I-JEPA detection. Returns result dict."""
    import torch
    from PIL import Image
    from src.datasets import eval_transform
    from src.utils import get_device
    from src.ijepa_anomaly_detector import IJEPAAnomalyDetector
    import numpy as np, torch.nn.functional as F, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils import tensor_to_numpy

    if not is_trained(category):
        return {"error": f"I-JEPA not trained for '{category}'. "
                         f"Run: python run_ijepa.py --categories {category} --epochs 100"}

    device = get_device()

    # Load model (cached via backend helper).
    # _find_ijepa_checkpoint returns either a .pth path or "__pretrained_vit__" sentinel.
    from backend import _find_ijepa_checkpoint, _get_ijepa_model, _PRETRAINED_SENTINEL
    ckpt_path = _find_ijepa_checkpoint(category)
    if not ckpt_path:
        return {"error": "Checkpoint not found. Run: python run_ijepa.py --categories "
                         f"{category} --epochs 100   OR   python rebuild_detectors.py"}
    model = _get_ijepa_model(ckpt_path, device)

    # Load detector
    det_path = os.path.join(CHECKPOINT_DIR, f"ijepa_detector_{category}.pkl")
    detector = IJEPAAnomalyDetector.load(det_path)

    # Load image
    img    = Image.open(image_path).convert("RGB")
    tensor = eval_transform(img).unsqueeze(0).to(device)

    score, patch_scores = detector.score(model, tensor, device)

    # Metrics / threshold
    m = get_metrics(category)
    threshold  = m.get("best_threshold", float(np.percentile(patch_scores, 90)))
    roc_auc    = m.get("roc_auc")
    f1         = m.get("f1_at_best_threshold")
    avg_prec   = m.get("average_precision")
    is_anomaly = score >= threshold
    label      = "ANOMALY" if is_anomaly else "NORMAL"

    # Build heatmap PNG
    grid = 14
    hmap = patch_scores.reshape(grid, grid).astype(np.float32)
    hmap_t = torch.tensor(hmap).unsqueeze(0).unsqueeze(0)
    hmap_up = F.interpolate(hmap_t, size=(224, 224), mode="bilinear", align_corners=False)
    hmap_np = hmap_up.squeeze().numpy()

    os.makedirs(HEATMAP_DIR, exist_ok=True)
    uid  = uuid.uuid4().hex[:10]
    fname = f"ijepa_{category}_{uid}.png"
    fpath = os.path.join(HEATMAP_DIR, fname)

    img_np  = tensor_to_numpy(tensor.cpu())
    hm_norm = hmap_np.copy()
    if hm_norm.max() > hm_norm.min():
        hm_norm = (hm_norm - hm_norm.min()) / (hm_norm.max() - hm_norm.min())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")

    axes[0].imshow(img_np); axes[0].set_title("Input Image", color="#c9d1d9", fontsize=11); axes[0].axis("off")
    im = axes[1].imshow(hm_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Patch Anomaly Scores", color="#c9d1d9", fontsize=11); axes[1].axis("off")
    cb = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#c9d1d9")

    axes[2].imshow(img_np)
    axes[2].imshow(hm_norm, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title("Overlay", color="#c9d1d9", fontsize=11); axes[2].axis("off")

    col = "#f85149" if is_anomaly else "#3fb950"
    fig.suptitle(f"{label}  |  Score: {score:.4f}  |  Threshold: {threshold:.4f}",
                 fontsize=13, fontweight="bold", color=col)
    plt.tight_layout()
    plt.savefig(fpath, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "score":     round(float(score), 6),
        "threshold": round(float(threshold), 6),
        "label":     label,
        "is_anomaly": bool(is_anomaly),
        "heatmap":   fname,
        "roc_auc":   round(roc_auc, 4) if roc_auc else None,
        "f1":        round(f1, 4)       if f1       else None,
        "avg_prec":  round(avg_prec, 4) if avg_prec else None,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  I-JEPA Anomaly Detection App")
    print("  Open: http://127.0.0.1:5001")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
