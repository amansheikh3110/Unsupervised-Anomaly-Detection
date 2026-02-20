"""
Web UI for Anomaly Detection Demo.

Run: python app.py
Then open http://127.0.0.1:5000
"""

import os
import uuid
from flask import Flask, send_from_directory, request, jsonify
from werkzeug.utils import secure_filename

import backend

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/categories")
def api_categories():
    return jsonify(backend.list_categories())


@app.route("/api/dataset")
def api_dataset():
    category = request.args.get("category", "leather")
    split = request.args.get("split")  # train | test | None
    items = backend.list_dataset_images(category, split)
    return jsonify(items)


@app.route("/api/image")
def api_image():
    """Serve image from project (e.g. data/leather/...). Path must be under data/."""
    path = request.args.get("path", "")
    path = path.lstrip("/").replace("\\", "/")
    if not path.startswith("data/") or ".." in path:
        return "Forbidden", 403
    full = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
    if not os.path.isfile(full):
        return "Not Found", 404
    return send_from_directory(os.path.dirname(full), os.path.basename(full))


@app.route("/api/check", methods=["POST"])
def api_check():
    """Run anomaly check. Body: form with 'image' file and 'category'."""
    category = request.form.get("category", "leather")
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Allowed: png, jpg, jpeg"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    safe_name = f"upload_{uuid.uuid4().hex[:12]}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(save_path)
    try:
        result = backend.run_anomaly_check(save_path, category)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.isfile(save_path):
            os.remove(save_path)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/metrics/<category>")
def api_metrics(category):
    m = backend.get_category_metrics(category)
    return jsonify(m)


@app.route("/heatmaps/<path:filename>")
def serve_heatmap(filename):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "heatmaps")
    return send_from_directory(folder, filename)


if __name__ == "__main__":
    print("Starting Anomaly Detection Demo UI at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
