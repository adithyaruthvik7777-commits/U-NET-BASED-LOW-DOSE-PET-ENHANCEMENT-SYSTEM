import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename

from unet_model import load_model, get_device
from utils.image_processing import (
    load_pet_image,
    apply_colormap_and_save,
    compute_histogram_bins,
    save_kaggle_style_comparison,
    DATASET_STATS,
    normalize_colormap,
)
from utils.enhance_pipeline import enhance_pet
from utils.metrics import calculate_metrics

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "pet_unet_secret_key_2026")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload limit

# Faster single-pass inference on Render / low-RAM hosts (avoids HTML gateway timeouts)
if os.environ.get("RENDER") or os.environ.get("FAST_INFERENCE", "").lower() in ("1", "true", "yes"):
    os.environ["FAST_INFERENCE"] = "1"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
SAMPLE_FOLDER = os.path.join(BASE_DIR, "static", "sample_images")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_unet_pet.pth")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"dcm", "ima", "img", "npy", "npz", "png", "jpg", "jpeg", "tif", "tiff"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def resolve_sample_path(sample_id):
    """
    Resolve preset sample from real dataset exports in static/sample_images.
    Returns (low_path, full_path). Always runs live U-Net on low_path.
    Dataset: https://www.kaggle.com/datasets/skarthik112/paired-low-dose-dataset-1to50
    """
    candidates = [
        f"{sample_id}.npy",
        f"{sample_id}.ima",
        f"{sample_id}.IMA",
        f"{sample_id}.png",
        f"sample_{sample_id}.npy",
        f"sample_{sample_id}.ima",
        f"sample_{sample_id}.png",
        "sample_low_dose_pet.npy",
        "sample_low_dose_pet.png",
    ]
    low_path = None
    for name in candidates:
        path = os.path.join(SAMPLE_FOLDER, name)
        if os.path.exists(path):
            low_path = path
            break

    full_path = None
    if low_path:
        base, ext = os.path.splitext(os.path.basename(low_path))
        full_candidates = [
            f"{base}_full.npy",
            f"{base}_full.ima",
            f"{base}_full.png",
            f"{base}_full{ext}",
            f"full_{base}{ext}",
            f"{sample_id}_full.npy",
            f"{sample_id}_full.ima",
            f"{sample_id}_full.png",
            "sample_low_dose_pet_full.npy",
            "sample_low_dose_pet_full.png",
        ]
        for name in full_candidates:
            path = os.path.join(SAMPLE_FOLDER, name)
            if os.path.exists(path):
                full_path = path
                break

    return low_path, full_path


# Load Model globally on startup
unet_instance, current_device = load_model(MODEL_PATH)

# Cap CPU threads — reduces peak RAM on Render
try:
    import torch as _torch
    _torch.set_num_threads(1)
    _torch.set_num_interop_threads(1)
except Exception:
    pass


# ==========================================
# PAGE ROUTES
# ==========================================

@app.route("/")
def index():
    return render_template("index.html", active_page="home", dataset_stats=DATASET_STATS)


@app.route("/about")
def about():
    return render_template("about.html", active_page="about", dataset_stats=DATASET_STATS)


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html", active_page="how_it_works", dataset_stats=DATASET_STATS)


@app.route("/enhance")
def enhance():
    return render_template("enhance.html", active_page="enhance", dataset_stats=DATASET_STATS)


@app.route("/model-info")
def model_info():
    return render_template("model_info.html", active_page="model_info", dataset_stats=DATASET_STATS)


@app.route("/methodology")
def methodology():
    return render_template("methodology.html", active_page="methodology", dataset_stats=DATASET_STATS)


@app.route("/tech-stack")
def tech_stack():
    return render_template("tech_stack.html", active_page="tech_stack", dataset_stats=DATASET_STATS)


@app.route("/architecture")
def architecture():
    return render_template("architecture.html", active_page="architecture", dataset_stats=DATASET_STATS)


# ==========================================
# API ENDPOINTS
# ==========================================

@app.route("/api/health")
def api_health():
    """Lightweight health check for Render."""
    return jsonify({
        "ok": True,
        "model_weights": os.path.exists(MODEL_PATH),
        "device": str(current_device),
        "fast_inference": os.environ.get("FAST_INFERENCE", "0") == "1",
    })


@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    """
    Handles image upload or sample selection, DICOM .IMA header parsing,
    PyTorch U-Net inference, postprocessing, histogram calculation, and JSON rendering.
    """
    import gc
    import cv2

    file_path = None
    full_dose_path = None
    colormap = normalize_colormap(request.form.get("colormap", "gray"))
    fast = os.environ.get("FAST_INFERENCE", "0") == "1"

    sample_id = request.form.get("sample_id")
    if sample_id:
        file_path, full_dose_path = resolve_sample_path(sample_id)
    elif "file" in request.files:
        file = request.files["file"]
        if file and file.filename != "" and allowed_file(file.filename):
            unique_id = uuid.uuid4().hex[:8]
            safe_name = secure_filename(file.filename) or "scan.ima"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], f"upload_{unique_id}_{safe_name}")
            file.save(file_path)

    if not file_path or not os.path.exists(file_path):
        return jsonify({"success": False, "error": "No valid file uploaded or sample selected."}), 400

    try:
        unique_id = uuid.uuid4().hex[:8]

        low_arr, metadata = load_pet_image(file_path)
        # Shrink early to cut RAM / time on Render
        if fast or max(low_arr.shape[:2]) > 256:
            low_arr = cv2.resize(low_arr, (256, 256), interpolation=cv2.INTER_AREA)

        full_arr = None
        if (not fast) and full_dose_path and os.path.exists(full_dose_path):
            full_arr, _ = load_pet_image(full_dose_path)
            if full_arr.shape[:2] != low_arr.shape[:2]:
                full_arr = cv2.resize(full_arr, (low_arr.shape[1], low_arr.shape[0]), interpolation=cv2.INTER_AREA)

        enhanced_arr, low_work = enhance_pet(unet_instance, current_device, low_arr)
        gc.collect()

        low_filename = f"low_{unique_id}.png"
        high_filename = f"high_{unique_id}.png"
        compare_filename = f"compare_{unique_id}.png"
        low_save_path = os.path.join(app.config["UPLOAD_FOLDER"], low_filename)
        high_save_path = os.path.join(app.config["UPLOAD_FOLDER"], high_filename)
        compare_save_path = os.path.join(app.config["UPLOAD_FOLDER"], compare_filename)

        apply_colormap_and_save(low_work, low_save_path, colormap=colormap)
        apply_colormap_and_save(enhanced_arr, high_save_path, colormap=colormap)

        full_filename = None
        full_dose_url = None
        if full_arr is not None:
            full_filename = f"full_{unique_id}.png"
            full_save_path = os.path.join(app.config["UPLOAD_FOLDER"], full_filename)
            apply_colormap_and_save(full_arr, full_save_path, colormap=colormap)
            full_dose_url = url_for("static", filename=f"uploads/{full_filename}")

        if fast:
            # Lightweight side-by-side (avoid heavy comparison renderer on Render)
            import numpy as np
            from PIL import Image
            from utils.image_processing import _to_rgb_uint8
            pair = np.concatenate(
                [_to_rgb_uint8(low_work, colormap), _to_rgb_uint8(enhanced_arr, colormap)],
                axis=1,
            )
            Image.fromarray(pair).save(compare_save_path)
        else:
            save_kaggle_style_comparison(
                low_work,
                enhanced_arr,
                compare_save_path,
                full_arr=full_arr,
                colormap=colormap,
                dataset_stats=DATASET_STATS,
            )

        if full_arr is not None:
            metrics = calculate_metrics(full_arr, enhanced_arr)
            metrics["reference"] = "full_dose_gt"
        else:
            # Fast path: avoid heavy skimage if needed — calculate_metrics is OK at 256²
            metrics = calculate_metrics(low_work, enhanced_arr)
            metrics["reference"] = "low_vs_enhanced"

        if fast:
            hist_low = {"labels": [], "values": []}
            hist_high = {"labels": [], "values": []}
            try:
                hist_low = compute_histogram_bins(low_work)
                hist_high = compute_histogram_bins(enhanced_arr)
            except Exception:
                pass
        else:
            hist_low = compute_histogram_bins(low_work)
            hist_high = compute_histogram_bins(enhanced_arr)

        return jsonify({
            "success": True,
            "low_dose_url": url_for("static", filename=f"uploads/{low_filename}"),
            "high_dose_url": url_for("static", filename=f"uploads/{high_filename}"),
            "full_dose_url": full_dose_url,
            "comparison_url": url_for("static", filename=f"uploads/{compare_filename}"),
            "low_filename": low_filename,
            "high_filename": high_filename,
            "full_filename": full_filename,
            "metrics": metrics,
            "metadata": metadata,
            "dataset_stats": DATASET_STATS,
            "histograms": {"low": hist_low, "high": hist_high},
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        gc.collect()


@app.route("/download/<filename>")
def download_file(filename):
    """
    Download enhanced High-Dose PET image file.
    """
    safe_name = secure_filename(filename)
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        safe_name,
        as_attachment=True,
        download_name=f"Enhanced_HighDose_PET_{safe_name}"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting PET Image Enhancement System on http://127.0.0.1:{port} ...")
    print("Dataset:", DATASET_STATS)
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
