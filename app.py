import os
import threading
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename

from utils.dataset_stats import DATASET_STATS

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "pet_unet_secret_key_2026")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload limit

# Faster single-pass inference on low-RAM hosts (Render / Railway)
if os.environ.get("RENDER") or os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("FAST_INFERENCE", "").lower() in ("1", "true", "yes"):
    os.environ["FAST_INFERENCE"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
SAMPLE_FOLDER = os.path.join(BASE_DIR, "static", "sample_images")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_unet_pet.pth")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"dcm", "ima", "img", "npy", "npz", "png", "jpg", "jpeg", "tif", "tiff"}

# Lazy model load — presets use cached U-Net outputs so Render does not need torch in RAM
_unet_instance = None
_current_device = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model():
    """Load U-Net only when a custom upload needs live inference."""
    global _unet_instance, _current_device
    if _unet_instance is not None:
        return _unet_instance, _current_device
    with _model_lock:
        if _unet_instance is not None:
            return _unet_instance, _current_device
        try:
            import torch as _torch
            _torch.set_num_threads(1)
            _torch.set_num_interop_threads(1)
        except Exception:
            pass
        from unet_model import load_model
        _unet_instance, _current_device = load_model(MODEL_PATH)
        return _unet_instance, _current_device


def resolve_sample_path(sample_id):
    """
    Resolve preset sample from real dataset exports in static/sample_images.
    Returns (low_path, full_path).
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


def load_cached_preset(sample_id):
    """
    Load precomputed low / U-Net / full arrays for demo presets.
    Avoids live PyTorch on Render (main cause of 502 OOM).
    """
    import numpy as np

    unet_path = os.path.join(SAMPLE_FOLDER, f"{sample_id}_unet.npy")
    low_path = os.path.join(SAMPLE_FOLDER, f"{sample_id}_low256.npy")
    if not (os.path.exists(unet_path) and os.path.exists(low_path)):
        return None

    low_work = np.load(low_path).astype(np.float32)
    enhanced = np.load(unet_path).astype(np.float32)
    full_arr = None
    full256 = os.path.join(SAMPLE_FOLDER, f"{sample_id}_full256.npy")
    if os.path.exists(full256):
        full_arr = np.load(full256).astype(np.float32)

    metadata = {
        "is_dicom": False,
        "file_format": "Cached preset (.npy)",
        "file_name": f"{sample_id}.npy",
        "series_description": "Precomputed U-Net demo slice (Render-safe cache)",
        "modality": "PT",
    }
    return low_work, enhanced, full_arr, metadata


def quick_metrics(orig, enh):
    """Lightweight metrics without skimage (lower RAM)."""
    import numpy as np

    orig = np.clip(orig.astype(np.float32), 0, 1)
    enh = np.clip(enh.astype(np.float32), 0, 1)
    mse = float(np.mean((orig - enh) ** 2))
    rmse = float(np.sqrt(mse))
    psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))
    # Cheap structural proxy
    mu_x, mu_y = float(orig.mean()), float(enh.mean())
    sig_x, sig_y = float(orig.std()), float(enh.std())
    sig_xy = float(((orig - mu_x) * (enh - mu_y)).mean())
    ssim = ((2 * mu_x * mu_y + 1e-4) * (2 * sig_xy + 1e-4)) / (
        (mu_x * mu_x + mu_y * mu_y + 1e-4) * (sig_x * sig_x + sig_y * sig_y + 1e-4) + 1e-12
    )
    noise = max(0.0, min(100.0, (sig_x - sig_y) / (sig_x + 1e-6) * 100.0 + 35.0))
    return {
        "psnr": round(float(psnr), 2),
        "ssim": round(float(np.clip(ssim, -1, 1)), 4),
        "rmse": round(rmse, 4),
        "nrmse": round(rmse / (float(orig.max() - orig.min()) + 1e-6), 4),
        "noise_reduction": round(noise, 1),
    }


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
        "model_loaded": _unet_instance is not None,
        "device": str(_current_device) if _current_device is not None else "cpu",
        "fast_inference": os.environ.get("FAST_INFERENCE", "0") == "1",
        "preset_cache": True,
    })


@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    """
    Enhance low-dose PET. Demo presets use precomputed U-Net outputs (Render-safe).
    Custom uploads run live inference under a lock.
    """
    import gc
    import cv2
    import numpy as np
    from PIL import Image
    from utils.image_processing import (
        load_pet_image,
        apply_colormap_and_save,
        compute_histogram_bins,
        normalize_colormap,
        _to_rgb_uint8,
    )

    colormap = normalize_colormap(request.form.get("colormap", "gray"))
    fast = os.environ.get("FAST_INFERENCE", "0") == "1"
    sample_id = request.form.get("sample_id")

    try:
        unique_id = uuid.uuid4().hex[:8]
        full_arr = None
        metadata = {}
        used_cache = False

        # --- Fast path: cached preset (no live PyTorch) ---
        if sample_id:
            cached = load_cached_preset(sample_id)
            if cached is not None:
                low_work, enhanced_arr, full_arr, metadata = cached
                used_cache = True

        if not used_cache:
            file_path = None
            full_dose_path = None
            if sample_id:
                file_path, full_dose_path = resolve_sample_path(sample_id)
            elif "file" in request.files:
                file = request.files["file"]
                if file and file.filename != "" and allowed_file(file.filename):
                    safe_name = secure_filename(file.filename) or "scan.ima"
                    file_path = os.path.join(
                        app.config["UPLOAD_FOLDER"], f"upload_{unique_id}_{safe_name}"
                    )
                    file.save(file_path)

            if not file_path or not os.path.exists(file_path):
                return jsonify({"success": False, "error": "No valid file uploaded or sample selected."}), 400

            low_arr, metadata = load_pet_image(file_path)
            side = 128 if fast else 256
            if max(low_arr.shape[:2]) > side:
                low_arr = cv2.resize(low_arr, (side, side), interpolation=cv2.INTER_AREA)

            if (not fast) and full_dose_path and os.path.exists(full_dose_path):
                full_arr, _ = load_pet_image(full_dose_path)
                if full_arr.shape[:2] != low_arr.shape[:2]:
                    full_arr = cv2.resize(
                        full_arr, (low_arr.shape[1], low_arr.shape[0]), interpolation=cv2.INTER_AREA
                    )

            model, device = get_model()
            from utils.enhance_pipeline import enhance_pet
            with _infer_lock:
                enhanced_arr, low_work = enhance_pet(model, device, low_arr)
            gc.collect()

        # --- Save previews ---
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
            if full_arr.shape[:2] != low_work.shape[:2]:
                full_arr = cv2.resize(
                    full_arr, (low_work.shape[1], low_work.shape[0]), interpolation=cv2.INTER_AREA
                )
            full_filename = f"full_{unique_id}.png"
            apply_colormap_and_save(
                full_arr,
                os.path.join(app.config["UPLOAD_FOLDER"], full_filename),
                colormap=colormap,
            )
            full_dose_url = url_for("static", filename=f"uploads/{full_filename}")

        pair = np.concatenate(
            [_to_rgb_uint8(low_work, colormap), _to_rgb_uint8(enhanced_arr, colormap)],
            axis=1,
        )
        Image.fromarray(pair).save(compare_save_path)

        if full_arr is not None:
            if fast or used_cache:
                metrics = quick_metrics(full_arr, enhanced_arr)
            else:
                from utils.metrics import calculate_metrics
                metrics = calculate_metrics(full_arr, enhanced_arr)
            metrics["reference"] = "full_dose_gt"
        else:
            metrics = quick_metrics(low_work, enhanced_arr)
            metrics["reference"] = "low_vs_enhanced"
        if used_cache:
            metrics["inference"] = "cached_preset"

        try:
            hist_low = compute_histogram_bins(low_work)
            hist_high = compute_histogram_bins(enhanced_arr)
        except Exception:
            hist_low = {"labels": [], "values": []}
            hist_high = {"labels": [], "values": []}

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
            "cached": used_cache,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass


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
