"""
Download the paired Low/Full-Dose PET dataset and export demo pairs for the web app.

Dataset:
  https://www.kaggle.com/datasets/skarthik112/paired-low-dose-dataset-1to50

Trained weights (Kaggle notebook output):
  kaggle kernels output adithyapanidepu/notebooka154a97ccb -p ./models
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "static" / "sample_images"
MODEL_DIR = BASE_DIR / "models"
DATA_CACHE_HINT = Path.home() / ".cache" / "kagglehub" / "datasets" / "skarthik112" / "paired-low-dose-dataset-1to50"

DATASET_SLUG = "skarthik112/paired-low-dose-dataset-1to50"
KERNEL_SLUG = "adithyapanidepu/notebooka154a97ccb"

LOW_NAMES = ("Low_Dose", "low_dose", "LowDose", "LOW_DOSE", "LD", "ld")
HIGH_NAMES = (
    "Full_Dose", "full_dose", "FullDose", "FULL_DOSE",
    "High_Dose", "high_dose", "HighDose", "HIGH_DOSE", "FD", "fd", "HD", "hd",
)

VALID_EXTS = {".dcm", ".ima", ".img", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".npz", ""}


def download_dataset() -> Path:
    import kagglehub
    print(f"Downloading dataset: {DATASET_SLUG}")
    path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"Dataset path: {path}")
    return path


def download_model_weights() -> Path | None:
    """Download best_unet_pet.pth from the Kaggle notebook output if possible."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / "best_unet_pet.pth"
    if dest.exists():
        print(f"Using existing local weights: {dest} ({dest.stat().st_size} bytes)")
        return dest
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading kernel output: {KERNEL_SLUG}")
        api.kernels_output(KERNEL_SLUG, path=str(MODEL_DIR), quiet=False)
        candidates = list(MODEL_DIR.rglob("best_unet_pet.pth"))
        if candidates:
            best = max(candidates, key=lambda p: p.stat().st_size)
            if best.resolve() != dest.resolve():
                shutil.copy2(best, dest)
            print(f"Model saved: {dest} ({dest.stat().st_size} bytes)")
            return dest
    except Exception as e:
        print(f"[WARN] Could not download kernel output via API: {e}")
        print("Place best_unet_pet.pth manually into models/ (from Kaggle Output tab).")
    return dest if dest.exists() else None


def find_named_dirs(root: Path, names: tuple[str, ...]) -> list[Path]:
    wanted = {n.lower() for n in names}
    found = []
    if not root.is_dir():
        return found
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower() in wanted:
                found.append(Path(dirpath) / d)
    return found


def list_images(folder: Path) -> dict[str, Path]:
    mapping = {}
    if not folder.is_dir():
        return mapping
    for root, _, files in os.walk(folder):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext not in VALID_EXTS and ext != "":
                continue
            if f.startswith("."):
                continue
            abs_path = Path(root) / f
            rel = abs_path.relative_to(folder).as_posix()
            mapping[rel] = abs_path
    return mapping


def resolve_pair_dirs(data_root: Path) -> tuple[Path, Path]:
    low_dirs = find_named_dirs(data_root, LOW_NAMES)
    high_dirs = find_named_dirs(data_root, HIGH_NAMES)
    best = None
    for low_dir in low_dirs:
        for high_dir in high_dirs:
            if low_dir.resolve() == high_dir.resolve():
                continue
            n = min(len(list_images(low_dir)), len(list_images(high_dir)))
            if best is None or n > best[0]:
                best = (n, low_dir, high_dir)
    if best is None or best[0] == 0:
        raise FileNotFoundError(
            f"Could not find Low_Dose / Full_Dose image folders under {data_root}"
        )
    print(f"Low : {best[1]} ({best[0]}+ paired)")
    print(f"Full: {best[2]}")
    return best[1], best[2]


def build_pairs(low_dir: Path, high_dir: Path) -> list[tuple[Path, Path]]:
    low_map = list_images(low_dir)
    high_map = list_images(high_dir)
    common = sorted(set(low_map) & set(high_map))
    if common:
        return [(low_map[n], high_map[n]) for n in common]
    # basename fallback
    low_b = {}
    high_b = {}
    for rel, p in low_map.items():
        low_b.setdefault(Path(rel).name, p)
    for rel, p in high_map.items():
        high_b.setdefault(Path(rel).name, p)
    names = sorted(set(low_b) & set(high_b))
    return [(low_b[n], high_b[n]) for n in names]


def load_pet_array(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in (".dcm", ".ima", ".img", ""):
        import pydicom
        ds = pydicom.dcmread(str(path), force=True)
        img = ds.pixel_array.astype(np.float32)
    elif ext == ".npy":
        img = np.load(path).astype(np.float32)
    elif ext == ".npz":
        z = np.load(path)
        img = z[z.files[0]].astype(np.float32)
    else:
        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    # Exact Kaggle Cell1 normalization
    img = img - float(img.min())
    img = img / (float(img.max()) + 1e-8)
    return img.astype(np.float32)


def export_demo_pairs(pairs: list[tuple[Path, Path]], n: int = 4) -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    stems = [
        "brain_slice_12",
        "thorax_slice_25",
        "abdomen_slice_38",
        "pelvis_slice_50",
    ]

    # Prefer pairs where low-dose is clearly noisier than full-dose (better demos)
    scored = []
    step = max(1, len(pairs) // 80)
    for i in range(0, len(pairs), step):
        low_path, high_path = pairs[i]
        try:
            low = load_pet_array(low_path)
            full = load_pet_array(high_path)
            mask = low > 0.02
            if mask.sum() < 200:
                continue
            from scipy.ndimage import uniform_filter
            def _ls(img):
                m = uniform_filter(img.astype(np.float64), 7)
                m2 = uniform_filter(img.astype(np.float64) ** 2, 7)
                return np.sqrt(np.maximum(m2 - m ** 2, 0))
            ratio = float(_ls(low)[mask].mean() / (_ls(full)[mask].mean() + 1e-8))
            scored.append((ratio, i, low_path, high_path, low, full))
        except Exception:
            continue
    scored.sort(reverse=True, key=lambda x: x[0])
    # Prefer moderate noise gap (same anatomy as Kaggle screenshots; avoid empty/outlier slices)
    preferred = [s for s in scored if 1.25 <= s[0] <= 3.0]
    if len(preferred) >= n:
        chosen_meta = preferred[:n]
    elif len(scored) >= n:
        chosen_meta = scored[:n]
    else:
        idxs = np.linspace(0, len(pairs) - 1, n, dtype=int)
        chosen_meta = []
        for i in idxs:
            lp, hp = pairs[int(i)]
            chosen_meta.append((1.0, int(i), lp, hp, load_pet_array(lp), load_pet_array(hp)))

    try:
        from create_sample_pet_images import create_dicom_ima_file, HAS_PYDICOM
    except Exception:
        HAS_PYDICOM = False
        create_dicom_ima_file = None

    for stem, (_ratio, _i, low_path, high_path, low, full) in zip(stems, chosen_meta):
        print(f"Exporting {stem} (noise ratio~{_ratio:.2f}) from:\n  LOW : {low_path}\n  FULL: {high_path}")

        np.save(SAMPLE_DIR / f"{stem}.npy", low)
        np.save(SAMPLE_DIR / f"{stem}_full.npy", full)

        import cv2
        low256 = cv2.resize(low, (256, 256), interpolation=cv2.INTER_AREA)
        full256 = cv2.resize(full, (256, 256), interpolation=cv2.INTER_AREA)
        Image.fromarray((low256 * 255).astype(np.uint8)).save(SAMPLE_DIR / f"{stem}.png")
        Image.fromarray((full256 * 255).astype(np.uint8)).save(SAMPLE_DIR / f"{stem}_full.png")

        if HAS_PYDICOM and create_dicom_ima_file is not None:
            create_dicom_ima_file(str(SAMPLE_DIR / f"{stem}.ima"), low256, patient_id=stem.upper())
            create_dicom_ima_file(str(SAMPLE_DIR / f"{stem}_full.ima"), full256, patient_id=stem.upper() + "_FD")

        for label, src in (("low_raw", low_path), ("full_raw", high_path)):
            if src.suffix.lower() in (".ima", ".dcm", ".img") and src.stat().st_size < 8_000_000:
                shutil.copy2(src, SAMPLE_DIR / f"{stem}_{label}{src.suffix.lower()}")

    # Default fallback sample
    if chosen_meta:
        low = chosen_meta[0][4]
        import cv2
        low256 = cv2.resize(low, (256, 256), interpolation=cv2.INTER_AREA)
        np.save(SAMPLE_DIR / "sample_low_dose_pet.npy", low256)
        Image.fromarray((low256 * 255).astype(np.uint8)).save(SAMPLE_DIR / "sample_low_dose_pet.png")
        full = chosen_meta[0][5]
        full256 = cv2.resize(full, (256, 256), interpolation=cv2.INTER_AREA)
        np.save(SAMPLE_DIR / "sample_low_dose_pet_full.npy", full256)
        Image.fromarray((full256 * 255).astype(np.uint8)).save(SAMPLE_DIR / "sample_low_dose_pet_full.png")

    print(f"[SUCCESS] Exported {len(chosen_meta)} real dataset pairs into {SAMPLE_DIR}")


def find_existing_dataset() -> Path | None:
    env = os.environ.get("PET_DATASET_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    local = BASE_DIR / "data" / "paired-low-dose-dataset-1to50"
    if local.is_dir():
        return local
    pointer = BASE_DIR / "data" / "dataset_root.txt"
    if pointer.exists():
        p = Path(pointer.read_text(encoding="utf-8").strip())
        if p.is_dir():
            # kagglehub may point at .../versions — prefer .../versions/1
            if (p / "1").is_dir():
                return p / "1"
            return p
    if DATA_CACHE_HINT.exists():
        versions = sorted([p for p in DATA_CACHE_HINT.iterdir() if p.is_dir()], reverse=True)
        for v in versions:
            if (v / "PAIR_LOW_DOSE_HIGH_DOSE").is_dir() or list(v.glob("**/Low_Dose")):
                return v
            if (v / "1").is_dir():
                return v / "1"
    return None


def main():
    print("=" * 60)
    print("Paired Low-Dose / Full-Dose PET (1:50)")
    print(f"Dataset : https://www.kaggle.com/datasets/{DATASET_SLUG}")
    print(f"Weights : kaggle kernels output {KERNEL_SLUG}")
    print("=" * 60)

    download_model_weights()

    data_root = find_existing_dataset()
    if data_root is None:
        data_root = download_dataset()
    else:
        print(f"Using existing dataset at: {data_root}")

    # If kagglehub left an archive, try download again (extracts)
    if not list(data_root.rglob("*")) or (
        data_root.is_file() or (data_root / "1.archive").exists()
    ):
        try:
            data_root = download_dataset()
        except Exception as e:
            print(f"[WARN] re-download: {e}")

    low_dir, high_dir = resolve_pair_dirs(data_root)
    pairs = build_pairs(low_dir, high_dir)
    print(f"Paired images found: {len(pairs)}")
    if not pairs:
        raise RuntimeError("No paired Low/Full images found.")

    export_demo_pairs(pairs, n=4)

    # Write pointer file for the app
    pointer = BASE_DIR / "data" / "dataset_root.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(data_root.resolve()), encoding="utf-8")
    print(f"Wrote dataset pointer: {pointer}")


if __name__ == "__main__":
    main()
