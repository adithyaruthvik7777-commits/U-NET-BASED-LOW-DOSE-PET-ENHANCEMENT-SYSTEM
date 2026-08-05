###### CELL 1



# =====================================================
# STEP 1: IMPORT LIBRARIES
# =====================================================

import os
import numpy as np
import pydicom
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# =====================================================
# STEP 2: SET RANDOM SEED
# =====================================================

np.random.seed(42)
torch.manual_seed(42)

# =====================================================
# STEP 3: DEFINE DATASET PATH (PAIRED LOW / FULL DOSE)
# =====================================================

import zipfile
import shutil
from collections import Counter

VALID_EXTS = (
    ".dcm", ".DCM", ".ima", ".IMA", ".img", ".IMG",
    ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".tif", ".TIF", ".tiff", ".TIFF",
    ".npy", ".NPY", ".npz", ".NPZ",
)

# Dataset: https://www.kaggle.com/datasets/skarthik112/paired-low-dose-dataset-1to50
# Model output: best_unet_pet.pth from notebook adithyapanidepu/notebooka154a97ccb
USER_DATA_ROOT = "/kaggle/input/datasets/skarthik112/paired-low-dose-dataset-1to50"
CANDIDATE_DIRS = [
    os.path.join(USER_DATA_ROOT, "PAIR_LOW_DOSE_HIGH_DOSE"),
    USER_DATA_ROOT,
    "/kaggle/input/paired-low-dose-dataset-1to50/PAIR_LOW_DOSE_HIGH_DOSE",
    "/kaggle/input/paired-low-dose-dataset-1to50",
]

LOW_NAME_CANDIDATES = ("Low_Dose", "low_dose", "LowDose", "LOW_DOSE", "LD", "ld")
HIGH_NAME_CANDIDATES = (
    "Full_Dose", "full_dose", "FullDose", "FULL_DOSE",
    "High_Dose", "high_dose", "HighDose", "HIGH_DOSE", "FD", "fd", "HD", "hd",
)

EXTRACT_DIR = "/kaggle/working/paired_dataset_extracted"


def is_image_file(name):
    """True for known image extensions, or extensionless names (common DICOM)."""
    if name.startswith("."):
        return False
    lower = name.lower()
    if lower.endswith((".zip", ".tar", ".gz", ".7z", ".rar", ".csv", ".txt", ".json", ".md")):
        return False
    ext = os.path.splitext(name)[1]
    if ext == "":
        return True  # likely DICOM without extension
    return name.endswith(VALID_EXTS) or ext.lower() in {
        e.lower() for e in VALID_EXTS
    }


def folder_inventory(folder, max_samples=12):
    """Return (n_files, n_dirs, ext_counts, sample_names) for debugging."""
    n_files, n_dirs = 0, 0
    ext_counts = Counter()
    samples = []
    if not os.path.isdir(folder):
        return 0, 0, ext_counts, samples
    for root, dirs, files in os.walk(folder):
        n_dirs += len(dirs)
        for f in files:
            n_files += 1
            ext = os.path.splitext(f)[1] or "(no_ext)"
            ext_counts[ext] += 1
            if len(samples) < max_samples:
                samples.append(os.path.relpath(os.path.join(root, f), folder))
    return n_files, n_dirs, ext_counts, samples


def list_image_files(folder):
    """Map relative path -> absolute path for image-like files under folder."""
    mapping = {}
    if not os.path.isdir(folder):
        return mapping
    for root, _, files in os.walk(folder):
        for f in files:
            if not is_image_file(f):
                continue
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, folder).replace("\\", "/")
            mapping[rel_path] = abs_path
    return mapping


def find_named_dirs(root, name_candidates):
    """Find all directories whose basename matches any candidate (case-insensitive)."""
    wanted = {n.lower() for n in name_candidates}
    found = []
    if not os.path.isdir(root):
        return found
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower() in wanted:
                found.append(os.path.join(dirpath, d))
    return found


def find_zip_files(root):
    zips = []
    if not os.path.isdir(root):
        return zips
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".zip"):
                zips.append(os.path.join(dirpath, f))
    return zips


def extract_zips_if_needed(root):
    """If image folders are empty but zips exist, extract into /kaggle/working."""
    zips = find_zip_files(root)
    if not zips:
        return None

    print(f"Found {len(zips)} zip archive(s). Extracting to {EXTRACT_DIR} ...")
    if os.path.isdir(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    for zpath in zips:
        print("  Extracting:", zpath)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(EXTRACT_DIR)
    return EXTRACT_DIR


def pick_best_pair(search_roots):
    """Among search roots, choose Low/Full folders with the most paired-looking files."""
    best = None  # (score, data_root, low_dir, high_dir, n_low, n_high)

    for root in search_roots:
        if not os.path.isdir(root):
            continue

        low_dirs = find_named_dirs(root, LOW_NAME_CANDIDATES)
        high_dirs = find_named_dirs(root, HIGH_NAME_CANDIDATES)

        # Also try direct children named Low_Dose / Full_Dose
        for name in LOW_NAME_CANDIDATES:
            p = os.path.join(root, name)
            if os.path.isdir(p) and p not in low_dirs:
                low_dirs.append(p)
        for name in HIGH_NAME_CANDIDATES:
            p = os.path.join(root, name)
            if os.path.isdir(p) and p not in high_dirs:
                high_dirs.append(p)

        for low_dir in low_dirs:
            for high_dir in high_dirs:
                if os.path.abspath(low_dir) == os.path.abspath(high_dir):
                    continue
                n_low = len(list_image_files(low_dir))
                n_high = len(list_image_files(high_dir))
                score = min(n_low, n_high)
                if score <= 0:
                    continue
                if best is None or score > best[0]:
                    best = (score, root, low_dir, high_dir, n_low, n_high)

    return best


def discover_input_roots():
    """List /kaggle/input mounts so we can find the attached dataset."""
    roots = []
    for base in ("/kaggle/input", "/kaggle/input/datasets"):
        if not os.path.isdir(base):
            continue
        try:
            for name in sorted(os.listdir(base)):
                path = os.path.join(base, name)
                if os.path.isdir(path):
                    roots.append(path)
        except OSError:
            pass
    return roots


def resolve_dataset():
    """Locate Low_Dose / Full_Dose folders that actually contain images."""
    search_roots = []
    for cand in CANDIDATE_DIRS:
        if os.path.isdir(cand):
            search_roots.append(cand)

    # Always inspect the user-provided root first
    print("=" * 50)
    print("DATASET PATH PROBE")
    print("=" * 50)
    print("/kaggle/input exists:", os.path.isdir("/kaggle/input"))
    print("Attached input datasets:", discover_input_roots())

    for cand in CANDIDATE_DIRS:
        exists = os.path.isdir(cand)
        print(f"\nCandidate: {cand}")
        print(f"  exists: {exists}")
        if not exists:
            continue
        try:
            top = os.listdir(cand)
        except OSError as e:
            print(f"  listdir error: {e}")
            continue
        print(f"  top-level ({len(top)}): {top[:30]}")
        n_files, n_dirs, ext_counts, samples = folder_inventory(cand)
        print(f"  walk totals: files={n_files}, dirs={n_dirs}")
        print(f"  extensions : {dict(ext_counts.most_common(15))}")
        print(f"  samples    : {samples[:8]}")

    best = pick_best_pair(search_roots)

    # If folders exist but are empty, try extracting zip archives
    if best is None:
        for cand in list(search_roots):
            extracted = extract_zips_if_needed(cand)
            if extracted is not None:
                search_roots.append(extracted)
                best = pick_best_pair(search_roots)
                if best is not None:
                    break

    # Last resort: search every attached Kaggle input dataset
    if best is None:
        extra = discover_input_roots()
        best = pick_best_pair(extra + [EXTRACT_DIR])

    if best is None:
        raise FileNotFoundError(
            "Low_Dose / Full_Dose folders were found empty (0 image files).\n"
            "Your dataset mount exists but contains no readable images under those names.\n\n"
            "Likely causes:\n"
            "  1) Dataset not fully uploaded (empty Low_Dose / Full_Dose stubs)\n"
            "  2) Images are inside archives that could not be extracted\n"
            "  3) Folder names differ from Low_Dose / Full_Dose\n\n"
            "Check the DATASET PATH PROBE printout above for real folder/file names,\n"
            "then set LOW_DIR / HIGH_DIR manually to those paths."
        )

    _, data_dir, low_dir, high_dir, n_low, n_high = best
    print("\nResolved dataset:")
    print("  DATA_DIR :", data_dir)
    print("  LOW_DIR  :", low_dir, f"({n_low} images)")
    print("  HIGH_DIR :", high_dir, f"({n_high} images)")
    return data_dir, low_dir, high_dir


DATA_DIR, LOW_DIR, HIGH_DIR = resolve_dataset()

# =====================================================
# STEP 4: BUILD PAIRED FILE LIST
# =====================================================

low_files_map = list_image_files(LOW_DIR)
high_files_map = list_image_files(HIGH_DIR)

# Prefer full relative-path match; fall back to basename-only match.
common_names = sorted(set(low_files_map.keys()) & set(high_files_map.keys()))

if len(common_names) == 0:
    low_by_base = {}
    for rel, path in low_files_map.items():
        low_by_base.setdefault(os.path.basename(rel), path)
    high_by_base = {}
    for rel, path in high_files_map.items():
        high_by_base.setdefault(os.path.basename(rel), path)
    common_names = sorted(set(low_by_base.keys()) & set(high_by_base.keys()))
    paired_files = [(low_by_base[n], high_by_base[n]) for n in common_names]
else:
    paired_files = [
        (low_files_map[name], high_files_map[name])
        for name in common_names
    ]

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)
print("Dataset Root      :", DATA_DIR)
print("Low Dose Folder   :", LOW_DIR)
print("Full Dose Folder  :", HIGH_DIR)
print("Low Dose Images   :", len(low_files_map))
print("Full Dose Images  :", len(high_files_map))
print("Paired Images     :", len(paired_files))

if len(paired_files) == 0:
    sample_low = list(low_files_map.keys())[:5]
    sample_high = list(high_files_map.keys())[:5]
    raise RuntimeError(
        "No paired images found (filename mismatch or empty folders).\n"
        f"Check folders:\n  {LOW_DIR}\n  {HIGH_DIR}\n"
        f"Sample low names : {sample_low}\n"
        f"Sample high names: {sample_high}"
    )

# =====================================================
# STEP 5: TRAIN / VALIDATION / TEST SPLIT (70/15/15)
# =====================================================

train_pairs, temp_pairs = train_test_split(
    paired_files,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val_pairs, test_pairs = train_test_split(
    temp_pairs,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

print(f"Training Pairs    : {len(train_pairs)}")
print(f"Validation Pairs  : {len(val_pairs)}")
print(f"Testing Pairs     : {len(test_pairs)}")

# =====================================================
# STEP 6: LOAD PET IMAGE
# =====================================================

def load_pet(path):
    """
    Load PET image (DICOM .dcm/.ima, PNG/JPG, NPY) and normalize to [0,1].
    """
    ext = os.path.splitext(path)[1].lower()

    # .IMA is DICOM (Siemens-style extension used by this dataset)
    if ext in (".dcm", ".ima", ".img", ""):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
    elif ext == ".npy":
        img = np.load(path).astype(np.float32)
    elif ext == ".npz":
        img = np.load(path)["arr_0"].astype(np.float32)
    else:
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
        except Exception:
            img = plt.imread(path).astype(np.float32)
            if img.ndim == 3:
                img = img.mean(axis=2)

    img = img - img.min()
    img = img / (img.max() + 1e-8)

    return img

# =====================================================
# STEP 7: VISUALIZE ONE PAIRED SAMPLE
# =====================================================

sample_low_path, sample_high_path = train_pairs[0]

sample_low = load_pet(sample_low_path)
sample_high = load_pet(sample_high_path)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(sample_low, cmap="gray")
plt.title("Paired Low-Dose PET")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sample_high, cmap="gray")
plt.title("Paired Full-Dose PET")
plt.axis("off")

plt.tight_layout()
plt.show()
