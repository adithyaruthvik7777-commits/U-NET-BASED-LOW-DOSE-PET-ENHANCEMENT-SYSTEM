"""
Generate realistic paired Low-Dose / Full-Dose axial PET demo slices
for the web app presets (matches 1:50 dose-reduction look from Kaggle).
"""
import os
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.dataset import FileDataset
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def create_dicom_ima_file(filename, pixel_array, patient_id="SIEMENS_PET_001"):
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "PET^ANONYMOUS"
    ds.PatientID = patient_id
    ds.PatientSex = "M"
    ds.PatientAge = "058Y"
    ds.Modality = "PT"
    ds.Manufacturer = "Siemens Healthineers (Biograph Vision)"
    ds.StationName = "SIEMENS_PET_CT"
    ds.SeriesDescription = "PAIR_LOW_DOSE_HIGH_DOSE_1to50"
    ds.Rows, ds.Columns = pixel_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.WindowCenter = "500"
    ds.WindowWidth = "1000"
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.SliceThickness = "2.0"
    ds.PixelSpacing = ["1.56", "1.56"]

    uint_arr = (np.clip(pixel_array, 0, 1) * 4095).astype(np.uint16)
    ds.PixelData = uint_arr.tobytes()
    ds.save_as(filename)


def _ellipse_mask(y, x, cy, cx, ry, rx):
    return ((x - cx) ** 2) / (rx ** 2 + 1e-8) + ((y - cy) ** 2) / (ry ** 2 + 1e-8) <= 1.0


def generate_axial_pet_phantom(slice_num=25, size=256, seed=None):
    """
    Synthesize an axial torso PET-like activity map (soft tissue + lungs + spine + arms).
    Returns (full_dose_clean, low_dose_noisy) in [0, 1].
    """
    rng = np.random.default_rng(seed if seed is not None else slice_num + 42)
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2

    # Body outline
    body = _ellipse_mask(y, x, cy, cx, 105, 78)
    # Arms
    arm_l = _ellipse_mask(y, x, cy, 28, 42, 18)
    arm_r = _ellipse_mask(y, x, cy, size - 28, 42, 18)

    soft = np.zeros((size, size), dtype=np.float32)
    soft[body] = 0.28
    soft[arm_l] = 0.22
    soft[arm_r] = 0.22

    # Lungs (lower activity)
    lung_l = _ellipse_mask(y, x, cy - 8, cx - 28, 38, 22)
    lung_r = _ellipse_mask(y, x, cy - 8, cx + 28, 38, 22)
    soft[lung_l] = 0.06
    soft[lung_r] = 0.06

    # Mediastinum / heart
    heart = _ellipse_mask(y, x, cy + 5, cx + 6, 22, 16)
    soft[heart] = 0.55

    # Spine
    spine = _ellipse_mask(y, x, cy + 18, cx, 18, 10)
    soft[spine] = 0.72

    # FDG hotspots (vary by slice)
    hot1 = np.exp(-((x - (cx - 20 + slice_num % 7)) ** 2 + (y - (cy - 30)) ** 2) / (2 * 10 ** 2)) * 0.95
    hot2 = np.exp(-((x - (cx + 25)) ** 2 + (y - (cy + 35 + slice_num % 5)) ** 2) / (2 * 8 ** 2)) * 0.8
    soft = soft + hot1 + hot2
    soft = soft * (body | arm_l | arm_r)
    soft = np.clip(soft, 0.0, 1.0)

    # Mild spatial blur for full-dose smoothness
    try:
        from scipy.ndimage import gaussian_filter
        full = gaussian_filter(soft, sigma=0.8)
    except ImportError:
        full = soft.copy()
    full = np.clip(full, 0.0, 1.0)

    # 1:50 low-dose Poisson noise (very grainy, matches Kaggle low-dose look)
    count_scale = 6.0 + (slice_num % 5) * 0.4
    lam = np.clip(full * count_scale, 0.0, None)
    low = rng.poisson(lam).astype(np.float32) / count_scale
    # Extra Gaussian readout noise
    low = low + rng.normal(0.0, 0.035, size=low.shape).astype(np.float32)
    low = np.clip(low, 0.0, 1.0)
    low = low * (body | arm_l | arm_r)

    return full.astype(np.float32), low.astype(np.float32)


def _save_pair(out_dir, stem, full, low, patient_id):
    low_ima = os.path.join(out_dir, f"{stem}.ima")
    full_ima = os.path.join(out_dir, f"{stem}_full.ima")
    low_npy = os.path.join(out_dir, f"{stem}.npy")
    full_npy = os.path.join(out_dir, f"{stem}_full.npy")
    low_png = os.path.join(out_dir, f"{stem}.png")
    full_png = os.path.join(out_dir, f"{stem}_full.png")

    np.save(low_npy, low)
    np.save(full_npy, full)
    Image.fromarray((low * 255).astype(np.uint8)).save(low_png)
    Image.fromarray((full * 255).astype(np.uint8)).save(full_png)

    if HAS_PYDICOM:
        create_dicom_ima_file(low_ima, low, patient_id=patient_id)
        create_dicom_ima_file(full_ima, full, patient_id=patient_id + "_FD")
    else:
        Image.fromarray((low * 255).astype(np.uint8)).save(low_ima.replace(".ima", "_fallback.png"))


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "static", "sample_images")
    os.makedirs(out_dir, exist_ok=True)

    samples_info = [
        ("brain_slice_12", 12, "Brain/Upper PET"),
        ("thorax_slice_25", 25, "Thoracic PET"),
        ("abdomen_slice_38", 38, "Abdominal PET"),
        ("pelvis_slice_50", 50, "Pelvic PET"),
    ]

    for stem, slice_idx, desc in samples_info:
        full, low = generate_axial_pet_phantom(slice_idx, size=256, seed=slice_idx * 17)
        _save_pair(out_dir, stem, full, low, patient_id=f"SIEMENS_PET_SLICE_{slice_idx}")
        print(f"[OK] {desc}: {stem}.ima + {stem}_full.ima")

    # Default fallback used by older sample_id paths
    full, low = generate_axial_pet_phantom(25, size=256, seed=99)
    np.save(os.path.join(out_dir, "sample_low_dose_pet.npy"), low)
    Image.fromarray((low * 255).astype(np.uint8)).save(os.path.join(out_dir, "sample_low_dose_pet.png"))
    np.save(os.path.join(out_dir, "sample_low_dose_pet_full.npy"), full)
    Image.fromarray((full * 255).astype(np.uint8)).save(os.path.join(out_dir, "sample_low_dose_pet_full.png"))

    print("[SUCCESS] Generated paired Low/Full-Dose Siemens .IMA preset samples.")
