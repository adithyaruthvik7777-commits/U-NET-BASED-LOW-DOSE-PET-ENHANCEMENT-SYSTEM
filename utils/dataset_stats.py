"""Lightweight dataset constants — safe to import at app startup (no torch/cv2)."""

DATASET_STATS = {
    "low_dose_images": 9006,
    "full_dose_images": 9006,
    "paired_images": 9006,
    "training_pairs": 6304,
    "validation_pairs": 1351,
    "testing_pairs": 1351,
    "dataset_url": "https://www.kaggle.com/datasets/skarthik112/paired-low-dose-dataset-1to50",
    "model_kernel": "adithyapanidepu/notebooka154a97ccb",
    "mean_ssim": 0.9320,
    "mean_ssim_pct": 93.2,
    "mean_psnr_db": 30.05,
}
