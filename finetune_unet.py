"""
Fine-tune best_unet_pet.pth on paired low/full-dose PET to maximize SSIM/PSNR.
Stage-2 defaults: larger subset, SSIM-heavy loss, SWA, cosine LR.
"""
from __future__ import annotations

import argparse
import copy
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from unet_model import UNet, get_device
from download_kaggle_dataset import (
    find_existing_dataset,
    resolve_pair_dirs,
    build_pairs,
    load_pet_array,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_IN = BASE_DIR / "models" / "best_unet_pet.pth"
MODEL_OUT = BASE_DIR / "models" / "best_unet_pet.pth"
BACKUP = BASE_DIR / "models" / "best_unet_pet_before_finetune.pth"
STAGE_BACKUP = BASE_DIR / "models" / "best_unet_pet_stage1.pth"

# Defaults tuned for further SSIM gains on CPU
NUM_TRAIN = 800
NUM_VAL = 120
EPOCHS = 6
BATCH_SIZE = 2
LR = 1.5e-5
SEED = 42
MAX_SIDE = 256


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PairDataset(Dataset):
    def __init__(self, pairs, max_side=MAX_SIDE, augment=False):
        self.pairs = pairs
        self.max_side = max_side
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        import cv2

        low_path, high_path = self.pairs[idx]
        low = load_pet_array(Path(low_path))
        high = load_pet_array(Path(high_path))
        if low.shape[0] != self.max_side or low.shape[1] != self.max_side:
            low = cv2.resize(low, (self.max_side, self.max_side), interpolation=cv2.INTER_AREA)
            high = cv2.resize(high, (self.max_side, self.max_side), interpolation=cv2.INTER_AREA)

        if self.augment:
            if random.random() < 0.5:
                low = np.flip(low, axis=1).copy()
                high = np.flip(high, axis=1).copy()
            if random.random() < 0.5:
                low = np.flip(low, axis=0).copy()
                high = np.flip(high, axis=0).copy()
            if random.random() < 0.25:
                k = random.choice([1, 2, 3])
                low = np.rot90(low, k).copy()
                high = np.rot90(high, k).copy()

        low_t = torch.as_tensor(low, dtype=torch.float32).unsqueeze(0)
        high_t = torch.as_tensor(high, dtype=torch.float32).unsqueeze(0)
        return low_t, high_t


def ssim_loss(pred, target, C1=0.01 ** 2, C2=0.03 ** 2):
    """Differentiable SSIM loss (1 - SSIM), window≈11 via avg pools."""
    mu_x = F.avg_pool2d(pred, 11, 1, 5)
    mu_y = F.avg_pool2d(target, 11, 1, 5)
    sigma_x = F.avg_pool2d(pred * pred, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 11, 1, 5) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-8
    )
    return 1.0 - ssim_map.clamp(-1, 1).mean()


def edge_loss(pred, target):
    """Encourage matching gradients (lesion / organ boundaries)."""
    kx = pred.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = pred.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    px = F.conv2d(pred, kx, padding=1)
    py = F.conv2d(pred, ky, padding=1)
    tx = F.conv2d(target, kx, padding=1)
    ty = F.conv2d(target, ky, padding=1)
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def combined_loss(pred, target):
    # SSIM-heavy for structural accuracy; light edge term for sharpness
    return 0.30 * F.l1_loss(pred, target) + 0.55 * ssim_loss(pred, target) + 0.15 * edge_loss(pred, target)


@torch.no_grad()
def eval_metrics(model, loader, device):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    model.eval()
    psnrs, ssims = [], []
    for low, high in loader:
        low, high = low.to(device), high.to(device)
        pred = torch.clamp(model(low), 0, 1)
        for i in range(pred.shape[0]):
            p = pred[i, 0].cpu().numpy()
            g = high[i, 0].cpu().numpy()
            psnrs.append(peak_signal_noise_ratio(g, p, data_range=1.0))
            ssims.append(structural_similarity(g, p, data_range=1.0, win_size=7))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def average_state_dicts(dicts):
    avg = copy.deepcopy(dicts[0])
    for k in avg:
        stacked = torch.stack([d[k].float() for d in dicts], dim=0)
        avg[k] = stacked.mean(dim=0)
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=NUM_TRAIN)
    parser.add_argument("--val", type=int, default=NUM_VAL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print("Device:", device)

    data_root = find_existing_dataset()
    if data_root is None:
        raise FileNotFoundError("Dataset not found. Run download_kaggle_dataset.py first.")
    if (Path(data_root) / "1").is_dir():
        data_root = Path(data_root) / "1"
    low_dir, high_dir = resolve_pair_dirs(Path(data_root))
    pairs = build_pairs(low_dir, high_dir)
    random.shuffle(pairs)
    # Fixed split after shuffle: val first so train growth does not reshuffle val
    val_pairs = pairs[: args.val]
    train_pairs = pairs[args.val : args.val + args.train]
    print(f"Fine-tune train={len(train_pairs)} val={len(val_pairs)} of {len(pairs)} total")

    train_loader = DataLoader(
        PairDataset(train_pairs, max_side=args.max_side, augment=True),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        PairDataset(val_pairs, max_side=args.max_side, augment=False),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )

    model = UNet().to(device)
    if MODEL_IN.exists():
        state = torch.load(MODEL_IN, map_location=device)
        model.load_state_dict(state)
        print("Loaded", MODEL_IN)
        if not BACKUP.exists():
            shutil.copy2(MODEL_IN, BACKUP)
            print("Backup saved:", BACKUP)
        shutil.copy2(MODEL_IN, STAGE_BACKUP)
        print("Stage checkpoint:", STAGE_BACKUP)
    else:
        print("[WARN] No pretrained weights; training from scratch")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_ssim = -1.0
    recent_states = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for low, high in bar:
            low, high = low.to(device), high.to(device)
            opt.zero_grad()
            pred = model(low)
            loss = combined_loss(pred, high)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()

        psnr, ssim = eval_metrics(model, val_loader, device)
        print(
            f"Epoch {epoch}: train_loss={running / len(train_loader):.4f}  "
            f"val_PSNR={psnr:.2f}  val_SSIM={ssim:.4f} ({ssim * 100:.1f}%)"
        )
        recent_states.append(copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))
        if len(recent_states) > 3:
            recent_states.pop(0)

        if ssim > best_ssim:
            best_ssim = ssim
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  Saved best model -> {MODEL_OUT} (SSIM={ssim:.4f})")

    # SWA over last up-to-3 epochs; keep if it beats best single checkpoint
    if len(recent_states) >= 2:
        swa = average_state_dicts(recent_states)
        model.load_state_dict(swa)
        model.to(device)
        psnr, ssim = eval_metrics(model, val_loader, device)
        print(f"SWA: val_PSNR={psnr:.2f}  val_SSIM={ssim:.4f} ({ssim * 100:.1f}%)")
        if ssim > best_ssim:
            best_ssim = ssim
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  Saved SWA model -> {MODEL_OUT} (SSIM={ssim:.4f})")

    print("Done. Best val SSIM:", best_ssim)


if __name__ == "__main__":
    main()
