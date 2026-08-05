"""
High-quality PET enhancement inference:
- Test-time augmentation (flips + mild scale averaging)
- Optional residual blend and NLM refine (refine off by default; hurts SSIM)
"""
from __future__ import annotations

import os

import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None

import cv2

from utils.image_processing import preprocess_for_unet, postprocess_unet_output


def _flip_np(arr, mode):
    if mode == "h":
        return np.flip(arr, axis=1).copy()
    if mode == "v":
        return np.flip(arr, axis=0).copy()
    if mode == "hv":
        return np.flip(np.flip(arr, axis=0), axis=1).copy()
    return arr


def _unflip_np(arr, mode):
    return _flip_np(arr, mode)  # flips are involutions


def predict_single(model, device, low_arr):
    # On Render/CPU: force 256² to avoid OOM / empty gateway responses
    target = (256, 256) if os.environ.get("FAST_INFERENCE", "0") == "1" else None
    input_tensor, low_work, pad_hw = preprocess_for_unet(low_arr, target_size=target)
    orig_hw = (low_work.shape[0], low_work.shape[1])
    if HAS_TORCH and hasattr(input_tensor, "to"):
        input_tensor = input_tensor.to(device)
        with torch.inference_mode():
            output_tensor = model(input_tensor)
    else:
        output_tensor = model(input_tensor)
    raw = postprocess_unet_output(
        output_tensor,
        orig_shape=orig_hw,
        low_arr_input=low_work,
        pad_hw=pad_hw,
        refine=False,
    )
    return raw.astype(np.float32), low_work


def predict_with_tta(model, device, low_arr, use_scales=True):
    """
    Average U-Net predictions over flips (+ optional mild scale TTA).
    Empirically scale TTA gives a small SSIM/PSNR bump vs flips alone.
    """
    modes = ["id", "h", "v", "hv"]
    preds = []
    low_work = None
    for mode in modes:
        inp = low_arr if mode == "id" else _flip_np(low_arr, mode)
        pred, work = predict_single(model, device, inp)
        if mode != "id":
            pred = _unflip_np(pred, mode)
        preds.append(pred)
        if low_work is None:
            low_work = work if mode == "id" else low_arr.astype(np.float32)

    if use_scales:
        h, w = low_arr.shape[:2]
        for scale in (0.9, 1.1):
            rs = cv2.resize(low_arr, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
            pred, _ = predict_single(model, device, rs)
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            preds.append(pred)

    stacked = np.stack(preds, axis=0)
    return np.clip(stacked.mean(axis=0), 0.0, 1.0).astype(np.float32), low_work


def residual_blend(enhanced_arr, low_arr, alpha=0.95):
    """Slight residual blend toward low-dose; alpha~0.95 improves PSNR with similar SSIM."""
    low = low_arr.astype(np.float32)
    if low.shape != enhanced_arr.shape:
        low = cv2.resize(low, (enhanced_arr.shape[1], enhanced_arr.shape[0]), interpolation=cv2.INTER_AREA)
    return np.clip(low + alpha * (enhanced_arr.astype(np.float32) - low), 0.0, 1.0).astype(np.float32)


def adaptive_refine(enhanced_arr, low_arr=None):
    """
    Multi-hypothesis refine tuned for max SSIM vs full-dose PET.
    Best grid result on demo pairs: NLM h≈16–20 with ~80% denoise blend.
    """
    base = np.clip(enhanced_arr.astype(np.float32), 0.0, 1.0)
    candidates = []

    for strength, blend in ((16, 0.80), (20, 0.80), (18, 0.70), (24, 0.75), (12, 0.65)):
        img_u8 = (base * 255.0).astype(np.uint8)
        try:
            den = cv2.fastNlMeansDenoising(
                img_u8, None, h=int(strength), templateWindowSize=7, searchWindowSize=21
            )
            den = cv2.bilateralFilter(den, d=7, sigmaColor=45, sigmaSpace=45)
        except Exception:
            den = cv2.bilateralFilter(img_u8, d=9, sigmaColor=55, sigmaSpace=55)
        den = den.astype(np.float32) / 255.0
        blur = cv2.GaussianBlur((den * 255).astype(np.uint8), (0, 0), 0.8).astype(np.float32) / 255.0
        den = np.clip(den + 0.15 * (den - blur), 0.0, 1.0)
        candidates.append(np.clip((1.0 - blend) * base + blend * den, 0.0, 1.0))

    def score(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge = float(np.mean(np.sqrt(gx * gx + gy * gy)))
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        noise = float(np.std(img - blur))
        return edge / (noise + 1e-6)

    best = max(candidates, key=score)
    return best.astype(np.float32)


def enhance_pet(model, device, low_arr, use_tta=None, use_refine=False, residual_alpha=0.95, use_scales=None):
    """
    Enhancement pipeline. On Render/CPU hosts set FAST_INFERENCE=1 for a single
    forward pass (avoids timeouts). Otherwise flip TTA; scale TTA only when not fast.
    """
    fast = os.environ.get("FAST_INFERENCE", "0") == "1"
    if use_tta is None:
        use_tta = not fast
    if use_scales is None:
        use_scales = not fast

    if use_tta:
        raw, low_work = predict_with_tta(model, device, low_arr, use_scales=use_scales)
    else:
        raw, low_work = predict_single(model, device, low_arr)

    enhanced = residual_blend(raw, low_work, alpha=residual_alpha) if residual_alpha is not None else raw
    if use_refine:
        enhanced = adaptive_refine(enhanced, low_arr=low_work)
    return enhanced, low_work
