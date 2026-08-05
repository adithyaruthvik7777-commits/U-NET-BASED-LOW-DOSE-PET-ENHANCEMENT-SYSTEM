# =====================================================
# CELL 5: LOAD MODEL + EVALUATE U-NET
# =====================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model_path = "/kaggle/working/best_unet_pet.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found!\n"
        f"Expected: {model_path}\n\n"
        f"Run the training cell (Cell 4) first."
    )

checkpoint = torch.load(model_path, map_location=device)

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print("✅ Best U-Net model loaded successfully!")

# =====================================================
# VISUALIZE ONE TEST SAMPLE
# =====================================================

low_img, high_img = next(iter(test_loader))

low_img = low_img.to(device)
high_img = high_img.to(device)

with torch.no_grad():
    restored = model(low_img)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(low_img[0,0].cpu().numpy(), cmap="gray")
plt.title("Paired Low-Dose PET")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(restored[0,0].cpu().numpy(), cmap="gray")
plt.title("U-Net Output")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(high_img[0,0].cpu().numpy(), cmap="gray")
plt.title("Paired Full-Dose PET")
plt.axis("off")

plt.tight_layout()
plt.show()

# =====================================================
# EVALUATION
# =====================================================

psnr_list = []
ssim_list = []
rmse_list = []
nrmse_list = []

with torch.no_grad():

    for low_img, high_img in test_loader:

        low_img = low_img.to(device)
        high_img = high_img.to(device)

        pred = model(low_img)

        for i in range(pred.shape[0]):

            pred_np = pred[i,0].cpu().numpy()
            gt_np = high_img[i,0].cpu().numpy()

            pred_np = np.clip(np.nan_to_num(pred_np),0,1)
            gt_np = np.clip(np.nan_to_num(gt_np),0,1)

            psnr = peak_signal_noise_ratio(
                gt_np,
                pred_np,
                data_range=1.0
            )

            ssim = structural_similarity(
                gt_np,
                pred_np,
                data_range=1.0,
                win_size=7
            )

            mse = mean_squared_error(
                gt_np.flatten(),
                pred_np.flatten()
            )

            rmse = np.sqrt(mse)
            nrmse = rmse/(gt_np.max()-gt_np.min()+1e-8)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            rmse_list.append(rmse)
            nrmse_list.append(nrmse)

print("="*60)
print("FINAL TEST RESULTS")
print("="*60)

print(f"Average PSNR  : {np.mean(psnr_list):.4f} dB")
print(f"Average SSIM  : {np.mean(ssim_list):.4f}")
print(f"Average RMSE  : {np.mean(rmse_list):.6f}")
print(f"Average NRMSE : {np.mean(nrmse_list):.6f}")

# =====================================================
# TRAINING CURVES
# =====================================================

plt.figure(figsize=(7,5))
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
plt.plot(val_losses)
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
plt.plot(psnr_history)
plt.title("PSNR")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
plt.plot(ssim_history)
plt.title("SSIM")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.grid(True)
plt.show()

print("\n✅ PROJECT COMPLETED SUCCESSFULLY!")