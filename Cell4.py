### CELL 4




# =====================================================
# STEP 25: IMPORT REQUIRED LIBRARIES
# =====================================================
# Training loop is unchanged.
# Batches are real (low_dose, full_dose) pairs from the new dataset.

from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np
import torch

# =====================================================
# STEP 26: INITIALIZE TRAINING VARIABLES
# =====================================================

# Defined in Cell 3; keep a fallback if Cell 3 was not run in this session.
if "NUM_EPOCHS" not in globals():
    NUM_EPOCHS = 30

best_loss = float("inf")

train_losses = []
val_losses = []

psnr_history = []
ssim_history = []

# =====================================================
# STEP 27: START TRAINING
# =====================================================

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
    print("=" * 60)

    # -------------------------------------------------
    # TRAINING PHASE
    # -------------------------------------------------

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(train_loader)

    for low_img, high_img in progress_bar:

        low_img = low_img.to(device)
        high_img = high_img.to(device)

        optimizer.zero_grad()

        pred = model(low_img)

        loss = criterion(pred, high_img)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(
            Train_Loss=f"{loss.item():.6f}"
        )

    train_loss = running_loss / len(train_loader)

    train_losses.append(train_loss)

    # -------------------------------------------------
    # VALIDATION PHASE
    # -------------------------------------------------

    model.eval()

    val_running_loss = 0.0

    psnr_epoch = []
    ssim_epoch = []

    with torch.no_grad():

        for low_img, high_img in val_loader:

            low_img = low_img.to(device)
            high_img = high_img.to(device)

            pred = model(low_img)

            loss = criterion(pred, high_img)

            val_running_loss += loss.item()

            # -----------------------------------------
            # Compute metrics for each image in batch
            # -----------------------------------------

            batch_size = pred.shape[0]

            for i in range(batch_size):

                pred_np = pred[i, 0].cpu().numpy()
                gt_np = high_img[i, 0].cpu().numpy()

                pred_np = np.clip(pred_np, 0, 1)
                gt_np = np.clip(gt_np, 0, 1)

                psnr_epoch.append(
                    peak_signal_noise_ratio(
                        gt_np,
                        pred_np,
                        data_range=1.0
                    )
                )

                ssim_epoch.append(
                    structural_similarity(
                        gt_np,
                        pred_np,
                        data_range=1.0,
                        win_size=7
                    )
                )

    val_loss = val_running_loss / len(val_loader)

    val_losses.append(val_loss)

    avg_psnr = np.mean(psnr_epoch)
    avg_ssim = np.mean(ssim_epoch)

    psnr_history.append(avg_psnr)
    ssim_history.append(avg_ssim)

    scheduler.step()

    # -------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------

    print(f"Training Loss   : {train_loss:.6f}")
    print(f"Validation Loss : {val_loss:.6f}")
    print(f"Validation PSNR : {avg_psnr:.4f}")
    print(f"Validation SSIM : {avg_ssim:.4f}")

    # -------------------------------------------------
    # SAVE BEST MODEL
    # -------------------------------------------------

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            "/kaggle/working/best_unet_pet.pth"
        )

        print("✅ Best model saved successfully!")

# =====================================================
# STEP 28: TRAINING COMPLETED
# =====================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"Best Validation Loss : {best_loss:.6f}")
print("Saved Model          : /kaggle/working/best_unet_pet.pth")