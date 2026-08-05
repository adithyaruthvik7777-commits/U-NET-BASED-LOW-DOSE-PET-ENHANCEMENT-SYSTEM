# =====================================================
# CELL 6: SAVE PROJECT FILES
# =====================================================
# Saving results for the paired Low-Dose / Full-Dose U-Net project.

import os
import shutil

save_dir = "/kaggle/working/project_results"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# Copy trained model
# -----------------------------
model_path="/kaggle/working/best_unet_pet.pth"

if os.path.exists(model_path):
    shutil.copy(model_path, os.path.join(save_dir,"best_unet_pet.pth"))
    print("✅ Model copied.")
else:
    print("❌ Model file not found.")

# -----------------------------
# Save final metrics
# -----------------------------
with open(os.path.join(save_dir, "final_results.txt"), "w") as f:
    f.write("FINAL TEST RESULTS\n")
    f.write("Dataset: PAIRED Low-Dose / Full-Dose\n")
    f.write("=============================\n")
    f.write(f"Average PSNR  : {np.mean(psnr_list):.4f} dB\n")
    f.write(f"Average SSIM  : {np.mean(ssim_list):.4f}\n")
    f.write(f"Average RMSE  : {np.mean(rmse_list):.6f}\n")
    f.write(f"Average NRMSE : {np.mean(nrmse_list):.6f}\n")

print("✅ Metrics saved.")

# -----------------------------
# Save training loss graph
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(train_losses, marker="o")
plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=300)
plt.close()

# -----------------------------
# Save validation loss graph
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(val_losses, marker="o")
plt.title("Validation Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "validation_loss.png"), dpi=300)
plt.close()

# -----------------------------
# Save PSNR graph
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(psnr_history, marker="o")
plt.title("PSNR vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "psnr_graph.png"), dpi=300)
plt.close()

# -----------------------------
# Save SSIM graph
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(ssim_history, marker="o")
plt.title("SSIM vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ssim_graph.png"), dpi=300)
plt.close()

print("\n===================================")
print("✅ ALL FILES SAVED SUCCESSFULLY!")
print("===================================")
print("Saved folder:", save_dir)
print("\nFiles:")
print("- best_unet_pet.pth")
print("- final_results.txt")
print("- training_loss.png")
print("- validation_loss.png")
print("- psnr_graph.png")
print("- ssim_graph.png")