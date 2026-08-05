#### CELL 2

!pip install -q monai

# =====================================================
# STEP 8: IMPORT REQUIRED LIBRARIES
# =====================================================

from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, ScaleIntensity, EnsureType

# =====================================================
# STEP 9: DEFINE MONAI TRANSFORMS
# =====================================================

monai_transform = Compose([
    ScaleIntensity(),
    EnsureType()
])

# =====================================================
# STEP 10: CREATE CUSTOM PAIRED PET DATASET
# =====================================================

class PETDataset(Dataset):
    """
    Loads real paired Low-Dose and Full-Dose PET images.
    Each item in pair_list is: (low_dose_path, full_dose_path)
    """

    def __init__(self, pair_list):
        self.pair_list = pair_list

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):

        low_path, high_path = self.pair_list[idx]

        # -----------------------------
        # Load Paired PET Images
        # -----------------------------
        low_img = load_pet(low_path)
        high_img = load_pet(high_path)

        # -----------------------------
        # Apply MONAI Preprocessing
        # -----------------------------
        high_img = monai_transform(high_img)
        low_img = monai_transform(low_img)

        # -----------------------------
        # Convert to PyTorch Tensors
        # -----------------------------
        high_img = torch.as_tensor(
            high_img,
            dtype=torch.float32
        ).unsqueeze(0)

        low_img = torch.as_tensor(
            low_img,
            dtype=torch.float32
        ).unsqueeze(0)

        return low_img, high_img

# =====================================================
# STEP 11: CREATE DATASETS
# =====================================================

train_dataset = PETDataset(train_pairs)
val_dataset = PETDataset(val_pairs)
test_dataset = PETDataset(test_pairs)

print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)
print("Training Samples   :", len(train_dataset))
print("Validation Samples :", len(val_dataset))
print("Testing Samples    :", len(test_dataset))

# =====================================================
# STEP 12: CREATE DATALOADERS
# =====================================================

BATCH_SIZE = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("\n✅ DataLoaders created successfully!")

# =====================================================
# STEP 13: VERIFY BATCH SHAPES
# =====================================================

low_batch, high_batch = next(iter(train_loader))

print("\nBatch Shapes")
print("Low Dose  :", low_batch.shape)
print("Full Dose :", high_batch.shape)

# Expected output:
# Low Dose  : torch.Size([4, 1, H, W])
# Full Dose : torch.Size([4, 1, H, W])

# =====================================================
# STEP 14: VISUALIZE ONE TRAINING SAMPLE
# =====================================================

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(low_batch[0, 0].cpu().numpy(), cmap="gray")
plt.title("Paired Low-Dose PET")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(high_batch[0, 0].cpu().numpy(), cmap="gray")
plt.title("Paired Full-Dose PET")
plt.axis("off")

plt.tight_layout()
plt.show()

# =====================================================
# STEP 15: SET DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing Device:", device)
print("Batch Size  :", BATCH_SIZE)
print("✅ Paired Low/Full-Dose pipeline ready!")
