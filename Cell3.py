### CELL 3


# =====================================================
# STEP 16: IMPORT REQUIRED LIBRARIES
# =====================================================
# Model / training setup is unchanged.
# It still learns: Low-Dose PET  ->  Full-Dose PET
# (now using real paired data from Cell 1 & 2)

import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================
# STEP 17: DEFINE U-NET MODEL
# =====================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(1,64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128,256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottom = DoubleConv(256,512)

        self.up3 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv3 = DoubleConv(512,256)

        self.up2 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv2 = DoubleConv(256,128)

        self.up1 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv1 = DoubleConv(128,64)

        self.final = nn.Conv2d(64,1,1)

    def forward(self,x):

        c1=self.down1(x)
        p1=self.pool1(c1)

        c2=self.down2(p1)
        p2=self.pool2(c2)

        c3=self.down3(p2)
        p3=self.pool3(c3)

        b=self.bottom(p3)

        u3=self.up3(b)
        u3=torch.cat([u3,c3],dim=1)
        c4=self.conv3(u3)

        u2=self.up2(c4)
        u2=torch.cat([u2,c2],dim=1)
        c5=self.conv2(u2)

        u1=self.up1(c5)
        u1=torch.cat([u1,c1],dim=1)
        c6=self.conv1(u1)

        out=self.final(c6)

        return out

# =====================================================
# STEP 18: INITIALIZE MODEL
# =====================================================

model = UNet().to(device)

print("="*60)
print("U-NET MODEL INITIALIZED")
print("="*60)

# =====================================================
# STEP 19: DISPLAY MODEL INFORMATION
# =====================================================

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")

# =====================================================
# STEP 20: LOSS FUNCTION
# =====================================================

criterion = nn.L1Loss()

print("Loss Function : L1 Loss")

# =====================================================
# STEP 21: OPTIMIZER
# =====================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

print("Optimizer : Adam")

# =====================================================
# STEP 22: LR SCHEDULER / EPOCHS
# =====================================================

NUM_EPOCHS = 30

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)

print("="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print("Epochs :",NUM_EPOCHS)
print("Learning Rate :",1e-4)
print("Batch Size :",BATCH_SIZE)
print("Device :",device)