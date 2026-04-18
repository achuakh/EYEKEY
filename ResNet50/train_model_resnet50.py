import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import glob
import random

# ---------------------------------------------------------
# 1. THE ARCHITECTURE (ResNet50-UNet)
# ---------------------------------------------------------
class ResNetUNet(nn.Module):
    def __init__(self):
        super(ResNetUNet, self).__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) 
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) 
        self.layer2 = self.base_layers[5] 
        self.layer3 = self.base_layers[6] 
        self.layer4 = self.base_layers[7] 

        self.up4 = nn.ConvTranspose2d(2048, 1024, 2, 2)
        self.dec4 = self.conv_block(2048, 1024)
        self.up3 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec3 = self.conv_block(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec2 = self.conv_block(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 64, 2, 2)
        self.dec1 = self.conv_block(128, 64)
        
        self.final_up = nn.ConvTranspose2d(64, 32, 2, 2)
        self.final_conv = nn.Conv2d(32, 1, 1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e0 = self.layer0(x); e1 = self.layer1(e0); e2 = self.layer2(e1)
        e3 = self.layer3(e2); e4 = self.layer4(e3)

        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))
        
        return self.final_conv(self.final_up(d1))

# --- LOSS FUNCTIONS ---
def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
    
def edge_loss(pred, target):
    pred = torch.sigmoid(pred)
    # Using fixed tensors for Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3).to(pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3).to(pred.device)
    
    edge_pred_x = torch.nn.functional.conv2d(pred, sobel_x, padding=1)
    edge_pred_y = torch.nn.functional.conv2d(pred, sobel_y, padding=1)
    edge_gt_x = torch.nn.functional.conv2d(target, sobel_x, padding=1)
    edge_gt_y = torch.nn.functional.conv2d(target, sobel_y, padding=1)
    
    return torch.nn.functional.mse_loss(edge_pred_x, edge_gt_x) + torch.nn.functional.mse_loss(edge_pred_y, edge_gt_y) 

def aggressive_tversky_loss(pred_logits, target, alpha=0.5, beta=0.5, smooth=1e-6):
    pred = torch.sigmoid(pred_logits)
    tp = (pred * target).sum(dim=(2, 3))
    fp = ((1 - target) * pred).sum(dim=(2, 3))
    fn = (target * (1 - pred)).sum(dim=(2, 3))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky.mean()

# ---------------------------------------------------------
# 2. DATASET WITH SPILL SUPPRESSION & AUGMENTATION
# ---------------------------------------------------------
class MattingDataset(Dataset):
    def __init__(self, base_dirs, bg_dir=None, target_size=(1024, 1024)):
        self.img_paths = []
        self.msk_paths = []
        self.bg_paths = [] # <--- FIX: Initialize this!
        
        if bg_dir:
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                self.bg_paths.extend(glob.glob(os.path.join(bg_dir, ext)))

        for base_path in base_dirs:
            curr_imgs = sorted(glob.glob(os.path.join(base_path, 'images', "*.jpg")))
            curr_msks = sorted(glob.glob(os.path.join(base_path, 'masks', "*.png")))
            if len(curr_imgs) == len(curr_msks):
                self.img_paths.extend(curr_imgs)
                self.msk_paths.extend(curr_msks)
            else:
                print(f"⚠️ Mismatch in {base_path}!")

        self.target_size = target_size
        self.color_aug = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def suppress_spill_tensor(self, img_tensor):
        r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]
        avg_rb = (r + b) / 2.0
        g = torch.where(g > avg_rb, avg_rb, g)
        return torch.stack([r, g, b])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_pil = Image.open(self.img_paths[idx]).convert("RGB").resize(self.target_size)
        msk_pil = Image.open(self.msk_paths[idx]).convert("L").resize(self.target_size)
        
        img_t = transforms.ToTensor()(img_pil)
        msk_t = transforms.ToTensor()(msk_pil)
        img_t = self.suppress_spill_tensor(img_t)

        # AUGMENTATION: This makes the model "Background Invariant"
        if self.bg_paths and random.random() > 0.1:
            bg_pil = Image.open(random.choice(self.bg_paths)).convert("RGB").resize(self.target_size)
            bg_t = transforms.ToTensor()(bg_pil)
            img_t = img_t * msk_t + bg_t * (1 - msk_t)

        return self.normalize(img_t), msk_t

# ---------------------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------------------
device = torch.device("cuda")
model = ResNetUNet().to(device)

# --- CRITICAL DEFINITIONS ---
num_epochs = 50  # <--- Epoch Runs
optimizer = optim.Adam(model.parameters(), lr=0.0001)
bce_criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
scaler = torch.amp.GradScaler('cuda')
best_loss = float('inf')

# --- STEP A: List your data sources ---
# This keeps everything organized without having to rename files
master_paths = [
    '/home/andrew/CODE/EYEKEY/dataset/train',        # Set 1: Original
    '/home/andrew/CODE/EYEKEY/dataset/train2',       # Set 2: More Variants
    '/home/andrew/CODE/EYEKEY/dataset/train_green'   # Set 3: Green Top (Hard Mode)
]

# --- STEP B: Instantiate the new Dataset ---
# Initialize with the background library path
train_set = MattingDataset(
    base_dirs=master_paths, 
    bg_dir='/home/andrew/CODE/EYEKEY/dataset/background_library', 
    target_size=(1024, 1024)
)

# --- STEP C: Feed it to the Loader ---
loader = DataLoader(train_set, batch_size=24, shuffle=True, num_workers=8, pin_memory=True)

print(f"🚀 Total training pool: {len(train_set)} images. Starting master bake...")

print(f"🚀 High-Precision Mode: 1024x1024 Resolution | AMP Enabled")

# --- UPDATED LOSS RECIPE ---
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        # Consistent modern AMP syntax
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            
            # Use the higher-precision Tversky loss you defined
            loss_bce = bce_criterion(outputs, masks)
            loss_at = aggressive_tversky_loss(outputs, masks)
            loss_e = edge_loss(outputs, masks)
            
            # The Final Formula: Higher weight on edges for the green-on-green set
            total_loss = loss_bce + loss_at + (5.0 * loss_e)
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += total_loss.item()
    
    scheduler.step()
    avg_epoch_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]}")
    
    # --- BEST LOSS TRACKER ---
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), 'models/resnet_unet_BEST.pth')
        print(f"⭐ New Best Loss! Model saved to resnet_unet_BEST.pth")
    
    # --- MILESTONE SAVES ---
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'models/resnet_1024_ep{epoch+1}.pth')

torch.save(model.state_dict(), 'models/resnet_unet_FINAL.pth')
print("✨ Master's Thesis Model Complete!")