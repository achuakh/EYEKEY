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

def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1e-6):
    """
    Tversky index: beta (0.7) penalizes False Negatives (missing arms) harder.
    alpha (0.3) helps control False Positives (blue blobs).
    """
    pred = torch.sigmoid(pred)
    
    # Flatten for calculation
    pred = pred.view(-1)
    target = target.view(-1)
    
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()
    
    # Tversky index formula
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky

def focal_loss(logits, target, gamma=2.0, alpha=0.25):
    """
    Safe for AMP: Combined Sigmoid + BCE.
    'logits' are the raw outputs from your ResNet before any sigmoid.
    """
    # 1. Use the version safe for float16 autocasting
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    
    # 2. We still need the probability (p_t) for the focal weight calculation
    # Autocast is fine with sigmoid here as long as it's not immediately passed to BCE
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1 - probs) * (1 - target)
    
    loss = alpha * (1 - p_t)**gamma * bce
    return loss.mean()

# --- THE RECALL-MAXIMIZER LOSS FUNCTIONS ---
def weighted_bce_focal_loss(logits, target, pos_weight=3.0, gamma=2.0):
    """
    pos_weight=15.0 forces the AI to prioritize 'Andrew' pixels over background.
    Safe for AMP (uses logits directly).
    """
    # Weighting the '1' class (Andrew) much higher than '0' (Background)
    weight = torch.tensor([pos_weight]).to(logits.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=weight, reduction='none'
    )
    
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1 - probs) * (1 - target)
    loss = (1 - p_t)**gamma * bce
    return loss.mean()

def aggressive_tversky_loss(pred_logits, target, alpha=0.5, beta=0.5, smooth=1e-6):
    """
    beta=0.9 makes a 'missing arm' 9x more painful than a 'blue blob'.
    """
    pred = torch.sigmoid(pred_logits)
    pred = pred.view(-1)
    target = target.view(-1)
    
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()
    
    # Tversky index favors recall (beta) over precision (alpha)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky


# ---------------------------------------------------------
# 2. DATASET WITH SPILL SUPPRESSION & AUGMENTATION
# ---------------------------------------------------------
class MattingDataset(Dataset):
    def __init__(self, img_dir, msk_dir, bg_dir=None, target_size=(1024, 1024)):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.msk_paths = sorted(glob.glob(os.path.join(msk_dir, "*.png")))
        self.bg_paths = []
        if bg_dir:
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                self.bg_paths.extend(glob.glob(os.path.join(bg_dir, ext)))
        
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
        
        img_pil = self.color_aug(img_pil)
        img_t = transforms.ToTensor()(img_pil)
        msk_t = transforms.ToTensor()(msk_pil)

        img_t = self.suppress_spill_tensor(img_t)

        if self.bg_paths and random.random() > 0.1:
            bg_pil = Image.open(random.choice(self.bg_paths)).convert("RGB").resize(self.target_size)
            bg_t = transforms.ToTensor()(bg_pil)
            img_t = img_t * msk_t + bg_t * (1 - msk_t)

        img_t = self.normalize(img_t)
        return img_t, msk_t

# ---------------------------------------------------------
# 3. TRAINING LOOP - TUNE
# ---------------------------------------------------------
device = torch.device("cuda")
model = ResNetUNet().to(device)

# --- LOAD EXISTING MODEL WEIGHTS ---
model.load_state_dict(torch.load('models/t_resnet_unet_FINAL_95c.pth'))

# --- CRITICAL DEFINITIONS ---
num_epochs = 10  # <--- Epoch Runs
optimizer = optim.Adam(model.parameters(), lr=0.000005)
bce_criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
scaler = torch.amp.GradScaler('cuda')

base_path = '/home/andrew/CODE/EYEKEY/dataset'
train_set = MattingDataset(
    img_dir=os.path.join(base_path, 'train/images'),
    msk_dir=os.path.join(base_path, 'train/masks'),
    bg_dir=os.path.join(base_path, 'background_library'),
    target_size=(1024, 1024)
)
loader = DataLoader(train_set, batch_size=12, shuffle=True, num_workers=6, pin_memory=True)

print(f"🚀 High-Precision Mode: 1024x1024 Resolution | AMP Enabled")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            
            # 1. Neutralize the 'Andrew' Bias (Drop to 1.1)
            loss_wf = weighted_bce_focal_loss(outputs, masks, pos_weight=1.1)
            
            # 2. Aggressive Tversky (RE-INSERT THIS LINE)
            # This handles the internal connectivity and helps kill the halo.
            loss_at = aggressive_tversky_loss(outputs, masks, alpha=0.5, beta=0.5)
            
            # 3. Aggressive Edge Snap (Bump to 4.0x)
            loss_e = edge_loss(outputs, masks)
            
            # The Final 'Zero-Tolerance' Formula
            # Now loss_at is defined and won't cause a crash!
            total_loss = (1.0 * loss_wf) + (1.0 * loss_at) + (4.0 * loss_e)
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += total_loss.item()
    
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/len(loader):.4f} | LR: {scheduler.get_last_lr()[0]}")
    
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'models/t_resnet_1024_ep{epoch+1}.pth')

torch.save(model.state_dict(), 'models/t_resnet_unet_FINAL.pth')
print("✨ Master's Thesis Tune Model Complete!")