#!/usr/bin/env python
# coding: utf-8

# # Load the dataset

# ## Define the u-net

# In[1]:


import os
import sys


import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import cv2
import numpy as np

# 2. PREVENT WORKER COLLISION
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 3. VERIFY AND WARM UP
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Forces a single allocation to lock the driver
    _ = torch.zeros(1).to("cuda")
    print(f"✅ A6000 Linked: {torch.cuda.get_device_name(0)}")
    print(f"📊 Capacity: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


# In[ ]:


class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        # Encoder (Downscaling: Learning "What" is in the image)
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck (The deepest representation)
        self.bottleneck = self.conv_block(128, 256)

        # Decoder (Upscaling: Learning "Where" the mask should be)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128) # Concatenation (Skip Connection)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Final Output (1 channel for the probability: 0.0 to 1.0)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        """Standard double-convolution block with Batch Normalization."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        # 2. Bottleneck
        b = self.bottleneck(self.pool(e2))

        # 3. Decoding with Skip Connections
        # We 'cat' (concatenate) the upsampled features with the encoder features
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))

        # Sigmoid squashes output to a 0-1 range for a probability mask
        return self.final(d1)

print("🧠 SimpleUNet class defined.")

class ProUNet(nn.Module):
    def __init__(self):
        super(ProUNet, self).__init__()

        # 4 Levels of depth instead of 2
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self.conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat((d4, e4), dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat((d3, e3), dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))

        return self.final(d1)

print("🧠 ProUNet class defined with deeper architecture.")


# In[ ]:


class MattingDataset(Dataset):
    def __init__(self, img_dir, msk_dir, bg_dir=None, target_size=(512, 512)):
        # Collect and sort foreground paths
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.msk_paths = sorted(glob.glob(os.path.join(msk_dir, "*.png")))
        
        # --- THE FIX: Look for multiple formats ---
        self.bg_paths = []
        if bg_dir:
            # Grab jpg, jpeg, and png (both lower and uppercase)
            valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
            for ext in valid_extensions:
                self.bg_paths.extend(glob.glob(os.path.join(bg_dir, ext)))
                
        self.target_size = target_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load Foreground and Mask, immediately resize them
        img = Image.open(self.img_paths[idx]).convert("RGB").resize(self.target_size)
        msk = Image.open(self.msk_paths[idx]).convert("L").resize(self.target_size)

        # 2. DYNAMIC BACKGROUND REPLACEMENT
        # If we have backgrounds, 70% of the time we swap the background
        if self.bg_paths and random.random() > 0.3:
            # Pick a random background and resize it to match
            bg_path = random.choice(self.bg_paths)
            bg = Image.open(bg_path).convert("RGB").resize(self.target_size)
            
            # This digitally pastes the foreground over the random background using the mask
            img = Image.composite(img, bg, msk)

        # 3. Convert to Tensors for PyTorch
        return self.transform(img), self.transform(msk)

# --- INITIALIZE THE DATA BRIDGE ---
base_path = '/home/andrew/CODE/EYEKEY/dataset'
img_dir = os.path.join(base_path, 'train/images')
msk_dir = os.path.join(base_path, 'train/masks')

# Point this to a folder full of random images (landscapes, cities, rooms, etc.)
# It does not matter what resolution they are; the dataset will resize them.
bg_dir = os.path.join(base_path, 'background_library') 

# Pass the bg_dir into the dataset
dataset = MattingDataset(img_dir, msk_dir, bg_dir=bg_dir, target_size=(512, 512))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

print(f"📦 Data Bridge Established: Found {len(dataset)} samples ready for training.")
if dataset.bg_paths:
    print(f"🌍 Background Augmentation ACTIVE: Loaded {len(dataset.bg_paths)} random backgrounds.")


# ## Training

# In[ ]:


# debug test

try:
    test_tensor = torch.zeros((100, 100)).to("cuda")
    print("✅ Smoke test passed: CUDA is responding correctly.")
    del test_tensor
except Exception as e:
    print(f"❌ CUDA still failing: {e}")


# In[ ]:


import torch.optim as optim
import random

# 1. Setup Device (Use GPU if you have one in WSL2, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProUNet().to(device) # Using the U-Net we defined earlier

# 2. Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss() 

print(f"🚀 Training on {device}...")

# 3. The Loop
model.train()
for epoch in range(30):
    epoch_loss = 0
    for images, masks in dataloader:
        # FORCE everything to the A6000 and ensure FLOAT32
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32) # BCE needs float masks!

        optimizer.zero_grad(set_to_none=True) # More memory efficient than zero_grad()

        outputs = model(images)

        # Ensure shapes match (B, 1, 512, 512)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Optional: cleanup intermediate tensors
        del outputs, loss

    current_loss = epoch_loss/len(dataloader)
    print(f"Epoch [{epoch+1}], Loss: {current_loss:.4f}")
    
    # ---------------------------------------------------------
    # 💾 ANDREW's FAILSAFE: Checkpoint every 5 epochs
    # ---------------------------------------------------------
    if (epoch + 1) % 5 == 0:
        os.makedirs('models', exist_ok=True)
        ckpt_path = f'models/prounet_checkpoint_ep{epoch+1}.pth'
        torch.save(model.state_dict(), ckpt_path)
        print(f"   -> Checkpoint safely written to: {ckpt_path}")

    torch.cuda.empty_cache() # Flush after every epoch

print("✨ Training Complete! The brain has finished its first workout.")

# ---------------------------------------------------------
# 💾 THE FINAL SAVE
# ---------------------------------------------------------
os.makedirs('models', exist_ok=True)
final_path = 'models/prounet_FINAL_epoch30.pth'
torch.save(model.state_dict(), final_path)
print(f"🎉 SUCCESS: Master's Thesis model weights saved to {final_path}")