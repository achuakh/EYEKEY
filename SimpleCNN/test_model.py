import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np

# ---------------------------------------------------------
# 1. THE ARCHITECTURE
# ---------------------------------------------------------
class ProUNet(nn.Module):
    def __init__(self):
        super(ProUNet, self).__init__()
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

# ---------------------------------------------------------
# 2. THE GEOMETRY BRIDGE (Rectify & Un-Rectify)
# ---------------------------------------------------------
def live_rectify(frame_rgb):
    """Rotates 90 CW and pads width to make a perfect square for the AI."""
    frame = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
    h, w = frame.shape[:2]
    pad = (h - w) // 2
    if pad > 0:
        return cv2.copyMakeBorder(frame, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return frame

def live_unrectify(mask_512, original_shape):
    """Reverses the process: scales up, crops padding, and rotates back CCW."""
    orig_h, orig_w = original_shape[:2] # e.g., 1080, 1920
    rotated_h, rotated_w = orig_w, orig_h # e.g., 1920, 1080
    pad = (rotated_h - rotated_w) // 2 # e.g., 420
    square_size = rotated_h # e.g., 1920
    
    # Scale 512x512 back to the large padded square (e.g., 1920x1920)
    # Note: cv2.resize expects (width, height)
    mask_square = cv2.resize(mask_512, (square_size, square_size), interpolation=cv2.INTER_LINEAR)
    
    # Crop the left and right padding off
    if pad > 0:
        mask_cropped = mask_square[:, pad : pad + rotated_w]
    else:
        mask_cropped = mask_square
        
    # Rotate CCW back to original sideways orientation
    mask_final = cv2.rotate(mask_cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return mask_final

# ---------------------------------------------------------
# 3. LOAD THE HARDWARE AND WEIGHTS
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProUNet().to(device)

weights_path = 'models/prounet_FINAL_epoch30.pth'
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print(f"✅ Loaded weights from: {weights_path}")

# ---------------------------------------------------------
# 4. PREPARE THE IMAGE (Simulating NDI input)
# ---------------------------------------------------------
test_image_path = 'dataset/train/images/img_2c4749bc.jpg' # <-- CHANGE TO YOUR TEST IMAGE
img_bgr = cv2.imread(test_image_path)
original_shape = img_bgr.shape

# Convert to RGB and Rectify (just like the live loop)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rectified = live_rectify(img_rgb)

# Convert to tensor
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((512, 512)),
    T.ToTensor(),
])
input_tensor = transform(img_rectified).unsqueeze(0).to(device)

# ---------------------------------------------------------
# 5. RUN INFERENCE
# ---------------------------------------------------------
print("🧠 Processing image...")
with torch.no_grad():
    raw_logits = model(input_tensor)
    pred_mask = (torch.sigmoid(raw_logits) > 0.5).float()

# Extract the 512x512 mask array
mask_np = pred_mask.squeeze().cpu().numpy()
binary_mask_512 = (mask_np * 255).astype(np.uint8)

# ---------------------------------------------------------
# 6. UN-RECTIFY & SAVE
# ---------------------------------------------------------
final_broadcast_mask = live_unrectify(binary_mask_512, original_shape)

# Cut out the background on the ORIGINAL image
result = cv2.bitwise_and(img_bgr, img_bgr, mask=final_broadcast_mask)

# Create a side-by-side image: [Original | Mask | Cutout]
mask_bgr = cv2.cvtColor(final_broadcast_mask, cv2.COLOR_GRAY2BGR)

# Stack them horizontally (they are back to their original sideways landscape orientation)
preview = np.hstack((img_bgr, mask_bgr, result))

output_file = 'test_result_unrectified.jpg'
# Optional: resize the final preview so it's not massively wide when you open it
preview_small = cv2.resize(preview, (preview.shape[1] // 2, preview.shape[0] // 2))

cv2.imwrite(output_file, preview_small)
print(f"🎉 Success! Check {output_file} to see the broadcast-ready result.")