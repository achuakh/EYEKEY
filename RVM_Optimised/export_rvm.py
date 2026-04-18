import torch
import torchvision.transforms.functional as F_t

# 1. THE HACK: Replace the 'normalize' function with a "dumb" version 
# that has no 'if' statements. This prevents the "Guard" error.
original_normalize = F_t.normalize

def dumb_normalize(tensor, mean, std, inplace=False):
    dtype = tensor.dtype
    device = tensor.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=dtype, device=device).view(-1, 1, 1)
    return tensor.sub(mean).div(std)

F_t.normalize = dumb_normalize

# 2. Load the model
print("Loading RVM MobilenetV3...")
model = torch.hub.load('PeterL1n/RobustVideoMatting', 'mobilenetv3')
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
model = model.eval().cuda()

# 3. Initialize 1080p buffers
print("Initializing 1080p buffers...")
src = torch.randn(1, 3, 1080, 1920).cuda()
with torch.no_grad():
    _, _, *rec_states = model(src)

# 4. Export to ONNX
print("Exporting to ONNX (using monkey-patch)...")
with torch.no_grad():
    torch.onnx.export(
        model, 
        (src, *rec_states), 
        "rvm_mobilenet.onnx",
        export_params=True,
        opset_version=17, # Updated to 17 to match your environment's default
        do_constant_folding=True,
        input_names=['src', 'r1', 'r2', 'r3', 'r4'],
        output_names=['fgr', 'pha', 'ro1', 'ro2', 'ro3', 'ro4']
    )

# 5. Restore original function
F_t.normalize = original_normalize
print("✅ SUCCESS! C:\\eyekey\\rvm_mobilenet.onnx generated.")