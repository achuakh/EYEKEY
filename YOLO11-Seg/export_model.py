from ultralytics import YOLO

# 1. Load your high-res PyTorch variant
model = YOLO('yolo11x-seg.pt')

# 2. Export with the explicit 1024 dimension
# imgsz=1024: This is the "secret sauce" for finer finger detection
# workspace=32: Excellent choice for the A6000's 48GB VRAM
model.export(
    format='engine', 
    imgsz=1024, 
    device=0, 
    half=True, 
    workspace=32
)