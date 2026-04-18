# Real-Time Spatio-Temporal Matting for Virtual Production
![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)

This repository documents the evolutionary development of a deep-learning pipeline for infrastructure-agnostic human subject isolation. Transitioning from spatial segmentation to recurrent temporal matting, the system is optimized for high-fidelity virtual production on **NVIDIA RTX A6000** hardware.

## 📊 Hardware Benchmarks
The following table illustrates the performance benchmarks at 1080p resolution. Moving to a native Windows environment with TensorRT optimization allowed the pipeline to meet the 16.6ms frame budget required for professional broadcast.

| Architecture | Resolution | Precision | Device | FPS | Latency | GPU Util % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet50-UNet** | 1024x1024 | FP32 | RTX A6000 | ~12 | ~83ms | 90% |
| **YOLO11x-seg** | 1080p | FP16 | RTX A6000 | 40+ | <25ms | 55% |
| **RVM (Standard)** | 1080p | FP16 | RTX A6000 | 30 | ~33ms | 35% |
| **RVM (TensorRT)** | 1080p | FP16 | RTX A6000 | 60* | <16ms | 50% |

*\*Projected achievement following TensorRT engine compilation.*

## 🚀 The Evolutionary Path
1. **SimpleCNN**: Proof-of-concept silhouette extraction at 512x512.
2. **ResNet50**: Scaling to 1024px with Tversky & Edge Loss refinement.
3. **YOLO11-Seg**: First real-time NDI implementation achieving 40+ FPS.
4. **Robust Video Matting (RVM)**: Final recurrent solution resolving temporal jitter and the "Perspective Trap".

## 🛠️ Installation & Prerequisites
- **Python**: 3.10 (Strict requirement for NDI binary compatibility).
- **NDI**: Ensure **NDI 6 SDK/Runtime** is installed on the Windows host.
- **Setup**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt

## 🧠 Pre-trained Models
To run the production applications, download the optimized engines from the [v1.0.0 Release](https://github.com/achuakh/EYEKEY/releases/tag/v1.0.0) and place them in their respective source folders.

| Model File | Resolution | Target Folder |
| :--- | :--- | :--- |
| `rvm_mobilenetv3.pth` | 1080p | `/src/RVM/` |
| `rvm_a6000.engine` | 1080p | `/src/RVM_Optimised/` |
| `yolo11x-seg-1024.engine` | 1024px | `/src/YOLO11-Seg/` |
