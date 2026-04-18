import sys
import os
import time
import numpy as np
import cv2
import NDIlib as ndi
import torch
import torch.nn.functional as F
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QComboBox, QLabel)
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

# --- [HARDENED MATTE WORKER] ---
class MatteWorker(QThread):
    fps_signal = pyqtSignal(float)
    status_signal = pyqtSignal(str)

    def __init__(self, source_name, model_path):
        super().__init__()
        self.source_name = source_name
        self.model_path = model_path
        self.running = True
        self.out_buffer = np.zeros((1080, 1920, 4), dtype=np.uint8)
    
    def stop(self):
        self.running = False

    def run(self):
        import NDIlib as ndi
        import torch
        import torch.nn.functional as F
        
        try:
            # 1. LOAD THE ENGINE
            model = YOLO(self.model_path, task='segment')
            engine_imgsz = 1024 if "1024" in self.model_path else 640
            
            # 2. SENDER: FORCED IDENTITY
            s_settings = ndi.SendCreate()
            s_settings.ndi_name = "python" # Force the name to 'python'
            self.ndi_send = ndi.send_create(s_settings)
            
            # 3. RECEIVER SETUP
            r_settings = ndi.RecvCreateV3(color_format=ndi.RECV_COLOR_FORMAT_BGRX_BGRA)
            r_settings.bandwidth = ndi.RECV_BANDWIDTH_HIGHEST 
            ndi_recv = ndi.recv_create_v3(r_settings)
            
            source = ndi.Source()
            source.ndi_name = self.source_name.encode('ascii')
            ndi.recv_connect(ndi_recv, source)
            
            # Connection Handshake
            self.status_signal.emit("CONNECTING...")
            connected = False
            for _ in range(30):
                t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 200)
                if t == ndi.FRAME_TYPE_VIDEO:
                    connected = True
                    ndi.recv_free_video_v2(ndi_recv, v)
                    break
                if not self.running: return
                time.sleep(0.05)

            if not connected:
                self.status_signal.emit("ERROR: TIMEOUT")
                return

            self.status_signal.emit(f"LIVE: {engine_imgsz}p")
            out_v = ndi.VideoFrameV2()
            prev_mask_gpu = None

            while self.running:
                t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 33) 
                if t != ndi.FRAME_TYPE_VIDEO: continue
                
                start_time = time.time()
                h, w = v.yres, v.xres
                
                # --- [GPU PIPELINE] ---
                raw_pixels = np.frombuffer(v.data, dtype=np.uint8).reshape(h, w, 4)
                frame_tensor = torch.from_numpy(raw_pixels).to(device=0, non_blocking=True)
                
                # Letterbox Logic
                active_h = int(engine_imgsz * (9 / 16))
                pad = (engine_imgsz - active_h) // 2
                
                img_gpu = frame_tensor[:, :, :3].permute(2, 0, 1).unsqueeze(0).half() / 255.0
                img_letterbox = F.interpolate(img_gpu, size=(active_h, engine_imgsz), mode='bilinear')
                img_final = F.pad(img_letterbox, (0, 0, pad, pad), "constant", 0)
                
                results = model(img_final, classes=[0], verbose=False, conf=0.55)
                
                if results[0].masks is not None:
                    mask_gpu = results[0].masks.data[0].half().unsqueeze(0).unsqueeze(0) 
                    mask_gpu = mask_gpu[:, :, pad : pad + active_h, :] # Reverse Crop
                    
                    if prev_mask_gpu is not None and prev_mask_gpu.shape == mask_gpu.shape:
                        mask_gpu = (mask_gpu * 0.4) + (prev_mask_gpu * 0.6)
                    prev_mask_gpu = mask_gpu

                    mask_gpu = F.interpolate(mask_gpu, size=(h, w), mode='bilinear')
                    mask_gpu = (mask_gpu - 0.12).clamp(0, 1) 
                    mask_gpu = torch.pow(mask_gpu, 2.5)      
                    mask_gpu = F.avg_pool2d(mask_gpu, kernel_size=11, stride=1, padding=5)
                    
                    alpha = (torch.clamp(mask_gpu.squeeze() * 2.0, 0, 1) * 255).to(torch.uint8)
                    frame_tensor[:, :, 3] = alpha
                else:
                    frame_tensor[:, :, 3] = 0
                
                self.out_buffer[:h, :w, :] = frame_tensor.cpu().numpy()

                out_v.xres, out_v.yres = w, h
                out_v.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
                out_v.frame_rate_N, out_v.frame_rate_D = 60000, 1001
                out_v.line_stride_in_bytes = w * 4
                out_v.data = self.out_buffer
                
                ndi.send_send_video_v2(self.ndi_send, out_v)
                self.fps_signal.emit(1.0 / (time.time() - start_time))
                ndi.recv_free_video_v2(ndi_recv, v)

        except Exception as e:
            print(f"PIPELINE ERROR: {str(e)}")
        finally:
            # --- CRITICAL DESTRUCTION ORDER ---
            self.running = False
            if 'ndi_recv' in locals(): ndi.recv_destroy(ndi_recv)
            if hasattr(self, 'ndi_send'): 
                ndi.send_destroy(self.ndi_send)
                print("LOG: NDI Sender 'python' destroyed.")

# --- [UI WITH REFRESH BUTTON] ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDI AI-Matte (A6000 Final Stable)")
        self.setMinimumWidth(450)
        
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.source_combo = QComboBox()
        layout.addWidget(QLabel("Select iPhone Source:"))
        layout.addWidget(self.source_combo)
        
        # RESTORED: Refresh Button
        self.refresh_btn = QPushButton("🔄 Refresh NDI List")
        self.refresh_btn.clicked.connect(self.refresh_sources)
        layout.addWidget(self.refresh_btn)

        self.model_combo = QComboBox()
        self.model_combo.addItem("yolo11x-seg-640.engine")
        self.model_combo.addItem("yolo11x-seg-1024.engine")
        layout.addWidget(QLabel("Select TensorRT Engine:"))
        layout.addWidget(self.model_combo)

        self.start_btn = QPushButton("Start Pipeline")
        self.start_btn.clicked.connect(self.start_pipeline)
        layout.addWidget(self.start_btn)
        
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        self.fps_label = QLabel("FPS: 0.0")
        layout.addWidget(self.fps_label)

        self.find_settings = ndi.FindCreate()
        self.ndi_find = ndi.find_create_v2(self.find_settings)
        self.worker = None
        self.refresh_sources()

    def refresh_sources(self):
        ndi.find_wait_for_sources(self.ndi_find, 1000)
        sources = ndi.find_get_current_sources(self.ndi_find)
        self.source_combo.clear()
        for s in sources:
            self.source_combo.addItem(s.ndi_name)

    def start_pipeline(self):
        # 1. STOP & CLEANUP
        if self.worker is not None and self.worker.isRunning():
            self.status_label.setText("Cleaning network names...")
            self.worker.stop() # Calls wait() internally
            
            # Essential: Give NDI 500ms to clear the name from the network registry
            time.sleep(0.5) 
            
            torch.cuda.empty_cache() 
            print("LOG: Network name 'python' should now be free.")

        # 2. START NEW KERNEL
        source = self.source_combo.currentText()
        model_file = self.model_combo.currentText()
        if not source or not model_file: return
        
        self.worker = MatteWorker(source, model_file)
        self.worker.status_signal.connect(self.status_label.setText)
        self.worker.fps_signal.connect(lambda f: self.fps_label.setText(f"FPS: {f:.2f}"))
        self.worker.start()

    def closeEvent(self, event):
        if self.worker is not None: self.worker.stop()
        ndi.find_destroy(self.ndi_find)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())