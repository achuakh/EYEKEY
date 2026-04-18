import sys
import time
import numpy as np
import NDIlib as ndi
import torch
import torch.nn.functional as F
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QComboBox, QLabel, QSlider)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

# --- [RVM WORKER WITH DYNAMIC PARAMS] ---
class RVMWorker(QThread):
    fps_signal = pyqtSignal(float)
    status_signal = pyqtSignal(str)

    def __init__(self, source_name, model_path):
        super().__init__()
        self.source_name = source_name
        self.model_path = model_path
        self.running = True
        
        # Real-time Parameters (Checkpoint 1 Defaults)
        self.gain = 1.25
        self.gamma = 0.9
        self.ratio = 0.3 
        self._last_ratio = 0.3 
        
        self.out_buffer = np.zeros((1080, 1920, 4), dtype=np.uint8)

    def stop(self):
        self.running = False

    def run(self):
        try:
            # 1. LOAD MODEL
            model = torch.hub.load('PeterL1n/RobustVideoMatting', 'mobilenetv3')
            model.load_state_dict(torch.load(self.model_path))
            model = model.eval().cuda().half()
            
            # 2. NDI SETUP
            s_settings = ndi.SendCreate()
            s_settings.ndi_name = "python-rvm"
            self.ndi_send = ndi.send_create(s_settings)
            
            r_settings = ndi.RecvCreateV3(color_format=ndi.RECV_COLOR_FORMAT_BGRX_BGRA)
            ndi_recv = ndi.recv_create_v3(r_settings)
            source = ndi.Source()
            source.ndi_name = self.source_name.encode('ascii')
            ndi.recv_connect(ndi_recv, source)
            
            # 3. HANDSHAKE
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

            self.status_signal.emit("LIVE: RVM-CP1")
            rec_states = [None] * 4 
            out_v = ndi.VideoFrameV2()

            while self.running:
                t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 33)
                if t != ndi.FRAME_TYPE_VIDEO: continue
                
                start_time = time.time()
                h, w = v.yres, v.xres
                
                # --- GPU UPLOAD ---
                raw_pixels = np.frombuffer(v.data, dtype=np.uint8).reshape(h, w, 4)
                frame_tensor = torch.from_numpy(raw_pixels).to(device=0, non_blocking=True)
                
                img = frame_tensor[:, :, :3].permute(2, 0, 1).unsqueeze(0).half() / 255.0
                
                # --- DYNAMIC RATIO RESET ---
                # If the ratio changes, we must reset states to prevent shape errors
                if self.ratio != self._last_ratio:
                    rec_states = [None] * 4
                    self._last_ratio = self.ratio

                # --- INFERENCE ---
                with torch.no_grad():
                    fgr, pha, *rec_states = model(img, *rec_states, downsample_ratio=self.ratio)
                
                # --- ALPHA REFINEMENT (DYNAMIC) ---
                pha_boosted = torch.clamp(pha * self.gain, 0, 1)
                
                # Apply the Checkpoint 1 70/30 Soft-Blend
                pha_final = (pha_boosted * 0.7) + (pha * 0.3)
                pha_final = torch.pow(pha_final, self.gamma)
                
                alpha = (pha_final.squeeze() * 255).to(torch.uint8)
                frame_tensor[:, :, 3] = alpha
                
                # --- BROADCAST ---
                self.out_buffer[:h, :w, :] = frame_tensor.cpu().numpy()
                out_v.xres, out_v.yres = w, h
                out_v.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
                out_v.data = self.out_buffer
                ndi.send_send_video_v2(self.ndi_send, out_v)
                
                self.fps_signal.emit(1.0 / (time.time() - start_time))
                ndi.recv_free_video_v2(ndi_recv, v)

        except Exception as e:
            print(f"PIPELINE ERROR: {str(e)}")
        finally:
            self.running = False
            if 'ndi_recv' in locals(): ndi.recv_destroy(ndi_recv)
            if hasattr(self, 'ndi_send'): ndi.send_destroy(self.ndi_send)

# --- [GUI WITH REAL-TIME SLIDERS] ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVM Checkpoint 1 (A6000 Live Control)")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # NDI Controls
        self.source_combo = QComboBox()
        layout.addWidget(QLabel("Select NDI Input Source:"))
        layout.addWidget(self.source_combo)
        
        self.refresh_btn = QPushButton("🔄 Refresh NDI")
        self.refresh_btn.clicked.connect(self.refresh_sources)
        layout.addWidget(self.refresh_btn)

        # --- REAL-TIME SLIDERS ---
        
        # 1. GAIN SLIDER (1.0 to 2.0)
        self.gain_label = QLabel("Alpha Gain: 1.25")
        layout.addWidget(self.gain_label)
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(100, 200) # 1.00 to 2.00
        self.gain_slider.setValue(125)
        self.gain_slider.valueChanged.connect(self.update_params)
        self.gain_slider.setToolTip(
            "Forces uncertain 'gray' pixels to pure white. \n"
            "Increase this if your palm or shirt is turning transparent (Ghosting)."
        )
        layout.addWidget(self.gain_slider)

        # 2. GAMMA SLIDER (0.5 to 1.5)
        self.gamma_label = QLabel("Alpha Gamma: 0.90")
        layout.addWidget(self.gamma_label)
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(50, 150) # 0.50 to 1.50
        self.gamma_slider.setValue(90)
        self.gamma_slider.valueChanged.connect(self.update_params)
        self.gamma_slider.setToolTip(
            "Adjusts the transparency curve. \n"
            "Lower values (<1.0) thicken the matte; higher values (>1.0) thin it out."
        )
        layout.addWidget(self.gamma_slider)

        # 3. DOWNSAMPLE RATIO (0.1 to 0.6)
        self.ratio_label = QLabel("Downsample Ratio: 0.30")
        layout.addWidget(self.ratio_label)
        self.ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.ratio_slider.setRange(10, 60) # 0.10 to 0.60
        self.ratio_slider.setValue(30)
        self.ratio_slider.valueChanged.connect(self.update_params)
        self.ratio_slider.setToolTip(
            "Adjusts the AI's 'field of view.' \n"
            "0.3 is the sweet spot. Drop to 0.2 if moving very close to the camera \n"
            "to help the AI recognize large hand gestures."
        )
        layout.addWidget(self.ratio_slider)

        # Pipeline Controls
        self.start_btn = QPushButton("Start RVM Pipeline")
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

    def update_params(self):
        if self.worker:
            self.worker.gain = self.gain_slider.value() / 100.0
            self.worker.gamma = self.gamma_slider.value() / 100.0
            self.worker.ratio = self.ratio_slider.value() / 100.0
            
            self.gain_label.setText(f"Alpha Gain: {self.worker.gain:.2f}")
            self.gamma_label.setText(f"Alpha Gamma: {self.worker.gamma:.2f}")
            self.ratio_label.setText(f"Downsample Ratio: {self.worker.ratio:.2f}")

    def refresh_sources(self):
        ndi.find_wait_for_sources(self.ndi_find, 1000)
        sources = ndi.find_get_current_sources(self.ndi_find)
        self.source_combo.clear()
        for s in sources:
            self.source_combo.addItem(s.ndi_name)

    def start_pipeline(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            time.sleep(0.5)
            torch.cuda.empty_cache()

        source = self.source_combo.currentText()
        if not source: return
        
        self.worker = RVMWorker(source, "rvm_mobilenetv3.pth")
        self.update_params() # Apply current slider values immediately
        self.worker.status_signal.connect(self.status_label.setText)
        self.worker.fps_signal.connect(lambda f: self.fps_label.setText(f"FPS: {f:.2f}"))
        self.worker.start()

    def closeEvent(self, event):
        if self.worker: self.worker.stop()
        ndi.find_destroy(self.ndi_find)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())