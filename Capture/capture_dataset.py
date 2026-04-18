import NDIlib as ndi
import cv2
import numpy as np
import os
import time
import threading

# 1. INITIALIZATION & DIRECTORIES (Vocal Version)
import os

# Using absolute paths to remove all doubt
BASE_DIR = os.path.expanduser('~/CODE/EYEKEY/dataset/train2')
REAL_IMG_DIR = os.path.join(BASE_DIR, 'images')
REAL_MSK_DIR = os.path.join(BASE_DIR, 'masks')

try:
    os.makedirs(REAL_IMG_DIR, exist_ok=True)
    os.makedirs(REAL_MSK_DIR, exist_ok=True)
    print(f"✅ Directory check passed:")
    print(f"   📂 Images: {os.path.abspath(REAL_IMG_DIR)}")
    print(f"   📂 Masks:  {os.path.abspath(REAL_MSK_DIR)}")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not create directories. {e}")
    exit()

# 2. THE STORAGE TAPE (RAM)
# We store raw data here to avoid ALL disk I/O during capture
video_buffer = [] 
buffer_lock = threading.Lock()
latest_fill = None
latest_key = None

# 3. THE SIPHON THREADS
def siphon_stream(ndi_source, is_fill):
    global latest_fill, latest_key
    recv_create = ndi.RecvCreateV3()
    recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
    recv = ndi.recv_create_v3(recv_create)
    ndi.recv_connect(recv, ndi_source)
    
    while True:
        t, v, _, _ = ndi.recv_capture_v3(recv, 10)
        if t == ndi.FRAME_TYPE_VIDEO:
            with buffer_lock:
                if is_fill:
                    latest_fill = np.copy(v.data)
                else:
                    latest_key = np.copy(v.data)
            ndi.recv_free_video_v2(recv, v)
        time.sleep(0.001)

# 4. START NETWORKING
if not ndi.initialize():
    exit()

ndi_find = ndi.find_create_v2()
time.sleep(2)
sources = ndi.find_get_current_sources(ndi_find)

threading.Thread(target=siphon_stream, args=(sources[0], True), daemon=True).start()
threading.Thread(target=siphon_stream, args=(sources[1], False), daemon=True).start()

# 5. THE CLEAN BURST CAPTURE
print(f"🚀 BURST MODE ACTIVE. Using sources: {sources[0].ndi_name} and {sources[1].ndi_name}")

CAPTURE_INTERVAL = 0.2  # <--- ADD THIS LINE (0.2 = 5 frames per second)

try:
    while True:
        start_time = time.time()
        
        with buffer_lock:
            # Only record if both Fill and Key have arrived
            if latest_fill is not None and latest_key is not None:
                video_buffer.append((latest_fill, latest_key))
                # This line only prints the count, no 'False' warnings
                print(f"\r📦 Total Dataset Frames in RAM: {len(video_buffer)}", end="")
            else:
                # Silent wait for the NDI threads to catch the first frames
                pass

        elapsed = time.time() - start_time
        time.sleep(max(0, CAPTURE_INTERVAL - elapsed))

except KeyboardInterrupt:
    print(f"\n\n⏹️ Capture stopped.")
    
    if not video_buffer:
        print("❌ ERROR: No frames were captured to RAM. Check NDI sources!")
    else:
        # 6. THE FLUSH
        print(f"💾 Flushing {len(video_buffer)} frames to Disk...")
    start_idx = len(os.listdir(REAL_IMG_DIR))
    
    for i, (f_data, k_data) in enumerate(video_buffer):
        current_idx = start_idx + i
        
        # Process and Save
        fill_rgb = cv2.cvtColor(f_data, cv2.COLOR_BGRA2BGR)
        key_gray = cv2.cvtColor(k_data, cv2.COLOR_BGRA2GRAY)
        
        cv2.imwrite(os.path.join(REAL_IMG_DIR, f"{current_idx:04d}.jpg"), fill_rgb)
        cv2.imwrite(os.path.join(REAL_MSK_DIR, f"{current_idx:04d}.png"), key_gray)
        
        if i % 50 == 0:
            print(f"  -> Progress: {i}/{len(video_buffer)}")

    print("✨ SUCCESS. All frames safely written to the dataset folder.")

finally:
    ndi.destroy()