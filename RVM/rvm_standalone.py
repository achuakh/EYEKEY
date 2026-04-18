import torch
import numpy as np
import NDIlib as ndi
import time
import torch.nn.functional as F

# We'll stick to 'run_rvm_final' so it matches your existing call
def run_rvm_final(source_name, model_path):
    print(f"LOG: Initializing RVM on A6000...")
    try:
        # Load architecture from Torch Hub
        model = torch.hub.load('PeterL1n/RobustVideoMatting', 'mobilenetv3')
        model.load_state_dict(torch.load(model_path))
        model = model.eval().cuda().half()
        print("LOG: Model weights loaded onto A6000.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    # NDI RECEIVER SETUP
    r_settings = ndi.RecvCreateV3(color_format=ndi.RECV_COLOR_FORMAT_BGRX_BGRA)
    ndi_recv = ndi.recv_create_v3(r_settings)
    source = ndi.Source()
    source.ndi_name = source_name.encode('ascii')
    ndi.recv_connect(ndi_recv, source)
    
    # NDI SENDER SETUP (python-rvm)
    s_settings = ndi.SendCreate()
    s_settings.ndi_name = "python-rvm"
    ndi_send = ndi.send_create(s_settings)

    # --- HANDSHAKE LOGIC ---
    print(f"LOG: Waiting for {source_name}...")
    connected = False
    for i in range(60): 
        t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 200)
        if t == ndi.FRAME_TYPE_VIDEO:
            print(f"SUCCESS: Connected! {v.xres}x{v.yres}")
            connected = True
            ndi.recv_free_video_v2(ndi_recv, v)
            break
        if i % 10 == 0: print("...searching for NDI packets...")
        time.sleep(0.05)

    if not connected:
        print("ERROR: Connection Timeout.")
        return

    # PROCESSING BUFFERS
    rec_states = [None] * 4 
    out_buffer = np.zeros((1080, 1920, 4), dtype=np.uint8)
    out_v = ndi.VideoFrameV2()

    print("--- [RVM PIPELINE ACTIVE] ---")
    
    try:
        while True:
            t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 33)
            if t != ndi.FRAME_TYPE_VIDEO: continue
            
            start_time = time.time()
            h, w = v.yres, v.xres
            
            # 1. GPU UPLOAD
            raw_pixels = np.frombuffer(v.data, dtype=np.uint8).reshape(h, w, 4)
            frame_tensor = torch.from_numpy(raw_pixels).to(device=0, non_blocking=True)
            
            # 2. PRE-PROCESS
            img = frame_tensor[:, :, :3].permute(2, 0, 1).unsqueeze(0).half() / 255.0
            
            # 3. RVM INFERENCE (0.3 Sweet Spot for A6000)
            with torch.no_grad():
                fgr, pha, *rec_states = model(img, *rec_states, downsample_ratio=0.4)
            
            # --- STEP 4: THE 'SOFT-CORE' RECOVERY ---
            # 1. We ditch the Max Pool (Dilation) as it creates the hard edge.
            # 2. We use a milder gain (1.2) to help the palm.
            pha_boosted = torch.clamp(pha * 1.25, 0, 1)
            
            # 3. THE ALPHA BLEND
            # We blend the 'Boosted' version with the 'Original' version.
            # This ensures the core of your body is opaque (from boosted)
            # but the edges remain soft and anti-aliased (from original).
            # 0.7 weight on boosted fills the palm; 0.3 on original keeps the hair.
            pha_final = (pha_boosted * 0.7) + (pha * 0.3)
            
            # 4. SUBTLE GAMMA (Back to 0.9)
            # 1.0 is neutral. 0.9 slightly thickens the matte without sharpening the edge.
            pha_final = torch.pow(pha_final, 0.9)
            
            alpha = (pha_final.squeeze() * 255).to(torch.uint8)
            frame_tensor[:, :, 3] = alpha
            
            # 5. SYNC DOWNLOAD & SEND
            out_buffer[:h, :w, :] = frame_tensor.cpu().numpy()
            out_v.xres, out_v.yres = w, h
            out_v.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
            out_v.data = out_buffer
            ndi.send_send_video_v2(ndi_send, out_v)
            
            ndi.recv_free_video_v2(ndi_recv, v)
            
            fps = 1.0 / (time.time() - start_time)
            print(f"\rLIVE | FPS: {fps:.2f} | GPU: ~35%", end="")

    except KeyboardInterrupt:
        print("\nLOG: Stopped by user.")
    finally:
        ndi.recv_destroy(ndi_recv)
        ndi.send_destroy(ndi_send)

if __name__ == "__main__":
    # Updated to match your NameError traceback and your scan
    run_rvm_final("IPHONE 247F (HX Camera)", "rvm_mobilenetv3.pth")