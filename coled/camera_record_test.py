import pyrealsense2 as rs
import cv2
import sys
import numpy as np
import time

# --- Configuration for Recording ---
RECORDING_DURATION_SEC = 10
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
BRIGHTNESS_THRESHOLD = 20 
# -----------------------------------

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("Error: No Intel RealSense device connected.")
    sys.exit(1)

pipelines = []
video_writers = []
device_names = []

print(f"Initializing {len(devices)} device(s)...")

for i, device in enumerate(devices):
    sn = device.get_info(rs.camera_info.serial_number)
    name = device.get_info(rs.camera_info.name)
    device_names.append(sn)
    
    # 1. Create a unique pipeline and config for each device
    pipe = rs.pipeline()
    cfg = rs.config()
    
    # 2. Enable device by specific serial number
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
    
    # 3. Start the pipeline
    pipe.start(cfg)
    pipelines.append(pipe)
    
    # 4. Create a unique video writer for each device
    out_filename = f'cam_{sn}_{i}.avi'
    writer = cv2.VideoWriter(out_filename, FOURCC, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    video_writers.append(writer)
    
    print(f"  - Started: {name} (SN: {sn}) -> Saving to {out_filename}")

print(f"\nRecording for {RECORDING_DURATION_SEC} seconds...")

start_time = time.time()
frame_count = 0

try:
    while (time.time() - start_time) < RECORDING_DURATION_SEC:
        frames_list = []
        
        for i, pipe in enumerate(pipelines):
            # Wait for frames from this specific camera
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            
            # Logic for individual camera (Brightness check)
            avg_brightness = np.mean(color_image)
            if avg_brightness > BRIGHTNESS_THRESHOLD:
                # You can add per-camera trigger logic here if needed
                pass

            # Write to the specific file for this camera
            video_writers[i].write(color_image)
            
            # Label the image for the preview window
            cv2.putText(color_image, f"SN: {device_names[i]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frames_list.append(color_image)

        # 5. Display all cameras side-by-side (Horizontal concatenation)
        if frames_list:
            display_image = np.hstack(frames_list)
            cv2.imshow('Multi-Camera Stream', display_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

finally:
    # Cleanup all resources
    print("\nClosing streams...")
    for pipe in pipelines:
        pipe.stop()
    for writer in video_writers:
        writer.release()
    cv2.destroyAllWindows()
    print("Recording finished.")