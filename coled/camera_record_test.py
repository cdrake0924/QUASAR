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
# -----------------------------------

def initialize_cameras():
    ctx = rs.context()
    initial_devices = ctx.query_devices()
    
    if len(initial_devices) == 0:
        print("Error: No Intel RealSense devices found.")
        sys.exit(1)

    print(f"Found {len(initial_devices)} devices. Sending hardware reset...")
    
    # Step 1: Force a hardware reset on all found devices
    for dev in initial_devices:
        try:
            dev.hardware_reset()
        except Exception as e:
            print(f"Reset failed for a device: {e}")

    # Step 2: WAIT for the OS to re-enumerate the cameras
    # This is the most important part. 5 seconds is usually safe.
    print("Waiting 5 seconds for cameras to reboot and re-register...")
    time.sleep(5)

    # Step 3: Re-query the devices now that they are fresh
    devices = ctx.query_devices()
    pipelines = []
    video_writers = []
    device_names = []

    print(f"Initializing {len(devices)} device(s)...")

    for i, device in enumerate(devices):
        sn = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        device_names.append(sn)
        
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(sn)
        cfg.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
        
        try:
            # Staggered start to prevent USB power/bandwidth spikes
            pipe.start(cfg)
            pipelines.append(pipe)
            
            out_filename = f'cam_{sn}_{i}_test.avi'
            writer = cv2.VideoWriter(out_filename, FOURCC, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            video_writers.append(writer)
            
            print(f"  - Started: {name} (SN: {sn}) -> Saving to {out_filename}")
            time.sleep(1) # Give the bus a moment before the next camera starts
        except Exception as e:
            print(f"  - Failed to start {sn}: {e}")

    return pipelines, video_writers, device_names

# --- Main Execution ---

pipelines, video_writers, device_names = initialize_cameras()

intrinsics = pipelines[0].get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print(intrinsics)

if not pipelines:
    print("No pipelines were started. Exiting.")
    sys.exit(1)

print(f"\nRecording for {RECORDING_DURATION_SEC} seconds... Press 'q' to stop.")

start_time = time.time()

try:
    while (time.time() - start_time) < RECORDING_DURATION_SEC:
        frames_list = []
        
        for i, pipe in enumerate(pipelines):
            # Using try_wait_for_frames prevents a full crash if one camera lags
            success, frames = pipe.try_wait_for_frames(timeout_ms=100)
            if not success:
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            
            # Write to the specific file
            video_writers[i].write(color_image)
            
            # Add label for display
            cv2.putText(color_image, f"SN: {device_names[i]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frames_list.append(color_image)

        if frames_list:
            display_image = np.hstack(frames_list)
            cv2.imshow('Multi-Camera Stream', display_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nClosing streams and releasing resources...")
    for pipe in pipelines:
        try:
            pipe.stop()
        except:
            pass
    for writer in video_writers:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")