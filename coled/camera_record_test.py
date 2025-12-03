import pyrealsense2 as rs
import cv2
import sys
import numpy as np
import time

# --- Configuration for Recording ---
RECORDING_DURATION_SEC = 10  # Set the duration for recording in seconds
OUTPUT_FILENAME = 'test_video.avi'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
BRIGHTNESS_THRESHOLD = 20 # Average pixel brightness for synchronization
# -----------------------------------

ctx = rs.context()
devices = ctx.query_devices()

# 1. Check if the list of devices is empty
if len(devices) == 0:
    print("Error: No Intel RealSense device connected.")
    print("Please check USB connection and ensure the camera is powered on.")
    sys.exit(1)

# 2. If devices are found, print their information
print(f"Found {len(devices)} Intel RealSense device(s):")
for device in devices:
    print(f"  - Name: {device.get_info(rs.camera_info.name)}")
    print(f"  - Serial Number: {device.get_info(rs.camera_info.serial_number)}")
    print("---")

pipeline = rs.pipeline()
config = rs.config()

# Configure color stream
config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)

# Start streaming
profile = pipeline.start(config)

# Initialize Video Writer
video_writer = cv2.VideoWriter(OUTPUT_FILENAME, FOURCC, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
print(f"Recording started. Saving to '{OUTPUT_FILENAME}' for {RECORDING_DURATION_SEC} seconds...")

start_time = time.time()
trigger_frame_index = -1
trigger_frame_found = False
frame_count = 0
try:
    while (time.time() - start_time) < RECORDING_DURATION_SEC:
        # Get frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        avg_brightness = np.mean(color_image)
        print(avg_brightness)

        # Write the frame to the video file
        video_writer.write(color_image)

        #trigger when brightness threshold changes
        #shows the frame index where that is found
        if not trigger_frame_found and avg_brightness > BRIGHTNESS_THRESHOLD:
            trigger_frame_found = True
            trigger_frame_index = frame_count
            trigger_timestamp_ms = color_frame.get_timestamp()
            print(f"\n✨ Trigger Detected! ✨")
            print(f"Frame Index: {trigger_frame_index}")
            print(f"Timestamp: {trigger_timestamp_ms / 1000.0:.3f} seconds")
            print(f"Average Brightness: {avg_brightness:.2f}\n")
        frame_count += 1
        
        #Display while recording for testing
        cv2.imshow('Recording Stream', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming and close video writer
    pipeline.stop()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Recording finished and saved to '{OUTPUT_FILENAME}'.")