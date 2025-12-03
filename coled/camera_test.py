import pyrealsense2 as rs
import cv2
import sys
import numpy as np


# --- CHECK FOR CONNECTED DEVICES ---
ctx = rs.context()
devices = ctx.query_devices()

# 1. Check if the list of devices is empty
if len(devices) == 0:
    print("Error: No Intel RealSense device connected.")
    print("Please check USB connection and ensure the camera is powered on.")
    sys.exit(1) # Exit the script if no camera is found

# 2. If devices are found, print their information
print(f"Found {len(devices)} Intel RealSense device(s):")
for device in devices:
    print(f"  - Name: {device.get_info(rs.camera_info.name)}")
    print(f"  - Serial Number: {device.get_info(rs.camera_info.serial_number)}")
    print("---")

# The rest of your original code starts here:
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# ... (rest of your configuration and streaming code)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        #depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame.get_data(), alpha=0.03), cv2.COLORMAP_JET)
        color_image = np.asanyarray(color_frame.get_data())

        # Render images
        cv2.imshow('Color Stream', color_image)
        #cv2.imshow('Depth Stream', depth_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
