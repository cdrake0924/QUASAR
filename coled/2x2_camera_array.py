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
CHECKERBOARD_SIZE = (8,6)
SQUARE_SIZE = 30 #mm
DEPTH = 1524 #mm
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

#I got this from  google gemini
#It is supposed to be code to get the focal length from meta data from Intel
intrinsics = pipelines[0].get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print(intrinsics)
focal_length = intrinsics.fx
print(focal_length)

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

        gray_l = cv2.cvtColor(frames_list[0], cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frames_list[1], cv2.COLOR_BGR2GRAY)

        found_checker_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD_SIZE, None)
        found_checker_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD_SIZE, None)

        if(found_checker_l and found_checker_r):
            disparity = abs(corners_l[0][0][0] - corners_r[0][0][0])
            if(disparity > 0):
                baseline = (DEPTH * disparity) / focal_length
                print(f"Calculated Distance = {baseline:.2f} mm")


        frame_count += 1
    all_disparities = [abs(corners_l[i][0][0] - corners_r[i][0][0]) for i in range(len(corners_l))]
    avg_disparity = sum(all_disparities) / len(all_disparities)
    avg_baseline = (DEPTH * avg_disparity) / focal_length
    print(f"Average Baseline: {avg_baseline:.2f}")

    # --- NEW: Reprojection Error Calculation ---
    print("\nEvaluating Calibration Quality...")
    
    # 1. Define the real-world coordinates of the checkerboard squares
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # Scale by your square size (30mm)

    # 2. Use the intrinsics you got from the RealSense
    # We convert the RealSense intrinsics object into a standard OpenCV matrix
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                              [0, intrinsics.fy, intrinsics.ppy],
                              [0, 0, 1]], dtype=np.float32)
    print(camera_matrix)
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)

    # 3. Calculate error for the last captured frame
    if found_checker_l:
        # SolvePnP finds the pose of the camera (Rotation and Translation)
        ret, rvec, tvec = cv2.solvePnP(objp, corners_l, camera_matrix, dist_coeffs)
        
        # Project the 3D points back onto the 2D image plane
        imgpoints_projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        
        # Calculate the distance between detected corners and projected points
        error = cv2.norm(corners_l, imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
        print(f"Reprojection Error (Left Cam): {error:.4f} pixels")
        
        if error > 1.0:
            print("Warning: High error! FoundationStereo point clouds may be distorted.")

finally:
    # Cleanup all resources
    print("\nClosing streams...")
    for pipe in pipelines:
        pipe.stop()
    for writer in video_writers:
        writer.release()
    cv2.destroyAllWindows()
    print("Recording finished.")