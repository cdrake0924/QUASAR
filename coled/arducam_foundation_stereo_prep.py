import cv2
import numpy as np
import os
import sys
import time

"""
arducam_foundation_stereo_prep.py

Run this AFTER arducam_stereo_calibration.py.

Does three things required before FoundationStereo:
  1. Loads stereo calibration and computes rectification maps
  2. Captures a live stereo image pair, undistorts and rectifies both frames
  3. Writes the K.txt intrinsic file FoundationStereo expects:
       Line 1: flattened 1x9 camera matrix (from the rectified projection matrix)
       Line 2: baseline in meters

Usage:
  # Live capture (default)
  python arducam_foundation_stereo_prep.py

  # From existing images
  python arducam_foundation_stereo_prep.py --left left.png --right right.png

Output files (saved to --out-dir, default: foundation_stereo_input/):
  left_rect.png   — rectified + undistorted left image
  right_rect.png  — rectified + undistorted right image
  K.txt           — intrinsic file for FoundationStereo
"""

# --- Configuration ---
FRAME_WIDTH   = 1920
FRAME_HEIGHT  = 1080
FPS           = 15
CAM0_INDEX    = 1       # Left camera
CAM1_INDEX    = 2       # Right camera

STEREO_CALIB_FILE = "arducam_stereo_calib.npz"
OUTPUT_DIR        = "foundation_stereo_input"
# ---------------------


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Camera {index} opened: {w}x{h} @ {fps:.1f} fps")
    return cap


def compute_rectification_maps(mtx0, dist0, mtx1, dist1, R, T, img_size):
    """
    Compute the rectification rotation matrices and projection matrices
    for both cameras, then build the pixel remap arrays.

    stereoRectify produces:
      R0, R1  : rotation to apply to each camera to make them coplanar
      P0, P1  : new projection matrices in the rectified coordinate system
      Q       : disparity-to-depth mapping matrix (useful later for 3D reconstruction)
      map0x/y : pixel remap arrays for cam0
      map1x/y : pixel remap arrays for cam1

    CALIB_ZERO_DISPARITY puts the principal points at the same pixel location
    in both rectified images — generally gives the best results.
    """
    R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
        mtx0, dist0,
        mtx1, dist1,
        img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0          # alpha=0 crops to only valid pixels (no black borders)
    )

    map0x, map0y = cv2.initUndistortRectifyMap(
        mtx0, dist0, R0, P0, img_size, cv2.CV_32FC1
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        mtx1, dist1, R1, P1, img_size, cv2.CV_32FC1
    )

    return map0x, map0y, map1x, map1y, P0, P1, Q


def rectify_images(img0, img1, map0x, map0y, map1x, map1y):
    """Apply rectification maps to both images."""
    rect0 = cv2.remap(img0, map0x, map0y, cv2.INTER_LINEAR)
    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    return rect0, rect1


def write_K_txt(P0, baseline_m: float, output_path: str):
    """
    Write the intrinsic file FoundationStereo expects.

    Format (K.txt):
      Line 1: 9 space-separated floats — the flattened 3x3 camera matrix
              taken from the LEFT rectified projection matrix P0
      Line 2: baseline in meters

    P0 is a 3x4 matrix; we take the top-left 3x3 block as K.
    After rectification P0[:3,:3] == P1[:3,:3] (same focal length, same
    principal point), which is why we only need to write one matrix.
    """
    K = P0[:3, :3]
    flat = K.flatten()
    with open(output_path, 'w') as f:
        f.write(' '.join(f'{v:.6f}' for v in flat) + '\n')
        f.write(f'{baseline_m:.6f}\n')
    print(f"  K.txt written: {output_path}")
    print(f"    Intrinsic matrix K:\n{K}")
    print(f"    Baseline: {baseline_m*1000:.2f} mm ({baseline_m:.6f} m)")


def capture_live_pair(cap0, cap1):
    """
    Show a live side-by-side preview and capture one frame pair on spacebar.
    Press 'q' to quit without capturing.
    """
    print("\nLive preview — press SPACE to capture a frame pair, 'q' to quit.")
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            time.sleep(0.05)
            continue

        # Downsample for display only (1920x1080 side-by-side is very wide)
        scale = 0.4
        disp0 = cv2.resize(frame0, None, fx=scale, fy=scale)
        disp1 = cv2.resize(frame1, None, fx=scale, fy=scale)
        combined = np.hstack([disp0, disp1])
        cv2.putText(combined, "CAM 0 (LEFT)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, "CAM 1 (RIGHT)", (disp0.shape[1] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Stereo Preview — SPACE to capture, Q to quit", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.destroyAllWindows()
            return frame0, frame1
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None, None


def draw_epipolar_lines(rect0, rect1, n_lines=10):
    """
    Draw horizontal epipolar lines on both rectified images side-by-side.
    After correct rectification, matching points lie on the same horizontal line.
    This is a visual sanity check — the lines should pass through corresponding
    features in both images at the same Y coordinate.
    """
    h = rect0.shape[0]
    vis0 = rect0.copy()
    vis1 = rect1.copy()
    step = h // (n_lines + 1)
    for i in range(1, n_lines + 1):
        y = i * step
        color = (0, int(255 * i / n_lines), int(255 * (1 - i / n_lines)))
        cv2.line(vis0, (0, y), (vis0.shape[1], y), color, 1)
        cv2.line(vis1, (0, y), (vis1.shape[1], y), color, 1)
    combined = np.hstack([vis0, vis1])
    scale = 0.4
    combined = cv2.resize(combined, None, fx=scale, fy=scale)
    cv2.imshow("Epipolar lines — press any key to close", combined)
    print("\nEpipolar line check: corresponding features should lie on the same")
    print("colored horizontal line in both images. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rectify stereo pair and generate K.txt for FoundationStereo."
    )
    parser.add_argument(
        "--stereo-calib", default=STEREO_CALIB_FILE,
        help=f"Path to stereo calibration .npz (default: {STEREO_CALIB_FILE})"
    )
    parser.add_argument(
        "--left", default=None,
        help="Path to existing left image (skips live capture)"
    )
    parser.add_argument(
        "--right", default=None,
        help="Path to existing right image (skips live capture)"
    )
    parser.add_argument(
        "--cam0-index", type=int, default=CAM0_INDEX,
        help=f"Left camera index (default: {CAM0_INDEX})"
    )
    parser.add_argument(
        "--cam1-index", type=int, default=CAM1_INDEX,
        help=f"Right camera index (default: {CAM1_INDEX})"
    )
    parser.add_argument(
        "--out-dir", default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}/)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load stereo calibration ------------------------------------------
    if not os.path.exists(args.stereo_calib):
        print(f"Error: stereo calibration file not found: '{args.stereo_calib}'")
        print("Run arducam_stereo_calibration.py first.")
        sys.exit(1)

    print(f"Loading stereo calibration from: {args.stereo_calib}")
    data   = np.load(args.stereo_calib)
    mtx0   = data['mtx0'];  dist0 = data['dist0']
    mtx1   = data['mtx1'];  dist1 = data['dist1']
    R      = data['R']
    T      = data['T']

    baseline_mm = float(np.linalg.norm(T))
    baseline_m  = baseline_mm / 1000.0
    print(f"  Baseline: {baseline_mm:.2f} mm ({baseline_m:.4f} m)")

    # ---- Get image pair ---------------------------------------------------
    if args.left and args.right:
        print(f"\nLoading images: {args.left}, {args.right}")
        img0 = cv2.imread(args.left)
        img1 = cv2.imread(args.right)
        if img0 is None or img1 is None:
            print("Error: could not read one or both image files.")
            sys.exit(1)
    else:
        print(f"\nOpening cameras {args.cam0_index} (left) and {args.cam1_index} (right)...")
        cap0 = open_camera(args.cam0_index)
        cap1 = open_camera(args.cam1_index)
        img0, img1 = capture_live_pair(cap0, cap1)
        cap0.release()
        cap1.release()
        if img0 is None:
            print("No image captured. Exiting.")
            sys.exit(0)

    img_size = (img0.shape[1], img0.shape[0])  # (width, height)

    # ---- Compute rectification maps ---------------------------------------
    print("\nComputing rectification maps...")
    map0x, map0y, map1x, map1y, P0, P1, Q = compute_rectification_maps(
        mtx0, dist0, mtx1, dist1, R, T, img_size
    )

    # ---- Rectify images ---------------------------------------------------
    print("Rectifying images...")
    rect0, rect1 = rectify_images(img0, img1, map0x, map0y, map1x, map1y)

    # ---- Save rectified images --------------------------------------------
    left_out  = os.path.join(args.out_dir, "left_rect.png")
    right_out = os.path.join(args.out_dir, "right_rect.png")
    cv2.imwrite(left_out,  rect0)
    cv2.imwrite(right_out, rect1)
    print(f"\nSaved rectified images:")
    print(f"  Left:  {left_out}")
    print(f"  Right: {right_out}")

    # ---- Write K.txt for FoundationStereo ---------------------------------
    k_txt_path = os.path.join(args.out_dir, "K.txt")
    write_K_txt(P0, baseline_m, k_txt_path)

    # ---- Also save Q matrix (useful for depth reconstruction later) -------
    np.save(os.path.join(args.out_dir, "Q.npy"), Q)
    print(f"  Q matrix saved: {os.path.join(args.out_dir, 'Q.npy')}")

    # ---- Epipolar line sanity check ---------------------------------------
    draw_epipolar_lines(rect0, rect1)

    # ---- Print FoundationStereo command -----------------------------------
    print("\n--- Ready for FoundationStereo ---")
    print("Run:")
    print(f"  python scripts/run_demo.py \\")
    print(f"    --left_file  {left_out} \\")
    print(f"    --right_file {right_out} \\")
    print(f"    --intrinsic_file {k_txt_path} \\")
    print(f"    --out_dir output/ \\")
    print(f"    --get_pc 1 --scale 1")