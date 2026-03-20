import cv2
import numpy as np
import os
import sys
import time

# --- Configuration ---
CHECKERBOARD_SIZE = (8, 6)    # Must match what was used for intrinsic calibration
SQUARE_SIZE_MM    = 30        # Must match what was used for intrinsic calibration
FRAME_WIDTH       = 1920
FRAME_HEIGHT      = 1080
FPS               = 15
CAPTURE_COUNT     = 20        # Number of simultaneous frame pairs to collect
CAPTURE_INTERVAL  = 1.0       # Seconds between captures (longer than intrinsic — need both cameras stable)
DW                = (11, 11)    # Corner refinement window

# Camera indices
CAM0_INDEX = 1
CAM1_INDEX = 2

# Intrinsic calibration files (produced by arducam_intrinsic_calibration.py)
CAM0_INTRINSIC = "arducam_intrinsic_calib_0.npz"
CAM1_INTRINSIC = "arducam_intrinsic_calib_1.npz"

# Output
OUTPUT_FILE = "arducam_stereo_calib.npz"
# ---------------------


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera via DirectShow (Windows) and set resolution."""
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


def capture_stereo_frames(cap0: cv2.VideoCapture,
                           cap1: cv2.VideoCapture,
                           checkerboard: tuple,
                           n_frames: int,
                           interval_sec: float,
                           save_dir: str):
    """
    Simultaneously capture checkerboard frames from both cameras.

    A pair is only saved when the checkerboard is detected in BOTH frames
    at the same time — this is critical for stereo calibration accuracy.

    Returns
    -------
    paths0, paths1 : lists of saved file paths for cam0 and cam1
    """
    os.makedirs(os.path.join(save_dir, "cam0"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "cam1"), exist_ok=True)

    paths0, paths1 = [], []
    last_capture_time = 0.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

    print(f"\nCapturing {n_frames} stereo frame pairs.")
    print("The checkerboard must be visible to BOTH cameras simultaneously.")
    print("Press 'q' to stop early.\n")

    while len(paths0) < n_frames:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("  Warning: failed to grab frame from one or both cameras.")
            time.sleep(0.05)
            continue

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        found0, corners0 = cv2.findChessboardCorners(gray0, checkerboard, None)
        found1, corners1 = cv2.findChessboardCorners(gray1, checkerboard, None)

        display0 = frame0.copy()
        display1 = frame1.copy()

        now = time.time()
        both_found = found0 and found1

        if both_found:
            corners0 = cv2.cornerSubPix(gray0, corners0, DW, (-1, -1), criteria)
            corners1 = cv2.cornerSubPix(gray1, corners1, DW, (-1, -1), criteria)
            cv2.drawChessboardCorners(display0, checkerboard, corners0, found0)
            cv2.drawChessboardCorners(display1, checkerboard, corners1, found1)

            if (now - last_capture_time) >= interval_sec:
                idx = len(paths0)
                p0 = os.path.join(save_dir, "cam0", f"stereo_{idx:03d}.png")
                p1 = os.path.join(save_dir, "cam1", f"stereo_{idx:03d}.png")
                cv2.imwrite(p0, frame0)
                cv2.imwrite(p1, frame1)
                paths0.append(p0)
                paths1.append(p1)
                last_capture_time = now
                print(f"  Captured pair {len(paths0)}/{n_frames}")
                label = f"CAPTURED {len(paths0)}/{n_frames}"
                color = (0, 255, 0)
            else:
                label = f"Both found — {len(paths0)}/{n_frames}"
                color = (0, 200, 255)
        else:
            # Show which camera(s) can't see the board
            if not found0:
                cv2.putText(display0, "NOT FOUND", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if not found1:
                cv2.putText(display1, "NOT FOUND", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            label = f"Need both cameras — {len(paths0)}/{n_frames}"
            color = (0, 0, 255)

        cv2.putText(display0, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display1, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show both cameras side by side
        combined = np.hstack([display0, display1])
        cv2.putText(combined, "CAM 0", (10, FRAME_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "CAM 1", (FRAME_WIDTH + 10, FRAME_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Stereo Calibration Capture", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("  Stopped early by user.")
            break

    cv2.destroyAllWindows()
    print(f"\nTotal stereo pairs captured: {len(paths0)}")
    return paths0, paths1


def computeStereoCalibration(paths0: list, paths1: list,
                              mtx0, dist0, mtx1, dist1,
                              checkerboard: tuple):
    """
    Run stereo calibration from paired checkerboard image paths.

    Uses the intrinsics from each camera as a starting point and refines
    the relative pose (R, T) between them.

    Returns
    -------
    R  : 3x3 rotation matrix from cam0 to cam1
    T  : 3x1 translation vector from cam0 to cam1 (in mm)
    E  : essential matrix
    F  : fundamental matrix
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

    # Build world-coordinate grid (same as intrinsic calibration)
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints  = []
    imgpoints0 = []
    imgpoints1 = []
    img_shape  = None

    print(f"\nProcessing {len(paths0)} stereo frame pair(s)...")

    for p0, p1 in zip(paths0, paths1):
        img0 = cv2.imread(p0)
        img1 = cv2.imread(p1)
        if img0 is None or img1 is None:
            print(f"  Warning: could not read pair ({p0}, {p1}), skipping.")
            continue

        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img_shape = gray0.shape[::-1]

        found0, corners0 = cv2.findChessboardCorners(gray0, checkerboard, None)
        found1, corners1 = cv2.findChessboardCorners(gray1, checkerboard, None)

        if found0 and found1:
            corners0 = cv2.cornerSubPix(gray0, corners0, DW, (-1, -1), criteria)
            corners1 = cv2.cornerSubPix(gray1, corners1, DW, (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints0.append(corners0)
            imgpoints1.append(corners1)
        else:
            print(f"  Warning: checkerboard not found in pair, skipping.")

    if len(objpoints) < 4:
        raise RuntimeError(
            f"Only {len(objpoints)} valid pair(s) found. "
            "Need at least 4 for reliable stereo calibration."
        )

    print(f"Running stereo calibration on {len(objpoints)} valid pair(s)...")

    # cv2.CALIB_FIX_INTRINSIC tells stereoCalibrate to trust the intrinsics
    # we already computed and only solve for R and T
    flags = cv2.CALIB_FIX_INTRINSIC

    ret, mtx0, dist0, mtx1, dist1, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints0,
        imgpoints1,
        mtx0, dist0,
        mtx1, dist1,
        img_shape,
        criteria=criteria,
        flags=flags
    )

    # Baseline = magnitude of T (distance between cameras in mm)
    baseline_mm = np.linalg.norm(T)

    print("\n--- Stereo Calibration Results ---")
    print(f"RMS reprojection error: {ret:.4f} px")
    if ret > 1.0:
        print("  Warning: error > 1 px — recapture with more varied board positions.")
    else:
        print("  Calibration looks good.")

    print(f"\nBaseline (distance between cameras): {baseline_mm:.2f} mm  "
          f"({baseline_mm/25.4:.2f} inches)")

    print("\nRotation matrix (R) — cam0 to cam1:")
    print(R)
    print("\nTranslation vector (T) in mm — cam0 to cam1:")
    print(T)

    return R, T, E, F, ret


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stereo extrinsic calibration for two Arducam cameras."
    )
    parser.add_argument(
        "--cam0-intrinsic", default=CAM0_INTRINSIC,
        help=f"Path to cam0 intrinsic .npz file (default: {CAM0_INTRINSIC})"
    )
    parser.add_argument(
        "--cam1-intrinsic", default=CAM1_INTRINSIC,
        help=f"Path to cam1 intrinsic .npz file (default: {CAM1_INTRINSIC})"
    )
    parser.add_argument(
        "--cam0-index", type=int, default=CAM0_INDEX,
        help=f"Camera index for cam0 (default: {CAM0_INDEX})"
    )
    parser.add_argument(
        "--cam1-index", type=int, default=CAM1_INDEX,
        help=f"Camera index for cam1 (default: {CAM1_INDEX})"
    )
    parser.add_argument(
        "--save-dir", default="stereo_frames",
        help="Directory for captured stereo frame pairs (default: stereo_frames/)"
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE,
        help=f"Path for the output .npz file (default: {OUTPUT_FILE})"
    )
    args = parser.parse_args()

    # ---- Load intrinsics ---------------------------------------------------
    print("Loading intrinsic calibration files...")
    for path in [args.cam0_intrinsic, args.cam1_intrinsic]:
        if not os.path.exists(path):
            print(f"Error: intrinsic file not found: '{path}'")
            print("Run arducam_intrinsic_calibration.py for each camera first.")
            sys.exit(1)

    cam0_data = np.load(args.cam0_intrinsic)
    cam1_data = np.load(args.cam1_intrinsic)
    mtx0, dist0 = cam0_data['mtx'], cam0_data['dist']
    mtx1, dist1 = cam1_data['mtx'], cam1_data['dist']
    print(f"  Cam0 intrinsics loaded from: {args.cam0_intrinsic}")
    print(f"  Cam1 intrinsics loaded from: {args.cam1_intrinsic}")

    # ---- Open both cameras -------------------------------------------------
    print(f"\nOpening cameras {args.cam0_index} and {args.cam1_index}...")
    cap0 = open_camera(args.cam0_index)
    cap1 = open_camera(args.cam1_index)

    # ---- Capture stereo pairs ----------------------------------------------
    paths0, paths1 = capture_stereo_frames(
        cap0, cap1,
        CHECKERBOARD_SIZE, CAPTURE_COUNT, CAPTURE_INTERVAL,
        args.save_dir
    )
    cap0.release()
    cap1.release()

    if len(paths0) < 4:
        print("Error: too few stereo pairs captured. Exiting.")
        sys.exit(1)

    # ---- Run stereo calibration --------------------------------------------
    R, T, E, F, rms = computeStereoCalibration(
        paths0, paths1,
        mtx0, dist0, mtx1, dist1,
        CHECKERBOARD_SIZE
    )

    # ---- Save results ------------------------------------------------------
    np.savez(
        args.output,
        R=R, T=T, E=E, F=F,
        mtx0=mtx0, dist0=dist0,
        mtx1=mtx1, dist1=dist1,
        rms=rms
    )
    print(f"\nStereo calibration saved to: {args.output}")
    print("\nLoad with:")
    print("  data = np.load('arducam_stereo_calib.npz')")
    print("  R, T = data['R'], data['T']   # extrinsics")
    print("  mtx0, dist0 = data['mtx0'], data['dist0']")
    print("  mtx1, dist1 = data['mtx1'], data['dist1']")