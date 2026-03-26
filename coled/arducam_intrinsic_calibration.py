import cv2
import numpy as np
import glob
import os
import sys
import time

# --- Configuration ---
CHECKERBOARD_SIZE = (8, 6)       # Number of inner corners (cols, rows)
SQUARE_SIZE_MM    = 30           # Physical size of each square in mm
FRAME_WIDTH       = 1280
FRAME_HEIGHT      = 720
FPS               = 15
CAPTURE_COUNT     = 20           # How many valid frames to collect before calibrating
CAPTURE_INTERVAL  = 0.5          # Minimum seconds between captures (avoid blurry duplicates)

# Corner sub-pixel refinement window — use smaller values (e.g. (4,4)) for low-res images
DW = (11, 11)

# Where to save the result
OUTPUT_DIR  = "intrinsic"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "arducam_intrinsic_calib.npz")
# ---------------------


def find_arducam_index(max_index: int = 10) -> int:
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"  Found camera at index {idx}")
                return idx
    return -1


def open_arducam(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}.")

    # MJPG often unlocks higher resolutions on USB webcams/Arducam devices.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Camera opened: {w}x{h} @ {fps:.1f} fps")
    if w < FRAME_WIDTH or h < FRAME_HEIGHT:
        print(
            f"  Warning: camera {index} did not accept requested {FRAME_WIDTH}x{FRAME_HEIGHT}. "
            "Calibration may be invalid if you expected a different resolution."
        )
    return cap


def computeIntrinsic(images: list, checkerboard: tuple, dW: tuple):
    """
    Compute camera intrinsics from a list of checkerboard image file paths.

    Follows the instructor-provided pattern from the course:
      - objpoints: real-world 3-D corner coordinates (Z=0 plane, units = squares)
      - imgpoints: detected 2-D sub-pixel corner coordinates

    Parameters
    ----------
    images      : list of file paths to calibration images
    checkerboard: (cols, rows) inner corner count, e.g. (8, 6)
    dW          : corner-refinement half-window, e.g. (8, 8)

    Returns
    -------
    mtx  : 3x3 camera matrix
    dist : distortion coefficients (k1, k2, p1, p2, k3)
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

    objpoints = []  # 3-D world points across all images
    imgpoints = []  # 2-D image points across all images

    # Build the template world-coordinate grid (Z = 0)
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    # Scale so units are millimetres
    objp *= SQUARE_SIZE_MM

    img_shape = None

    print(f"\nProcessing {len(images)} calibration image(s)...")
    print("Press any key to advance through each image.")

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"  Warning: could not read '{fname}', skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (width, height)

        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, dW, (-1, -1), criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
            cv2.putText(img, "OK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 2)
        else:
            cv2.putText(img, "NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 2)
            print(f"  Warning: checkerboard not found in '{os.path.basename(fname)}'")

        cv2.imshow("Calibration Frames", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    if len(objpoints) < 4:
        raise RuntimeError(
            f"Only {len(objpoints)} valid frame(s) found. "
            "Need at least 4 for reliable calibration."
        )

    print(f"\nRunning calibration on {len(objpoints)} valid frame(s)...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    # --- Reprojection error ---
    total_error = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        total_error += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    mean_error = total_error / len(objpoints)

    print("\n--- Intrinsic Calibration Results ---")
    print("Camera matrix (mtx):\n", mtx)
    print("\nDistortion coefficients (dist):\n", dist)
    print(f"\nMean reprojection error: {mean_error:.4f} px")
    if mean_error > 1.0:
        print("  Warning: error > 1 px — consider recapturing with a flatter board "
              "or more varied angles.")
    else:
        print("  Calibration looks good.")

    return mtx, dist


def capture_calibration_frames(cap: cv2.VideoCapture,
                                checkerboard: tuple,
                                n_frames: int,
                                interval_sec: float,
                                save_dir: str) -> list:
    """
    Live-capture calibration frames from the Arducam.

    Shows a live feed; automatically saves a frame whenever:
      - the checkerboard is detected, AND
      - at least `interval_sec` have passed since the last capture.

    Press 'q' to stop early (will calibrate with whatever was captured).

    Returns a list of saved file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    last_capture_time = 0.0

    print(f"\nCapturing {n_frames} frames. Move the board to different angles and distances.")
    print("Press 'q' to stop early.\n")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

    while len(saved_paths) < n_frames:
        ret, frame = cap.read()
        if not ret:
            print("  Warning: failed to grab frame.")
            time.sleep(0.05)
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        now = time.time()
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, DW, (-1, -1), criteria)
            cv2.drawChessboardCorners(display, checkerboard, corners2, found)

            if (now - last_capture_time) >= interval_sec:
                fname = os.path.join(save_dir, f"calib_{len(saved_paths):03d}.png")
                cv2.imwrite(fname, frame)
                saved_paths.append(fname)
                last_capture_time = now
                cv2.putText(display,
                            f"CAPTURED {len(saved_paths)}/{n_frames}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                print(f"  Captured frame {len(saved_paths)}/{n_frames}")
            else:
                cv2.putText(display,
                            f"Board found — {len(saved_paths)}/{n_frames}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        else:
            cv2.putText(display,
                        f"No board — {len(saved_paths)}/{n_frames}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Arducam — Calibration Capture", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("  Stopped early by user.")
            break

    cv2.destroyAllWindows()
    print(f"\nTotal frames captured: {len(saved_paths)}")
    return saved_paths


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Arducam intrinsic calibration using a checkerboard."
    )
    parser.add_argument(
        "--images", nargs="*", default=None,
        help="Glob pattern or list of existing calibration image paths. "
             "If omitted, frames are captured live from the camera."
    )
    parser.add_argument(
        "--camera-index", type=int, default=None,
        help="Force a specific /dev/video index (auto-detected if omitted)."
    )
    parser.add_argument(
        "--save-dir", default="intrinsic/calib_frames",
        help="Directory for captured calibration frames (default: calib_frames/)."
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE,
        help=f"Path for the output .npz file (default: {OUTPUT_FILE})."
    )
    args = parser.parse_args()

    # ---- Determine image source -----------------------------------------
    if args.images:
        # Expand globs if the shell didn't
        image_paths = []
        for pattern in args.images:
            expanded = glob.glob(pattern)
            image_paths.extend(expanded if expanded else [pattern])
        image_paths = sorted(image_paths)
        print(f"Using {len(image_paths)} provided image(s).")
    else:
        # Live capture
        print("No images provided — starting live capture from Arducam.")
        if args.camera_index is not None:
            cam_idx = args.camera_index
        else:
            print("Auto-detecting Arducam...")
            cam_idx = find_arducam_index()

        if cam_idx < 0:
            print("Error: no camera found. Connect the Arducam and try again.")
            sys.exit(1)

        print(f"Opening camera index {cam_idx}...")
        cap = open_arducam(cam_idx)

        image_paths = capture_calibration_frames(
            cap, CHECKERBOARD_SIZE, CAPTURE_COUNT, CAPTURE_INTERVAL, args.save_dir
        )
        cap.release()

        if len(image_paths) < 4:
            print("Error: too few frames captured for calibration. Exiting.")
            sys.exit(1)

    # ---- Run calibration -------------------------------------------------
    mtx, dist = computeIntrinsic(image_paths, CHECKERBOARD_SIZE, DW)

    # ---- Save results ----------------------------------------------------
    np.savez(args.output, mtx=mtx, dist=dist)
    print(f"\nCalibration saved to: {args.output}")
    print("Load with:  data = np.load('arducam_intrinsic_calib.npz')")
    print("            mtx, dist = data['mtx'], data['dist']")