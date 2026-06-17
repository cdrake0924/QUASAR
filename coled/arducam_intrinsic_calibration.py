import cv2
import numpy as np
import glob
import os
import sys
import time

try:
    from quad_utils import (
        POSITIONS,
        load_layout,
        intrinsic_path as quad_intrinsic_path,
    )
    _HAVE_QUAD_UTILS = True
except Exception:
    _HAVE_QUAD_UTILS = False

# --- Configuration ---
CHECKERBOARD_SIZE = (8, 6)       # Number of inner corners (cols, rows)
SQUARE_SIZE_MM    = 30           # Physical size of each square in mm
# Defaults; can be overridden per-run via --width/--height/--fps.
# IMPORTANT: intrinsics MUST be calibrated at the same resolution you intend
# to capture with downstream. UVC cameras change sensor crop / focal length
# between resolutions (especially across aspect ratios), so a 1280x720
# calibration is NOT generally valid at 640x480.
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


def open_arducam(index: int,
                  width: int = FRAME_WIDTH,
                  height: int = FRAME_HEIGHT,
                  fps: int = FPS) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}.")

    # MJPG often unlocks higher resolutions on USB webcams/Arducam devices.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Camera opened: {w}x{h} @ {actual_fps:.1f} fps")
    if w != width or h != height:
        print(
            f"  Warning: camera {index} delivered {w}x{h} instead of "
            f"{width}x{height}. Calibration will be valid for {w}x{h} only — "
            "use the same resolution for extrinsic capture downstream."
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


def _run_single(camera_index: int | None,
                images: list | None,
                save_dir: str,
                output_path: str,
                label: str = "",
                width: int = FRAME_WIDTH,
                height: int = FRAME_HEIGHT,
                fps: int = FPS) -> None:
    """
    End-to-end intrinsic calibration for ONE camera. Captures (or loads) frames,
    runs calibrateCamera, and writes the .npz to output_path.
    """
    banner = f" ({label}) " if label else " "
    print(f"\n========= Intrinsic calibration{banner}=========")
    print(f"Output: {output_path}")
    print(f"Save dir: {save_dir}")
    print(f"Capture resolution: {width}x{height} @ {fps} fps")

    if images:
        image_paths = []
        for pattern in images:
            expanded = glob.glob(pattern)
            image_paths.extend(expanded if expanded else [pattern])
        image_paths = sorted(image_paths)
        print(f"Using {len(image_paths)} provided image(s).")
    else:
        print("No images provided — starting live capture from Arducam.")
        if camera_index is None:
            print("Auto-detecting Arducam...")
            camera_index = find_arducam_index()

        if camera_index is None or camera_index < 0:
            raise RuntimeError("No camera found. Connect the Arducam and try again.")

        print(f"Opening camera index {camera_index}...")
        cap = open_arducam(camera_index, width=width, height=height, fps=fps)
        try:
            image_paths = capture_calibration_frames(
                cap, CHECKERBOARD_SIZE, CAPTURE_COUNT, CAPTURE_INTERVAL, save_dir
            )
        finally:
            cap.release()

        if len(image_paths) < 4:
            raise RuntimeError(
                f"Too few frames captured ({len(image_paths)}). Need at least 4."
            )

    mtx, dist = computeIntrinsic(image_paths, CHECKERBOARD_SIZE, DW)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, mtx=mtx, dist=dist)
    print(f"\nCalibration saved to: {output_path}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Arducam intrinsic calibration using a checkerboard. "
                    "Supports single-camera (legacy) and 4-camera (quad) workflows."
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
        "--save-dir", default=None,
        help="Directory for captured calibration frames. "
             "Default: intrinsic/calib_frames/ (legacy) "
             "or intrinsic/calib_frames/<position>/ (quad)."
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Path for the output .npz file. "
             f"Default (legacy): {OUTPUT_FILE}. "
             f"Default (quad): intrinsic/cam_<position>_intr.npz."
    )
    parser.add_argument(
        "--position", choices=POSITIONS if _HAVE_QUAD_UTILS else ["tl", "tr", "bl", "br"],
        default=None,
        help="Quad-rig position to calibrate. Reads camera_layout.json for the "
             "OS index and writes intrinsic/cam_<position>_intr.npz."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Quad mode: calibrate all 4 positions (tl, tr, bl, br) sequentially "
             "using indices from camera_layout.json. Pauses between each."
    )
    parser.add_argument(
        "--layout", default="camera_layout.json",
        help="Path to camera_layout.json (used with --position or --all)."
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help=f"Capture/calibration width (default: {FRAME_WIDTH}, or layout's "
             "frame_width when --position/--all is used). The .npz this script "
             "writes is ONLY valid at this exact resolution; use the same value "
             "when capturing extrinsics downstream."
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help=f"Capture/calibration height (default: {FRAME_HEIGHT})."
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help=f"Capture fps (default: {FPS})."
    )
    args = parser.parse_args()

    def _resolve_dims(layout: dict | None) -> tuple[int, int, int]:
        """CLI flag > layout > module default."""
        if layout is not None:
            lw, lh, lfps = layout["frame_width"], layout["frame_height"], layout["fps"]
        else:
            lw, lh, lfps = FRAME_WIDTH, FRAME_HEIGHT, FPS
        return (
            args.width  if args.width  is not None else lw,
            args.height if args.height is not None else lh,
            args.fps    if args.fps    is not None else lfps,
        )

    # -------- Quad mode (--all) -----------------------------------------
    if args.all:
        if not _HAVE_QUAD_UTILS:
            print("Error: quad_utils.py not importable; cannot use --all.")
            sys.exit(1)
        layout = load_layout(args.layout)
        w, h, fps_v = _resolve_dims(layout)
        print(f"Quad mode: calibrating all 4 positions from '{args.layout}' "
              f"at {w}x{h}@{fps_v} fps.")
        for pos in POSITIONS:
            input(f"\n>>> Position the {pos.upper()} camera's view of the board, "
                  f"then press ENTER to start capture for '{pos}' (index "
                  f"{layout[pos]})... ")
            save_dir = args.save_dir or os.path.join("intrinsic", "calib_frames", pos)
            output   = args.output   or quad_intrinsic_path(pos)
            try:
                _run_single(
                    camera_index=layout[pos],
                    images=None,
                    save_dir=save_dir,
                    output_path=output,
                    label=pos,
                    width=w, height=h, fps=fps_v,
                )
            except Exception as e:
                print(f"  Error calibrating '{pos}': {e}")
                resp = input("  Continue with next camera? [y/N]: ").strip().lower()
                if resp != "y":
                    sys.exit(1)
        print("\nAll 4 intrinsic calibrations done.")
        sys.exit(0)

    # -------- Quad mode (--position) ------------------------------------
    if args.position is not None:
        if not _HAVE_QUAD_UTILS:
            print("Error: quad_utils.py not importable; cannot use --position.")
            sys.exit(1)
        layout = load_layout(args.layout)
        w, h, fps_v = _resolve_dims(layout)
        pos = args.position
        cam_idx  = args.camera_index if args.camera_index is not None else layout[pos]
        save_dir = args.save_dir or os.path.join("intrinsic", "calib_frames", pos)
        output   = args.output   or quad_intrinsic_path(pos)
        _run_single(
            camera_index=cam_idx,
            images=args.images,
            save_dir=save_dir,
            output_path=output,
            label=pos,
            width=w, height=h, fps=fps_v,
        )
        sys.exit(0)

    # -------- Legacy single-camera mode ---------------------------------
    w, h, fps_v = _resolve_dims(None)
    save_dir = args.save_dir or "intrinsic/calib_frames"
    output   = args.output   or OUTPUT_FILE
    _run_single(
        camera_index=args.camera_index,
        images=args.images,
        save_dir=save_dir,
        output_path=output,
        width=w, height=h, fps=fps_v,
    )
    print("Load with:  data = np.load(output_path)")
    print("            mtx, dist = data['mtx'], data['dist']")