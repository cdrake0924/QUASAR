"""
Stage 1 — Intrinsic Calibration (quasar/perception)

Finds each camera's internal optical properties (focal length, principal
point, lens distortion) and bundles them into a 3x3 camera matrix K.

Workflow:
  - Reads device indices from camera.json (positions -> OS indices).
  - Processes one camera at a time, in the order:
        top_left, top_right, bot_left, bot_right
  - Opens a live preview and auto-captures when a valid checkerboard is found.
    Captures are gated on frame coverage (the board must be seen across all
    regions of the frame) so the distortion and principal point are well
    constrained. After TARGET_IMAGES views it runs cv2.calibrateCamera().
  - Saves K_{n}.txt and dist_{n}.txt into intrinsics/.

A quality gate rejects a camera's calibration (does NOT save K/dist) when the
principal point is far from center, the FOV/focal is implausible, fx/fy
mismatch, distortion is extreme, or reprojection error is high — the symptoms
of an under-constrained solve that silently breaks triangulation/MVS later.

Run:
    python intrinsics.py
    python intrinsics.py --fresh        # clear intrinsics/ before capturing
    python intrinsics.py --force        # save even if the quality gate fails
"""

import argparse
import glob
import json
import os
import shutil
import time

import cv2
import numpy as np


# --- Configuration ----------------------------------------------------------

# Inner-corner grid of the checkerboard (columns, rows). Default 9x6 inner
# corners == a 10x7 square board. Edit this to match your printed pattern.
CHECKERBOARD = (8, 6)

# Capture resolution used everywhere in this project.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Number of valid checkerboard views to collect per camera. More views with
# good spatial coverage give a far more stable distortion / principal point
# estimate than the bare minimum.
TARGET_IMAGES = 20

# Physical size of one checkerboard square. The absolute scale does not affect
# the intrinsic matrix, so a unit square is fine for K. Used only to build the
# object-point grid.
SQUARE_SIZE = 1.0

# Frame-coverage gating. The frame is divided into a COVERAGE_GRID grid of
# cells; the board centroid must land in many different cells before we accept
# enough images. This forces the board toward the edges/corners, which is where
# lens distortion is strongest and least observed otherwise.
COVERAGE_GRID = (3, 3)            # (columns, rows)
MAX_CAPTURES_PER_CELL = 3         # cap per cell to force spatial spread
MIN_CENTROID_SHIFT_PX = 40.0      # min board movement between captures

# Minimum seconds between two auto-captures so we don't grab nearly identical
# frames in a fraction of a second.
CAPTURE_COOLDOWN_SEC = 1.0

# Reject if the calibrated principal point sits farther than this fraction of
# the image dimension from the center (a symptom of poor edge/corner coverage).
PRINCIPAL_POINT_TOLERANCE = 0.15

# Reject a calibration whose reprojection error exceeds this (px).
REPROJ_FAIL_PX = 1.0

# Order in which cameras are calibrated.
POSITION_ORDER = ["top_left", "top_right", "bot_left", "bot_right"]

HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA_JSON = os.path.join(HERE, "camera.json")
OUTPUT_DIR = os.path.join(HERE, "intrinsics")

# Corner sub-pixel refinement termination criteria.
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# --- Camera I/O --------------------------------------------------------------

def load_camera_indices():
    """Load camera.json and return an ordered list of (position, index)."""
    if not os.path.exists(CAMERA_JSON):
        raise FileNotFoundError(
            f"camera.json not found at {CAMERA_JSON}. Create it first, e.g.:\n"
            '  {\n'
            '    "top_left":  1,\n'
            '    "top_right": 2,\n'
            '    "bot_left":  3,\n'
            '    "bot_right": 4\n'
            '  }'
        )
    with open(CAMERA_JSON, "r") as f:
        mapping = json.load(f)

    cameras = []
    for position in POSITION_ORDER:
        if position not in mapping:
            raise KeyError(
                f"camera.json is missing the '{position}' key. "
                f"Expected keys: {POSITION_ORDER}."
            )
        cameras.append((position, int(mapping[position])))
    return cameras


def open_camera(index):
    """
    Open a webcam at the given OS index at FRAME_WIDTH x FRAME_HEIGHT.

    On Windows the DirectShow backend is the most reliable for USB UVC
    cameras, with Media Foundation as a fallback. MJPG is requested first
    because it unlocks higher resolutions / framerates on most USB cameras.
    """
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("DEFAULT", cv2.CAP_ANY),
    ]

    for name, flag in backends:
        cap = cv2.VideoCapture(index, flag)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        ok, _ = cap.read()
        if not ok:
            cap.release()
            continue

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Camera {index} opened via {name}: {actual_w}x{actual_h}")
        return cap

    raise RuntimeError(
        f"Cannot open camera at index {index}. Check the value in camera.json "
        "and that the device is connected and not in use by another program."
    )


# --- Calibration -------------------------------------------------------------

def build_object_points():
    """One set of 3D points for the checkerboard, lying flat on Z=0."""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD[0], 0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def board_centroid(corners):
    """Mean (x, y) pixel position of the detected corners."""
    return corners.reshape(-1, 2).mean(axis=0)


def board_scale_bucket(corners, image_size):
    """
    Classify the board's apparent size as 'far', 'mid', or 'near' from the
    fraction of the frame its bounding box covers. Used as a distance-variety
    hint — calibration needs both near and far views.
    """
    w, h = image_size
    pts = corners.reshape(-1, 2)
    bbox = (pts[:, 0].max() - pts[:, 0].min()) * \
           (pts[:, 1].max() - pts[:, 1].min())
    frac = bbox / float(w * h)
    if frac < 0.06:
        return "far"
    if frac > 0.18:
        return "near"
    return "mid"


def cell_of(centroid, image_size):
    """Map a pixel position to a (col, row) cell in the coverage grid."""
    w, h = image_size
    gx, gy = COVERAGE_GRID
    col = min(int(centroid[0] / w * gx), gx - 1)
    row = min(int(centroid[1] / h * gy), gy - 1)
    return (col, row)


def draw_coverage_hud(display, image_size, cell_counts, captured,
                      scale_counts):
    """Overlay the coverage grid, per-cell counts, and progress text."""
    w, h = image_size
    gx, gy = COVERAGE_GRID
    for c in range(1, gx):
        x = int(w * c / gx)
        cv2.line(display, (x, 0), (x, h), (80, 80, 80), 1)
    for r in range(1, gy):
        y = int(h * r / gy)
        cv2.line(display, (0, y), (w, y), (80, 80, 80), 1)

    for col in range(gx):
        for row in range(gy):
            count = cell_counts.get((col, row), 0)
            full = count >= MAX_CAPTURES_PER_CELL
            color = (0, 200, 0) if full else (
                (0, 165, 255) if count > 0 else (0, 0, 200))
            cx = int(w * (col + 0.5) / gx)
            cy = int(h * (row + 0.5) / gy)
            cv2.putText(display, str(count), (cx - 6, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(display, f"Captured: {captured}/{TARGET_IMAGES}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    scale_text = (f"near {scale_counts.get('near', 0)}  "
                  f"mid {scale_counts.get('mid', 0)}  "
                  f"far {scale_counts.get('far', 0)}")
    cv2.putText(display, scale_text, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
    cv2.putText(display, "TILT the board 30-45 deg; vary near/far",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 0), 1)


def collect_images(cap, camera_number):
    """
    Show a live preview and auto-capture TARGET_IMAGES valid checkerboard
    views, gated on frame coverage so the board is seen across the whole frame.
    Returns lists of object points and image points ready for
    cv2.calibrateCamera(), along with the image size (w, h).

    Saves each captured frame to intrinsics/img_{camera_number}_{n}.jpg.
    """
    objp = build_object_points()
    object_points = []
    image_points = []
    image_size = None

    captured = 0
    last_capture_time = 0.0
    last_centroid = None
    cell_counts = {}
    scale_counts = {}
    window = f"Intrinsics - camera {camera_number} (ESC to abort)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    find_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    print(f"  Need {TARGET_IMAGES} views spread across the whole frame "
          "(corners included). IMPORTANT: tilt the board at strong angles "
          "and capture at both near and far distances — flat-on views alone "
          "make the focal length diverge.")

    while captured < TARGET_IMAGES:
        ok, frame = cap.read()
        if not ok:
            print("  Warning: dropped frame, retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD, find_flags
        )

        display = frame.copy()
        now = time.time()

        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA
            )
            cv2.drawChessboardCorners(
                display, CHECKERBOARD, corners_refined, found
            )

            centroid = board_centroid(corners_refined)
            cell = cell_of(centroid, image_size)
            moved_enough = (
                last_centroid is None
                or np.linalg.norm(centroid - last_centroid)
                >= MIN_CENTROID_SHIFT_PX
            )
            cell_has_room = (
                cell_counts.get(cell, 0) < MAX_CAPTURES_PER_CELL
            )
            cooled_down = (now - last_capture_time) >= CAPTURE_COOLDOWN_SEC

            if cooled_down and moved_enough and cell_has_room:
                object_points.append(objp.copy())
                image_points.append(corners_refined)
                cell_counts[cell] = cell_counts.get(cell, 0) + 1
                bucket = board_scale_bucket(corners_refined, image_size)
                scale_counts[bucket] = scale_counts.get(bucket, 0) + 1

                photo_number = captured + 1
                filename = f"img_{camera_number}_{photo_number}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)

                captured += 1
                last_capture_time = now
                last_centroid = centroid
                print(f"    Captured {captured}/{TARGET_IMAGES} "
                      f"(cell {cell}, {bucket}) -> {filename}")

        draw_coverage_hud(display, image_size, cell_counts, captured,
                          scale_counts)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("  Aborted by user.")
            break

    cv2.destroyWindow(window)
    return object_points, image_points, image_size


def calibrate(object_points, image_points, image_size):
    """Run cv2.calibrateCamera and return (K, dist, reprojection_error).

    CALIB_FIX_K3 drops the 3rd radial-distortion term. These are mild ~50 deg
    lenses, so k3 only adds a free parameter that couples with focal length and
    can let the solve diverge when views lack perspective variety.
    """
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None,
        flags=cv2.CALIB_FIX_K3
    )

    # Compute mean reprojection error explicitly for a clear per-camera report.
    total_error = 0.0
    total_points = 0
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], K, dist
        )
        err = cv2.norm(image_points[i], projected, cv2.NORM_L2)
        total_error += err * err
        total_points += len(projected)
    mean_error = np.sqrt(total_error / total_points) if total_points else rms

    return K, dist, mean_error


def assess_calibration(K, dist, image_size):
    """
    Return a list of human-readable problems if the calibration looks like it
    diverged. Empty list means it looks plausible.

    A diverged solve (too few perspective/tilt variations) typically shows up
    as a wildly wrong focal length, mismatched fx/fy, or huge distortion terms.
    """
    import math

    w, _ = image_size
    fx, fy = float(K[0, 0]), float(K[1, 1])
    d = np.asarray(dist).reshape(-1)
    issues = []

    hfov = 2.0 * math.degrees(math.atan((w / 2.0) / fx))
    if hfov < 30.0 or hfov > 120.0:
        issues.append(
            f"implausible horizontal FOV {hfov:.0f} deg (fx={fx:.0f}); "
            "expected roughly 40-70 deg for these lenses"
        )
    if abs(fx - fy) / max(fx, fy) > 0.15:
        issues.append(f"fx/fy mismatch (fx={fx:.0f}, fy={fy:.0f})")
    if abs(d[0]) > 1.5 or (d.size > 1 and abs(d[1]) > 3.0):
        issues.append(f"extreme distortion coeffs {np.round(d[:3], 2)}")

    return issues


def gate_issues(K, dist, image_size, reproj):
    """
    All hard-fail reasons that make a calibration unusable downstream. An empty
    list means the calibration passes the gate.

    Combines the divergence checks (assess_calibration) with the two strongest
    symptoms of poor coverage: an off-center principal point and high
    reprojection error. A bad principal point in particular silently wrecks
    triangulation in MVS (rays are mis-aimed), so it must block the save.
    """
    w, h = image_size
    cx, cy = float(K[0, 2]), float(K[1, 2])
    issues = list(assess_calibration(K, dist, image_size))

    off_x = abs(cx - w / 2.0) / w
    off_y = abs(cy - h / 2.0) / h
    if off_x > PRINCIPAL_POINT_TOLERANCE or off_y > PRINCIPAL_POINT_TOLERANCE:
        issues.append(
            f"principal point ({cx:.1f}, {cy:.1f}) too far from center "
            f"({w / 2:.0f}, {h / 2:.0f}) — off ({off_x * 100:.0f}%, "
            f"{off_y * 100:.0f}%); board did not cover the frame edges/corners"
        )
    if reproj > REPROJ_FAIL_PX:
        issues.append(f"reprojection error {reproj:.2f} px > {REPROJ_FAIL_PX} px")

    return issues


def write_calibration_report(rows, image_size):
    """Write intrinsics/calibration_report.txt. rows: position -> dict."""
    path = os.path.join(OUTPUT_DIR, "calibration_report.txt")
    w, h = image_size if image_size else (FRAME_WIDTH, FRAME_HEIGHT)
    with open(path, "w") as f:
        f.write("Intrinsic calibration report\n")
        f.write(f"  resolution: {w}x{h}  (center {w / 2:.0f}, {h / 2:.0f})\n")
        f.write(f"  principal-point tolerance: {PRINCIPAL_POINT_TOLERANCE} of "
                f"image dim\n")
        f.write(f"  reprojection fail gate: {REPROJ_FAIL_PX} px\n\n")
        for position in POSITION_ORDER:
            if position not in rows:
                continue
            r = rows[position]
            f.write(f"{position} (camera {r['camera_number']}): {r['verdict']}\n")
            if r.get("K") is not None:
                K = r["K"]
                f.write(f"  fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} "
                        f"cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}  "
                        f"reproj={r['reproj']:.3f} px\n")
            for issue in r.get("issues", []):
                f.write(f"  - {issue}\n")
            f.write("\n")
    print(f"  Saved {path}")
    return path


def save_results(camera_number, K, dist):
    """Save K and distortion coefficients as space-delimited text."""
    k_path = os.path.join(OUTPUT_DIR, f"K_{camera_number}.txt")
    dist_path = os.path.join(OUTPUT_DIR, f"dist_{camera_number}.txt")

    np.savetxt(k_path, K, fmt="%.6f")
    np.savetxt(dist_path, np.asarray(dist).reshape(1, -1), fmt="%.6f")

    print(f"  Saved {k_path}")
    print(f"  Saved {dist_path}")


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 intrinsic calibration (with quality gate).")
    parser.add_argument("--force", action="store_true",
                        help="Save K/dist even if the quality gate fails.")
    parser.add_argument("--fresh", action="store_true",
                        help="Clear intrinsics/ (K_*, dist_*, img_*, report) "
                             "before capturing.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.fresh:
        for pattern in ("K_*.txt", "dist_*.txt", "img_*.jpg",
                        "calibration_report.txt"):
            for stale in glob.glob(os.path.join(OUTPUT_DIR, pattern)):
                os.remove(stale)
        print("  --fresh: cleared previous intrinsics outputs.\n")

    cameras = load_camera_indices()

    print("Intrinsic calibration")
    print(f"  Checkerboard inner corners: {CHECKERBOARD}")
    print(f"  Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Cameras (in order): {cameras}\n")

    focal_lengths = {}
    radial_k1 = {}
    rows = {}
    last_image_size = None

    for i, (position, camera_number) in enumerate(cameras):
        print(f"=== Camera '{position}' (index {camera_number}) ===")
        cap = open_camera(camera_number)
        try:
            object_points, image_points, image_size = collect_images(
                cap, camera_number
            )
        finally:
            cap.release()
        last_image_size = image_size or last_image_size

        if len(object_points) < TARGET_IMAGES:
            print(f"  Only collected {len(object_points)} images for "
                  f"'{position}'. Skipping calibration for this camera.\n")
            rows[position] = {"camera_number": camera_number, "K": None,
                              "reproj": float("nan"), "verdict": "SKIPPED",
                              "issues": ["too few images collected"]}
        else:
            K, dist, error = calibrate(object_points, image_points, image_size)
            print(f"  Reprojection error: {error:.4f} pixels")
            print("  K =")
            print(K)

            issues = gate_issues(K, dist, image_size, error)
            verdict = "OK" if not issues else "FAIL"
            rows[position] = {"camera_number": camera_number, "K": K,
                              "reproj": float(error), "verdict": verdict,
                              "issues": issues}

            if issues and not args.force:
                print("  >>> CALIBRATION REJECTED — K/dist NOT saved:")
                for issue in issues:
                    print(f"        - {issue}")
                print("      Recollect this camera: get the board into all 4 "
                      "image CORNERS/edges, tilt 30-45 deg (pitch & yaw), and "
                      "capture near AND far. (--force saves anyway.)")
            else:
                if issues:
                    print("  WARNING: saving despite gate issues (--force):")
                    for issue in issues:
                        print(f"        - {issue}")
                focal_lengths[position] = float(K[0, 0])
                radial_k1[position] = float(np.asarray(dist).reshape(-1)[0])
                save_results(camera_number, K, dist)

        # Do not auto-advance to the next camera.
        if i < len(cameras) - 1:
            input("\n  Press Enter to continue to the next camera... ")
            print()

    cv2.destroyAllWindows()

    write_calibration_report(rows, last_image_size)
    saved = [p for p in rows if rows[p]["verdict"] == "OK"]
    failed = [p for p in rows if rows[p]["verdict"] not in ("OK",)]
    print("\n=== Intrinsics gate summary ===")
    for position in POSITION_ORDER:
        if position in rows:
            print(f"  {position:10s}: {rows[position]['verdict']}")
    if failed and not args.force:
        print(f"  {len(failed)} camera(s) NOT saved: {failed}. Recollect them "
              "and re-run before Stage 2.")

    # Cross-camera consistency: these are identical cameras, so their focal
    # lengths and distortion should be close. An outlier means that camera's
    # calibration is still under-constrained (usually too little tilt/distance
    # variety) and will throw off its extrinsic pose — recollect it.
    if len(focal_lengths) >= 2:
        median_fx = float(np.median(list(focal_lengths.values())))
        median_k1 = float(np.median(list(radial_k1.values())))
        print("\nCross-camera consistency (identical cameras should match):")
        for position in focal_lengths:
            fx = focal_lengths[position]
            k1 = radial_k1[position]
            fx_dev = abs(fx - median_fx) / median_fx
            outlier = fx_dev > 0.15 or abs(k1 - median_k1) > 0.3
            flag = "  <-- OUTLIER, recollect this camera" if outlier else ""
            print(f"  {position}: fx = {fx:.1f} ({fx_dev * 100:.0f}% from "
                  f"median), k1 = {k1:.2f}{flag}")

    print("\nAll cameras processed.")


if __name__ == "__main__":
    main()
