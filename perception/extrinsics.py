"""
Stage 2 — Extrinsic Calibration (quasar/perception)

Finds the position and orientation (rotation R, translation t) of each camera
relative to `top_left`, which is treated as the world origin.

Workflow:
  - Loads device indices and position labels from camera.json.
  - Loads each camera's intrinsics (K and dist) from intrinsics/.
  - Opens all 4 cameras at once and shows a tiled 2x2 live preview.
  - When a checkerboard is detected in ALL 4 cameras in the same loop
    iteration, one synchronized image-set is captured automatically.
  - Press Q when done collecting (aim for 15-20+ synchronized sets).
  - Each camera's pose vs top_left is solved with cv2.stereoCalibrate over all
    sets, then refined by greedily dropping the worst-reprojecting sets until
    the RMS hits a target (best-subset selection). A final quality gate refuses
    to save poses whose RMS is too high for fixed-pose triangulation/MVS.

Outputs:
  - extrinsics/<position>/img_<n>.jpg   (captured frames)
  - extrinsics/K.txt                    (human-readable R / t + RMS per camera)
  - extrinsics/poses.npz                (R_<position>, t_<position> arrays)
  - extrinsics/calibration_report.txt   (per-pair RMS, sets used, pass/fail)

Run:
    python extrinsics.py
    python extrinsics.py --fresh        # clear extrinsics/ before capturing
    python extrinsics.py --force        # save even if the RMS gate fails
"""

import argparse
import json
import os
import shutil
import time

import cv2
import numpy as np


# --- Configuration ----------------------------------------------------------

# Inner-corner grid of the checkerboard (columns, rows). Must match the board
# used in Stage 1.
CHECKERBOARD = (8, 6)

# Physical size of one checkerboard square, in MILLIMETERS. This sets the
# metric scale of the recovered translations and the printed pair distances.
# Set it to your printed board's square size so the distances come out in mm.
SQUARE_SIZE_MM = 30

# Capture resolution used everywhere in this project.
FRAME_WIDTH = 800
FRAME_HEIGHT = 600
FPS = 15

# Minimum seconds between two auto-captures so successive synchronized sets
# are actually different views of the board.
CAPTURE_COOLDOWN_SEC = 1.0

# The reference camera. Its pose is identity R, zero t by definition.
REFERENCE = "top_left"

# Order of positions. Also the 2x2 tile layout (row-major).
POSITION_ORDER = ["top_left", "top_right", "bot_left", "bot_right"]

HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA_JSON = os.path.join(HERE, "camera.json")
INTRINSICS_DIR = os.path.join(HERE, "intrinsics")
OUTPUT_DIR = os.path.join(HERE, "extrinsics")

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- Quality gate / best-subset selection ------------------------------------
# Stop dropping sets once a pair's stereo RMS reaches this (px).
RMS_TARGET_PX = 0.6
# Gate: refuse to save poses if any pair's RMS exceeds this (px). At this level
# the relative pose is too inaccurate for fixed-pose triangulation — every
# triangulated point blows past COLMAP's ~4 px filter and fusion yields 0.
RMS_FAIL_PX = 1.0
# Never refine below this many sets (a small subset overfits / is unstable).
MIN_KEEP_SETS = 10


def verdict(rms):
    """Map an RMS (px) to OK / WARN / FAIL."""
    if rms <= RMS_TARGET_PX:
        return "OK"
    return "WARN" if rms <= RMS_FAIL_PX else "FAIL"


# --- Loading -----------------------------------------------------------------

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


def load_intrinsics(camera_number):
    """Load K (3x3) and distortion coefficients for one camera."""
    k_path = os.path.join(INTRINSICS_DIR, f"K_{camera_number}.txt")
    dist_path = os.path.join(INTRINSICS_DIR, f"dist_{camera_number}.txt")

    if not os.path.exists(k_path) or not os.path.exists(dist_path):
        raise FileNotFoundError(
            f"Missing intrinsics for camera {camera_number}. Expected "
            f"{k_path} and {dist_path}. Run intrinsics.py (Stage 1) first."
        )

    K = np.loadtxt(k_path, dtype=np.float64).reshape(3, 3)
    dist = np.loadtxt(dist_path, dtype=np.float64).reshape(1, -1)
    return K, dist


def open_camera(index):
    """
    Open a webcam at the given OS index at FRAME_WIDTH x FRAME_HEIGHT.

    Tries DirectShow first (most reliable for USB UVC cameras on Windows),
    then Media Foundation, then any backend. MJPG is requested before the
    resolution because it unlocks higher modes on most USB cameras.
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
        "and that the device is connected and not in use by another program. "
        "Opening 4 cameras at once can also saturate USB bandwidth — try "
        "spreading them across separate USB controllers."
    )


# --- Geometry helpers --------------------------------------------------------

def build_object_points():
    """3D checkerboard corners on the Z=0 plane, scaled to millimeters."""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD[0], 0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp


def camera_center(R_rel, t_rel):
    """Camera center in the reference frame: C = -R_rel^T @ t_rel."""
    return (-R_rel.T @ t_rel).reshape(3)


# --- Capture -----------------------------------------------------------------

def make_tile(frame, found, corners, label):
    """Annotate a single camera frame for the tiled preview."""
    tile = frame.copy()
    if found:
        cv2.drawChessboardCorners(tile, CHECKERBOARD, corners, found)
    color = (0, 255, 0) if found else (0, 0, 255)
    cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return tile


def collect_sets(caps):
    """
    Live tiled preview. Returns (sets, image_size) where sets is a list of
    synchronized sets, each a dict {position: refined_corners}, and image_size
    is the (width, height) of the captured frames. Frames are also saved to
    disk per camera.
    """
    find_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    sets = []
    image_size = None
    last_capture_time = 0.0
    window = "Extrinsics - all cameras (press Q when done)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print("  Hold the checkerboard so it is fully visible in ALL 4 cameras.")
    print("  Sets are captured automatically. Aim for 15-20. Press Q to stop.")

    while True:
        frames = {}
        grays = {}
        corners_by_pos = {}
        found_by_pos = {}

        read_ok = True
        for position, cap in caps.items():
            ok, frame = cap.read()
            if not ok:
                read_ok = False
                break
            frames[position] = frame
            grays[position] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not read_ok:
            print("  Warning: dropped frame, retrying...")
            continue

        if image_size is None:
            ref_gray = grays[POSITION_ORDER[0]]
            image_size = (ref_gray.shape[1], ref_gray.shape[0])

        for position in POSITION_ORDER:
            found, corners = cv2.findChessboardCorners(
                grays[position], CHECKERBOARD, find_flags
            )
            if found:
                corners = cv2.cornerSubPix(
                    grays[position], corners, (11, 11), (-1, -1),
                    SUBPIX_CRITERIA
                )
            found_by_pos[position] = found
            corners_by_pos[position] = corners

        all_found = all(found_by_pos[p] for p in POSITION_ORDER)
        now = time.time()

        if all_found and (now - last_capture_time) >= CAPTURE_COOLDOWN_SEC:
            set_number = len(sets) + 1
            this_set = {}
            for position in POSITION_ORDER:
                this_set[position] = corners_by_pos[position]
                filename = f"img_{set_number}.jpg"
                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, position, filename),
                    frames[position]
                )
            sets.append(this_set)
            last_capture_time = now
            print(f"    Captured synchronized set {set_number} "
                  f"(all 4 cameras).")

        # Build the 2x2 tiled preview.
        tiles = {
            p: make_tile(frames[p], found_by_pos[p], corners_by_pos[p],
                         f"{p}  {'OK' if found_by_pos[p] else '...'}")
            for p in POSITION_ORDER
        }
        top = np.hstack([tiles["top_left"], tiles["top_right"]])
        bottom = np.hstack([tiles["bot_left"], tiles["bot_right"]])
        grid = np.vstack([top, bottom])
        cv2.putText(
            grid, f"Sets captured: {len(sets)}", (8, grid.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
        cv2.imshow(window, grid)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cv2.destroyWindow(window)
    return sets, image_size


# --- Solve -------------------------------------------------------------------

def _stereo_calibrate(objpoints, ip_ref, ip_c, K_ref, dist_ref, K_c, dist_c,
                      image_size):
    """Run stereoCalibrate with fixed intrinsics; return (rms, R, T)."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    result = cv2.stereoCalibrate(
        objpoints, ip_ref, ip_c, K_ref, dist_ref, K_c, dist_c, image_size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    rms, R, T = result[0], result[5], result[6]
    return (float(rms), np.asarray(R, dtype=np.float64),
            np.asarray(T, dtype=np.float64).reshape(3, 1))


def _per_set_errors(objp, ip_ref, ip_c, K_ref, dist_ref, K_c, dist_c, R, T):
    """Mean reprojection error (px) in cam-c for each set, given R, T.

    The board pose is recovered in the reference camera (solvePnP), mapped into
    cam-c via the relative pose R, T, then the board corners are reprojected and
    compared to the detected corners. A set the relative pose cannot explain
    (motion during the async grab, a misdetection) shows a large error.
    """
    errs = []
    for c_ref, c_c in zip(ip_ref, ip_c):
        ok, rvec, tvec = cv2.solvePnP(objp, c_ref, K_ref, dist_ref)
        if not ok:
            errs.append(np.inf)
            continue
        R_ref, _ = cv2.Rodrigues(rvec)
        R_c = R @ R_ref
        t_c = R @ tvec + T
        proj, _ = cv2.projectPoints(objp, cv2.Rodrigues(R_c)[0], t_c, K_c, dist_c)
        e = np.linalg.norm(proj.reshape(-1, 2) - c_c.reshape(-1, 2),
                           axis=1).mean()
        errs.append(float(e))
    return np.asarray(errs)


def solve_pair(objp, ip_ref, ip_c, K_ref, dist_ref, K_c, dist_c, image_size):
    """
    Robust relative pose of one camera vs the reference.

    Solves over all sets, then (if the RMS is above target) greedily drops the
    worst-reprojecting sets — largest error first — re-solving each time until
    the RMS reaches RMS_TARGET_PX or only MIN_KEEP_SETS sets remain. Keeps the
    lowest-RMS solution seen. Returns (R, T, rms, n_used, n_total).
    """
    n = len(ip_ref)
    objpoints = [objp for _ in range(n)]
    rms, R, T = _stereo_calibrate(objpoints, ip_ref, ip_c, K_ref, dist_ref,
                                  K_c, dist_c, image_size)
    if n <= MIN_KEEP_SETS or rms <= RMS_TARGET_PX:
        return R, T, rms, n, n

    # Rank sets best-first by reprojection error under the full-set solve.
    order = np.argsort(
        _per_set_errors(objp, ip_ref, ip_c, K_ref, dist_ref, K_c, dist_c, R, T)
    )
    best = (R, T, rms, n)
    for keep in range(n - 1, MIN_KEEP_SETS - 1, -1):
        idx = order[:keep]
        op = [objp for _ in range(keep)]
        sr, sR, sT = _stereo_calibrate(
            op, [ip_ref[i] for i in idx], [ip_c[i] for i in idx],
            K_ref, dist_ref, K_c, dist_c, image_size,
        )
        if sr < best[2]:
            best = (sR, sT, sr, keep)
        if sr <= RMS_TARGET_PX:
            return sR, sT, sr, keep, n
    R, T, rms, keep = best
    return R, T, rms, keep, n


def solve_extrinsics(sets, intrinsics, image_size):
    """
    Solve every non-reference camera's pose vs the reference with robust
    best-subset stereoCalibrate (fixed, pre-calibrated intrinsics).

    stereoCalibrate jointly optimizes the relative pose over all corner
    correspondences; the returned R, T satisfy X_cam = R @ X_ref + T (the pose
    of `cam` in the reference frame). Returns (poses, stats) where
    poses = {position: (R, t)} (reference = identity / zero) and
    stats = {position: (rms, n_used, n_total)}.
    """
    objp = build_object_points()
    imgpoints = {p: [s[p] for s in sets] for p in POSITION_ORDER}
    K_ref, dist_ref = intrinsics[REFERENCE]

    poses = {REFERENCE: (np.eye(3), np.zeros((3, 1)))}
    stats = {}

    print("\nStereo calibration (each camera vs reference, robust subset):")
    for position in POSITION_ORDER:
        if position == REFERENCE:
            continue
        K_c, dist_c = intrinsics[position]
        R, T, rms, n_used, n_total = solve_pair(
            objp, imgpoints[REFERENCE], imgpoints[position],
            K_ref, dist_ref, K_c, dist_c, image_size,
        )
        poses[position] = (R, T.reshape(3, 1))
        stats[position] = (rms, n_used, n_total)
        print(f"  {position:10s} vs {REFERENCE}: RMS = {rms:.4f} px "
              f"[{verdict(rms)}]  (used {n_used}/{n_total} sets)")

    return poses, stats


# --- Output ------------------------------------------------------------------

def save_outputs(poses, stats):
    """Write extrinsics/K.txt (human-readable + RMS) and extrinsics/poses.npz."""
    k_path = os.path.join(OUTPUT_DIR, "K.txt")
    with open(k_path, "w") as f:
        for position in POSITION_ORDER:
            R, t = poses[position]
            R_list = np.array2string(
                R, separator=", ",
                formatter={"float_kind": lambda x: f"{x:.6f}"}
            ).replace("\n", "")
            t_list = np.array2string(
                t.reshape(3), separator=", ",
                formatter={"float_kind": lambda x: f"{x:.6f}"}
            )
            f.write(f"{position}\n")
            if position in stats:
                rms, n_used, n_total = stats[position]
                f.write(f"# stereo RMS: {rms:.4f} px [{verdict(rms)}] "
                        f"(used {n_used}/{n_total} sets)\n")
            f.write(f"R: {R_list}\n")
            f.write(f"t: {t_list}\n\n")
    print(f"  Saved {k_path}")

    npz_path = os.path.join(OUTPUT_DIR, "poses.npz")
    arrays = {}
    for position in POSITION_ORDER:
        R, t = poses[position]
        arrays[f"R_{position}"] = R
        arrays[f"t_{position}"] = t.reshape(3)
    np.savez(npz_path, **arrays)
    print(f"  Saved {npz_path}")


def write_calibration_report(stats):
    """Write extrinsics/calibration_report.txt and return the worst RMS (px)."""
    path = os.path.join(OUTPUT_DIR, "calibration_report.txt")
    worst = max((s[0] for s in stats.values()), default=0.0)
    with open(path, "w") as f:
        f.write("Extrinsic calibration report\n")
        f.write(f"  square_size_mm:        {SQUARE_SIZE_MM}\n")
        f.write(f"  RMS target (px):       {RMS_TARGET_PX}\n")
        f.write(f"  RMS fail gate (px):    {RMS_FAIL_PX}\n\n")
        f.write(f"  {'pair':24s}{'RMS_px':>9s}{'sets':>10s}  verdict\n")
        for position in POSITION_ORDER:
            if position not in stats:
                continue
            rms, n_used, n_total = stats[position]
            pair = f"{REFERENCE}<->{position}"
            f.write(f"  {pair:24s}{rms:9.4f}{f'{n_used}/{n_total}':>10s}  "
                    f"{verdict(rms)}\n")
        overall = "PASS" if worst <= RMS_FAIL_PX else "FAIL"
        f.write(f"\n  worst RMS: {worst:.4f} px  ->  {overall}\n")
    print(f"  Saved {path}")
    return worst


def print_pair_distances(poses):
    """Print the distance between every pair of camera centers."""
    centers = {p: camera_center(R, t) for p, (R, t) in poses.items()}

    print("\nCamera pair distances:")
    for i in range(len(POSITION_ORDER)):
        for j in range(i + 1, len(POSITION_ORDER)):
            a, b = POSITION_ORDER[i], POSITION_ORDER[j]
            measured = float(np.linalg.norm(centers[a] - centers[b]))
            print(f"  {a} <-> {b}: {measured:.2f} mm")
    print("  (distances are in the unit of SQUARE_SIZE_MM)")


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 extrinsic calibration (with RMS quality gate).")
    parser.add_argument("--force", action="store_true",
                        help="Save poses even if the RMS gate fails.")
    parser.add_argument("--fresh", action="store_true",
                        help="Clear extrinsics/ (captured images + old "
                             "K.txt/poses.npz/report) before capturing.")
    args = parser.parse_args()

    cameras = load_camera_indices()

    # Prepare output subfolders.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.fresh:
        for position, _ in cameras:
            sub = os.path.join(OUTPUT_DIR, position)
            if os.path.isdir(sub):
                shutil.rmtree(sub)
        for name in ("K.txt", "poses.npz", "calibration_report.txt"):
            stale = os.path.join(OUTPUT_DIR, name)
            if os.path.exists(stale):
                os.remove(stale)
        print("  --fresh: cleared previous captures and outputs in "
              f"{OUTPUT_DIR}.")
    for position, _ in cameras:
        os.makedirs(os.path.join(OUTPUT_DIR, position), exist_ok=True)

    # Load intrinsics before opening any stream.
    intrinsics = {}
    for position, camera_number in cameras:
        intrinsics[position] = load_intrinsics(camera_number)
    print("Loaded intrinsics for all 4 cameras.")

    print("Extrinsic calibration")
    print(f"  Checkerboard inner corners: {CHECKERBOARD}")
    print(f"  Square size: {SQUARE_SIZE_MM} mm")
    print(f"  Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Reference camera: {REFERENCE}\n")

    # Open all cameras (small stagger helps USB bandwidth negotiation).
    caps = {}
    try:
        for position, camera_number in cameras:
            print(f"Opening '{position}' (index {camera_number})...")
            caps[position] = open_camera(camera_number)
            time.sleep(0.25)

        sets, image_size = collect_sets(caps)
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()

    print(f"\nCollected {len(sets)} synchronized set(s).")
    if len(sets) < 4:
        print("  Too few sets for a reliable solve. Aim for 15-20 and rerun.")
        if not sets:
            return

    poses, stats = solve_extrinsics(sets, intrinsics, image_size)

    print("\nRecovered poses (relative to top_left):")
    for position in POSITION_ORDER:
        R, t = poses[position]
        print(f"  {position}: t = {t.reshape(3)}")

    # Quality gate.
    worst = write_calibration_report(stats)
    print("\n=== Calibration quality gate ===")
    for position in POSITION_ORDER:
        if position in stats:
            rms, n_used, n_total = stats[position]
            print(f"  {REFERENCE} <-> {position:10s}: {rms:.4f} px "
                  f"[{verdict(rms)}]  (used {n_used}/{n_total})")
    print(f"  Worst RMS: {worst:.4f} px  (target <= {RMS_TARGET_PX}, "
          f"fail > {RMS_FAIL_PX})")

    if worst > RMS_FAIL_PX and not args.force:
        print("\n  CALIBRATION REJECTED — poses.npz NOT written.")
        print("  The relative pose(s) are too inaccurate for fixed-pose "
              "triangulation/MVS (points exceed COLMAP's reprojection filter "
              "and fusion yields 0).")
        print("  Recapture: hold the board completely still at each pose, fill "
              "more of the frame, vary the tilt, and pay attention to the "
              "FAIL camera(s). Then re-run extrinsics.py.")
        print("  (Pass --force to save these poses anyway.)")
        return

    save_outputs(poses, stats)
    print_pair_distances(poses)
    if worst > RMS_FAIL_PX:
        print("\nExtrinsic calibration saved with --force despite a FAIL gate.")
    else:
        print("\nExtrinsic calibration complete.")


if __name__ == "__main__":
    main()
