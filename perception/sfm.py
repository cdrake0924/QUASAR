"""
Stage 3 — SfM / Sparse Point Cloud (quasar/perception)

Runs COLMAP Structure-from-Motion on a STATIC capture of the scene to produce a
sparse point cloud and a set of registered camera poses in sfm/sparse/0/. This
sparse model is the fixed pose set reused by every frame of dynamic MVS.

Pipeline (COLMAP CLI via subprocess; no pycolmap bindings):
  1. feature_extractor  - one call per camera position so each position gets a
                          single camera locked to its calibrated PINHOLE
                          intrinsics (fx, fy, cx, cy).
  2. exhaustive_matcher
  3. mapper             - intrinsics held fixed (we trust the calibration).
  4. model_converter    - export sparse/0 to TXT for inspection.

Depends on: camera.json, intrinsics/K_*.txt, intrinsics/dist_*.txt,
            extrinsics/poses.npz (used as a cross-check after mapping).

Before running, capture a short static clip (nothing moving) and export frames
into sfm/images/ named  {position}_{frame:06d}.jpg  e.g. top_left_000001.jpg.

Run:
    python sfm.py
    python sfm.py --no-undistort        # feed raw frames (skip undistortion)
    python sfm.py --no-gpu              # for COLMAP built without CUDA
    python sfm.py --fresh               # wipe database/sparse and restart

Notes / decisions (see chat): the README suggests locking intrinsics via a
cameras.txt plus --single_camera_per_folder. That flag cannot inject custom
intrinsics, so to actually honor the calibration we instead run the extractor
once per position with --image_list_path + --ImageReader.single_camera 1 +
--ImageReader.camera_params. We also undistort the frames first (using the
calibrated dist_*.txt) so the distortion-free PINHOLE model is valid; pass
--no-undistort to skip. A reference sfm/cameras.txt is still written.
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

import cv2
import numpy as np


# --- Paths / constants -------------------------------------------------------

POSITION_ORDER = ["top_left", "top_right", "bot_left", "bot_right"]

HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA_JSON = os.path.join(HERE, "camera.json")
INTRINSICS_DIR = os.path.join(HERE, "intrinsics")
EXTRINSICS_DIR = os.path.join(HERE, "extrinsics")

SFM_DIR = os.path.join(HERE, "sfm")
IMAGES_INPUT_DIR = os.path.join(SFM_DIR, "images")
IMAGES_PINHOLE_DIR = os.path.join(SFM_DIR, "images_pinhole")
DATABASE_PATH = os.path.join(SFM_DIR, "database.db")
SPARSE_DIR = os.path.join(SFM_DIR, "sparse")
CAMERAS_TXT = os.path.join(SFM_DIR, "cameras.txt")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# --- Loading / validation ----------------------------------------------------

def load_camera_indices():
    """Load camera.json -> ordered list of (position, device_index)."""
    if not os.path.exists(CAMERA_JSON):
        raise FileNotFoundError(f"camera.json not found at {CAMERA_JSON}.")
    with open(CAMERA_JSON, "r") as f:
        mapping = json.load(f)
    cameras = []
    for position in POSITION_ORDER:
        if position not in mapping:
            raise KeyError(f"camera.json missing '{position}'.")
        cameras.append((position, int(mapping[position])))
    return cameras


def load_intrinsics(camera_number):
    """Load K (3x3) and distortion coefficients for one camera."""
    k_path = os.path.join(INTRINSICS_DIR, f"K_{camera_number}.txt")
    dist_path = os.path.join(INTRINSICS_DIR, f"dist_{camera_number}.txt")
    if not os.path.exists(k_path) or not os.path.exists(dist_path):
        raise FileNotFoundError(
            f"Missing intrinsics for camera {camera_number} "
            f"({k_path} / {dist_path}). Run intrinsics.py (Stage 1) first."
        )
    K = np.loadtxt(k_path, dtype=np.float64).reshape(3, 3)
    dist = np.loadtxt(dist_path, dtype=np.float64).reshape(1, -1)
    return K, dist


def find_colmap(explicit):
    """Resolve the colmap binary or fail with install guidance."""
    binary = explicit or shutil.which("colmap")
    if not binary:
        raise RuntimeError(
            "COLMAP was not found on PATH. Install it and ensure 'colmap' is "
            "runnable, or pass --colmap C:\\path\\to\\colmap.exe.\n"
            "  Windows: download from https://github.com/colmap/colmap/releases"
            " and add the folder to PATH.\n"
            "  Ubuntu:  sudo apt install colmap\n"
            "  Docs:    https://colmap.github.io/install.html"
        )
    return binary


def discover_frames():
    """
    Scan sfm/images/ for {position}_{frame}.ext files, grouped by position.
    Returns {position: [filenames sorted]}.
    """
    if not os.path.isdir(IMAGES_INPUT_DIR):
        raise FileNotFoundError(
            f"{IMAGES_INPUT_DIR} does not exist. Export your static-scene "
            "frames there, named {position}_{frame:06d}.jpg, e.g. "
            "top_left_000001.jpg."
        )

    groups = {p: [] for p in POSITION_ORDER}
    for name in sorted(os.listdir(IMAGES_INPUT_DIR)):
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        matched = None
        for position in POSITION_ORDER:
            if name.startswith(position + "_"):
                matched = position
                break
        if matched:
            groups[matched].append(name)

    missing = [p for p in POSITION_ORDER if not groups[p]]
    if missing:
        raise RuntimeError(
            f"No frames found for: {missing}. Expected files like "
            f"'{missing[0]}_000001.jpg' in {IMAGES_INPUT_DIR}."
        )
    for position in POSITION_ORDER:
        n = len(groups[position])
        print(f"  {position}: {n} frame(s)")
        if n < 3:
            print(f"    NOTE: only {n} frame(s) for {position}; more views "
                  "(a few seconds of static clip) improve registration.")
    return groups


# --- Preparation -------------------------------------------------------------

def image_size_of(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read image {path}.")
    return img.shape[1], img.shape[0]  # (w, h)


def prepare_working_images(groups, intrinsics, undistort):
    """
    Build the directory COLMAP will read from. With undistort=True, each frame
    is undistorted with its camera's K/dist (so the PINHOLE model is valid) and
    written to sfm/images_pinhole/. With undistort=False, the raw sfm/images/
    folder is used as-is.

    Returns (work_dir, {position: (fx, fy, cx, cy)}, (w, h)).
    """
    sample = os.path.join(IMAGES_INPUT_DIR, groups[POSITION_ORDER[0]][0])
    width, height = image_size_of(sample)

    pinhole_params = {}
    for position in POSITION_ORDER:
        K, _ = intrinsics[position]
        pinhole_params[position] = (
            float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        )

    if not undistort:
        print("  Using raw frames (no undistortion).")
        return IMAGES_INPUT_DIR, pinhole_params, (width, height)

    print(f"  Undistorting frames into {IMAGES_PINHOLE_DIR} ...")
    if os.path.isdir(IMAGES_PINHOLE_DIR):
        shutil.rmtree(IMAGES_PINHOLE_DIR)
    os.makedirs(IMAGES_PINHOLE_DIR, exist_ok=True)

    for position in POSITION_ORDER:
        K, dist = intrinsics[position]
        for name in groups[position]:
            src = os.path.join(IMAGES_INPUT_DIR, name)
            img = cv2.imread(src)
            if img is None:
                raise RuntimeError(f"Could not read {src}.")
            # newCameraMatrix=K keeps the PINHOLE params equal to K.
            undistorted = cv2.undistort(img, K, dist, None, K)
            cv2.imwrite(os.path.join(IMAGES_PINHOLE_DIR, name), undistorted)

    return IMAGES_PINHOLE_DIR, pinhole_params, (width, height)


def write_cameras_txt(pinhole_params, image_size):
    """Write a reference COLMAP cameras.txt (PINHOLE) from the calibration."""
    w, h = image_size
    with open(CAMERAS_TXT, "w") as f:
        f.write("# Camera list (reference; PINHOLE intrinsics from Stage 1)\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx, fy, cx, cy]\n")
        for cam_id, position in enumerate(POSITION_ORDER, start=1):
            fx, fy, cx, cy = pinhole_params[position]
            f.write(f"{cam_id} PINHOLE {w} {h} "
                    f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    print(f"  Wrote {CAMERAS_TXT}")


# --- COLMAP invocation -------------------------------------------------------

def run(cmd):
    """Run a COLMAP command, streaming its output. Raise on failure."""
    print("\n$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"COLMAP command failed (exit {result.returncode}): {cmd[1]}"
        )


def feature_extraction(colmap, work_dir, groups, pinhole_params, use_gpu):
    """One extractor call per position so each gets its calibrated camera."""
    list_dir = os.path.join(SFM_DIR, "_image_lists")
    os.makedirs(list_dir, exist_ok=True)

    for position in POSITION_ORDER:
        list_path = os.path.join(list_dir, f"{position}.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(groups[position]) + "\n")

        fx, fy, cx, cy = pinhole_params[position]
        run([
            colmap, "feature_extractor",
            "--database_path", DATABASE_PATH,
            "--image_path", work_dir,
            "--image_list_path", list_path,
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}",
            "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
        ])


def feature_matching(colmap, use_gpu):
    run([
        colmap, "exhaustive_matcher",
        "--database_path", DATABASE_PATH,
        "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
    ])


def mapping(colmap, work_dir):
    os.makedirs(SPARSE_DIR, exist_ok=True)
    run([
        colmap, "mapper",
        "--database_path", DATABASE_PATH,
        "--image_path", work_dir,
        "--output_path", SPARSE_DIR,
        # Trust the calibrated intrinsics: do not let bundle adjustment move
        # focal length / principal point / distortion.
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
    ])


def model_num_images(model_dir):
    """Read the registered-image count from a COLMAP images.bin header."""
    bin_path = os.path.join(model_dir, "images.bin")
    if not os.path.exists(bin_path):
        return 0
    try:
        with open(bin_path, "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]
    except Exception:
        return 0


def select_best_model():
    """
    COLMAP may emit several sub-models (sparse/0, sparse/1, ...), numbered by
    creation order rather than size. Pick the one with the most registered
    images.
    """
    if not os.path.isdir(SPARSE_DIR):
        raise RuntimeError(
            f"Mapper produced no output in {SPARSE_DIR}. Reconstruction failed "
            "— see COLMAP output above (often too few matches / low texture)."
        )
    candidates = []
    for name in sorted(os.listdir(SPARSE_DIR)):
        model_dir = os.path.join(SPARSE_DIR, name)
        if os.path.isdir(model_dir) and \
                os.path.exists(os.path.join(model_dir, "images.bin")):
            candidates.append((model_dir, model_num_images(model_dir)))
    if not candidates:
        raise RuntimeError(
            f"No reconstruction found under {SPARSE_DIR}. Reconstruction "
            "failed — see COLMAP output above."
        )
    candidates.sort(key=lambda x: x[1], reverse=True)
    if len(candidates) > 1:
        summary = ", ".join(f"{os.path.basename(d)}({n})"
                            for d, n in candidates)
        print(f"  COLMAP produced {len(candidates)} sub-models: {summary}")
        print("  Using the largest. (Multiple models mean some views did not "
              "connect into a single reconstruction.)")
    return candidates[0][0]


def convert_model(colmap, model_dir):
    run([
        colmap, "model_converter",
        "--input_path", model_dir,
        "--output_path", model_dir,
        "--output_type", "TXT",
    ])
    return model_dir


# --- Reporting / validation --------------------------------------------------

def quat_to_rot(qw, qx, qy, qz):
    """COLMAP quaternion (w, x, y, z) -> rotation matrix (world->cam)."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),
         2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),
         2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
         1 - 2 * (qx * qx + qy * qy)],
    ])


def parse_images_txt(model_dir):
    """Return list of (name, camera_center) from COLMAP images.txt."""
    path = os.path.join(model_dir, "images.txt")
    entries = []
    with open(path, "r") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    # images.txt: data lines come in pairs; the first of each pair is the pose.
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        R = quat_to_rot(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        center = -R.T @ t
        entries.append((name, center))
    return entries


def count_points(model_dir):
    path = os.path.join(model_dir, "points3D.txt")
    n = 0
    with open(path, "r") as f:
        for ln in f:
            if not ln.startswith("#") and ln.strip():
                n += 1
    return n


def position_of(name):
    for position in POSITION_ORDER:
        if name.startswith(position + "_"):
            return position
    return None


def calibrated_centers():
    """Camera centers (in the top_left frame) from extrinsics/poses.npz."""
    npz_path = os.path.join(EXTRINSICS_DIR, "poses.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    centers = {}
    for position in POSITION_ORDER:
        R = data[f"R_{position}"]
        t = data[f"t_{position}"].reshape(3)
        centers[position] = -R.T @ t
    return centers


def report_and_validate(model_dir):
    entries = parse_images_txt(model_dir)
    n_points = count_points(model_dir)

    registered_positions = {}
    for name, center in entries:
        p = position_of(name)
        if p is None:
            continue
        registered_positions.setdefault(p, []).append(center)

    print("\n=== Reconstruction summary ===")
    print(f"  Registered images: {len(entries)}")
    print(f"  3D points: {n_points}")
    present = [p for p in POSITION_ORDER if p in registered_positions]
    print(f"  Camera positions registered: {len(present)}/4  {present}")
    if len(present) < 3:
        print("  WARNING: fewer than 3 of 4 camera positions registered. "
              "Extrinsic calibration or the static capture may need redoing "
              "(more overlap / texture / frames).")

    # Best-effort cross-check vs. the calibrated extrinsics (scale-normalized,
    # since COLMAP's reconstruction has an arbitrary scale).
    try:
        cal = calibrated_centers()
        if cal and len(present) >= 3:
            colmap_centers = {
                p: np.mean(registered_positions[p], axis=0) for p in present
            }
            pairs = [(a, b) for i, a in enumerate(present)
                     for b in present[i + 1:]]
            d_colmap = {pair: float(np.linalg.norm(
                colmap_centers[pair[0]] - colmap_centers[pair[1]]))
                for pair in pairs}
            d_cal = {pair: float(np.linalg.norm(
                cal[pair[0]] - cal[pair[1]])) for pair in pairs}
            ratios = [d_cal[p] / d_colmap[p] for p in pairs
                      if d_colmap[p] > 1e-9]
            if ratios:
                scale = float(np.median(ratios))
                print("\n  Cross-check vs. calibrated extrinsics "
                      "(COLMAP scaled to mm):")
                for pair in pairs:
                    if d_colmap[pair] <= 1e-9:
                        continue
                    scaled = d_colmap[pair] * scale
                    err = abs(scaled - d_cal[pair]) / d_cal[pair] * 100
                    print(f"    {pair[0]} <-> {pair[1]}: "
                          f"{scaled:.1f} mm vs {d_cal[pair]:.1f} mm "
                          f"({err:.0f}%)")
    except Exception as exc:  # never let validation break the run
        print(f"  (extrinsic cross-check skipped: {exc})")


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 3 SfM via COLMAP.")
    parser.add_argument("--colmap", default=None,
                        help="Path to the colmap binary (default: PATH).")
    parser.add_argument("--no-undistort", action="store_true",
                        help="Feed raw frames instead of undistorting first.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable CUDA (COLMAP built without GPU support).")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing database/sparse before running.")
    args = parser.parse_args()

    colmap = find_colmap(args.colmap)
    cameras = load_camera_indices()
    intrinsics = {p: load_intrinsics(idx) for p, idx in cameras}

    os.makedirs(SFM_DIR, exist_ok=True)
    if args.fresh:
        for path in (DATABASE_PATH, SPARSE_DIR):
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
        print("Cleared previous database/sparse.")

    if os.path.exists(DATABASE_PATH):
        print(f"NOTE: {DATABASE_PATH} already exists; COLMAP will append. Use "
              "--fresh for a clean run.")

    print("Stage 3 — SfM")
    print(f"  COLMAP: {colmap}")
    print("Discovering frames...")
    groups = discover_frames()

    print("Preparing images / intrinsics...")
    work_dir, pinhole_params, image_size = prepare_working_images(
        groups, intrinsics, undistort=not args.no_undistort
    )
    write_cameras_txt(pinhole_params, image_size)

    use_gpu = not args.no_gpu
    print("\nRunning COLMAP feature extraction (per camera)...")
    feature_extraction(colmap, work_dir, groups, pinhole_params, use_gpu)
    print("\nRunning COLMAP exhaustive matching...")
    feature_matching(colmap, use_gpu)
    print("\nRunning COLMAP mapper...")
    mapping(colmap, work_dir)
    print("\nExporting model to TXT...")
    model_dir = select_best_model()
    convert_model(colmap, model_dir)

    report_and_validate(model_dir)
    print(f"\nSparse model ready at {model_dir}")


if __name__ == "__main__":
    main()
