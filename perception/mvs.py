"""
Stage 4 — MVS / Dense Point Cloud (quasar/perception)

Multi-View Stereo. The camera poses are FIXED (recovered once by the static
Stage-3 SfM pass) and reused for every synchronized dynamic frame. For each
frame-set (one image per camera at time T) COLMAP's PatchMatch estimates a
dense depth map per view and fuses them into a single dense point cloud.

Pipeline per frame (COLMAP CLI via subprocess):
  1. feature_extractor   - detect SIFT features in the 4 raw views
  2. exhaustive_matcher  - match features across the 4 views
  3. point_triangulator  - triangulate a sparse cloud AT THE FIXED POSES
  4. image_undistorter   - undistort the 4 raw views + lay out a dense workspace
  5. patch_match_stereo  - dense depth/normal maps (CUDA GPU required)
  6. stereo_fusion       - fuse depth maps -> fused.ply

Depends on: sfm/sparse/0/ (fixed poses), intrinsics/K_*.txt + dist_*.txt,
            camera.json.

----------------------------------------------------------------------------
Design note (why this differs slightly from the README):

The README says "copy sfm/sparse/0/ as the fixed pose set". That assumes the
SfM model holds one image per camera. In practice the Stage-3 static capture
produced ~10 frames per camera, so sfm/sparse/0 contains ~40 images that don't
match a dynamic frame's 4 images. We instead lift one fixed pose per camera
position from the refined SfM model and reuse those poses for every frame.

Crucially, COLMAP's dense stereo needs a sparse model WITH 3D points: PatchMatch
derives each view's depth-search range and stereo source images from sparse
co-visibility, and stereo_fusion needs consistent geometry to fuse. A poses-only
model with an empty points3D.txt makes PatchMatch produce geometrically
inconsistent depth maps that fusion silently discards (0 points).

So per frame we run feature_extractor + exhaustive_matcher on the 4 views, then
point_triangulator with the FIXED rig poses + calibrated OPENCV intrinsics. This
yields a small but real sparse cloud (typically 100s of points) that anchors
the dense step. Depth ranges are then auto-derived from each frame's own points,
which also handles dynamic content moving in depth.
----------------------------------------------------------------------------

Input layout (raw synchronized dynamic footage):
    mvs/frames/
    |-- 000001/
    |   |-- top_left.jpg
    |   |-- top_right.jpg
    |   |-- bot_left.jpg
    |   `-- bot_right.jpg
    |-- 000002/
    |   `-- ...

Output: mvs/frame_{n:06d}/fused.ply  (one dense cloud per frame)

Run:
    python mvs.py
    python mvs.py --start_frame 10 --end_frame 20
    python mvs.py --max_image_size 640
    python mvs.py --fresh                 # delete existing mvs/frame_* workspaces
"""

import argparse
import os
import shutil
import sqlite3

import cv2
import numpy as np

# Reuse Stage-3 helpers (no side effects at import; main() is __main__-guarded).
from sfm import (
    POSITION_ORDER,
    load_camera_indices,
    load_intrinsics,
    find_colmap,
    run,
    select_best_model,
    SPARSE_DIR as SFM_SPARSE_DIR,
)


# --- Paths -------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
MVS_DIR = os.path.join(HERE, "mvs")
FRAMES_DIR = os.path.join(MVS_DIR, "frames")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# A fused 3D point must be supported by at least this many consistent pixel
# observations. COLMAP's default is 5, which is impossible for a 4-camera rig
# (a point can be seen by at most 4 cameras), so fusion would always yield 0.
MIN_NUM_PIXELS = 2


# --- Fixed-rig model ---------------------------------------------------------

def parse_sfm_poses(model_dir):
    """
    From sfm/sparse/0/images.txt pick, per camera position, the single image
    with the most 3D-point observations (best-constrained pose).

    Returns {position: (qw, qx, qy, qz, tx, ty, tz)} using COLMAP's raw
    world->camera convention (exactly as stored in images.txt).
    """
    images_txt = os.path.join(model_dir, "images.txt")
    if not os.path.exists(images_txt):
        raise FileNotFoundError(
            f"{images_txt} not found. Run sfm.py (Stage 3) first; it exports "
            "the model to TXT."
        )
    with open(images_txt, "r") as f:
        lines = [ln for ln in f if not ln.startswith("#")]

    best = {}  # position -> (n_obs, pose_tuple)
    for i in range(0, len(lines) - 1, 2):
        header = lines[i].split()
        if len(header) < 10:
            continue
        pose = tuple(float(x) for x in header[1:8])  # qw qx qy qz tx ty tz
        name = header[9]
        position = _position_of(name)
        if position is None:
            continue
        pts = lines[i + 1].split()
        # second line is X Y POINT3D_ID triplets; count valid observations
        n_obs = sum(1 for j in range(2, len(pts), 3) if pts[j] != "-1")
        if position not in best or n_obs > best[position][0]:
            best[position] = (n_obs, pose)

    missing = [p for p in POSITION_ORDER if p not in best]
    if missing:
        raise RuntimeError(
            f"SfM model is missing camera positions {missing}. All 4 cameras "
            "must be registered in Stage 3 before running MVS."
        )
    return {p: best[p][1] for p in POSITION_ORDER}


def _position_of(name):
    for position in POSITION_ORDER:
        if name.startswith(position):
            return position
    return None


def read_db_images(db_path):
    """Return [(image_id, name, camera_id), ...] as COLMAP assigned them."""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
    finally:
        con.close()
    return rows


def write_pose_model(model_dir, db_rows, intrinsics, poses, image_size):
    """
    Write a poses-only COLMAP input model (TXT) for point_triangulator.

    Camera/image IDs are taken straight from the feature database so the model
    and database agree; intrinsics are our calibrated OPENCV parameters and the
    poses are the fixed rig poses (matched to each row by camera position).
    points3D.txt is empty — point_triangulator fills it in.
    """
    os.makedirs(model_dir, exist_ok=True)
    w, h = image_size

    with open(os.path.join(model_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for _image_id, name, cam_id in db_rows:
            position = _position_of(name)
            K, dist = intrinsics[position]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            d = np.array(dist).reshape(-1)
            k1 = float(d[0]) if d.size > 0 else 0.0
            k2 = float(d[1]) if d.size > 1 else 0.0
            p1 = float(d[2]) if d.size > 2 else 0.0
            p2 = float(d[3]) if d.size > 3 else 0.0
            f.write(f"{cam_id} OPENCV {w} {h} "
                    f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} "
                    f"{k1:.8f} {k2:.8f} {p1:.8f} {p2:.8f}\n")

    with open(os.path.join(model_dir, "images.txt"), "w") as f:
        f.write("# Image list: IMAGE_ID, QW,QX,QY,QZ, TX,TY,TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, name, cam_id in db_rows:
            position = _position_of(name)
            qw, qx, qy, qz, tx, ty, tz = poses[position]
            f.write(f"{image_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{tx:.10f} {ty:.10f} {tz:.10f} {cam_id} {name}\n")
            f.write("\n")  # no 2D observations; triangulation provides them

    # Empty points3D.txt — point_triangulator populates it.
    open(os.path.join(model_dir, "points3D.txt"), "w").close()


def count_model_points(model_dir):
    """Count points in a TXT model's points3D.txt (0 if absent)."""
    path = os.path.join(model_dir, "points3D.txt")
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for ln in f if ln.strip() and not ln.startswith("#"))


# --- Dynamic frame discovery -------------------------------------------------

def find_view_file(folder, position):
    """Locate {position}.<ext> inside a frame folder (any supported ext)."""
    for ext in IMAGE_EXTS:
        candidate = os.path.join(folder, f"{position}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def discover_dynamic_frames(start_frame, end_frame):
    """
    Return sorted list of (frame_number, folder_path) for frame folders that
    contain all 4 camera views, filtered to [start_frame, end_frame].
    """
    if not os.path.isdir(FRAMES_DIR):
        raise FileNotFoundError(
            f"{FRAMES_DIR} does not exist. Create one folder per synchronized "
            "moment, e.g. mvs/frames/000001/top_left.jpg (+ top_right, "
            "bot_left, bot_right)."
        )
    frames = []
    for name in sorted(os.listdir(FRAMES_DIR)):
        folder = os.path.join(FRAMES_DIR, name)
        if not os.path.isdir(folder):
            continue
        try:
            num = int(name)
        except ValueError:
            print(f"  Skipping non-numeric frame folder '{name}'.")
            continue
        if start_frame is not None and num < start_frame:
            continue
        if end_frame is not None and num > end_frame:
            continue
        views = {p: find_view_file(folder, p) for p in POSITION_ORDER}
        missing = [p for p, v in views.items() if v is None]
        if missing:
            print(f"  Skipping frame {name}: missing views {missing}.")
            continue
        frames.append((num, folder, views))
    if not frames:
        raise RuntimeError(
            "No complete frame folders found in "
            f"{FRAMES_DIR} for the requested range."
        )
    return frames


# --- Per-frame MVS -----------------------------------------------------------

def count_ply_points(path):
    """Read the vertex count from a PLY header."""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "rb") as f:
            for _ in range(40):
                line = f.readline().decode("ascii", "ignore").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except Exception:
        return 0
    return 0


def write_patchmatch_config(dense_dir):
    """Overwrite dense/stereo/patch-match.cfg so every view uses all the other
    views as stereo source images."""
    cfg_path = os.path.join(dense_dir, "stereo", "patch-match.cfg")
    with open(cfg_path, "w") as f:
        for position in POSITION_ORDER:
            f.write(f"{position}.jpg\n")
            f.write("__all__\n")


def process_frame(colmap, num, views, intrinsics, poses, image_size,
                  max_image_size, use_gpu):
    """Run the full per-frame pipeline (features -> triangulation -> dense
    fusion). Returns the fused point count, or raises on failure (caller
    records it)."""
    frame_dir = os.path.join(MVS_DIR, f"frame_{num:06d}")
    images_dir = os.path.join(frame_dir, "images")
    model_in = os.path.join(frame_dir, "model_in")    # poses-only input model
    sparse_dir = os.path.join(frame_dir, "sparse")    # triangulated model
    dense_dir = os.path.join(frame_dir, "dense")
    db_path = os.path.join(frame_dir, "database.db")
    fused_ply = os.path.join(frame_dir, "fused.ply")

    # Clean any stale per-frame workspace, then lay it out fresh.
    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # Copy the 4 raw views in, named {position}.jpg.
    for position in POSITION_ORDER:
        shutil.copy(views[position], os.path.join(images_dir, f"{position}.jpg"))

    gpu_flag = "1" if use_gpu else "0"

    # 1-2. Detect + match features across the 4 views (intrinsic-free).
    run([
        colmap, "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "0",
        "--FeatureExtraction.use_gpu", gpu_flag,
    ])
    run([
        colmap, "exhaustive_matcher",
        "--database_path", db_path,
        "--FeatureMatching.use_gpu", gpu_flag,
    ])

    # 3. Triangulate a sparse cloud at the FIXED rig poses + calibrated K.
    db_rows = read_db_images(db_path)
    write_pose_model(model_in, db_rows, intrinsics, poses, image_size)
    run([
        colmap, "point_triangulator",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--input_path", model_in,
        "--output_path", sparse_dir,
    ])
    run([
        colmap, "model_converter",
        "--input_path", sparse_dir,
        "--output_path", sparse_dir,
        "--output_type", "TXT",
    ])
    n_sparse = count_model_points(sparse_dir)
    print(f"  Triangulated sparse points: {n_sparse}")
    if n_sparse == 0:
        raise RuntimeError(
            "point_triangulator produced 0 points — the 4 views share too "
            "little matchable texture. Capture a more textured scene."
        )

    # 4. Undistort + lay out the dense workspace.
    run([
        colmap, "image_undistorter",
        "--image_path", images_dir,
        "--input_path", sparse_dir,
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ])

    # For a 4-camera rig every view should use the other three as stereo
    # sources; make that explicit rather than relying on co-visibility counts.
    write_patchmatch_config(dense_dir)

    # 5. Dense depth/normal maps. Depth ranges are auto-derived from the
    #    frame's own triangulated points (handles depth changes over time).
    run([
        colmap, "patch_match_stereo",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.max_image_size", str(max_image_size),
        "--PatchMatchStereo.geom_consistency", "1",
        "--PatchMatchStereo.gpu_index", "0" if use_gpu else "-1",
    ])

    # 6. Fuse. min_num_pixels must be <= 4 for a 4-camera rig (see constant).
    run([
        colmap, "stereo_fusion",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", fused_ply,
        "--StereoFusion.min_num_pixels", str(MIN_NUM_PIXELS),
    ])

    return count_ply_points(fused_ply)


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 4 dense MVS via COLMAP.")
    parser.add_argument("--colmap", default=None,
                        help="Path to the colmap binary (default: PATH).")
    parser.add_argument("--start_frame", type=int, default=None,
                        help="First frame number to process (inclusive).")
    parser.add_argument("--end_frame", type=int, default=None,
                        help="Last frame number to process (inclusive).")
    parser.add_argument("--max_image_size", type=int, default=640,
                        help="Max image dimension for PatchMatch (default 640).")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable CUDA. NOTE: PatchMatch requires a GPU; "
                             "this will almost certainly fail without CUDA.")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing mvs/frame_* workspaces first.")
    args = parser.parse_args()

    colmap = find_colmap(args.colmap)
    cameras = load_camera_indices()
    intrinsics = {p: load_intrinsics(idx) for p, idx in cameras}

    print("Stage 4 — MVS (dense)")
    print(f"  COLMAP: {colmap}")

    # Locate the Stage-3 SfM model (largest sub-model) and lift fixed poses.
    if not os.path.isdir(SFM_SPARSE_DIR):
        raise FileNotFoundError(
            f"{SFM_SPARSE_DIR} not found. Run sfm.py (Stage 3) first."
        )
    sfm_model = select_best_model()
    print(f"  SfM model: {sfm_model}")
    poses = parse_sfm_poses(sfm_model)

    os.makedirs(MVS_DIR, exist_ok=True)
    if args.fresh:
        for name in os.listdir(MVS_DIR):
            if name.startswith("frame_"):
                shutil.rmtree(os.path.join(MVS_DIR, name))
        print("  Cleared previous frame_* workspaces.")

    print("Discovering dynamic frames...")
    frames = discover_dynamic_frames(args.start_frame, args.end_frame)
    print(f"  {len(frames)} frame(s) to process.")

    # All frames share one resolution (the calibrated rig); read it once.
    sample_view = frames[0][2][POSITION_ORDER[0]]
    sample = cv2.imread(sample_view)
    if sample is None:
        raise RuntimeError(f"Could not read sample frame {sample_view}.")
    image_size = (sample.shape[1], sample.shape[0])
    print(f"  Frame resolution: {image_size[0]}x{image_size[1]}")

    use_gpu = not args.no_gpu
    if not use_gpu:
        print("  WARNING: --no-gpu set; PatchMatch needs CUDA and will likely "
              "fail.")

    point_counts = []
    failures = []
    for idx, (num, _folder, views) in enumerate(frames, start=1):
        print(f"\n=== Frame {num:06d} ({idx}/{len(frames)}) ===")
        try:
            n = process_frame(colmap, num, views, intrinsics, poses,
                              image_size, args.max_image_size, use_gpu)
            print(f"  -> fused.ply: {n} points")
            point_counts.append(n)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append(num)

    print("\n=== MVS summary ===")
    print(f"  Frames processed: {len(point_counts)}/{len(frames)}")
    if point_counts:
        print(f"  Avg points per cloud: {int(np.mean(point_counts)):,}")
        print(f"  Min / max points: {min(point_counts):,} / "
              f"{max(point_counts):,}")
    if failures:
        print(f"  FAILED frames: {failures}")
    print(f"\n  Dense clouds: {MVS_DIR}\\frame_XXXXXX\\fused.ply")


if __name__ == "__main__":
    main()
