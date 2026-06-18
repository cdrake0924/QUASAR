"""
Stage 4 — MVS / Dense Point Cloud (quasar/perception)

Multi-View Stereo. The camera poses are FIXED (recovered once by the static
Stage-3 SfM pass) and reused for every synchronized dynamic frame. For each
frame-set (one image per camera at time T) COLMAP's PatchMatch estimates a
dense depth map per view and fuses them into a single dense point cloud.

Pipeline per frame (COLMAP CLI via subprocess):
  1. image_undistorter   - undistort the 4 raw views + lay out a dense workspace
  2. patch_match_stereo  - dense depth/normal maps (CUDA GPU required)
  3. stereo_fusion       - fuse depth maps -> fused.ply

Depends on: sfm/sparse/0/ (fixed poses), intrinsics/K_*.txt + dist_*.txt,
            camera.json.

----------------------------------------------------------------------------
Design note (why this differs slightly from the README):

The README says "copy sfm/sparse/0/ as the fixed pose set". That assumes the
SfM model holds one image per camera. In practice the Stage-3 static capture
produced ~10 frames per camera, so sfm/sparse/0 contains ~40 images. COLMAP's
image_undistorter requires every image in the model to exist on disk, so a
40-image model cannot be reused against the 4 images of a dynamic frame.

Instead we build a *reusable fixed-rig model* once: 4 cameras + 4 images (the
single best-constrained pose per position, lifted from the refined SfM model),
named {position}.jpg to match the dynamic frames, with OPENCV intrinsics
(fx,fy,cx,cy,k1,k2,p1,p2) so image_undistorter genuinely undistorts the raw
frames. This honors the README's intent (fixed poses reused per frame) while
working with the real data.
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
RIG_DIR = os.path.join(MVS_DIR, "_rig")          # reusable fixed-pose model
RIG_SPARSE_DIR = os.path.join(RIG_DIR, "sparse")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


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


def _quat_to_rot(qw, qx, qy, qz):
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


def compute_depth_range(model_dir, poses):
    """
    Estimate a global [depth_min, depth_max] (in the SfM coordinate scale) by
    projecting the static 3D points into each fixed camera and taking robust
    percentiles of their z (camera-forward) coordinate. Returns (dmin, dmax)
    or None if no points are available.
    """
    points_txt = os.path.join(model_dir, "points3D.txt")
    if not os.path.exists(points_txt):
        return None
    xyz = []
    with open(points_txt, "r") as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            parts = ln.split()
            if len(parts) < 4:
                continue
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not xyz:
        return None
    X = np.array(xyz).T  # 3 x N

    depths = []
    for position in POSITION_ORDER:
        qw, qx, qy, qz, tx, ty, tz = poses[position]
        R = _quat_to_rot(qw, qx, qy, qz)
        t = np.array([tx, ty, tz]).reshape(3, 1)
        z = (R @ X + t)[2]
        depths.append(z[z > 0])
    depths = np.concatenate(depths) if depths else np.array([])
    if depths.size == 0:
        return None
    dmin = float(np.percentile(depths, 1))
    dmax = float(np.percentile(depths, 99))
    # pad the range so legitimately closer/farther dynamic content isn't clipped
    span = max(dmax - dmin, 1e-6)
    dmin = max(dmin - 0.25 * span, 1e-4)
    dmax = dmax + 0.25 * span
    return dmin, dmax


def build_rig_model(intrinsics, image_size, poses):
    """
    Write the reusable fixed-rig COLMAP model (TXT) to mvs/_rig/sparse/.
    4 OPENCV cameras (one per position) + 4 images named {position}.jpg, each
    carrying its fixed SfM pose. points3D.txt is intentionally empty (depth
    ranges are passed to PatchMatch explicitly).
    """
    if os.path.isdir(RIG_SPARSE_DIR):
        shutil.rmtree(RIG_SPARSE_DIR)
    os.makedirs(RIG_SPARSE_DIR, exist_ok=True)
    w, h = image_size

    # cameras.txt — OPENCV model: fx, fy, cx, cy, k1, k2, p1, p2
    with open(os.path.join(RIG_SPARSE_DIR, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(POSITION_ORDER)}\n")
        for cam_id, position in enumerate(POSITION_ORDER, start=1):
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

    # images.txt — 2 lines per image; the 2nd (points2D) line is empty.
    with open(os.path.join(RIG_SPARSE_DIR, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(POSITION_ORDER)}\n")
        for cam_id, position in enumerate(POSITION_ORDER, start=1):
            qw, qx, qy, qz, tx, ty, tz = poses[position]
            f.write(f"{cam_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{tx:.10f} {ty:.10f} {tz:.10f} {cam_id} {position}.jpg\n")
            f.write("\n")  # no 2D observations

    # points3D.txt — empty (header only).
    with open(os.path.join(RIG_SPARSE_DIR, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")

    print(f"  Fixed-rig model written to {RIG_SPARSE_DIR}")


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


def process_frame(colmap, num, views, max_image_size, depth_range, use_gpu):
    """Run undistort -> patch_match -> fusion for one frame. Returns point
    count, or raises on failure (caller records it)."""
    frame_dir = os.path.join(MVS_DIR, f"frame_{num:06d}")
    images_dir = os.path.join(frame_dir, "images")
    sparse_dir = os.path.join(frame_dir, "sparse")
    dense_dir = os.path.join(frame_dir, "dense")
    fused_ply = os.path.join(frame_dir, "fused.ply")

    # Clean any stale per-frame workspace, then lay it out fresh.
    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # Copy the 4 raw views in, named to match the rig model ({position}.jpg).
    for position in POSITION_ORDER:
        shutil.copy(views[position], os.path.join(images_dir, f"{position}.jpg"))
    # Copy the fixed-rig model in.
    for fn in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy(os.path.join(RIG_SPARSE_DIR, fn),
                    os.path.join(sparse_dir, fn))

    run([
        colmap, "image_undistorter",
        "--image_path", images_dir,
        "--input_path", sparse_dir,
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ])

    pm = [
        colmap, "patch_match_stereo",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.max_image_size", str(max_image_size),
        "--PatchMatchStereo.geom_consistency", "1",
        "--PatchMatchStereo.gpu_index", "0" if use_gpu else "-1",
    ]
    if depth_range is not None:
        dmin, dmax = depth_range
        pm += ["--PatchMatchStereo.depth_min", f"{dmin:.6f}",
               "--PatchMatchStereo.depth_max", f"{dmax:.6f}"]
    run(pm)

    run([
        colmap, "stereo_fusion",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", fused_ply,
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
    depth_range = compute_depth_range(sfm_model, poses)
    if depth_range:
        print(f"  Depth range (SfM scale): "
              f"{depth_range[0]:.3f} .. {depth_range[1]:.3f}")
    else:
        print("  Depth range: auto (no SfM points found)")

    os.makedirs(MVS_DIR, exist_ok=True)
    if args.fresh:
        for name in os.listdir(MVS_DIR):
            if name.startswith("frame_"):
                shutil.rmtree(os.path.join(MVS_DIR, name))
        print("  Cleared previous frame_* workspaces.")

    print("Discovering dynamic frames...")
    frames = discover_dynamic_frames(args.start_frame, args.end_frame)
    print(f"  {len(frames)} frame(s) to process.")

    # Build the reusable fixed-rig model from a sample frame's dimensions.
    sample_view = frames[0][2][POSITION_ORDER[0]]
    sample = cv2.imread(sample_view)
    if sample is None:
        raise RuntimeError(f"Could not read sample frame {sample_view}.")
    image_size = (sample.shape[1], sample.shape[0])
    print(f"  Frame resolution: {image_size[0]}x{image_size[1]}")
    build_rig_model(intrinsics, image_size, poses)

    use_gpu = not args.no_gpu
    if not use_gpu:
        print("  WARNING: --no-gpu set; PatchMatch needs CUDA and will likely "
              "fail.")

    point_counts = []
    failures = []
    for idx, (num, _folder, views) in enumerate(frames, start=1):
        print(f"\n=== Frame {num:06d} ({idx}/{len(frames)}) ===")
        try:
            n = process_frame(colmap, num, views, args.max_image_size,
                              depth_range, use_gpu)
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
