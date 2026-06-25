"""
Stage 4 — MVS / Dense Point Cloud (quasar/perception)

Multi-View Stereo on the FIXED rig poses from rig.py (Stage 3). For each
synchronized 4-image set, COLMAP's PatchMatch estimates a dense depth map per
view and fuses them into a single dense point cloud. Two modes:

  --mode static    process mvs/static/{position}.jpg          -> mvs/static_fused.ply
  --mode dynamic   process mvs/frames/{n:06d}/{position}.jpg  -> mvs/frame_{n:06d}/fused.ply

Both modes share the identical rig poses (rig/sparse/). The rig is fixed, so the
scene depth range is estimated ONCE (from a reference frame-set) and reused.

Depth range via point_triangulator
-----------------------------------
The rig model has poses but no 3D points, so PatchMatch has no depth prior.
Rather than guessing from the baseline, we triangulate a REAL sparse cloud using
the fixed rig poses (point_triangulator never moves the poses):

  feature_extractor -> exhaustive_matcher -> point_triangulator

point_triangulator matches the input model to the database BY IMAGE NAME, and
the model's IMAGE_ID / CAMERA_ID must equal the IDs feature_extractor wrote into
the database. So we read the SQLite DB for the real IDs, then write the
triangulation input model with the fixed rig poses under those IDs. The clean
rig/sparse/ model (IDs 1..4) is still used as-is for image_undistorter.

Per-workspace MVS (COLMAP CLI):
  1. feature_extractor + exhaustive_matcher + point_triangulator
                         - triangulate REAL sparse points at the fixed rig poses.
                           PatchMatch/fusion need points in the model; an
                           empty-points model yields 0 fused points.
  2. image_undistorter   - undistort the 4 raw views + lay out a dense workspace
  3. patch_match_stereo  - dense depth/normal maps (CUDA GPU required)
  4. stereo_fusion       - fuse depth maps -> fused.ply
                           (min_num_pixels must be <= 4 for a 4-camera rig)

Depends on: rig/sparse/ (run rig.py first), intrinsics/, extrinsics/, camera.json

Run:
    python mvs.py --mode static
    python mvs.py --mode dynamic
    python mvs.py --mode dynamic --start_frame 1 --end_frame 5
    python mvs.py --mode static --depth_min 200 --depth_max 3000
    python mvs.py --mode dynamic --fresh
"""

import argparse
import os
import shutil
import sqlite3

import cv2
import numpy as np

from common import (
    POSITION_ORDER,
    IMAGE_EXTS,
    MVS_DIR,
    STATIC_DIR,
    FRAMES_DIR,
    STATIC_FUSED_PLY,
    RIG_SPARSE_DIR,
    load_camera_indices,
    load_intrinsics,
    load_poses,
    find_colmap,
    run,
    rot_to_quat,
    count_ply_points,
)


# --- Paths -------------------------------------------------------------------

STATIC_WORK_DIR = os.path.join(MVS_DIR, "static_work")     # dense workspace
TRIANGULATE_DIR = os.path.join(MVS_DIR, "_triangulate")    # depth-range estimate

# A fused 3D point needs >= this many consistent views. COLMAP's default is 5,
# but a 4-camera rig sees any point in at most 4 views, so the default always
# fuses 0 points. Must be <= 4 for this rig.
MIN_NUM_PIXELS = 2


# --- Helpers -----------------------------------------------------------------

def opencv_params(K, dist):
    """Return (fx, fy, cx, cy, k1, k2, p1, p2) for the OPENCV camera model."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    d = np.asarray(dist).reshape(-1)
    k1 = float(d[0]) if d.size > 0 else 0.0
    k2 = float(d[1]) if d.size > 1 else 0.0
    p1 = float(d[2]) if d.size > 2 else 0.0
    p2 = float(d[3]) if d.size > 3 else 0.0
    return fx, fy, cx, cy, k1, k2, p1, p2


def find_view_file(folder, position):
    """Locate {position}.<ext> inside a folder (any supported extension)."""
    for ext in IMAGE_EXTS:
        candidate = os.path.join(folder, f"{position}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def position_of(name):
    base = os.path.basename(name)
    for position in POSITION_ORDER:
        if base.startswith(position):
            return position
    return None


def discover_static():
    """Return {position: path} for the 4 views in mvs/static/, or raise."""
    if not os.path.isdir(STATIC_DIR):
        raise FileNotFoundError(
            f"{STATIC_DIR} does not exist. Capture a static frame-set first "
            "(static_scene.py) so it holds top_left.jpg, top_right.jpg, "
            "bot_left.jpg, bot_right.jpg."
        )
    views = {p: find_view_file(STATIC_DIR, p) for p in POSITION_ORDER}
    missing = [p for p, v in views.items() if v is None]
    if missing:
        raise RuntimeError(f"mvs/static/ is missing views {missing}.")
    return views


def discover_dynamic(start_frame, end_frame):
    """Return sorted [(num, folder, views)] for complete dynamic frame folders."""
    if not os.path.isdir(FRAMES_DIR):
        raise FileNotFoundError(
            f"{FRAMES_DIR} does not exist. Capture a sequence first "
            "(dynamic_scene.py), e.g. mvs/frames/000001/top_left.jpg (+ the "
            "other three positions)."
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
            f"No complete frame folders found in {FRAMES_DIR} for the "
            "requested range."
        )
    return frames


# --- Depth range via point_triangulator --------------------------------------

def _read_db_image_ids(database_path):
    """Return {name: (image_id, camera_id)} from a COLMAP SQLite database."""
    conn = sqlite3.connect(database_path)
    try:
        rows = conn.execute(
            "SELECT image_id, name, camera_id FROM images"
        ).fetchall()
    finally:
        conn.close()
    return {name: (int(image_id), int(camera_id))
            for image_id, name, camera_id in rows}


def _write_triangulation_model(model_dir, db_ids, intrinsics, poses,
                               image_size):
    """
    Write a COLMAP TXT model (cameras/images/empty points) whose IMAGE_ID and
    CAMERA_ID match the database, with the FIXED rig poses, ready for
    point_triangulator.
    """
    os.makedirs(model_dir, exist_ok=True)
    w, h = image_size

    # One camera per database image (one image per position here).
    with open(os.path.join(model_dir, "cameras.txt"), "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for name, (_img_id, cam_id) in db_ids.items():
            position = position_of(name)
            if position is None:
                continue
            fx, fy, cx, cy, k1, k2, p1, p2 = opencv_params(*intrinsics[position])
            f.write(f"{cam_id} OPENCV {w} {h} "
                    f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} "
                    f"{k1:.8f} {k2:.8f} {p1:.8f} {p2:.8f}\n")

    with open(os.path.join(model_dir, "images.txt"), "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for name, (img_id, cam_id) in db_ids.items():
            position = position_of(name)
            if position is None:
                continue
            R, t = poses[position]
            qw, qx, qy, qz = rot_to_quat(R)
            tx, ty, tz = (float(v) for v in np.asarray(t).reshape(3))
            f.write(f"{img_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{tx:.10f} {ty:.10f} {tz:.10f} {cam_id} "
                    f"{os.path.basename(name)}\n")
            f.write("\n")

    with open(os.path.join(model_dir, "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")


def _depth_range_from_points(points_txt, poses):
    """Project triangulated points into each camera; robust depth percentiles."""
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
    X = np.asarray(xyz).T  # 3 x N

    depths = []
    for position in POSITION_ORDER:
        R, t = poses[position]
        z = (R @ X + np.asarray(t).reshape(3, 1))[2]
        depths.append(z[z > 0])
    depths = np.concatenate(depths) if depths else np.array([])
    if depths.size == 0:
        return None
    dmin = float(np.percentile(depths, 1))
    dmax = float(np.percentile(depths, 99))
    span = max(dmax - dmin, 1e-6)
    dmin = max(dmin - 0.25 * span, 1e-4)
    dmax = dmax + 0.25 * span
    return dmin, dmax


def triangulate_sparse(colmap, images_dir, out_dir, intrinsics, poses, use_gpu):
    """
    Triangulate a REAL sparse cloud from the 4 views already present in
    images_dir, using the FIXED rig poses (point_triangulator never moves them).

    feature_extractor (per position, with that camera's calibrated OPENCV
    params) -> exhaustive_matcher -> point_triangulator. Intermediates are
    written under out_dir; returns the triangulated TXT model directory
    (out_dir/sparse). Raises on a hard COLMAP failure.
    """
    database = os.path.join(out_dir, "database.db")
    lists_dir = os.path.join(out_dir, "_lists")
    model_in = os.path.join(out_dir, "model_in")
    model_out = os.path.join(out_dir, "sparse")
    os.makedirs(lists_dir, exist_ok=True)
    os.makedirs(model_out, exist_ok=True)

    sample = None
    for position in POSITION_ORDER:
        path = os.path.join(images_dir, f"{position}.jpg")
        if os.path.exists(path):
            sample = cv2.imread(path)
            break
    if sample is None:
        raise RuntimeError(f"No {{position}}.jpg views found in {images_dir}.")
    image_size = (sample.shape[1], sample.shape[0])

    # One extractor call per position so each image gets its own OPENCV camera
    # with the calibrated parameters (keeps the database IDs clean for
    # point_triangulator's name-based matching).
    for position in POSITION_ORDER:
        list_path = os.path.join(lists_dir, f"{position}.txt")
        with open(list_path, "w") as f:
            f.write(f"{position}.jpg\n")
        fx, fy, cx, cy, k1, k2, p1, p2 = opencv_params(*intrinsics[position])
        run([
            colmap, "feature_extractor",
            "--database_path", database,
            "--image_path", images_dir,
            "--image_list_path", list_path,
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_params",
            f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2}",
            "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
        ])

    run([
        colmap, "exhaustive_matcher",
        "--database_path", database,
        "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
    ])

    db_ids = _read_db_image_ids(database)
    if len(db_ids) < len(POSITION_ORDER):
        print("  WARNING: not all 4 views registered in the database.")
    _write_triangulation_model(model_in, db_ids, intrinsics, poses, image_size)

    run([
        colmap, "point_triangulator",
        "--database_path", database,
        "--image_path", images_dir,
        "--input_path", model_in,
        "--output_path", model_out,
    ])
    run([
        colmap, "model_converter",
        "--input_path", model_out,
        "--output_path", model_out,
        "--output_type", "TXT",
    ])
    return model_out


def estimate_depth_range(colmap, ref_views, intrinsics, poses, use_gpu):
    """
    Triangulate a sparse cloud from the reference views using the fixed rig
    poses, then derive a robust [depth_min, depth_max]. Returns (dmin, dmax) or
    None if triangulation produced nothing.
    """
    print("\n=== Estimating depth range (point_triangulator) ===")
    if os.path.isdir(TRIANGULATE_DIR):
        shutil.rmtree(TRIANGULATE_DIR)
    images_dir = os.path.join(TRIANGULATE_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    for position in POSITION_ORDER:
        shutil.copy(ref_views[position],
                    os.path.join(images_dir, f"{position}.jpg"))

    try:
        model_out = triangulate_sparse(
            colmap, images_dir, TRIANGULATE_DIR, intrinsics, poses, use_gpu
        )
    except Exception as exc:
        print(f"  triangulation failed ({exc}); depth range -> auto.")
        return None

    rng = _depth_range_from_points(
        os.path.join(model_out, "points3D.txt"), poses
    )
    if rng is None:
        print("  No triangulated points; depth range -> auto.")
    else:
        print(f"  Depth range: {rng[0]:.2f} .. {rng[1]:.2f} mm")
    return rng


# --- Per-workspace MVS -------------------------------------------------------

def write_patchmatch_config(dense_dir):
    """Force every view to use the other three as stereo source images.

    image_undistorter writes a co-visibility-based config; with only 4 cameras
    we want each reference view to always pair with all the others.
    """
    cfg_path = os.path.join(dense_dir, "stereo", "patch-match.cfg")
    with open(cfg_path, "w") as f:
        for position in POSITION_ORDER:
            f.write(f"{position}.jpg\n")
            f.write("__all__\n")


def count_model_points(points_txt):
    """Count entries in a COLMAP points3D.txt (0 if missing)."""
    if not os.path.exists(points_txt):
        return 0
    n = 0
    with open(points_txt, "r") as f:
        for ln in f:
            if ln.strip() and not ln.startswith("#"):
                n += 1
    return n


def process_workspace(colmap, views, work_dir, fused_ply, intrinsics, poses,
                      max_image_size, depth_range, use_gpu):
    """Triangulate -> undistort -> patch_match -> fusion for one 4-view set."""
    images_dir = os.path.join(work_dir, "images")
    sparse_dir = os.path.join(work_dir, "sparse")   # filled by triangulation
    dense_dir = os.path.join(work_dir, "dense")

    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(images_dir, exist_ok=True)

    for position in POSITION_ORDER:
        shutil.copy(views[position],
                    os.path.join(images_dir, f"{position}.jpg"))

    # Triangulate REAL sparse points at the fixed rig poses. COLMAP's dense
    # stereo needs points in the model: PatchMatch derives stereo source images
    # and depth priors from sparse co-visibility, and an empty-points model
    # produces geometrically inconsistent depth maps that fusion discards
    # (0 points). This is the per-workspace fix proven out previously.
    triangulate_sparse(colmap, images_dir, work_dir, intrinsics, poses, use_gpu)
    n_sparse = count_model_points(os.path.join(sparse_dir, "points3D.txt"))
    print(f"  Triangulated sparse points: {n_sparse}")
    if n_sparse == 0:
        raise RuntimeError(
            "point_triangulator produced 0 points — the 4 views share too "
            "little matchable texture. Capture a more textured scene."
        )

    run([
        colmap, "image_undistorter",
        "--image_path", images_dir,
        "--input_path", sparse_dir,
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ])

    # 4-camera rig: force each view to use the other three as stereo sources.
    write_patchmatch_config(dense_dir)

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

    os.makedirs(os.path.dirname(fused_ply) or ".", exist_ok=True)
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
    parser.add_argument("--mode", required=True, choices=["static", "dynamic"],
                        help="static: mvs/static/  dynamic: mvs/frames/")
    parser.add_argument("--colmap", default=None,
                        help="Path to the colmap binary (default: PATH).")
    parser.add_argument("--start_frame", type=int, default=None,
                        help="(dynamic) first frame number, inclusive.")
    parser.add_argument("--end_frame", type=int, default=None,
                        help="(dynamic) last frame number, inclusive.")
    parser.add_argument("--max_image_size", type=int, default=640,
                        help="Max image dimension for PatchMatch (default 640).")
    parser.add_argument("--depth_min", type=float, default=None,
                        help="Override auto depth_min (mm).")
    parser.add_argument("--depth_max", type=float, default=None,
                        help="Override auto depth_max (mm).")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable CUDA. NOTE: patch_match_stereo needs a "
                             "GPU and will almost certainly fail without CUDA.")
    parser.add_argument("--fresh", action="store_true",
                        help="(dynamic) delete existing mvs/frame_* workspaces "
                             "first.")
    args = parser.parse_args()

    colmap = find_colmap(args.colmap)
    cameras = load_camera_indices()
    intrinsics = {p: load_intrinsics(idx) for p, idx in cameras}
    poses = load_poses()

    if not os.path.isdir(RIG_SPARSE_DIR):
        raise FileNotFoundError(
            f"{RIG_SPARSE_DIR} not found. Run rig.py (Stage 3) first."
        )

    print(f"Stage 4 — MVS (dense), mode = {args.mode}")
    print(f"  COLMAP: {colmap}")
    use_gpu = not args.no_gpu
    if not use_gpu:
        print("  WARNING: --no-gpu set; PatchMatch needs CUDA and will likely "
              "fail.")

    # Build the job list and pick reference views for depth estimation.
    os.makedirs(MVS_DIR, exist_ok=True)
    jobs = []  # (label, views, work_dir, fused_ply)
    if args.mode == "static":
        views = discover_static()
        jobs.append(("static", views, STATIC_WORK_DIR, STATIC_FUSED_PLY))
        ref_views = views
    else:
        if args.fresh:
            for name in os.listdir(MVS_DIR):
                if name.startswith("frame_"):
                    shutil.rmtree(os.path.join(MVS_DIR, name))
            print("  Cleared previous frame_* workspaces.")
        print("Discovering dynamic frames...")
        frames = discover_dynamic(args.start_frame, args.end_frame)
        print(f"  {len(frames)} frame(s) to process.")
        for num, _folder, views in frames:
            frame_dir = os.path.join(MVS_DIR, f"frame_{num:06d}")
            fused = os.path.join(frame_dir, "fused.ply")
            jobs.append((f"frame_{num:06d}", views, frame_dir, fused))
        ref_views = frames[0][2]

    # Depth range: explicit override, else triangulate once and reuse.
    if args.depth_min is not None and args.depth_max is not None:
        depth_range = (args.depth_min, args.depth_max)
        print(f"  Depth range (override): {depth_range[0]:.2f} .. "
              f"{depth_range[1]:.2f} mm")
    else:
        depth_range = estimate_depth_range(
            colmap, ref_views, intrinsics, poses, use_gpu
        )
        if args.depth_min is not None or args.depth_max is not None:
            print("  NOTE: pass BOTH --depth_min and --depth_max to override; "
                  "using auto estimate.")

    # Process each workspace.
    point_counts = []
    failures = []
    for idx, (label, views, work_dir, fused_ply) in enumerate(jobs, start=1):
        print(f"\n=== {label} ({idx}/{len(jobs)}) ===")
        try:
            n = process_workspace(colmap, views, work_dir, fused_ply,
                                  intrinsics, poses, args.max_image_size,
                                  depth_range, use_gpu)
            print(f"  -> {os.path.relpath(fused_ply, MVS_DIR)}: {n} points")
            point_counts.append(n)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append(label)

    print("\n=== MVS summary ===")
    print(f"  Workspaces processed: {len(point_counts)}/{len(jobs)}")
    if point_counts:
        print(f"  Avg points per cloud: {int(np.mean(point_counts)):,}")
        print(f"  Min / max points: {min(point_counts):,} / "
              f"{max(point_counts):,}")
    if failures:
        print(f"  FAILED: {failures}")
    if args.mode == "static":
        print(f"\n  Dense cloud: {STATIC_FUSED_PLY}")
    else:
        print(f"\n  Dense clouds: {MVS_DIR}\\frame_XXXXXX\\fused.ply")


if __name__ == "__main__":
    main()
