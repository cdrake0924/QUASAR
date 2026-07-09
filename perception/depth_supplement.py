"""
Stage 4b - Depth Anything V2 point cloud supplement (Track A / static only).

COLMAP MVS on a 4-camera rig produces a sparse cloud (a few hundred to a few
thousand points): PatchMatch only keeps surfels that agree across views, so
single-view surfaces, textureless regions and depth discontinuities all leave
holes. NPBG++ rasterises directly from that cloud, so a hole renders blank.

This script fills those holes. For each of the 4 rectified static views it runs
Depth Anything V2 (a monocular depth estimator), aligns the *relative* monocular
depth to the metric MVS cloud with a per-view scale+shift (least squares over the
MVS points that project into that view), back-projects every pixel into world
space using the calibrated intrinsics/extrinsics, then merges the dense clouds
with the original MVS cloud (voxel downsample + statistical outlier removal via
Open3D) and overwrites ``mvs/static_fused.ply``.

The MVS points anchor the scale; the Depth Anything points fill the coverage.
Both NPBG++ and 3DGS read ``mvs/static_fused.ply``, so they pick up the denser
cloud automatically. Run AFTER ``python mvs.py --mode static`` and BEFORE
``npbgpp.py`` / ``gs3d.py``.

Depends on: mvs/static_fused.ply, mvs/static/{position}.jpg, intrinsics/K_*.txt,
intrinsics/dist_*.txt, extrinsics/poses.npz, camera.json.
"""

import argparse
import os
import sys

import cv2
import numpy as np

from common import (
    POSITION_ORDER,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    HERE,
    STATIC_DIR,
    STATIC_FUSED_PLY,
    load_camera_indices,
    load_intrinsics,
    load_poses,
)

PROJECT_ROOT = os.path.dirname(HERE)
DA_REPO = os.path.join(PROJECT_ROOT, "Depth-Anything-V2")
DEFAULT_CKPT = os.path.join(
    DA_REPO, "checkpoints", "depth_anything_v2_metric_indoor_vitl.pth"
)

# Encoder feature configs from the Depth-Anything-V2 repo.
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

_SETUP_HINT = (
    "Depth Anything V2 is not set up. From the project root run:\n"
    "  cd {root}\n"
    "  git clone https://github.com/DepthAnything/Depth-Anything-V2\n"
    "  cd Depth-Anything-V2\n"
    "  pip install -r requirements.txt\n"
    "  wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large"
    "/resolve/main/depth_anything_v2_metric_indoor_vitl.pth \\\n"
    "    -O checkpoints/depth_anything_v2_metric_indoor_vitl.pth"
)


# --- Depth Anything V2 model -------------------------------------------------

def load_depth_model(model_path, encoder, max_depth):
    """
    Load the Depth Anything V2 *metric* model from a checkpoint.

    The metric indoor checkpoint uses the ``metric_depth`` variant of the repo
    (its DPT head takes a ``max_depth``), so we add that folder to sys.path.
    Returns ``(model, device)``; call ``model.infer_image(bgr)`` to get a
    float32 (H, W) depth map.
    """
    metric_dir = os.path.join(DA_REPO, "metric_depth")
    if not os.path.isdir(metric_dir):
        raise FileNotFoundError(_SETUP_HINT.format(root=PROJECT_ROOT))
    if metric_dir not in sys.path:
        sys.path.insert(0, metric_dir)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - env issue
        raise ImportError(
            "PyTorch is required for depth_supplement.py. Install with "
            "`pip install torch torchvision`."
        ) from exc

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as exc:
        raise ImportError(
            "Could not import depth_anything_v2 from "
            f"{metric_dir}.\n" + _SETUP_HINT.format(root=PROJECT_ROOT)
        ) from exc

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            + _SETUP_HINT.format(root=PROJECT_ROOT)
        )
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder '{encoder}' (choose from {list(MODEL_CONFIGS)}).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(**{**MODEL_CONFIGS[encoder], "max_depth": max_depth})
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"  Depth Anything V2 ({encoder}, max_depth={max_depth}m) on {device}")
    return model, device


# --- Geometry ----------------------------------------------------------------

def project_points(points_world, R, t, K):
    """
    Project world points into a WORLD-TO-CAMERA view.

    Returns (u, v, z) pixel columns/rows and camera-space depth (same units as
    the world points, i.e. mm). Points behind the camera have z <= 0.
    """
    Xc = points_world @ R.T + t  # (N, 3) camera space
    z = Xc[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = fx * Xc[:, 0] / safe_z + cx
    v = fy * Xc[:, 1] / safe_z + cy
    return u, v, z


def backproject(depth, mask, K, R, t):
    """
    Back-project the masked pixels of a metric depth map into world space.

    ``depth`` is (H, W) in mm, ``mask`` is a boolean (H, W) of pixels to keep.
    Returns (M, 3) world points. World-to-camera pose is (R, t), so the inverse
    is ``X_world = (X_cam - t) @ R``.
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    u = uu[mask].astype(np.float64)
    v = vv[mask].astype(np.float64)
    d = depth[mask].astype(np.float64)
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    Xc = np.stack([x, y, d], axis=1)
    return (Xc - t) @ R


# --- Per-view depth supplement -----------------------------------------------

def supplement_view(position, K, dist, R, t, mvs_pts, model, low_pct, high_pct):
    """
    Run Depth Anything on one view, align it to MVS, and back-project.

    Returns (world_points (M,3), colors_rgb (M,3) float in [0,1]) or None if the
    view can't be aligned (too few MVS anchors).
    """
    img_path = os.path.join(STATIC_DIR, f"{position}.jpg")
    if not os.path.exists(img_path):
        # tolerate .png fallback
        alt = os.path.join(STATIC_DIR, f"{position}.png")
        if os.path.exists(alt):
            img_path = alt
        else:
            print(f"  [{position}] image not found ({img_path}) - skipping")
            return None

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [{position}] could not read image - skipping")
        return None
    if (img.shape[1], img.shape[0]) != (FRAME_WIDTH, FRAME_HEIGHT):
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

    # Undistort with the calibrated K/dist; K stays the new camera matrix, so
    # the pinhole model used for back-projection is exactly K.
    und = cv2.undistort(img, K, dist, None, K)

    depth_da = np.asarray(model.infer_image(und), dtype=np.float64)
    if depth_da.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        depth_da = cv2.resize(depth_da, (FRAME_WIDTH, FRAME_HEIGHT))
    H, W = depth_da.shape

    # --- alignment: project MVS points, sample DA depth, lstsq scale+shift ---
    u, v, z = project_points(mvs_pts, R, t, K)
    inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inside.sum() < 8:
        print(f"  [{position}] only {int(inside.sum())} MVS anchors in view - skipping")
        return None

    ui = np.clip(np.round(u[inside]).astype(int), 0, W - 1)
    vi = np.clip(np.round(v[inside]).astype(int), 0, H - 1)
    da_vals = depth_da[vi, ui]
    mvs_vals = z[inside]

    # Drop DA extremes (sky / foreground pops) from the fit.
    lo, hi = np.percentile(depth_da, [low_pct, 100.0 - high_pct])
    keep = (da_vals > lo) & (da_vals < hi)
    if keep.sum() >= 8:
        da_vals, mvs_vals = da_vals[keep], mvs_vals[keep]

    A = np.stack([da_vals, np.ones_like(da_vals)], axis=1)
    (s, b), *_ = np.linalg.lstsq(A, mvs_vals, rcond=None)
    resid = A @ np.array([s, b]) - mvs_vals
    rms = float(np.sqrt(np.mean(resid ** 2)))
    mean_depth = float(np.mean(mvs_vals))
    flag = "  <-- HIGH, MVS coverage in this view is thin" if rms > 0.2 * mean_depth else ""
    print(f"  [{position}] anchors={len(mvs_vals):5d}  s={s:.4g}  t={b:.1f}mm  "
          f"RMS={rms:.1f}mm ({100.0 * rms / mean_depth:.0f}% of mean depth){flag}")

    aligned = s * depth_da + b

    # Back-project valid pixels (positive depth, non-extreme DA values).
    valid = (aligned > 0) & (depth_da > lo) & (depth_da < hi)
    pts = backproject(aligned, valid, K, R, t)
    rgb = cv2.cvtColor(und, cv2.COLOR_BGR2RGB).astype(np.float64)[valid] / 255.0
    return pts, rgb


# --- Merge / IO --------------------------------------------------------------

def _import_open3d():
    try:
        import open3d as o3d  # noqa: F401
        return o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D is required for the merge step. Install with `pip install open3d`."
        ) from exc


def run(model_path, encoder, max_depth, voxel_size, outlier_neighbors,
        outlier_std, low_pct, high_pct, overwrite):
    o3d = _import_open3d()

    if not os.path.exists(STATIC_FUSED_PLY):
        raise FileNotFoundError(
            f"{STATIC_FUSED_PLY} not found. Run `python mvs.py --mode static` first."
        )

    print("Loading MVS cloud...")
    pcd_mvs = o3d.io.read_point_cloud(STATIC_FUSED_PLY)
    mvs_pts = np.asarray(pcd_mvs.points, dtype=np.float64)
    n_mvs = len(mvs_pts)
    if n_mvs == 0:
        raise ValueError(f"{STATIC_FUSED_PLY} has 0 points - nothing to anchor to.")
    if not pcd_mvs.has_colors():
        pcd_mvs.paint_uniform_color([0.6, 0.6, 0.6])
    print(f"  MVS-only points: {n_mvs}")

    print("Loading intrinsics / extrinsics...")
    poses = load_poses()
    cams = load_camera_indices()
    idx_by_pos = {pos: idx for pos, idx in cams}

    print("Loading Depth Anything V2...")
    model, _ = load_depth_model(model_path, encoder, max_depth)

    print("Supplementing views:")
    all_pts, all_rgb = [], []
    for position in POSITION_ORDER:
        if position not in idx_by_pos:
            print(f"  [{position}] not in camera.json - skipping")
            continue
        K, dist = load_intrinsics(idx_by_pos[position])
        R, t = poses[position]
        out = supplement_view(position, K, dist, R, t, mvs_pts, model,
                               low_pct, high_pct)
        if out is not None:
            pts, rgb = out
            all_pts.append(pts)
            all_rgb.append(rgb)

    if not all_pts:
        raise RuntimeError(
            "No views could be supplemented. Check that mvs/static/*.jpg exist "
            "and that the MVS cloud projects into the calibrated views."
        )

    da_pts = np.concatenate(all_pts, axis=0)
    da_rgb = np.concatenate(all_rgb, axis=0)
    print(f"  Depth Anything points (raw): {len(da_pts)}")

    pcd_da = o3d.geometry.PointCloud()
    pcd_da.points = o3d.utility.Vector3dVector(da_pts)
    pcd_da.colors = o3d.utility.Vector3dVector(np.clip(da_rgb, 0.0, 1.0))

    print("Merging + cleaning...")
    merged = pcd_mvs + pcd_da
    before = len(merged.points)
    merged = merged.voxel_down_sample(voxel_size)
    after_voxel = len(merged.points)
    merged, _ = merged.remove_statistical_outlier(
        nb_neighbors=outlier_neighbors, std_ratio=outlier_std)
    final = len(merged.points)

    out_path = STATIC_FUSED_PLY
    if not overwrite:
        out_path = os.path.join(
            os.path.dirname(STATIC_FUSED_PLY), "static_fused_supplemented.ply")
    o3d.io.write_point_cloud(out_path, merged)

    print("\n--- Summary ------------------------------------------------------")
    print(f"  MVS-only:                 {n_mvs}")
    print(f"  + Depth Anything:         {len(da_pts)}")
    print(f"  combined (pre-clean):     {before}")
    print(f"  after voxel {voxel_size:.1f}mm:      {after_voxel}")
    print(f"  after outlier removal:    {final}")
    print(f"  written -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4b: densify the MVS static cloud with Depth Anything V2.")
    parser.add_argument("--model_path", default=DEFAULT_CKPT,
                        help="Path to the Depth Anything V2 checkpoint "
                             f"(default: {DEFAULT_CKPT}).")
    parser.add_argument("--encoder", default="vitl", choices=list(MODEL_CONFIGS),
                        help="Depth Anything V2 encoder (default vitl).")
    parser.add_argument("--max_depth", type=float, default=20.0,
                        help="Metric model max depth in meters (default 20, indoor).")
    parser.add_argument("--voxel_size", type=float, default=5.0,
                        help="Voxel downsample size in mm (default 5; lower = denser).")
    parser.add_argument("--outlier_neighbors", type=int, default=20,
                        help="Statistical outlier removal neighbor count (default 20).")
    parser.add_argument("--outlier_std", type=float, default=2.0,
                        help="Outlier std ratio (default 2.0; lower = more aggressive).")
    parser.add_argument("--extreme_pct", type=float, default=2.0,
                        help="Percent of DA depth extremes to drop at each end "
                             "(default 2.0 => keep the 2..98 percentile band).")
    parser.add_argument("--no_overwrite", action="store_true",
                        help="Write mvs/static_fused_supplemented.ply instead of "
                             "overwriting mvs/static_fused.ply.")
    args = parser.parse_args()

    run(
        model_path=args.model_path,
        encoder=args.encoder,
        max_depth=args.max_depth,
        voxel_size=args.voxel_size,
        outlier_neighbors=args.outlier_neighbors,
        outlier_std=args.outlier_std,
        low_pct=args.extreme_pct,
        high_pct=args.extreme_pct,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
