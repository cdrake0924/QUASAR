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
import warnings

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

# Pristine MVS cloud is preserved here on first run, so re-running the supplement
# aligns against the sparse MVS anchors (never against a previous dense output).
# mvs.py deletes this when it regenerates static_fused.ply.
MVS_ONLY_PLY = os.path.splitext(STATIC_FUSED_PLY)[0] + "_mvsonly.ply"

PROJECT_ROOT = os.path.dirname(HERE)
DA_REPO = os.path.join(PROJECT_ROOT, "Depth-Anything-V2")
DEFAULT_CKPT = os.path.join(
    DA_REPO, "checkpoints", "depth_anything_v2_metric_hypersim_vitl.pth"
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
    "  wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"
    "/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth \\\n"
    "    -O checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"
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


def robust_fit(da, mvs, iters=5, k=2.5):
    """
    Iteratively reweighted least-squares line fit ``mvs ~ s*da + t``.

    Plain lstsq is wrecked by the handful of MVS outliers a 4-camera fused cloud
    always carries — with only a few hundred sparse anchors those outliers can
    flip the slope sign. Each iteration drops residual outliers by MAD and
    re-fits. Returns ``(s, t, inlier_mask)``.
    """
    mask = np.ones(len(da), dtype=bool)
    s, b = 0.0, float(np.median(mvs))
    for _ in range(iters):
        if mask.sum() < 8:
            break
        A = np.stack([da[mask], np.ones(int(mask.sum()))], axis=1)
        (s, b), *_ = np.linalg.lstsq(A, mvs[mask], rcond=None)
        resid = (s * da + b) - mvs
        med = np.median(resid[mask])
        mad = np.median(np.abs(resid[mask] - med)) * 1.4826
        if mad < 1e-6:
            break
        newmask = np.abs(resid - med) < k * mad
        if int(newmask.sum()) < 8 or np.array_equal(newmask, mask):
            break
        mask = newmask
    return s, b, mask


# --- Per-view depth supplement -----------------------------------------------

def supplement_view(position, K, dist, R, t, mvs_pts, model, low_pct, high_pct,
                    min_corr):
    """
    Run Depth Anything on one view, scale to mm (MVS-refined when reliable),
    and back-project.

    Returns a dict with the world points + colors and the geometry needed for
    cross-view consistency filtering:
      pos, pts (M,3), rgb (M,3) in [0,1], depth (H,W mm / NaN), K, R, t.
    Returns None only if the view's image is missing/unreadable.
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

    # DepthAnythingV2 metric (Hypersim/indoor) returns depth in METERS. Our
    # world/MVS frame is in mm, so the baseline conversion is simply x1000 -
    # the metric model already carries scale, we don't need MVS to establish it.
    depth_da = np.asarray(model.infer_image(und), dtype=np.float64)
    if depth_da.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        depth_da = cv2.resize(depth_da, (FRAME_WIDTH, FRAME_HEIGHT))
    H, W = depth_da.shape
    da_mm = depth_da * 1000.0

    # --- optional MVS refinement: fit mm ~ s*da_mm + t on in-view anchors -----
    # s should be ~1 if the metric prediction is accurate. We only *apply* the
    # fit when it is trustworthy (positive, sane scale, real correlation);
    # otherwise we fall back to the raw metric depth so the view still
    # contributes dense geometry instead of being dropped.
    lo, hi = np.percentile(depth_da, [low_pct, 100.0 - high_pct])
    u, v, z = project_points(mvs_pts, R, t, K)
    inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    s, b, reliable, r = 1.0, 0.0, False, 0.0
    if int(inside.sum()) >= 8:
        ui = np.clip(np.round(u[inside]).astype(int), 0, W - 1)
        vi = np.clip(np.round(v[inside]).astype(int), 0, H - 1)
        da_anchor = depth_da[vi, ui] * 1000.0
        mvs_vals = z[inside]
        keep = (depth_da[vi, ui] > lo) & (depth_da[vi, ui] < hi)
        if int(keep.sum()) >= 8:
            da_anchor, mvs_vals = da_anchor[keep], mvs_vals[keep]

        s_fit, b_fit, inl = robust_fit(da_anchor, mvs_vals)
        da_in, mvs_in = da_anchor[inl], mvs_vals[inl]
        if len(mvs_in) > 2:
            resid = (s_fit * da_in + b_fit) - mvs_in
            rms = float(np.sqrt(np.mean(resid ** 2)))
            mean_depth = float(np.mean(mvs_in))
            r = float(np.corrcoef(da_in, mvs_in)[0, 1])
            reliable = (0.2 < s_fit < 5.0) and (r >= min_corr)
            note = "MVS-refined" if reliable else "raw metric (fit unreliable)"
            print(f"  [{position}] anchors={len(mvs_in):5d}  s={s_fit:.3f}  "
                  f"t={b_fit:.1f}mm  r={r:.2f}  RMS={rms:.0f}mm "
                  f"({100.0 * rms / mean_depth:.0f}%)  -> {note}")
            if reliable:
                s, b = s_fit, b_fit
    else:
        print(f"  [{position}] {int(inside.sum())} anchors in view -> raw metric")

    aligned = s * da_mm + b

    # Back-project valid pixels (positive depth, non-extreme DA values).
    valid = (aligned > 0) & (depth_da > lo) & (depth_da < hi)
    image_rgb = cv2.cvtColor(und, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    pts = backproject(aligned, valid, K, R, t)
    rgb = image_rgb[valid]
    # Depth map (mm where valid, NaN elsewhere) so a reprojected point can't be
    # "confirmed"/fused against a masked or extreme pixel.
    depth_test = np.where(valid, aligned, np.nan)
    return {"pos": position, "pts": pts, "rgb": rgb, "depth": depth_test,
            "image_rgb": image_rgb, "K": K, "R": R, "t": t}


# --- Cross-view consistency --------------------------------------------------

def cross_view_filter(views, min_views, rel_tol, abs_tol):
    """
    Keep only back-projected points that ≥ min_views cameras agree on.

    Monocular depth is not multi-view consistent: each view places the same
    surface at a slightly different depth, so back-projecting all views stacks
    N offset copies of every surface (the "ghosting"). This is COLMAP-style
    geometric consistency applied to the DA clouds: each point (from its source
    view) is reprojected into every other view and confirmed if the point's
    depth there matches that view's own depth map within
    ``max(abs_tol, rel_tol * depth)``. A point survives if the number of
    agreeing views (source + confirmers) is >= min_views.

    Occlusion is tolerated: a point that lands *behind* another view's surface
    simply isn't counted by that view (not rejected). Ghost copies, which only
    their own source view believes in, fail to reach min_views and are dropped.
    """
    cams = [(v["K"], v["R"], v["t"], v["depth"]) for v in views]
    kept_pts, kept_rgb = [], []
    total_in = total_out = 0

    for i, v in enumerate(views):
        P = v["pts"]
        if len(P) == 0:
            continue
        agree = np.ones(len(P), dtype=np.int32)  # source view counts as 1
        for j, (Kj, Rj, tj, Dj) in enumerate(cams):
            if j == i:
                continue
            Hj, Wj = Dj.shape
            Xc = P @ Rj.T + tj
            zj = Xc[:, 2]
            fx, fy, cx, cy = Kj[0, 0], Kj[1, 1], Kj[0, 2], Kj[1, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = fx * Xc[:, 0] / zj + cx
                v_ = fy * Xc[:, 1] / zj + cy
            inside = (zj > 0) & (u >= 0) & (u < Wj) & (v_ >= 0) & (v_ < Hj)
            if not inside.any():
                continue
            idx = np.where(inside)[0]
            ui = np.clip(u[idx].astype(np.int32), 0, Wj - 1)
            vi = np.clip(v_[idx].astype(np.int32), 0, Hj - 1)
            dj = Dj[vi, ui]                       # NaN where view j was masked
            tol = np.maximum(abs_tol, rel_tol * dj)
            close = np.abs(zj[idx] - dj) <= tol   # NaN comparisons -> False
            agree[idx[close]] += 1

        keep = agree >= min_views
        total_in += len(P)
        total_out += int(keep.sum())
        kept_pts.append(P[keep])
        kept_rgb.append(v["rgb"][keep])
        print(f"  [{v['pos']}] kept {int(keep.sum())}/{len(P)} "
              f"({100.0 * keep.sum() / len(P):.0f}%) after consistency")

    pts = np.concatenate(kept_pts, axis=0) if kept_pts else np.zeros((0, 3))
    rgb = np.concatenate(kept_rgb, axis=0) if kept_rgb else np.zeros((0, 3))
    print(f"  Consistency filter: {total_out}/{total_in} points kept "
          f"(min_views={min_views}, rel_tol={rel_tol}, abs_tol={abs_tol}mm)")
    return pts, rgb


# --- Per-pixel cross-view depth fusion ---------------------------------------

def fuse_depth(views, min_views, rel_tol, abs_tol):
    """
    Fuse the per-view depth maps into one consensus surface per reference frame.

    The consistency *filter* only culls disagreeing points, so surviving points
    still come from N independently-scaled depth maps and form a thick (blurry),
    slightly-doubled shell. Fusion instead rebuilds a single depth per reference
    pixel: every view is rendered into the reference frame (nearest-z per pixel),
    and each pixel's fused depth is the mean of the views that agree with the
    per-pixel median within ``max(abs_tol, rel_tol * median)``. A pixel is kept
    only if >= min_views agree. Because every reference frame collapses to the
    same cross-view consensus, the frames coincide (voxel-merge to one surface)
    instead of stacking offset copies — killing both ghosting and shell blur,
    while holes are filled by whichever views cover them.
    """
    V = len(views)
    world = [v["pts"] for v in views]
    out_pts, out_rgb = [], []
    total_out = 0

    for i, ref in enumerate(views):
        Ki, Ri, ti = ref["K"], ref["R"], ref["t"]
        fx, fy, cx, cy = Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2]
        H, W = ref["depth"].shape

        # Render every view's cloud into this reference frame, nearest-z / pixel.
        stack = np.full((V, H, W), np.nan)
        for j in range(V):
            Xc = world[j] @ Ri.T + ti
            z = Xc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = fx * Xc[:, 0] / z + cx
                vv = fy * Xc[:, 1] / z + cy
            m = (z > 0) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
            if not m.any():
                continue
            ui = u[m].astype(np.int32)
            vi = vv[m].astype(np.int32)
            buf = np.full((H, W), np.inf)
            np.minimum.at(buf, (vi, ui), z[m])   # nearest surface occludes
            buf[np.isinf(buf)] = np.nan
            stack[j] = buf

        # Per-pixel consensus: median, then mean of views within tolerance.
        # Empty (no-data) pixels raise All-NaN warnings that are expected here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(stack, axis=0)             # (H, W)
            tol = np.maximum(abs_tol, rel_tol * med)
            agree = np.abs(stack - med) <= tol            # NaN -> False
            count = agree.sum(axis=0)
            fused = np.nanmean(np.where(agree, stack, np.nan), axis=0)
        keep = (count >= min_views) & np.isfinite(fused)

        ys, xs = np.where(keep)
        if len(ys) == 0:
            print(f"  [{ref['pos']}] fused 0 pixels")
            continue
        d = fused[ys, xs]
        Xc = np.stack([(xs - cx) * d / fx, (ys - cy) * d / fy, d], axis=1)
        Xw = (Xc - ti) @ Ri
        out_pts.append(Xw)
        out_rgb.append(ref["image_rgb"][ys, xs])
        total_out += len(ys)
        print(f"  [{ref['pos']}] fused {len(ys)} pixels "
              f"({100.0 * len(ys) / (H * W):.0f}% of frame)")

    pts = np.concatenate(out_pts, axis=0) if out_pts else np.zeros((0, 3))
    rgb = np.concatenate(out_rgb, axis=0) if out_rgb else np.zeros((0, 3))
    print(f"  Depth fusion: {total_out} consensus points "
          f"(min_views={min_views}, rel_tol={rel_tol}, abs_tol={abs_tol}mm)")
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
        outlier_std, low_pct, high_pct, min_corr, method, min_views,
        rel_tol, abs_tol, overwrite):
    o3d = _import_open3d()

    # Align against the pristine MVS cloud. If a preserved copy exists (from a
    # previous supplement run) use it, so re-runs never compound onto a dense
    # output. Otherwise the current static_fused.ply IS the MVS cloud.
    base_ply = MVS_ONLY_PLY if os.path.exists(MVS_ONLY_PLY) else STATIC_FUSED_PLY
    if not os.path.exists(base_ply):
        raise FileNotFoundError(
            f"{STATIC_FUSED_PLY} not found. Run `python mvs.py --mode static` first."
        )

    print("Loading MVS cloud...")
    pcd_mvs = o3d.io.read_point_cloud(base_ply)
    mvs_pts = np.asarray(pcd_mvs.points, dtype=np.float64)
    n_mvs = len(mvs_pts)
    if n_mvs == 0:
        raise ValueError(f"{base_ply} has 0 points - nothing to anchor to.")
    if not pcd_mvs.has_colors():
        pcd_mvs.paint_uniform_color([0.6, 0.6, 0.6])
    if base_ply == MVS_ONLY_PLY:
        print(f"  MVS-only points: {n_mvs}  (from preserved "
              f"{os.path.basename(MVS_ONLY_PLY)})")
    else:
        print(f"  MVS-only points: {n_mvs}")
        o3d.io.write_point_cloud(MVS_ONLY_PLY, pcd_mvs)
        print(f"  preserved MVS-only base -> {os.path.basename(MVS_ONLY_PLY)}")

    print("Loading intrinsics / extrinsics...")
    poses = load_poses()
    cams = load_camera_indices()
    idx_by_pos = {pos: idx for pos, idx in cams}

    print("Loading Depth Anything V2...")
    model, _ = load_depth_model(model_path, encoder, max_depth)

    print("Supplementing views:")
    views = []
    for position in POSITION_ORDER:
        if position not in idx_by_pos:
            print(f"  [{position}] not in camera.json - skipping")
            continue
        K, dist = load_intrinsics(idx_by_pos[position])
        R, t = poses[position]
        out = supplement_view(position, K, dist, R, t, mvs_pts, model,
                               low_pct, high_pct, min_corr)
        if out is not None:
            views.append(out)

    if not views:
        raise RuntimeError(
            "No views could be supplemented. Check that mvs/static/*.jpg exist "
            "and that the MVS cloud projects into the calibrated views."
        )

    raw_da = sum(len(v["pts"]) for v in views)
    print(f"  Depth Anything points (raw): {raw_da}")

    if method == "fusion" and len(views) >= 2:
        print("Per-pixel cross-view depth fusion:")
        da_pts, da_rgb = fuse_depth(views, min_views, rel_tol, abs_tol)
        if len(da_pts) == 0:
            raise RuntimeError(
                "Depth fusion produced no consensus points. Loosen "
                "--consistency_rel/--consistency_abs or lower --min_views.")
    elif method == "filter" and len(views) >= 2:
        print("Cross-view consistency filtering:")
        da_pts, da_rgb = cross_view_filter(views, min_views, rel_tol, abs_tol)
        if len(da_pts) == 0:
            raise RuntimeError(
                "Consistency filter removed all points. Loosen --consistency_rel/"
                "--consistency_abs or lower --min_views (or use --method none).")
    else:
        da_pts = np.concatenate([v["pts"] for v in views], axis=0)
        da_rgb = np.concatenate([v["rgb"] for v in views], axis=0)

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
    print(f"  Depth Anything (raw):     {raw_da}")
    print(f"  + DA ({method}):{' ' * max(1, 18 - len(method))}{len(da_pts)}")
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
    parser.add_argument("--min_corr", type=float, default=0.3,
                        help="Minimum DA-vs-MVS anchor correlation to accept a "
                             "view's alignment (default 0.3; below this, or a "
                             "non-positive scale, the view falls back to raw "
                             "metric depth).")
    parser.add_argument("--method", choices=["fusion", "filter", "none"],
                        default="fusion",
                        help="Cross-view handling: 'fusion' (default) rebuilds "
                             "one consensus depth per reference pixel (kills "
                             "ghosting + shell blur); 'filter' only culls "
                             "disagreeing points; 'none' keeps all raw points.")
    parser.add_argument("--min_views", type=int, default=2,
                        help="Min cameras that must agree on a point/pixel to keep "
                             "it (default 2; higher = fewer ghosts, sparser).")
    parser.add_argument("--consistency_rel", type=float, default=0.05,
                        help="Relative depth tolerance for cross-view agreement "
                             "(default 0.05 = 5%% of depth).")
    parser.add_argument("--consistency_abs", type=float, default=20.0,
                        help="Absolute depth tolerance floor in mm for cross-view "
                             "agreement (default 20mm).")
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
        min_corr=args.min_corr,
        method=args.method,
        min_views=args.min_views,
        rel_tol=args.consistency_rel,
        abs_tol=args.consistency_abs,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
