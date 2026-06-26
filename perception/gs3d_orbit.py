"""
gs3d_orbit.py — render a novel camera path through a trained 3DGS scene.

This runs INSIDE the gaussian-splatting repo's environment (the `gs` conda env
with torch + diff_gaussian_rasterization). It is launched by
`gs3d.py --mode orbit`, which passes the gs-env python via --python; you normally
don't call this directly.

Why this exists: the vanilla repo's render.py only re-renders the cameras in the
dataset (your 4 captured views). To actually evaluate novel-view synthesis you
need viewpoints the model never trained on. This loads the trained Gaussians,
builds a smooth elliptical path that loops around the centroid of the 4 capture
cameras (giving parallax beyond the captured cone), and rasterizes each frame
with the repo's own renderer — so the output is directly comparable, frame for
frame, against an equivalent NPBG++ path.

The novel cameras are MiniCam objects (image-free): render() only needs the
intrinsics + the view/projection transforms, which we build with the repo's own
getWorld2View2 / getProjectionMatrix so the convention matches training exactly.
"""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _look_at(eye, target, up_world):
    """Return (R_c2w, T) in COLMAP convention (x-right, y-down, z-forward)."""
    f = _normalize(target - eye)          # forward = camera +z
    r = _normalize(np.cross(f, up_world))  # right = camera +x
    d = _normalize(np.cross(f, r))         # down  = camera +y
    R_w2c = np.stack([r, d, f], axis=0)    # rows map world -> camera
    R_c2w = R_w2c.T
    T = -R_w2c @ eye                       # tvec (world -> camera translation)
    return R_c2w, T


def main():
    parser = ArgumentParser(description="Render a novel orbit path of a 3DGS scene.")
    # These mirror the repo's expected args; values are merged from the model's
    # saved cfg_args by get_combined_args (so -s/source_path comes for free).
    parser.add_argument("--gs_repo", required=True, help="Path to gaussian-splatting repo.")
    parser.add_argument("--out", required=True, help="Output directory for frames.")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--amp_scale", type=float, default=1.5,
                        help="Orbit radius as a multiple of the capture-camera "
                             "half-spread (1.0 ~ stay within the rig).")
    args, _ = parser.parse_known_args()

    sys.path.insert(0, args.gs_repo)
    from scene import Scene  # noqa: E402
    from scene.cameras import MiniCam  # noqa: E402
    from gaussian_renderer import render, GaussianModel  # noqa: E402
    from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402

    # Build the repo-style arg namespace (pulls model_path/source_path/sh_degree
    # etc. from the trained model's cfg_args).
    repo_parser = ArgumentParser()
    model = ModelParams(repo_parser, sentinel=True)
    pipeline = PipelineParams(repo_parser)
    repo_parser.add_argument("--iteration", default=-1, type=int)
    repo_parser.add_argument("--gs_repo")
    repo_parser.add_argument("--out")
    repo_parser.add_argument("--frames", type=int)
    repo_parser.add_argument("--amp_scale", type=float)
    combined = get_combined_args(repo_parser)
    dataset = model.extract(combined)
    pipe = pipeline.extract(combined)

    os.makedirs(args.out, exist_ok=True)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=combined.iteration,
                      shuffle=False)
        cams = scene.getTrainCameras()
        if not cams:
            raise RuntimeError("No training cameras found in the model.")

        bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg, dtype=torch.float32, device="cuda")

        # Capture-camera centers + a central reference for orientation/intrinsics.
        centers = np.stack([c.camera_center.detach().cpu().numpy() for c in cams])
        eye0 = centers.mean(axis=0)
        ref = cams[int(np.argmin(np.linalg.norm(centers - eye0, axis=1)))]

        R_ref = np.asarray(ref.R, dtype=np.float64)  # camera-to-world
        right_w = _normalize(R_ref[:, 0])
        up_w = _normalize(-R_ref[:, 1])  # world up = negative of camera 'down'

        # Look target: robust scene center (median resists floaters).
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        target = np.median(xyz, axis=0)

        # Orbit amplitude from the spread of capture centers in the right/up plane.
        rel = centers - eye0
        span_r = rel @ right_w
        span_u = rel @ up_w
        amp_r = args.amp_scale * (span_r.max() - span_r.min()) / 2.0
        amp_u = args.amp_scale * (span_u.max() - span_u.min()) / 2.0
        fallback = 0.1 * float(np.linalg.norm(target - eye0))
        amp_r = amp_r if amp_r > 1e-6 else fallback
        amp_u = amp_u if amp_u > 1e-6 else fallback

        try:
            from tqdm import tqdm
            frame_iter = tqdm(range(args.frames), desc="Orbit render")
        except Exception:
            frame_iter = range(args.frames)

        for i in frame_iter:
            ang = 2.0 * np.pi * i / args.frames
            eye = eye0 + amp_r * np.sin(ang) * right_w + amp_u * np.cos(ang) * up_w
            R_c2w, T = _look_at(eye, target, up_w)

            w2v = torch.tensor(getWorld2View2(R_c2w, T)).transpose(0, 1).cuda()
            proj = getProjectionMatrix(
                znear=ref.znear, zfar=ref.zfar, fovX=ref.FoVx, fovY=ref.FoVy
            ).transpose(0, 1).cuda()
            full = w2v.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
            cam = MiniCam(ref.image_width, ref.image_height, ref.FoVy, ref.FoVx,
                          ref.znear, ref.zfar, w2v, full)

            image = render(cam, gaussians, pipe, background)["render"]
            torchvision.utils.save_image(
                image, os.path.join(args.out, f"orbit_{i:04d}.png"))

    print(f"\n  Rendered {args.frames} novel views -> {args.out}")


if __name__ == "__main__":
    main()
