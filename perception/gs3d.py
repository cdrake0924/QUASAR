"""
Track A-2 — 3D Gaussian Splatting (quasar/perception)

Preparation + launch script for the vanilla 3DGS repo
(https://github.com/graphdeco-inria/gaussian-splatting), expected at
quasar/gaussian-splatting/ (i.e. ../gaussian-splatting relative to this file).

What it does
------------
prep   Build track_a/gs3d/input/ as a COLMAP dataset the 3DGS loader accepts:
         - The 3DGS loader only handles PINHOLE / SIMPLE_PINHOLE cameras, NOT
           our OPENCV (distortion) rig model. So we run COLMAP image_undistorter
           on the 4 static views + rig/sparse/ to get undistorted images and a
           PINHOLE model.
         - Layout: input/images/ (undistorted), input/sparse/0/ (PINHOLE model).
         - Initialization cloud: our dense MVS cloud (mvs/static_fused.ply) is
           copied to input/sparse/0/points3D.ply. The 3DGS reader loads that ply
           directly as the init point cloud when it exists (no --init_pcd flag
           needed). image_undistorter keeps the world frame, so the dense cloud
           stays aligned with the undistorted cameras.
train  Run prep, then ../gaussian-splatting/train.py -s input -m output.
render Run ../gaussian-splatting/render.py -m output and copy images to
       track_a/gs3d/renders/ for manual comparison against NPBG++.

Depends on: mvs/static_fused.ply, rig/sparse/, mvs/static/, COLMAP on PATH,
            and the cloned gaussian-splatting repo with its CUDA env.

Run:
    python gs3d.py --mode prep
    python gs3d.py --mode train
    python gs3d.py --mode render
"""

import argparse
import os
import shutil
import sys

from common import (
    POSITION_ORDER,
    HERE,
    RIG_SPARSE_DIR,
    STATIC_DIR,
    STATIC_FUSED_PLY,
    find_colmap,
    make_gif,
    run,
)


# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(HERE)
GS_REPO = os.path.join(PROJECT_ROOT, "gaussian-splatting")

TRACK_A_GS = os.path.join(HERE, "track_a", "gs3d")
INPUT_DIR = os.path.join(TRACK_A_GS, "input")
INPUT_IMAGES = os.path.join(INPUT_DIR, "images")
INPUT_SPARSE0 = os.path.join(INPUT_DIR, "sparse", "0")
OUTPUT_DIR = os.path.join(TRACK_A_GS, "output")
RENDERS_DIR = os.path.join(TRACK_A_GS, "renders")
ORBIT_DIR = os.path.join(RENDERS_DIR, "orbit")
_UNDIST_TMP = os.path.join(TRACK_A_GS, "_undistort")

ORBIT_SCRIPT = os.path.join(HERE, "gs3d_orbit.py")

CLONE_HINT = (
    "Clone it next to perception/, e.g.:\n"
    "  cd " + PROJECT_ROOT + "\n"
    "  git clone https://github.com/graphdeco-inria/gaussian-splatting "
    "--recursive\n"
    "  cd gaussian-splatting && pip install -r requirements.txt\n"
    "  pip install ./submodules/diff-gaussian-rasterization "
    "./submodules/simple-knn"
)


# --- Validation --------------------------------------------------------------

def require_inputs():
    """Ensure the calibration / MVS artifacts this stage consumes exist."""
    if not os.path.isdir(RIG_SPARSE_DIR):
        raise FileNotFoundError(
            f"{RIG_SPARSE_DIR} not found. Run rig.py (Stage 3) first.")
    if not os.path.isdir(STATIC_DIR):
        raise FileNotFoundError(
            f"{STATIC_DIR} not found. Capture a static set (static_scene.py).")
    missing = [p for p in POSITION_ORDER
               if not os.path.exists(os.path.join(STATIC_DIR, f"{p}.jpg"))]
    if missing:
        raise FileNotFoundError(f"mvs/static/ is missing views {missing}.")
    if not os.path.exists(STATIC_FUSED_PLY):
        raise FileNotFoundError(
            f"{STATIC_FUSED_PLY} not found. Run mvs.py --mode static first.")


def require_repo():
    """Ensure the gaussian-splatting repo is present."""
    train_py = os.path.join(GS_REPO, "train.py")
    if not os.path.exists(train_py):
        raise FileNotFoundError(
            f"gaussian-splatting repo not found at {GS_REPO}.\n" + CLONE_HINT)
    return train_py


# --- Prep --------------------------------------------------------------------

def prep(colmap, max_image_size):
    """Build the COLMAP-format 3DGS input dataset (undistorted + dense init)."""
    require_inputs()

    if os.path.isdir(INPUT_DIR):
        shutil.rmtree(INPUT_DIR)
    if os.path.isdir(_UNDIST_TMP):
        shutil.rmtree(_UNDIST_TMP)
    os.makedirs(INPUT_DIR, exist_ok=True)

    # 1) Undistort the raw views with the OPENCV rig model -> PINHOLE model.
    print("\n=== Undistorting static views (OPENCV -> PINHOLE) ===")
    run([
        colmap, "image_undistorter",
        "--image_path", STATIC_DIR,
        "--input_path", RIG_SPARSE_DIR,
        "--output_path", _UNDIST_TMP,
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ])

    undist_images = os.path.join(_UNDIST_TMP, "images")
    undist_sparse = os.path.join(_UNDIST_TMP, "sparse")
    if not os.path.isdir(undist_images) or not os.path.isdir(undist_sparse):
        raise RuntimeError(
            "image_undistorter did not produce images/ + sparse/. See the "
            "COLMAP output above.")

    # 2) Arrange as the 3DGS loader expects: images/ and sparse/0/.
    shutil.copytree(undist_images, INPUT_IMAGES)
    shutil.copytree(undist_sparse, INPUT_SPARSE0)

    # 3) Inject the dense MVS cloud as the initialization point cloud. The 3DGS
    #    reader loads sparse/0/points3D.ply directly when present.
    shutil.copy(STATIC_FUSED_PLY, os.path.join(INPUT_SPARSE0, "points3D.ply"))

    shutil.rmtree(_UNDIST_TMP, ignore_errors=True)

    n_imgs = len([f for f in os.listdir(INPUT_IMAGES)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"\n  Prepared {INPUT_DIR}")
    print(f"    images/        {n_imgs} undistorted view(s)")
    print(f"    sparse/0/      PINHOLE model from rig/sparse/")
    print(f"    sparse/0/points3D.ply  <- dense init from "
          f"{os.path.basename(STATIC_FUSED_PLY)}")


# --- Train / Render ----------------------------------------------------------

def train(python_bin, iterations, extra_args):
    train_py = require_repo()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmd = [python_bin, train_py, "-s", INPUT_DIR, "-m", OUTPUT_DIR,
           "--iterations", str(iterations)] + extra_args
    print("\n=== Training 3DGS ===")
    run(cmd)
    print(f"\n  Trained model -> {OUTPUT_DIR}")


def render(python_bin, extra_args):
    if not os.path.isdir(OUTPUT_DIR):
        raise FileNotFoundError(
            f"{OUTPUT_DIR} not found. Run 'python gs3d.py --mode train' first.")
    render_py = os.path.join(GS_REPO, "render.py")
    if not os.path.exists(render_py):
        raise FileNotFoundError(
            f"render.py not found in {GS_REPO}.\n" + CLONE_HINT)
    cmd = [python_bin, render_py, "-m", OUTPUT_DIR] + extra_args
    print("\n=== Rendering 3DGS ===")
    run(cmd)
    collect_renders()


def orbit(python_bin, frames, amp_scale, extra_args):
    """Render a novel elliptical camera path (true novel-view evaluation)."""
    if not os.path.isdir(OUTPUT_DIR):
        raise FileNotFoundError(
            f"{OUTPUT_DIR} not found. Run 'python gs3d.py --mode train' first.")
    require_repo()
    if os.path.isdir(ORBIT_DIR):
        shutil.rmtree(ORBIT_DIR)
    os.makedirs(ORBIT_DIR, exist_ok=True)
    cmd = [python_bin, ORBIT_SCRIPT, "-m", OUTPUT_DIR, "--gs_repo", GS_REPO,
           "--out", ORBIT_DIR, "--frames", str(frames),
           "--amp_scale", str(amp_scale)] + extra_args
    print("\n=== Rendering novel orbit path ===")
    run(cmd)
    print(f"\n  Novel views -> {ORBIT_DIR}")
    print("  Flip through them (or make a gif) to judge novel-view quality "
          "and compare against NPBG++.")


def gif(fps):
    """Assemble the rendered orbit frames into a single looping GIF."""
    frames = sorted(f for f in os.listdir(ORBIT_DIR)
                    if f.lower().endswith(".png")) if os.path.isdir(ORBIT_DIR) else []
    if not frames:
        raise FileNotFoundError(
            f"No orbit frames in {ORBIT_DIR}. Run 'python gs3d.py --mode orbit' "
            "first.")
    out = os.path.join(RENDERS_DIR, "gs3d_orbit.gif")
    _, n = make_gif([os.path.join(ORBIT_DIR, f) for f in frames], out, fps=fps)
    print(f"\n  Wrote {out} ({n} frames @ {fps} fps)")


def collect_renders():
    """Copy rendered PNGs from output/{train,test}/ours_*/renders into renders/."""
    os.makedirs(RENDERS_DIR, exist_ok=True)
    copied = 0
    for split in ("train", "test"):
        split_dir = os.path.join(OUTPUT_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        for ours in sorted(os.listdir(split_dir)):
            renders = os.path.join(split_dir, ours, "renders")
            if not os.path.isdir(renders):
                continue
            for name in os.listdir(renders):
                dst = f"{split}_{ours}_{name}"
                shutil.copy(os.path.join(renders, name),
                            os.path.join(RENDERS_DIR, dst))
                copied += 1
    if copied:
        print(f"\n  Copied {copied} render(s) -> {RENDERS_DIR}")
    else:
        print("\n  No renders found to copy. With only 4 training cameras and "
              "no eval split, render.py may have produced nothing — see the "
              "novel-view note in the README.")


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Track A-2: prepare + launch 3D Gaussian Splatting.")
    parser.add_argument("--mode", required=True,
                        choices=["prep", "train", "render", "orbit", "gif"])
    parser.add_argument("--colmap", default=None,
                        help="Path to colmap binary (default: PATH).")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter for the 3DGS repo "
                             "(default: this one). Use the env where the 3DGS "
                             "CUDA submodules are installed.")
    parser.add_argument("--iterations", type=int, default=30000,
                        help="Training iterations (default 30000).")
    parser.add_argument("--max_image_size", type=int, default=2000,
                        help="image_undistorter max size (default 2000; the "
                             "1024x768 inputs are never upscaled).")
    parser.add_argument("--frames", type=int, default=120,
                        help="orbit mode: number of novel frames (default 120).")
    parser.add_argument("--amp_scale", type=float, default=1.5,
                        help="orbit mode: path radius as a multiple of the "
                             "capture-camera half-spread (default 1.5).")
    parser.add_argument("--fps", type=int, default=20,
                        help="gif mode: frames per second (default 20).")
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="Extra args after -- are forwarded to the 3DGS "
                             "train.py / render.py.")
    args = parser.parse_args()
    extra = [a for a in args.rest if a != "--"]

    print("Track A-2 — 3D Gaussian Splatting")
    print(f"  Repo:   {GS_REPO}")
    print(f"  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")

    if args.mode == "prep":
        prep(find_colmap(args.colmap), args.max_image_size)
    elif args.mode == "train":
        prep(find_colmap(args.colmap), args.max_image_size)
        train(args.python, args.iterations, extra)
        print("\nNext: python gs3d.py --mode render")
    elif args.mode == "render":
        render(args.python, extra)
    elif args.mode == "orbit":
        orbit(args.python, args.frames, args.amp_scale, extra)
    elif args.mode == "gif":
        gif(args.fps)


if __name__ == "__main__":
    main()
