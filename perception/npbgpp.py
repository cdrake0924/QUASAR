"""
Track A-1 — Neural Point-Based Graphics++ (NPBG++)  (quasar/perception)

Preparation + launch script for the NPBG++ comparison method. It mirrors the
3DGS stage (gs3d.py): same MVS point cloud + same calibrated poses go in, so the
two methods are evaluated on identical inputs.

WHY THIS LOOKS DIFFERENT FROM gs3d.py
-------------------------------------
NPBG++ (https://github.com/rakhimovv/npbgpp) is a 2021 PyTorch-Lightning + Hydra
codebase pinned to torch 1.9.1 + a source-built PyTorch3D + a custom CUDA
extension. That stack does not build cleanly on native Windows, so TRAIN/RENDER
are meant to run inside WSL2 / Linux (see README "Track A-1"). PREP, on the other
hand, only needs OpenCV + the COLMAP CLI, so you can run it on Windows.

THE DATA CONTRACT (npbgplusplus/data/colmap_scene.py :: ColmapScene)
--------------------------------------------------------------------
The repo already ships a COLMAP loader, but it is stricter than the 3DGS loader:
  * it reads  <scene>/sparse/cameras.bin  and takes the FIRST camera only,
    expecting a PINHOLE model (exactly fx, fy, cx, cy) shared by every image;
  * it reads  <scene>/sparse/images.bin   for poses (world-to-camera, COLMAP
    convention — identical to what rig.py already writes);
  * it loads  <scene>/images/<name>.png   (PNG, named after the COLMAP image
    names, all the SAME size);
  * it loads  <scene>/mvs_pc.ply          as the point cloud.

Our rig has four cameras with DIFFERENT intrinsics/distortion, so prep rectifies
all four raw views to ONE common pinhole K (cv2.undistort with a shared new
camera matrix) and writes a single-camera COLMAP model. Undistorting only changes
intrinsics, not the world frame, so the poses + point cloud stay aligned.

Prep also emits a tailored Hydra config `configs/datasets/quasar_one_scene.yaml`
into the npbgpp repo. The stock `colmap_one_scene.yaml` hardcodes val/holdout
indices up to 95, which trips the loader's `0 <= idx < len(views)` asserts on a
4-view scene; the tailored config zeroes those out so all four views train.

Layout produced under track_a/npbgpp/:
  data/quasar/images/<pos>.png      4 undistorted views (shared pinhole)
  data/quasar/sparse/{cameras,images,points3D}.bin   single-PINHOLE model
  data/quasar/mvs_pc.ply            copy of mvs/static_fused.ply
  output/                           training run dir (hydra.run.dir)
  renders/                          collected eval renders

Run:
    python npbgpp.py --mode prep      # build data/quasar/ (Windows ok)
    python npbgpp.py --mode train     # fine-tune on our 4 views (run in WSL)
    python npbgpp.py --mode render    # feed-forward eval render (run in WSL)
"""

import argparse
import os
import shutil

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
    rot_to_quat,
    camera_center,
    find_colmap,
    make_gif,
    read_ply_xyz,
    run,
    count_ply_points,
)


# --- Paths -------------------------------------------------------------------

TRACK_A = os.path.join(HERE, "track_a")
NPBG_DIR = os.path.join(TRACK_A, "npbgpp")
DATA_ROOT = os.path.join(NPBG_DIR, "data")
SCENE_NAME = "quasar"
SCENE_ROOT = os.path.join(DATA_ROOT, SCENE_NAME)
IMAGES_DIR = os.path.join(SCENE_ROOT, "images")
SPARSE_DIR = os.path.join(SCENE_ROOT, "sparse")
PC_DST = os.path.join(SCENE_ROOT, "mvs_pc.ply")
OUTPUT_DIR = os.path.join(NPBG_DIR, "output")
RENDERS_DIR = os.path.join(NPBG_DIR, "renders")

# --- Orbit (novel-view) scene -------------------------------------------------
# A second scene that keeps the 4 real views as INPUT/source and adds N novel
# orbit poses as render TARGETS, so the feed-forward renderer paints viewpoints
# the rig never captured (the real novel-view test, mirroring gs3d --mode orbit).
ORBIT_SCENE_NAME = "quasar_orbit"
ORBIT_SCENE_ROOT = os.path.join(DATA_ROOT, ORBIT_SCENE_NAME)
ORBIT_IMAGES_DIR = os.path.join(ORBIT_SCENE_ROOT, "images")
ORBIT_SPARSE_DIR = os.path.join(ORBIT_SCENE_ROOT, "sparse")
ORBIT_PC_DST = os.path.join(ORBIT_SCENE_ROOT, "mvs_pc.ply")
ORBIT_OUTPUT_DIR = os.path.join(NPBG_DIR, "orbit_output")
ORBIT_FRAMES_DIR = os.path.join(RENDERS_DIR, "orbit")

# External repo: quasar/npbgpp (sibling of perception/, like gaussian-splatting).
GS_REPO_HINT = "https://github.com/rakhimovv/npbgpp"
REPO = os.path.normpath(os.path.join(HERE, "..", "npbgpp"))
CONFIG_DST = os.path.join(REPO, "configs", "datasets", "quasar_one_scene.yaml")
ORBIT_CONFIG_DST = os.path.join(
    REPO, "configs", "datasets", "quasar_orbit.yaml")


# --- Validation --------------------------------------------------------------

def require_inputs():
    """Fail early with a clear message if a prep dependency is missing."""
    missing = []
    for position in POSITION_ORDER:
        img = os.path.join(STATIC_DIR, f"{position}.jpg")
        if not os.path.exists(img):
            missing.append(img)
    if not os.path.exists(STATIC_FUSED_PLY):
        missing.append(STATIC_FUSED_PLY)
    if missing:
        raise FileNotFoundError(
            "Missing NPBG++ prep inputs:\n  " + "\n  ".join(missing) +
            "\nRun the static capture + mvs.py first."
        )


def require_repo():
    """Fail with clone instructions if the npbgpp repo is not present."""
    if not os.path.isdir(REPO) or not os.path.exists(
            os.path.join(REPO, "train_net.py")):
        raise FileNotFoundError(
            f"NPBG++ repo not found at {REPO}.\n"
            f"Clone it next to perception/:\n"
            f"  git clone {GS_REPO_HINT} {REPO}\n"
            "Then build its WSL2/Linux environment (see README 'Track A-1')."
        )


# --- Prep --------------------------------------------------------------------

def common_pinhole(intrinsics):
    """
    Pick one shared PINHOLE intrinsic for all four cameras.

    The cameras are nearly identical, so the mean focal length with the
    principal point pinned to the image centre gives a clean shared model with
    minimal black border after rectification. Returns a 3x3 K.
    """
    fxs = [intr[0][0, 0] for intr in intrinsics.values()]
    fys = [intr[0][1, 1] for intr in intrinsics.values()]
    fx = float(np.mean(fxs))
    fy = float(np.mean(fys))
    cx = (FRAME_WIDTH - 1) / 2.0
    cy = (FRAME_HEIGHT - 1) / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float64)


def write_model_txt(txt_dir, K_common, poses):
    """Write a single-PINHOLE-camera COLMAP TXT model (camera id 1, 4 images)."""
    os.makedirs(txt_dir, exist_ok=True)
    fx, fy = float(K_common[0, 0]), float(K_common[1, 1])
    cx, cy = float(K_common[0, 2]), float(K_common[1, 2])

    with open(os.path.join(txt_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {FRAME_WIDTH} {FRAME_HEIGHT} "
                f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    with open(os.path.join(txt_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(POSITION_ORDER)}\n")
        for img_id, position in enumerate(POSITION_ORDER, start=1):
            R, t = poses[position]
            qw, qx, qy, qz = rot_to_quat(R)
            tx, ty, tz = (float(v) for v in np.asarray(t).reshape(3))
            # All images share camera id 1; names are PNG (loader default ext).
            f.write(f"{img_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{tx:.10f} {ty:.10f} {tz:.10f} 1 {position}.png\n")
            f.write("\n")  # no 2D observations

    with open(os.path.join(txt_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")


def write_dataset_config():
    """
    Emit configs/datasets/quasar_one_scene.yaml into the npbgpp repo.

    Based on colmap_one_scene.yaml but adapted for a 4-view scene:
      * val/holdout indices emptied (the stock [5..95] indices crash the loader
        on 4 views), so all four views are used for training;
      * data_root is left as a placeholder — npbgpp.py always passes
        datasets.data_root=... on the CLI, so this works whether you run from
        Windows or WSL (paths differ between them);
      * random_shift off + full-frame so the tiny capture set isn't cropped away.
    """
    config = f"""# Auto-generated by perception/npbgpp.py --mode prep. Do not hand-edit;
# re-run prep to regenerate. Tailored colmap_one_scene for our 4-view rig.
scene_class_name: npbgplusplus.data.ColmapScene
scene_name: {SCENE_NAME}
train_num_samples: 2000
train_random_zoom: ~
train_random_shift: false
train_image_size: ~
data_root: PASS_VIA_CLI  # overridden by datasets.data_root=... at launch
scene_subroot: ${{datasets.data_root}}
images_subroot: ${{datasets.data_root}}
val_indices: []
holdout_indices: []
cache_images: false
selection_count: 3
noise_sigma: ~
n_point: ~
pc_name: mvs_pc.ply

train:
  - dataset_name: ${{datasets.scene_name}}_finetune
    dataset_class:
      _target_: npbgplusplus.data.ViewSceneWrapper
      scene_dataset:
        _target_: ${{datasets.scene_class_name}}
        _convert_: all
        scene_root: ${{datasets.scene_subroot}}/${{datasets.scene_name}}
        images_root: ${{datasets.images_subroot}}/${{datasets.scene_name}}/images
        pc_path: ${{datasets.scene_subroot}}/${{datasets.scene_name}}/${{datasets.pc_name}}
        num_samples: ${{datasets.train_num_samples}}
        random_zoom: ${{datasets.train_random_zoom}}
        random_shift: ${{datasets.train_random_shift}}
        image_size: ${{datasets.train_image_size}}
        exclude_indices: []
        noise_sigma: ${{datasets.noise_sigma}}
        n_point: ${{datasets.n_point}}
      selection_count: ${{datasets.selection_count}}

val: []

test:
  - dataset_name: ${{datasets.scene_name}}_eval
    dataset_class:
      _target_: npbgplusplus.data.ViewSceneWrapper
      scene_dataset:
        _target_: ${{datasets.scene_class_name}}
        _convert_: all
        scene_root: ${{datasets.scene_subroot}}/${{datasets.scene_name}}
        images_root: ${{datasets.images_subroot}}/${{datasets.scene_name}}/images
        pc_path: ${{datasets.scene_subroot}}/${{datasets.scene_name}}/${{datasets.pc_name}}
        noise_sigma: ${{datasets.noise_sigma}}
        n_point: ${{datasets.n_point}}
      selection_method: ""
"""
    os.makedirs(os.path.dirname(CONFIG_DST), exist_ok=True)
    with open(CONFIG_DST, "w") as f:
        f.write(config)
    return CONFIG_DST


def prep(colmap):
    """Build track_a/npbgpp/data/quasar/ in the format ColmapScene expects."""
    import cv2

    require_inputs()
    cameras = load_camera_indices()
    intrinsics = {p: load_intrinsics(idx) for p, idx in cameras}
    poses = load_poses()

    K_common = common_pinhole(intrinsics)
    print("Track A-1 — NPBG++ prep")
    print(f"  Shared pinhole K: fx={K_common[0,0]:.2f} fy={K_common[1,1]:.2f} "
          f"cx={K_common[0,2]:.1f} cy={K_common[1,2]:.1f} "
          f"({FRAME_WIDTH}x{FRAME_HEIGHT})")

    # Clean + recreate the scene tree.
    if os.path.isdir(SCENE_ROOT):
        shutil.rmtree(SCENE_ROOT)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(SPARSE_DIR, exist_ok=True)

    # Rectify each raw view from its own (K_i, dist_i) to the shared pinhole.
    for position, idx in cameras:
        K_i, dist_i = intrinsics[position]
        src = os.path.join(STATIC_DIR, f"{position}.jpg")
        img = cv2.imread(src)
        if img is None:
            raise RuntimeError(f"Could not read {src}")
        if (img.shape[1], img.shape[0]) != (FRAME_WIDTH, FRAME_HEIGHT):
            img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        und = cv2.undistort(img, K_i, dist_i, None, K_common)
        dst = os.path.join(IMAGES_DIR, f"{position}.png")
        cv2.imwrite(dst, und)
        print(f"    rectified {position}.jpg -> {position}.png")

    # Write the single-PINHOLE COLMAP model (TXT) and convert to BIN, since
    # ColmapScene reads cameras.bin / images.bin specifically.
    txt_dir = os.path.join(SCENE_ROOT, "_sparse_txt")
    write_model_txt(txt_dir, K_common, poses)
    run([colmap, "model_converter",
         "--input_path", txt_dir,
         "--output_path", SPARSE_DIR,
         "--output_type", "BIN"])
    shutil.rmtree(txt_dir, ignore_errors=True)

    # Point cloud.
    shutil.copy2(STATIC_FUSED_PLY, PC_DST)

    cfg = write_dataset_config()

    print("\n  Wrote:")
    print(f"    {IMAGES_DIR}  (4 PNG views)")
    print(f"    {os.path.join(SPARSE_DIR, 'cameras.bin')} / images.bin")
    print(f"    {PC_DST}  ({count_ply_points(PC_DST)} points)")
    print(f"    {cfg}")
    _report_geometry(poses)
    print("\nPrep done. Next (inside WSL): python npbgpp.py --mode train")


def _report_geometry(poses):
    centers = {p: camera_center(R, t) for p, (R, t) in poses.items()}
    print("\n  Camera centers (world frame, top_left = origin):")
    for position in POSITION_ORDER:
        c = centers[position]
        print(f"    {position:10s}: [{c[0]:9.2f}, {c[1]:9.2f}, {c[2]:9.2f}]")


# --- Train / Render (run inside WSL2 / Linux) --------------------------------

def _weights_or_die(weights):
    if not weights:
        raise SystemExit(
            "NPBG++ needs a PRETRAINED checkpoint (its descriptor-prediction "
            "network is pretrained, not learned per-scene).\n"
            "Download one from the repo's checkpoints link "
            "(https://disk.yandex.ru/d/-1kx0XUlRHNumQ) and pass it:\n"
            "  python npbgpp.py --mode train  --weights /path/to/npbgpp_*.ckpt\n"
            "  python npbgpp.py --mode render --weights /path/to/npbgpp_*.ckpt"
        )
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    return weights


def train(python, weights, epochs, extra):
    """
    Fine-tune NPBG++ on our 4 views (writes a checkpoint under output/).

    Runs the repo's train_net.py via Hydra. Must run inside the npbgpp WSL env
    (torch 1.9.1 + pytorch3d + the built CUDA extension).
    """
    require_repo()
    weights = _weights_or_die(weights)
    _single_gpu_dist_env()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmd = [
        python, "train_net.py",
        "trainer.gpus=1",
        f"hydra.run.dir={OUTPUT_DIR}",
        "datasets=quasar_one_scene",
        f"datasets.data_root={DATA_ROOT}",
        "system=npbgpp_sphere",
        "system.visibility_scale=1.0",
        f"weights_path={weights}",
        f"trainer.max_epochs={epochs}",
        "dataloader=small",
    ] + extra
    run(cmd_with_cwd(cmd))


def render(python, weights, extra):
    """
    Feed-forward / eval render of the scene views (NPBG++'s headline mode).

    eval_only=true: predict descriptors from the pretrained network + point
    cloud and rasterize the views, no per-scene optimisation. Collect outputs
    into renders/ for visual comparison against the 3DGS orbit.
    """
    require_repo()
    weights = _weights_or_die(weights)
    _single_gpu_dist_env()
    os.makedirs(RENDERS_DIR, exist_ok=True)
    cmd = [
        python, "train_net.py",
        "trainer.gpus=1",
        f"hydra.run.dir={OUTPUT_DIR}",
        "datasets=quasar_one_scene",
        f"datasets.data_root={DATA_ROOT}",
        "system=npbgpp_sphere",
        "system.visibility_scale=1.0",
        f"weights_path={weights}",
        "eval_only=true",
        "dataloader=small",
    ] + extra
    run(cmd_with_cwd(cmd))
    _collect_renders()


def _collect_renders():
    """Copy the network outputs (and GTs) from the hydra run dir into renders/."""
    import glob
    src = glob.glob(os.path.join(
        OUTPUT_DIR, "rendered", "*", "test_epoch*", "*_rendered.png"))
    src += glob.glob(os.path.join(
        OUTPUT_DIR, "rendered", "*", "test_epoch*", "*_gt.png"))
    os.makedirs(RENDERS_DIR, exist_ok=True)
    for p in src:
        shutil.copy2(p, os.path.join(RENDERS_DIR, os.path.basename(p)))
    if src:
        print(f"\nCollected {len(src)} images into {RENDERS_DIR} "
              f"(*_rendered.png vs *_gt.png) for comparison against 3DGS.")
    else:
        print(f"\nNo rendered images found under {OUTPUT_DIR}/rendered/ — "
              "check the run log.")


def _single_gpu_dist_env():
    """
    Make the repo's always-on DistributedSampler work on a single GPU under WSL.

    build.py uses torch DistributedSampler whenever CUDA is available, so a
    process group must exist. PyTorch-Lightning's ddp defaults to the NCCL
    backend, whose socket setup fails under WSL — so force the gloo backend
    (world_size=1, no real comm needed).
    """
    os.environ.setdefault("PL_TORCH_DISTRIBUTED_BACKEND", "gloo")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def cmd_with_cwd(cmd):
    """
    train_net.py uses Hydra, which resolves configs relative to its own CWD, so
    the command must run from inside the repo. common.run() doesn't take a cwd,
    so prepend a shell `cd`. On Windows this won't run (the env is WSL-only); it
    is printed so you can copy it into the WSL shell if needed.
    """
    # Return the raw list; we cd via os.chdir to keep common.run() unchanged.
    os.chdir(REPO)
    return cmd


# --- Orbit (novel-view path) -------------------------------------------------

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _look_at(eye, target, up_world):
    """
    World-to-camera (R_w2c, T) for a camera at `eye` looking at `target`.

    Uses the exact same construction as gs3d_orbit._look_at (COLMAP convention:
    x-right, y-down, z-forward) so the two methods trace a comparable path.
    """
    f = _normalize(target - eye)            # forward = camera +z
    r = _normalize(np.cross(f, up_world))   # right   = camera +x
    d = _normalize(np.cross(f, r))          # down    = camera +y
    R_w2c = np.stack([r, d, f], axis=0)     # rows map world -> camera
    T = -R_w2c @ eye
    return R_w2c, T


def compute_orbit_poses(poses, frames, amp_scale):
    """
    Build N novel world-to-camera poses on an ellipse around the scene.

    Mirrors gs3d_orbit: the path lives in the right/up plane of the rig's
    central camera, centred on the mean capture position, sized to the spread of
    the 4 cameras (x amp_scale), and aimed at the median point of the MVS cloud
    (median resists outliers). Returns a list of (R_w2c, T) numpy pairs.
    """
    centers = np.stack([camera_center(R, t) for R, t in poses.values()])
    eye0 = centers.mean(axis=0)
    # Reference camera = the one closest to the centroid (defines the plane).
    ref_pos = POSITION_ORDER[int(np.argmin(np.linalg.norm(centers - eye0, axis=1)))]
    R_ref = np.asarray(poses[ref_pos][0], dtype=np.float64).reshape(3, 3)
    c2w = R_ref.T                            # columns are camera axes in world
    right_w = _normalize(c2w[:, 0])
    up_w = _normalize(-c2w[:, 1])            # world up = negative camera 'down'

    target = np.median(read_ply_xyz(STATIC_FUSED_PLY), axis=0)

    rel = centers - eye0
    span_r = rel @ right_w
    span_u = rel @ up_w
    amp_r = amp_scale * (span_r.max() - span_r.min()) / 2.0
    amp_u = amp_scale * (span_u.max() - span_u.min()) / 2.0
    fallback = 0.1 * float(np.linalg.norm(target - eye0))
    amp_r = amp_r if amp_r > 1e-6 else fallback
    amp_u = amp_u if amp_u > 1e-6 else fallback

    out = []
    for i in range(frames):
        ang = 2.0 * np.pi * i / frames
        eye = eye0 + amp_r * np.sin(ang) * right_w + amp_u * np.cos(ang) * up_w
        out.append(_look_at(eye, target, up_w))
    return out


def write_orbit_model_txt(txt_dir, K_common, real_poses, orbit_poses):
    """
    Single-PINHOLE COLMAP TXT model: 4 real views (input) + N orbit views.

    Real views get ids 1..4 and their calibrated poses; orbit views get ids 5..
    and the novel poses, named orbit_XXXX.png. All share camera id 1. The render
    targets are selected later by name, not by id, so model_converter is free to
    reorder.
    """
    os.makedirs(txt_dir, exist_ok=True)
    fx, fy = float(K_common[0, 0]), float(K_common[1, 1])
    cx, cy = float(K_common[0, 2]), float(K_common[1, 2])

    with open(os.path.join(txt_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {FRAME_WIDTH} {FRAME_HEIGHT} "
                f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    n_total = len(POSITION_ORDER) + len(orbit_poses)
    with open(os.path.join(txt_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {n_total}\n")
        img_id = 1
        for position in POSITION_ORDER:
            R, t = real_poses[position]
            qw, qx, qy, qz = rot_to_quat(R)
            tx, ty, tz = (float(v) for v in np.asarray(t).reshape(3))
            f.write(f"{img_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{tx:.10f} {ty:.10f} {tz:.10f} 1 {position}.png\n")
            f.write("\n")
            img_id += 1
        for i, (R, t) in enumerate(orbit_poses):
            qw, qx, qy, qz = rot_to_quat(R)
            tx, ty, tz = (float(v) for v in np.asarray(t).reshape(3))
            f.write(f"{img_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{tx:.10f} {ty:.10f} {tz:.10f} 1 orbit_{i:04d}.png\n")
            f.write("\n")
            img_id += 1

    with open(os.path.join(txt_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")


def _read_images_bin_names(path):
    """
    Return image NAMEs in the order ColmapScene will iterate them.

    model_converter writes images.bin from an unordered map, so the on-disk
    order is not the id order. ColmapScene indexes views by this read order, so
    we replicate read_images_binary here to map orbit views to dataset indices.
    """
    import struct
    names = []
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            f.read(4)            # image_id (uint32)
            f.read(8 * 7)        # qvec (4d) + tvec (3d)
            f.read(4)            # camera_id (uint32)
            chars = bytearray()
            while True:
                c = f.read(1)
                if c in (b"\x00", b""):
                    break
                chars += c
            names.append(chars.decode("utf-8", "ignore"))
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n2d)     # each point2D: 2 doubles + int64
    return names


def write_orbit_config(target_indices):
    """
    Emit configs/datasets/quasar_orbit.yaml: test renders the orbit views.

    target_views_indices = the orbit views -> their complement (the 4 real
    views) becomes the input/source set the descriptors are aggregated from.
    selection_method "" means every target uses all input views. train/val are
    empty (orbit is eval-only).
    """
    idx_list = "[" + ", ".join(str(i) for i in target_indices) + "]"
    config = f"""# Auto-generated by perception/npbgpp.py --mode orbit-prep. Do not hand-edit.
# Novel-view orbit scene: 4 real views are input/source, orbit_* views are the
# render targets (the 4 real views are their complement of target_views_indices).
scene_class_name: npbgplusplus.data.ColmapScene
scene_name: {ORBIT_SCENE_NAME}
data_root: PASS_VIA_CLI  # overridden by datasets.data_root=... at launch
scene_subroot: ${{datasets.data_root}}
images_subroot: ${{datasets.data_root}}
cache_images: false
selection_count: 3
noise_sigma: ~
n_point: ~
pc_name: mvs_pc.ply

train: []
val: []

test:
  - dataset_name: ${{datasets.scene_name}}_eval
    dataset_class:
      _target_: npbgplusplus.data.ViewSceneWrapper
      scene_dataset:
        _target_: ${{datasets.scene_class_name}}
        _convert_: all
        scene_root: ${{datasets.scene_subroot}}/${{datasets.scene_name}}
        images_root: ${{datasets.images_subroot}}/${{datasets.scene_name}}/images
        pc_path: ${{datasets.scene_subroot}}/${{datasets.scene_name}}/${{datasets.pc_name}}
        target_views_indices: {idx_list}
        noise_sigma: ${{datasets.noise_sigma}}
        n_point: ${{datasets.n_point}}
      selection_method: ""
"""
    os.makedirs(os.path.dirname(ORBIT_CONFIG_DST), exist_ok=True)
    with open(ORBIT_CONFIG_DST, "w") as f:
        f.write(config)
    return ORBIT_CONFIG_DST


def orbit_prep(colmap, frames, amp_scale):
    """
    Build track_a/npbgpp/data/quasar_orbit/ — the novel-view orbit scene.

    Reuses the 4 rectified real views (from --mode prep) as input and appends N
    novel orbit poses as render targets with black placeholder images (the eval
    renderer paints them against a learned background, never the GT, so the
    placeholders are unused for output).
    """
    import cv2

    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(
            f"{IMAGES_DIR} not found. Run 'python npbgpp.py --mode prep' first "
            "(orbit reuses its rectified views + shared pinhole).")

    cameras = load_camera_indices()
    intrinsics = {p: load_intrinsics(idx) for p, idx in cameras}
    poses = load_poses()
    K_common = common_pinhole(intrinsics)

    orbit_poses = compute_orbit_poses(poses, frames, amp_scale)
    print("Track A-1 — NPBG++ orbit prep")
    print(f"  {frames} novel poses (amp_scale={amp_scale}) around the cloud "
          f"median, in the rig's right/up plane.")

    if os.path.isdir(ORBIT_SCENE_ROOT):
        shutil.rmtree(ORBIT_SCENE_ROOT)
    os.makedirs(ORBIT_IMAGES_DIR, exist_ok=True)
    os.makedirs(ORBIT_SPARSE_DIR, exist_ok=True)

    # Real views (input/source): copy the already-rectified PNGs.
    for position in POSITION_ORDER:
        shutil.copy2(os.path.join(IMAGES_DIR, f"{position}.png"),
                     os.path.join(ORBIT_IMAGES_DIR, f"{position}.png"))

    # Orbit views (targets): black placeholders sized to the capture frame.
    black = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    for i in range(frames):
        cv2.imwrite(os.path.join(ORBIT_IMAGES_DIR, f"orbit_{i:04d}.png"), black)

    # Single-PINHOLE model (real + orbit), TXT -> BIN.
    txt_dir = os.path.join(ORBIT_SCENE_ROOT, "_sparse_txt")
    write_orbit_model_txt(txt_dir, K_common, poses, orbit_poses)
    run([colmap, "model_converter",
         "--input_path", txt_dir,
         "--output_path", ORBIT_SPARSE_DIR,
         "--output_type", "BIN"])
    shutil.rmtree(txt_dir, ignore_errors=True)

    shutil.copy2(STATIC_FUSED_PLY, ORBIT_PC_DST)

    # Map orbit views to their dataset indices (read order of images.bin).
    names = _read_images_bin_names(os.path.join(ORBIT_SPARSE_DIR, "images.bin"))
    target_indices = [i for i, n in enumerate(names)
                      if n.startswith("orbit_")]
    if len(target_indices) != frames:
        raise RuntimeError(
            f"Expected {frames} orbit views in the model, found "
            f"{len(target_indices)}.")
    cfg = write_orbit_config(target_indices)

    print("\n  Wrote:")
    print(f"    {ORBIT_IMAGES_DIR}  (4 real + {frames} orbit placeholders)")
    print(f"    {os.path.join(ORBIT_SPARSE_DIR, 'images.bin')}  "
          f"({len(names)} images, {frames} targets)")
    print(f"    {cfg}")
    print("\nOrbit prep done. Next (inside WSL): "
          "python npbgpp.py --mode orbit --weights ~/ckpts/<ckpt>.ckpt")


def orbit_render(python, weights, fps, extra):
    """
    Feed-forward render of the novel orbit views, then assemble a GIF.

    Same eval path as --mode render, but pointed at quasar_orbit (whose test set
    is the orbit views). Collects orbit_*_rendered.png in order and writes
    renders/npbgpp_orbit.gif. Run inside the npbgpp WSL env.
    """
    require_repo()
    weights = _weights_or_die(weights)
    if not os.path.isdir(ORBIT_SCENE_ROOT):
        raise FileNotFoundError(
            f"{ORBIT_SCENE_ROOT} not found. Run 'python npbgpp.py "
            "--mode orbit-prep' first (on Windows).")
    _single_gpu_dist_env()
    os.makedirs(ORBIT_OUTPUT_DIR, exist_ok=True)
    cmd = [
        python, "train_net.py",
        "trainer.gpus=1",
        f"hydra.run.dir={ORBIT_OUTPUT_DIR}",
        "datasets=quasar_orbit",
        f"datasets.data_root={DATA_ROOT}",
        "system=npbgpp_sphere",
        "system.visibility_scale=1.0",
        f"weights_path={weights}",
        "eval_only=true",
        "dataloader=small",
    ] + extra
    run(cmd_with_cwd(cmd))
    _assemble_orbit_gif(fps)


def _assemble_orbit_gif(fps):
    """Collect orbit_*_rendered.png (in frame order) and write the orbit GIF."""
    import glob
    rendered = glob.glob(os.path.join(
        ORBIT_OUTPUT_DIR, "rendered", "*", "test_epoch*", "orbit_*_rendered.png"))
    if not rendered:
        print(f"\nNo orbit renders found under {ORBIT_OUTPUT_DIR}/rendered/ — "
              "check the run log.")
        return

    if os.path.isdir(ORBIT_FRAMES_DIR):
        shutil.rmtree(ORBIT_FRAMES_DIR)
    os.makedirs(ORBIT_FRAMES_DIR, exist_ok=True)

    def frame_no(path):
        base = os.path.basename(path)             # orbit_0007_rendered.png
        return int(base.split("_")[1])

    rendered = sorted(rendered, key=frame_no)
    ordered = []
    for p in rendered:
        dst = os.path.join(ORBIT_FRAMES_DIR, f"orbit_{frame_no(p):04d}.png")
        shutil.copy2(p, dst)
        ordered.append(dst)

    out = os.path.join(RENDERS_DIR, "npbgpp_orbit.gif")
    _, n = make_gif(ordered, out, fps=fps)
    print(f"\nCollected {n} orbit frames -> {ORBIT_FRAMES_DIR}")
    print(f"Wrote {out} ({n} frames @ {fps} fps) for comparison vs the "
          "3DGS orbit gif.")


# --- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Track A-1 — NPBG++ prep + launch.")
    parser.add_argument("--mode", required=True,
                        choices=["prep", "train", "render",
                                 "orbit-prep", "orbit"])
    parser.add_argument("--colmap", default=None,
                        help="Path to the COLMAP binary (prep only).")
    parser.add_argument("--python", default="python",
                        help="Python interpreter for the npbgpp WSL env "
                             "(train/render).")
    parser.add_argument("--weights", default=None,
                        help="Pretrained NPBG++ checkpoint (train/render).")
    parser.add_argument("--epochs", type=int, default=20,
                        help="train mode: fine-tune epochs (default 20).")
    parser.add_argument("--frames", type=int, default=60,
                        help="orbit-prep: number of novel frames (default 60, "
                             "matching the 3DGS orbit).")
    parser.add_argument("--amp_scale", type=float, default=1.5,
                        help="orbit-prep: path radius as a multiple of the "
                             "capture-camera half-spread (default 1.5).")
    parser.add_argument("--fps", type=int, default=20,
                        help="orbit mode: gif frames per second (default 20).")
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="Extra args after -- forwarded to train_net.py.")
    args = parser.parse_args()
    extra = [a for a in args.rest if a != "--"]

    print("Track A-1 — Neural Point-Based Graphics++ (NPBG++)")
    print(f"  Repo:   {REPO}")
    print(f"  Data:   {SCENE_ROOT}")
    print(f"  Output: {OUTPUT_DIR}")

    if args.mode == "prep":
        prep(find_colmap(args.colmap))
    elif args.mode == "train":
        train(args.python, args.weights, args.epochs, extra)
    elif args.mode == "render":
        render(args.python, args.weights, extra)
    elif args.mode == "orbit-prep":
        orbit_prep(find_colmap(args.colmap), args.frames, args.amp_scale)
    elif args.mode == "orbit":
        orbit_render(args.python, args.weights, args.fps, extra)


if __name__ == "__main__":
    main()
