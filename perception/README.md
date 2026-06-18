# quasar/perception — Project README

> **For Cursor:** This is a completely fresh project. Do NOT reference or import anything from `quasar/coled`. You may look at `quasar/coled` for context on what has already been attempted, but all code here must be written from scratch in `quasar/perception`. Start clean.

---

## Project Goal

Build a pipeline that takes synchronized multi-camera video footage from a 2×2 camera array and produces a 4D Gaussian splat scene — a dynamic, view-interpolatable 3D representation playable on a VR headset. The pipeline runs in five sequential stages, each with its own Python file.

---

## Project Structure

```
quasar/
└── perception/
    ├── README.md                  ← this file
    ├── camera.json                ← camera position mapping (edit manually)
    ├── intrinsics.py
    ├── extrinsics.py
    ├── static_scene.py            ← Stage 3 static-capture helper
    ├── dynamic_scene.py           ← Stage 4 dynamic-capture helper
    ├── sfm.py
    ├── mvs.py
    ├── scgs.py
    ├── intrinsics/
    │   ├── img_1_1.jpg
    │   ├── img_1_2.jpg
    │   ├── ...
    │   ├── img_4_10.jpg
    │   ├── K_1.txt
    │   ├── K_2.txt
    │   ├── K_3.txt
    │   └── K_4.txt
    ├── extrinsics/
    │   ├── K.txt                  ← combined extrinsic output
    │   ├── top_left/
    │   │   ├── img_1.jpg
    │   │   └── ...
    │   ├── top_right/
    │   ├── bot_left/
    │   └── bot_right/
    ├── sfm/
    │   ├── database.db
    │   ├── images/                ← symlinked or copied frames from all cameras
    │   └── sparse/
    │       └── 0/
    │           ├── cameras.bin
    │           ├── images.bin
    │           └── points3D.bin
    ├── mvs/
    │   ├── frames/                ← INPUT: raw dynamic footage, one folder per moment
    │   │   └── XXXXXX/
    │   │       ├── top_left.jpg
    │   │       ├── top_right.jpg
    │   │       ├── bot_left.jpg
    │   │       └── bot_right.jpg
    │   ├── _rig/sparse/           ← reusable fixed 4-camera pose model (auto-built)
    │   └── frame_XXXXXX/          ← one workspace per synchronized frame-set
    │       ├── images/
    │       ├── dense/
    │       └── fused.ply
    └── scgs/
        ├── input/                 ← symlink or copy of mvs/ output
        ├── output/                ← trained SC-GS scene
        └── renders/               ← exported novel view renders
```

---

## camera.json

Before running anything, edit `camera.json` to map physical camera positions to OS device indices. This file is read by all pipeline stages.

```json
{
  "top_left":  1,
  "top_right": 2,
  "bot_left":  3,
  "bot_right": 4
}
```

The integer values are the camera device indices (e.g. passed to `cv2.VideoCapture(index)`). Adjust these to match your physical rig.

---

## Dependencies

Install all dependencies before running any stage:

```bash
pip install opencv-python numpy pycolmap open3d torch torchvision
```

COLMAP must also be installed as a system binary and accessible on PATH:

```bash
# Ubuntu
sudo apt install colmap

# or build from source: https://colmap.github.io/install.html
```

SC-GS requires cloning its repository separately — see the SC-GS stage below.

---

---

# Stage 1 — Intrinsic Calibration

**File:** `intrinsics.py`
**Output folder:** `intrinsics/`

## What this does

Intrinsic calibration finds each camera's internal optical properties — focal length, principal point (optical center), and lens distortion coefficients. These are bundled into a 3×3 matrix called **K** (the camera matrix). Every downstream stage (extrinsics, COLMAP, SC-GS) depends on accurate intrinsics.

This is done using a checkerboard pattern. OpenCV detects the corners of the checkerboard in multiple images taken from different angles, then solves for K using the known physical geometry of the checkerboard squares.

## How to run

1. Print or display a checkerboard pattern. The default assumes a **9×6 inner corner** grid (i.e. 10×7 squares). If yours differs, edit the `CHECKERBOARD` constant at the top of the file.
2. Run the script: `python intrinsics.py`
3. The script processes one camera at a time. For each camera, a live preview window opens.
4. Hold the checkerboard in front of the camera and move it to different positions, angles, and distances. When the script detects a valid checkerboard, it captures the image automatically.
5. 10 images are collected per camera, then calibration runs immediately.
6. Repeat for all 4 cameras.

## Implementation notes for Cursor

- Resolution: **640×480** for all cameras throughout this project.
- Camera indices come from `camera.json`. Iterate in order: `top_left`, `top_right`, `bot_left`, `bot_right`.
- Image naming: `img_{camera_number}_{photo_number}.jpg` — e.g. `img_1_4.jpg`. Camera number is the integer value from `camera.json`, not the position key.
- Images save to `intrinsics/`.
- After collecting 10 images for a camera, immediately run `cv2.calibrateCamera()` on those images.
- Save the intrinsic matrix K as `intrinsics/K_{camera_number}.txt` using `numpy.savetxt`. Use a plain space-delimited format with 6 decimal places.
- Also save the distortion coefficients alongside K — name them `dist_{camera_number}.txt` — they are needed for undistortion in later stages.
- Print the reprojection error for each camera after calibration. If it is above 1.0 pixels, warn the user and suggest recollecting images.
- Do not auto-advance to the next camera. Print a prompt and wait for the user to press Enter before opening the next camera.

---

---

# Stage 2 — Extrinsic Calibration

**File:** `extrinsics.py`
**Output folder:** `extrinsics/`
**Depends on:** `camera.json`, `intrinsics/K_*.txt`, `intrinsics/dist_*.txt`

## What this does

Extrinsic calibration finds the **position and orientation of each camera relative to the others** — specifically, a rotation matrix R and translation vector t for each camera expressed relative to `top_left`, which is treated as the world origin.

This is done by showing the checkerboard to all 4 cameras simultaneously and using the known intrinsics to solve for each camera's pose in the shared world frame.

## Physical rig geometry

The camera array is a fixed 2×2 grid with known physical spacing:

```
top_left ——— ~180mm ——— top_right
    |                         |
  ~180mm                    ~180mm
    |                         |
bot_left  ——— ~180mm ——— bot_right
```

These measurements are a rough estimate and should not be used for any validation

## How to run

1. Run: `python extrinsics.py`
2. A live preview window shows all 4 camera feeds simultaneously (tiled 2×2).
3. Hold the checkerboard so it is **fully visible in all 4 cameras at once**.
4. When all 4 cameras detect the checkerboard clearly, an image is captured automatically from each camera.
5. Continue holding and repositioning the checkerboard. The script collects as many valid synchronized image-sets as possible (aim for at least 15–20).
6. Press `Q` when done collecting. Calibration runs automatically.

## Implementation notes for Cursor

- Load `camera.json` to get device indices and position labels.
- Load `intrinsics/K_{n}.txt` and `intrinsics/dist_{n}.txt` for each camera before opening streams.
- Resolution: **640×480**.
- Detection: use `cv2.findChessboardCorners()` followed by `cv2.cornerSubPix()` for refinement. Only capture when corners are found in **all 4 cameras in the same loop iteration**.
- Image saving: each camera gets its own subfolder matching its position key.
  - `extrinsics/top_left/img_1.jpg`, `extrinsics/top_left/img_2.jpg`, etc.
  - `extrinsics/top_right/img_1.jpg`, etc.
  - Image numbers are shared across cameras (image 3 from all cameras is the same moment in time).
- Extrinsic solve: use `cv2.solvePnP()` with each camera's intrinsics to get R and t for each camera relative to `top_left`. `top_left` is the reference frame — its R is identity, t is zero.
- Save output to `extrinsics/K.txt`. This file should contain, for each camera position, its rotation matrix R (3×3) and translation vector t (3×1) in a clearly labeled plain-text format, e.g.:
  ```
  top_left
  R: [[1,0,0],[0,1,0],[0,0,1]]
  t: [0,0,0]

  top_right
  R: ...
  t: ...
  ```
- Also save as `extrinsics/poses.npz` using `numpy.savez` with keys `R_top_left`, `t_top_left`, etc. for easy loading downstream.
- Print the translation magnitudes between camera pairs

---

---

# Stage 3 — SfM (Sparse Point Cloud)

**File:** `sfm.py`
**Output folder:** `sfm/`
**Depends on:** `camera.json`, `intrinsics/K_*.txt`, `intrinsics/dist_*.txt`, `extrinsics/poses.npz`

## What this does

Structure from Motion (SfM) finds matching visual features (SIFT keypoints) across all camera views, triangulates their 3D positions geometrically, and produces a sparse point cloud of the scene along with refined camera poses. This sparse cloud and pose set is the required input to MVS.

**Important:** COLMAP SfM assumes a static scene. Run this on a dedicated static capture (all cameras pointing at the scene with nothing moving) before your dynamic footage session. The camera poses recovered here will be frozen and reused for every frame of dynamic MVS.

## How to run

1. Before running, capture a static scene with all 4 cameras (rig and scene perfectly still). Use the `static_scene.py` helper below to write frames into `sfm/images/` with the correct `{position}_{frame_number:06d}.jpg` naming — e.g. `top_left_000001.jpg`. (You may also export them yourself by any means as long as the naming matches.)
2. Run: `python sfm.py`
3. The script runs COLMAP feature extraction, matching, and mapping using subprocess calls.
4. Output lands in `sfm/sparse/0/`.

## Helper — `static_scene.py`

A small capture utility for the static SfM pass. It opens all 4 cameras at once, shows a tiled 2×2 preview, and writes synchronized frames straight into `sfm/images/` named `{position}_{frame:06d}.jpg`, so the output drops directly into the Stage 3 pipeline.

**Depends on:** `camera.json`

Controls (interactive mode):
- `SPACE` / `C` — capture one synchronized set (all 4 cameras)
- `B` — capture a burst of `--num` sets spaced by `--interval`
- `Q` / `ESC` — finish and exit

Run:
```bash
python static_scene.py                 # interactive
python static_scene.py --num 10        # auto-capture 10 sets, then quit
python static_scene.py --interval 0.5  # seconds between auto-captured sets
python static_scene.py --fresh         # wipe sfm/images/ before capturing
```

Capture tips:
- Keep the rig **and** the scene perfectly still — the 3D comes from the 4 fixed viewpoints, not motion, so a handful of synchronized sets is plenty.
- Make sure the scene is **texture-rich**; blank/low-texture surfaces are the most common cause of COLMAP failing to register.
- Frame numbering continues after any frames already in `sfm/images/` (use `--fresh` to start over).

## Implementation notes for Cursor

- Use `subprocess.run()` to call COLMAP CLI commands in sequence. Do not use pycolmap bindings — use the CLI for clarity and debuggability.
- Before running COLMAP, write a `sfm/cameras.txt` file that pre-loads the known intrinsics from `intrinsics/K_*.txt` into COLMAP's format. This prevents COLMAP from re-solving intrinsics and locks them to your calibrated values. Use COLMAP camera model `PINHOLE` with parameters `fx, fy, cx, cy`.
- COLMAP commands to run in order:
  ```bash
  colmap feature_extractor \
    --database_path sfm/database.db \
    --image_path sfm/images \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera_per_folder 1

  colmap exhaustive_matcher \
    --database_path sfm/database.db

  colmap mapper \
    --database_path sfm/database.db \
    --image_path sfm/images \
    --output_path sfm/sparse \
    --Mapper.fix_existing_images 1
  ```
- After mapping, run `colmap model_converter` to export the sparse model to TXT format for human-readable inspection:
  ```bash
  colmap model_converter \
    --input_path sfm/sparse/0 \
    --output_path sfm/sparse/0 \
    --output_type TXT
  ```
- Print the number of registered images and 3D points after completion. If fewer than 3 of 4 cameras registered, warn the user — extrinsic calibration or static capture may need to be redone.

---

---

# Stage 4 — MVS (Dense Point Cloud)

**File:** `mvs.py`
**Output folder:** `mvs/`
**Depends on:** `sfm/sparse/0/`, `intrinsics/`, `extrinsics/`

## What this does

Multi-View Stereo (MVS) uses the fixed camera poses from SfM and runs dense depth estimation across all 4 views for every synchronized frame. For each frame-set (one image per camera at time T), COLMAP's PatchMatch algorithm estimates depth at every pixel, then fuses all 4 depth maps into a single dense point cloud — typically millions of points.

This is how the pipeline handles **dynamic content**: the camera poses are fixed (from the static SfM pass) but the depth estimation runs independently per frame, so moving objects are reconstructed correctly at each moment in time.

## How to run

1. Before running, organize your synchronized dynamic footage frames into `mvs/frames/`. Structure:
   ```
   mvs/frames/
   ├── 000001/
   │   ├── top_left.jpg
   │   ├── top_right.jpg
   │   ├── bot_left.jpg
   │   └── bot_right.jpg
   ├── 000002/
   │   └── ...
   ```
   Frame numbers should be zero-padded to 6 digits. Each folder is one synchronized moment in time. Use the `dynamic_scene.py` helper below to record this directly, or export frames yourself with matching names.
2. Run: `python mvs.py`
3. For each frame folder, the script prepares a COLMAP dense workspace and runs PatchMatch stereo and fusion.
4. Output: `mvs/frame_XXXXXX/fused.ply` — one PLY point cloud per frame.

## Helper — `dynamic_scene.py`

Records synchronized frames from all 4 cameras over time and writes them straight into `mvs/frames/{frame:06d}/{position}.jpg`, so the output drops directly into Stage 4. Unlike the static helper, this captures a moving scene as a time sequence; each loop iteration grabs all 4 cameras before decoding to keep the views as close to simultaneous as possible.

**Depends on:** `camera.json`

Controls (interactive mode):
- `R` / `SPACE` — start / stop recording (saves a set every `1/fps` while recording)
- `C` — capture a single synchronized set (one moment)
- `Q` / `ESC` — finish and exit

Run:
```bash
python dynamic_scene.py                  # interactive
python dynamic_scene.py --fps 15         # target 15 sets/sec while recording
python dynamic_scene.py --duration 5     # auto-record 5 seconds, then quit
python dynamic_scene.py --fresh          # wipe mvs/frames/ before capturing
```

Capture tips:
- Keep the **rig fixed** — do not move the cameras. The poses were solved once in Stage 3 and are reused for every frame; only the scene should move.
- Frame numbering continues after any folders already in `mvs/frames/` (use `--fresh` to start over).
- For a first validation, record a short clip and process only a few frames: `python mvs.py --start_frame 1 --end_frame 5`.

## Implementation notes for Cursor

**Fixed-rig reconciliation (important).** The naive plan — "copy `sfm/sparse/0/` straight into each frame workspace" — assumes the SfM model holds exactly one image per camera. In practice the Stage-3 static capture produces several frames per camera (e.g. ~10), so `sfm/sparse/0/` contains ~40 images. COLMAP's `image_undistorter` requires every image in the model to exist on disk, so a 40-image model cannot be reused against the 4 images of a dynamic frame.

Instead, `mvs.py` builds a **reusable fixed-rig model once** (`mvs/_rig/sparse/`):
- 4 cameras (one per position) using the **OPENCV** model with the calibrated `fx, fy, cx, cy, k1, k2, p1, p2`, so `image_undistorter` genuinely undistorts the **raw** dynamic frames.
- 4 images named `{position}.jpg`, each carrying the single best-constrained pose for that position, lifted from the refined SfM model (the image with the most 3D-point observations).
- An empty `points3D.txt`; the scene depth range is instead estimated from the static SfM points and passed explicitly to PatchMatch (`--PatchMatchStereo.depth_min/max`).

This honors the README's intent (fixed poses reused for every frame) while working with the real multi-frame static capture.

**Per-frame flow:**
- Iterate `mvs/frames/` in sorted order. Each `mvs/frames/{n:06d}/` holds `{position}.jpg` for all 4 positions (raw, distorted footage straight from the cameras).
- Create a workspace at `mvs/frame_{n:06d}/`, copy the 4 raw views into `images/`, and copy the fixed-rig model into `sparse/`.
- Run the COLMAP commands per frame:
  ```bash
  colmap image_undistorter \
    --image_path mvs/frame_{n:06d}/images \
    --input_path mvs/frame_{n:06d}/sparse \
    --output_path mvs/frame_{n:06d}/dense \
    --output_type COLMAP --max_image_size 640

  colmap patch_match_stereo \
    --workspace_path mvs/frame_{n:06d}/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.max_image_size 640 \
    --PatchMatchStereo.geom_consistency 1 \
    --PatchMatchStereo.depth_min <d> --PatchMatchStereo.depth_max <D>

  colmap stereo_fusion \
    --workspace_path mvs/frame_{n:06d}/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path mvs/frame_{n:06d}/fused.ply
  ```
- After all frames, print a summary: frames processed, average/min/max point count, any frames that failed (a single frame failing does not abort the run).
- **GPU note:** `patch_match_stereo` requires a CUDA GPU — there is no CPU fallback in COLMAP.
- **Performance note:** MVS is slow — plan for several minutes per frame. For a first test run, process only 5–10 frames (`--start_frame` / `--end_frame`) to validate the pipeline before the full sequence.
- CLI: `--start_frame`, `--end_frame`, `--max_image_size` (default 640), `--fresh` (wipe `mvs/frame_*`), `--colmap`, `--no-gpu`.

---

---

# Stage 5 — SC-GS (Sparse-Controlled Gaussian Splatting)

**File:** `scgs.py`
**Output folder:** `scgs/`
**Depends on:** `mvs/` (all `fused.ply` files), `sfm/sparse/0/`, `camera.json`

## What SC-GS is and why we use it

At this point you have a sequence of dense point clouds — one per frame — each representing the scene geometry at a moment in time. SC-GS turns this sequence into a **single learnable dynamic scene representation** that can be rendered from any novel viewpoint at any time.

SC-GS works in two layers:

**Control points (sparse skeleton).** A small set of 3D nodes (typically 512–2048) are placed throughout the scene. These act like a skeleton or cage — they define how the scene deforms over time. Each control point learns a trajectory through time.

**Dense Gaussians (the actual render primitives).** Millions of 3D Gaussian ellipsoids cover every surface. Each Gaussian is "attached" to nearby control points via skinning weights. When a control point moves, all the Gaussians bound to it move with it using ARAP (As-Rigid-As-Possible) deformation — a physics-inspired constraint that prevents surfaces from tearing or stretching unrealistically.

At render time, you pick a timestamp, the control points deform the Gaussians to the right positions, and the Gaussians are splatted (rasterized) into an image from your chosen viewpoint. This runs in real time.

**Why not plain 4DGS?** 4DGS stores separate Gaussians for every moment in time, which balloons storage to 6GB+ for a 10-second clip. SC-GS stores one canonical set of Gaussians plus a compact deformation trajectory — far more VR-friendly.

## Setup — clone SC-GS

SC-GS is an external repository that must be cloned alongside this project:

```bash
cd quasar/
git clone https://github.com/yihua7/SC-GS --recursive
cd SC-GS
pip install -r requirements.txt
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

`scgs.py` will call into the SC-GS codebase via subprocess. Do not import SC-GS modules directly — invoke it through its CLI.

## Data preparation

SC-GS expects its input in a format similar to NeRF/3DGS: a folder of images plus a `transforms.json` (or COLMAP-format `sparse/` folder) describing camera poses.

`scgs.py` prepares this from your MVS output:

1. Select a representative subset of frames (e.g. every 3rd frame) to keep training tractable.
2. For each selected frame, copy the 4 camera images into `scgs/input/images/` with names `{frame:06d}_{position}.jpg`.
3. Copy `sfm/sparse/0/` into `scgs/input/sparse/` as the camera pose reference.
4. The MVS point clouds (`fused.ply`) are merged into a single initialization cloud at `scgs/input/init_cloud.ply` using Open3D voxel downsampling (1cm voxels). This is passed to SC-GS as the initial Gaussian positions — dramatically speeding up training versus random initialization.

## Dynamic mask generation

SC-GS needs to know which regions of each image are moving (dynamic) vs. static background. This tells it where to spend control point budget.

Use background subtraction to generate these masks:

1. Compute a median background image from the static SfM frames (one per camera).
2. For each dynamic frame, subtract the background and threshold to produce a binary mask.
3. Save masks to `scgs/input/masks/{frame:06d}_{position}.png`.
4. Pass `--gt_alpha_mask_as_dynamic_mask` to SC-GS training so it reads these masks.

If background subtraction produces noisy masks, Open3D's `remove_statistical_outlier` on the point clouds is a useful secondary filter.

## Training

`scgs.py` prepares the data and then launches SC-GS training via subprocess:

```bash
python ../SC-GS/train_gui.py \
  --source_path scgs/input \
  --model_path scgs/output \
  --deform_type node \
  --node_num 512 \
  --gt_alpha_mask_as_dynamic_mask \
  --gs_with_motion_mask \
  --init_isotropic_gs_with_all_colmap_pcl \
  --W 640 \
  --H 480
```

**Training has two phases that happen automatically:**
- Phase 1 (~10,000 steps): control points pre-train on the scene geometry. Takes ~20–40 minutes.
- Phase 2 (~30,000 steps): dense Gaussians train jointly with the deformation field. Takes 1–3 hours.

A GUI window opens during training showing the current reconstruction quality. You can monitor progress live.

## Output and VR export

Trained output lands in `scgs/output/`. The key file is `point_cloud/iteration_30000/point_cloud.ply` — the final Gaussian scene.

To render novel views or export for VR:

```bash
python ../SC-GS/render.py \
  --model_path scgs/output \
  --skip_train \
  --render_traj
```

Renders save to `scgs/renders/`. For VR playback, the rendered frames (left-eye and right-eye views at slightly offset camera positions) can be encoded as a side-by-side stereoscopic video using ffmpeg:

```bash
ffmpeg -framerate 30 -i scgs/renders/frame_%06d.png \
  -c:v libx265 -crf 18 -tag:v hvc1 \
  output_vr.mp4
```

## Implementation notes for Cursor

- `scgs.py` is a **preparation and launch script** — it does not reimplement SC-GS. Its job is: load MVS outputs → prepare input directory → generate masks → invoke SC-GS training → report completion.
- Add a `--frames_step` CLI argument (default 3) to control how many frames are skipped between selected training frames.
- Add a `--node_num` CLI argument (default 512) to control SC-GS control point count. Increase to 1024 for scenes with complex independent motion (multiple people).
- After training completes, print the path to the output PLY and the render command.
- If the SC-GS repo is not found at `../SC-GS`, print a clear error with the clone instructions above rather than a Python traceback.

---

## Running the full pipeline

```bash
# One time setup
python intrinsics.py        # collect checkerboard images, calibrate, save K_*.txt

# Once per capture session
python extrinsics.py        # sync all cameras, calibrate relative poses, save K.txt

# Once per static scene (before dynamic capture)
python sfm.py               # run COLMAP SfM on static frames, save sparse model

# Once per dynamic capture
python mvs.py               # run COLMAP MVS per frame, save fused.ply sequence

# Train and render
python scgs.py              # prepare SC-GS input, train, export VR video
```

---

## Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Intrinsic reprojection error > 1.0px | Poor checkerboard images | Recollect — vary angles more, avoid motion blur |
| Extrinsic translation magnitude wrong | Wrong camera.json indices | Swap device indices in camera.json |
| COLMAP registers < 3 cameras | Insufficient feature overlap | Widen camera FOV or add more static frames |
| MVS produces sparse or holey clouds | Low texture in scene | Add texture to scene or use stronger lighting |
| SC-GS training loss doesn't decrease | Bad initialization cloud | Check that init_cloud.ply covers the scene; increase node_num |
| VR video has seams between eyes | Stereo baseline too large | Reduce inter-ocular distance in render script |
