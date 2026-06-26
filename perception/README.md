# quasar/perception — Project README

> **For Cursor:** This is a completely fresh project. Do NOT reference or import anything from `quasar/coled`. You may look at `quasar/coled` for context on what has already been attempted, but all code here must be written from scratch in `quasar/perception`. Start clean.

---

## Project Goal

Build a pipeline that takes multi-camera footage from a calibrated 2×2 camera array and evaluates three different scene reconstruction methods — Neural Point-Based Graphics++ (NPBG++), 3D Gaussian Splatting (3DGS), and Sparse-Controlled Gaussian Splatting (SC-GS) — to determine which produces the best results given the constraints of this rig.

The pipeline splits into two research tracks after MVS:

**Track A — Static novel-view synthesis (NPBG++ vs 3DGS).** A single synchronized frame from all 4 cameras is reconstructed using both methods. The two methods are compared by rendering novel views from each and **visually inspecting the outputs side by side** — there is no automated metrics script; comparison is done manually from the rendered images.

**Track B — Dynamic scene reconstruction (SC-GS).** A multi-frame synchronized sequence is reconstructed using SC-GS to model how the scene deforms over time. Findings are reported separately since this is a different problem category from Track A.

> **Why separate tracks?** NPBG++ and 3DGS are static methods — they reconstruct one moment in time. SC-GS is a dynamic method — it requires a sequence of frames to learn motion. Comparing all three on the same input would be an unfair and misleading benchmark. The research value is in understanding which static method suits this rig, and separately whether SC-GS can produce usable dynamic reconstructions from the 4 perfectly-synchronized cameras of this rig.

---

## Project Structure

```
quasar/
└── perception/
    ├── README.md                  ← this file
    ├── camera.json                ← camera position mapping (edit manually)
    ├── common.py                  ← shared helpers imported by all stages
    ├── intrinsics.py              ← Stage 1
    ├── extrinsics.py              ← Stage 2
    ├── rig.py                     ← Stage 3: build COLMAP rig model from calibration
    ├── static_scene.py            ← capture helper for Track A (one synced frame)
    ├── dynamic_scene.py           ← capture helper for Track B (multi-frame sequence)
    ├── mvs.py                     ← Stage 4: dense point cloud (static/dynamic modes)
    ├── npbgpp.py                  ← Track A-1: Neural Point-Based Graphics++ (NPBG++)
    ├── gs3d.py                    ← Track A-2: 3D Gaussian Splatting
    ├── scgs.py                    ← Track B: SC-GS dynamic reconstruction
    ├── intrinsics/
    │   ├── img_1_1.jpg
    │   ├── img_1_2.jpg
    │   ├── ...
    │   ├── img_4_10.jpg
    │   ├── K_1.txt
    │   ├── K_2.txt
    │   ├── K_3.txt
    │   ├── K_4.txt
    │   ├── dist_1.txt
    │   ├── dist_2.txt
    │   ├── dist_3.txt
    │   ├── dist_4.txt
    │   └── calibration_report.txt ← per-camera fx/fy/cx/cy, reproj, pass/fail gate
    ├── extrinsics/
    │   ├── K.txt                  ← human-readable R, t + stereo RMS per camera
    │   ├── poses.npz              ← machine-readable R, t per camera
    │   ├── calibration_report.txt ← per-pair RMS, sets used, pass/fail gate
    │   ├── top_left/
    │   │   ├── img_1.jpg
    │   │   └── ...
    │   ├── top_right/
    │   ├── bot_left/
    │   └── bot_right/
    ├── rig/
    │   └── sparse/                ← COLMAP-format model built from calibration (no SfM)
    │       ├── cameras.txt
    │       ├── images.txt
    │       └── points3D.txt       ← empty; sparse points + depth range come from point_triangulator at MVS time
    ├── mvs/
    │   ├── static/                ← INPUT for Track A: one synced frame-set
    │   │   ├── top_left.jpg
    │   │   ├── top_right.jpg
    │   │   ├── bot_left.jpg
    │   │   └── bot_right.jpg
    │   ├── frames/                ← INPUT for Track B: multi-frame sequence
    │   │   └── XXXXXX/
    │   │       ├── top_left.jpg
    │   │       ├── top_right.jpg
    │   │       ├── bot_left.jpg
    │   │       └── bot_right.jpg
    │   ├── static_fused.ply       ← dense point cloud for Track A
    │   └── frame_XXXXXX/          ← per-frame dense workspace for Track B
    │       ├── images/
    │       ├── dense/
    │       └── fused.ply
    ├── track_a/
    │   ├── npbgpp/
    │   │   ├── output/            ← trained NPBG++ descriptors + network
    │   │   └── renders/           ← novel view PNGs (inspect manually)
    │   └── gs3d/
    │       ├── output/            ← trained 3DGS scene (.ply)
    │       └── renders/           ← novel view PNGs (inspect manually)
    └── track_b/
        └── scgs/
            ├── input/
            ├── output/            ← trained SC-GS scene + deformation checkpoint
            └── renders/
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

## common.py — shared helpers

`common.py` holds the small utilities that more than one stage needs, so they live in exactly one place instead of being copy-pasted (or cross-imported between stage scripts). Every stage (`rig.py`, `mvs.py`, the Track scripts) imports from here. It has no side effects on import and no `main()`.

The helpers it exposes:

- **`POSITION_ORDER`** — the canonical camera ordering `["top_left", "top_right", "bot_left", "bot_right"]`. Use this everywhere so camera IDs and tile layouts stay consistent.
- **Path constants** — absolute paths to the project subfolders (`INTRINSICS_DIR`, `EXTRINSICS_DIR`, `RIG_DIR`, `RIG_SPARSE_DIR`, `MVS_DIR`, …) derived from the location of `common.py`, so scripts work regardless of the current working directory.
- **`IMAGE_EXTS`** — accepted image extensions `(".jpg", ".jpeg", ".png")`.
- **`load_camera_indices()`** — reads `camera.json` and returns an ordered list of `(position, device_index)` for the four positions, validating that all keys are present.
- **`load_intrinsics(camera_number)`** — loads `intrinsics/K_{n}.txt` (3×3) and `intrinsics/dist_{n}.txt` for one camera; raises a clear error if Stage 1 hasn't been run.
- **`load_poses()`** — loads `extrinsics/poses.npz` and returns `{position: (R, t)}` in the **world-to-camera** convention (`top_left` is identity / zero).
- **`find_colmap(explicit=None)`** — resolves the `colmap` binary from an explicit path or `PATH`, with install guidance on failure.
- **`run(cmd)`** — runs a subprocess command (used for all COLMAP CLI calls), streaming output and raising on a non-zero exit code.
- **`quat_to_rot(qw, qx, qy, qz)`** / **`rot_to_quat(R)`** — convert between rotation matrices and COLMAP-order quaternions `[w, x, y, z]`.
- **`camera_center(R, t)`** — world-space camera center `C = -R.T @ t` for a world-to-camera pose.
- **`count_ply_points(path)`** — reads the vertex count from a PLY header (for run summaries).

---

## Dependencies

Install all dependencies before running any stage:

```bash
pip install opencv-python numpy open3d torch torchvision
```

COLMAP must also be installed as a system binary and accessible on PATH:

```bash
# Ubuntu
sudo apt install colmap

# or build from source: https://colmap.github.io/install.html
```

Each reconstruction method requires cloning its own external repository — see the relevant stage sections below for setup instructions.

```bash
# Track A-1: NPBG++
git clone https://github.com/rakhimovv/npbgpp
cd npbgpp && pip install -r requirements.txt && cd ..

# Track A-2: 3D Gaussian Splatting
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install -r requirements.txt
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
cd ..

# Track B: SC-GS
git clone https://github.com/yihua7/SC-GS --recursive
cd SC-GS
pip install -r requirements.txt
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
cd ..
```

---

---

# Stage 1 — Intrinsic Calibration

**File:** `intrinsics.py`
**Output folder:** `intrinsics/`

## What this does

Intrinsic calibration finds each camera's internal optical properties — focal length, principal point (optical center), and lens distortion coefficients. These are bundled into a 3×3 matrix called **K** (the camera matrix). Every downstream stage (extrinsics, COLMAP, SC-GS) depends on accurate intrinsics.

This is done using a checkerboard pattern. OpenCV detects the corners of the checkerboard in multiple images taken from different angles, then solves for K using the known physical geometry of the checkerboard squares.

> **Why this stage is critical.** A bad `K` — especially an off-center principal point `(cx, cy)` — silently breaks MVS later: with the optical center wrong, every back-projected ray is mis-aimed, so triangulating the scene produces reprojection errors far above COLMAP's filter and dense fusion yields **0 points**. A flat checkerboard at limited depths can still produce a low *extrinsic* RMS with a bad `K`, so the error hides until MVS. For a 640×480 camera the principal point must land near **(320, 240)**; values like `cy=108` or `cy=397` mean the calibration is bad and must be redone.

## How to run

1. Print or display a checkerboard pattern. The default assumes an **8×6 inner corner** grid (i.e. 9×7 squares). If yours differs, edit the `CHECKERBOARD` constant at the top of the file. The same grid must be used in Stage 2 (`extrinsics.py` also uses `8×6`).
2. Run the script: `python intrinsics.py`
3. The script processes one camera at a time. For each camera, a live preview window opens with a coverage HUD.
4. Hold the checkerboard in front of the camera and move it to different positions, angles, and distances. When the script detects a valid checkerboard, it captures automatically. **Get the board into all four image corners/edges and tilt it 30–45° (pitch & yaw) at both near and far distances** — this is what constrains the principal point and distortion. Flat-on, center-only views are exactly what produces a bad `K`.
5. `TARGET_IMAGES` views are collected per camera (with coverage gating), then calibration runs immediately.
6. Repeat for all 4 cameras.

## Quality gate

After each camera's solve, a gate decides whether the calibration is good enough to save:

- It checks: principal point within `PRINCIPAL_POINT_TOLERANCE` (default **15%**) of center, plausible FOV/focal, `fx ≈ fy`, non-extreme distortion, and reprojection error ≤ `REPROJ_FAIL_PX` (default **1.0 px**).
- If a camera **FAIL**s, its `K_{n}.txt` / `dist_{n}.txt` are **not saved** — recollect that camera (more corner/edge coverage, more tilt) and re-run. Pass `--force` to save anyway.
- A summary and `intrinsics/calibration_report.txt` (per-camera fx/fy/cx/cy, reprojection, verdict, issues) are written at the end.

```
python intrinsics.py            # normal run (gate enforced)
python intrinsics.py --fresh    # clear intrinsics/ (K_*, dist_*, img_*, report) first
python intrinsics.py --force    # save even if the gate FAILs
```

## Implementation notes for Cursor

- Resolution: **640×480** for all cameras throughout this project.
- Camera indices come from `camera.json`. Iterate in order: `top_left`, `top_right`, `bot_left`, `bot_right`.
- Image naming: `img_{camera_number}_{photo_number}.jpg` — e.g. `img_1_4.jpg`. Camera number is the integer value from `camera.json`, not the position key.
- Images save to `intrinsics/`.
- After collecting `TARGET_IMAGES` images for a camera, immediately run `cv2.calibrateCamera()` (with `CALIB_FIX_K3`) on those images.
- Save the intrinsic matrix K as `intrinsics/K_{camera_number}.txt` using `numpy.savetxt` (space-delimited, 6 decimals) **only if the camera passes the gate** (or `--force`). Save distortion coefficients alongside as `dist_{camera_number}.txt`.
- Reprojection error, principal-point offset, FOV/`fx≈fy`, and distortion sanity feed the gate; a `FAIL` blocks the save and prints why.
- Do not auto-advance to the next camera. Print a prompt and wait for the user to press Enter before opening the next camera.

---

---

# Stage 2 — Extrinsic Calibration

**File:** `extrinsics.py`
**Output folder:** `extrinsics/`
**Depends on:** `camera.json`, `intrinsics/K_*.txt`, `intrinsics/dist_*.txt`

## What this does

Extrinsic calibration finds the **position and orientation of each camera relative to the others** — specifically, a rotation matrix R and translation vector t for each camera expressed relative to `top_left`, which is treated as the world origin.

This is done by showing the checkerboard to all 4 cameras simultaneously and using the known intrinsics to solve each non-reference camera's pose with `cv2.stereoCalibrate` (intrinsics held fixed) against `top_left`.

> **Accuracy matters a lot here.** These poses are used *directly* by `mvs.py` for fixed-pose triangulation. If a camera's relative pose is off by even a few degrees, every triangulated point exceeds COLMAP's reprojection filter and dense fusion produces **0 points**. To catch this, Stage 2 reports a **stereo RMS** per camera pair and applies a **quality gate** (see below) that refuses to save poses that are too inaccurate to use.

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
6. Press `Q` when done collecting. Calibration runs automatically and prints the quality gate result.

### Capture technique (this is what determines accuracy)

The cameras are **not hardware-synchronized** — they are read one after another in each loop iteration. If the board is moving when a set is grabbed, each camera sees it in a slightly different place, which corrupts the solve. To get a low RMS:

- **Hold the board completely still** at each pose; let it settle, then let the auto-capture fire. Move only *between* captures.
- **Fill more of the frame** and include **strong tilts** (not just flat/fronto-parallel) — tilt provides the depth constraint.
- Vary position across the cameras' shared overlap; avoid glare and motion blur.
- If one camera consistently fails the gate, give that camera extra well-tilted, board-still views.

## Quality gate and best-subset selection

- Each camera pair is first solved over all sets, then the solver **greedily drops the worst-reprojecting sets** (largest error first), re-solving until the RMS reaches the target or only `MIN_KEEP_SETS` remain. This rejects motion-corrupted/misdetected sets automatically.
- Each pair gets a verdict from its final stereo RMS:
  - `OK`   — RMS ≤ `RMS_TARGET_PX` (default **0.6 px**)
  - `WARN` — RMS ≤ `RMS_FAIL_PX` (default **1.0 px**)
  - `FAIL` — RMS > `RMS_FAIL_PX`
- **Gate:** if any pair is `FAIL`, `poses.npz` is **not written** (so MVS can't silently consume bad poses). Re-capture and re-run. Pass `--force` to save anyway.
- Thresholds and `MIN_KEEP_SETS` are constants at the top of `extrinsics.py`.

```
python extrinsics.py            # normal run (gate enforced)
python extrinsics.py --fresh    # clear extrinsics/ (old captures + outputs) first
python extrinsics.py --force    # save even if the gate FAILs
```

`--fresh` deletes the per-position image subfolders and the stale `K.txt` / `poses.npz` / `calibration_report.txt` before capturing, so a new run never mixes with old image-sets.

## Implementation notes for Cursor

- Load `camera.json` to get device indices and position labels.
- Load `intrinsics/K_{n}.txt` and `intrinsics/dist_{n}.txt` for each camera before opening streams.
- Resolution: **640×480**.
- Detection: use `cv2.findChessboardCorners()` followed by `cv2.cornerSubPix()` for refinement. Only capture when corners are found in **all 4 cameras in the same loop iteration**.
- Image saving: each camera gets its own subfolder matching its position key.
  - `extrinsics/top_left/img_1.jpg`, `extrinsics/top_left/img_2.jpg`, etc.
  - `extrinsics/top_right/img_1.jpg`, etc.
  - Image numbers are shared across cameras (image 3 from all cameras is the same moment in time).
- Extrinsic solve: use `cv2.stereoCalibrate()` with `CALIB_FIX_INTRINSIC` (each camera's pre-calibrated K/dist held fixed) to get R and t for each camera relative to `top_left`. `top_left` is the reference frame — its R is identity, t is zero. The returned `R, T` satisfy `X_cam = R · X_ref + T` (world-to-camera, since the reference is the world origin).
- Per-set filtering: rank sets by mapping the reference board pose (`solvePnP`) into the other camera and reprojecting; drop the largest-error sets first.
- Save output to `extrinsics/K.txt`. For each camera position, write its rotation matrix R (3×3), translation vector t (3×1), and a `# stereo RMS` comment line, e.g.:
  ```
  top_left
  # stereo RMS: 0.0000 px [OK] (used 40/40 sets)
  R: [[1,0,0],[0,1,0],[0,0,1]]
  t: [0,0,0]

  top_right
  # stereo RMS: 0.55 px [OK] (used 18/40 sets)
  R: ...
  t: ...
  ```
- Also save as `extrinsics/poses.npz` using `numpy.savez` with keys `R_top_left`, `t_top_left`, etc. for easy loading downstream (only written if the gate passes or `--force`).
- Write `extrinsics/calibration_report.txt` with the per-pair RMS, sets used, verdicts, and overall PASS/FAIL.
- Print the translation magnitudes between camera pairs.

---

---

---

---

# Stage 3 — Rig Model (replaces SfM)

**File:** `rig.py`
**Output folder:** `rig/`
**Depends on:** `camera.json`, `intrinsics/K_*.txt`, `intrinsics/dist_*.txt`, `extrinsics/poses.npz`

## What this does and why SfM is not needed

The original plan used COLMAP's Structure from Motion (SfM) to recover camera poses from images. SfM exists to solve for poses when you *don't* know where your cameras are. But you already know — extrinsic calibration gave you a precise rotation matrix R and translation vector t for every camera relative to `top_left`. Running SfM on top of that would be redundant and would introduce small optimisation drift away from your physically measured poses.

`rig.py` instead writes a valid COLMAP-format sparse model **directly from your calibration data** — no image matching, no feature extraction, no optimisation. COLMAP's MVS pipeline (`image_undistorter`, `patch_match_stereo`, `stereo_fusion`) only needs a sparse model for the camera parameters and poses, not for the SfM reconstruction itself. So this is a complete drop-in replacement.

## What it writes

COLMAP's text-format sparse model consists of three files:

**`rig/sparse/cameras.txt`** — one entry per camera, containing the camera model and intrinsic parameters. Uses the `OPENCV` model so distortion coefficients are passed through and `image_undistorter` can genuinely undistort the raw frames.

```
# Camera list with one line of data per camera:
# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 OPENCV 640 480 fx fy cx cy k1 k2 p1 p2
2 OPENCV 640 480 fx fy cx cy k1 k2 p1 p2
3 OPENCV 640 480 fx fy cx cy k1 k2 p1 p2
4 OPENCV 640 480 fx fy cx cy k1 k2 p1 p2
```

**`rig/sparse/images.txt`** — one entry per camera, containing its pose as a quaternion + translation. COLMAP stores poses as **world-to-camera** transforms, and the extrinsic calibration in `extrinsics/poses.npz` is **already world-to-camera** (`top_left` is identity / zero, the world origin). So `rig.py` writes `R` and `t` straight through — no inversion — it only has to convert the rotation matrix to a quaternion.

**`rig/sparse/points3D.txt`** — intentionally empty. There are no SfM triangulated points in the rig model itself. A real sparse cloud (and the PatchMatch depth range) is produced at MVS time by `colmap point_triangulator`, which triangulates feature matches **using these fixed poses without optimizing them**.

## How to run

```bash
python rig.py
```

No interactive steps. Reads calibration files, writes the three model files, prints the camera centers and inter-camera distances for a visual sanity-check.

## Implementation notes for Cursor

- Use the shared helpers from `common.py` (`POSITION_ORDER`, `load_camera_indices`, `load_intrinsics`, `load_poses`, `rot_to_quat`, `camera_center`, path constants).
- Read `intrinsics/K_{n}.txt` and `intrinsics/dist_{n}.txt` for each camera (n is the integer device index from `camera.json`).
- Read `extrinsics/poses.npz` (via `load_poses()`) for `R_{position}` and `t_{position}` per camera.
- `top_left` is the world origin: its R is identity, t is zero.
- The poses in `poses.npz` are **already world-to-camera**, which is exactly what COLMAP wants. Write `R` and `t` **directly** — do NOT invert them (no `R.T`, no `-R.T @ t`).
- Convert each R to a COLMAP quaternion `[w, x, y, z]` for `images.txt` via `rot_to_quat` (scipy returns `[x, y, z, w]`, so reorder).
- Use the `OPENCV` camera model in `cameras.txt` with params `fx fy cx cy k1 k2 p1 p2` from the calibration. Camera IDs are `1..4` in `POSITION_ORDER` order.
- Image names in `images.txt` are `{position}.jpg` (e.g. `top_left.jpg`) — these must match the filenames `mvs.py` places in each frame workspace.
- Write `points3D.txt` as an empty file with only the header comment line (points are triangulated later by `mvs.py`).
- **Print the camera centers** `C = -R.T @ t` for all 4 cameras, plus the inter-camera distances (in mm, since calibration was done in mm), so the user can verify the 2×2 layout is physically correct — e.g. confirm `top_left`→`top_right` ≈ 180 mm and that the centers form a plausible square — before running MVS.

---

---

---

---

# Stage 4 — MVS (Dense Point Cloud)

**File:** `mvs.py`
**Output folder:** `mvs/`
**Depends on:** `rig/sparse/`, `intrinsics/`, `extrinsics/`

## What this does

Multi-View Stereo (MVS) uses the fixed camera poses from `rig.py` and runs dense depth estimation across all 4 views. COLMAP's PatchMatch algorithm estimates depth at every pixel for each view, then fuses all 4 depth maps into a single dense point cloud.

`mvs.py` handles two modes depending on which track you are running:

**Track A (static):** Processes `mvs/static/` — a single folder of 4 images (one per camera, one moment in time). Output is `mvs/static_fused.ply`.

**Track B (dynamic):** Processes `mvs/frames/` — a sequence of frame-folders, each holding 4 images. Output is `mvs/frame_XXXXXX/fused.ply` per frame.

The camera poses are identical in both modes — the rig model is fixed and reused for every frame.

## Capture helpers

### `static_scene.py` — for Track A

Opens all 4 cameras, shows a tiled 2×2 preview, and on keypress saves one synchronized image per camera into `mvs/static/` with filenames `{position}.jpg`.

Controls:
- `SPACE` / `C` — capture and save, then exit
- `Q` / `ESC` — exit without saving

```bash
python static_scene.py
```

### `dynamic_scene.py` — for Track B

Records synchronized frames from all 4 cameras over time into `mvs/frames/{frame:06d}/{position}.jpg`.

Controls:
- `R` / `SPACE` — start / stop recording
- `C` — capture a single synchronized set
- `Q` / `ESC` — exit

```bash
python dynamic_scene.py                  # interactive
python dynamic_scene.py --fps 15         # target 15 sets/sec while recording
python dynamic_scene.py --duration 5     # auto-record 5 seconds, then quit
python dynamic_scene.py --fresh          # wipe mvs/frames/ before capturing
```

## How to run

```bash
python mvs.py --mode static              # Track A: process mvs/static/
python mvs.py --mode dynamic             # Track B: process all mvs/frames/
python mvs.py --mode dynamic --start_frame 1 --end_frame 5   # subset for testing
```

## Depth range via `point_triangulator`

The rig model has the fixed poses but no 3D points, so PatchMatch has no idea how near/far the scene is. Rather than guessing from the camera baseline, `mvs.py` triangulates a **real** sparse cloud using the known rig poses and derives the depth range from it. Because the poses are fixed (`point_triangulator` never moves them), this stays faithful to the calibration.

This runs **once** per `mvs.py` invocation against a reference image-set (the 4 static views in `--mode static`, or the first complete frame in `--mode dynamic`) and the resulting `[depth_min, depth_max]` is reused for every frame, since the rig and scene scale are fixed:

```bash
# 1. features on the 4 reference views (one camera per position, OPENCV intrinsics)
colmap feature_extractor --database_path <db> --image_path <ref_images> ...
# 2. match them
colmap exhaustive_matcher --database_path <db> ...
# 3. triangulate using the FIXED rig poses (poses are not optimized)
colmap point_triangulator \
  --database_path <db> \
  --image_path <ref_images> \
  --input_path  <rig_model_with_db_matched_ids> \
  --output_path <triangulated_model>
```

`mvs.py` then projects the triangulated points into each camera, takes robust percentiles (≈1st / 99th) of the forward-Z (camera depth), pads the range, and passes that to PatchMatch. `--depth_min` / `--depth_max` CLI flags override the auto-estimate if needed.

> **ID-matching note for Cursor:** `point_triangulator` matches the input model against the database **by image name**, and the model's `IMAGE_ID` / `CAMERA_ID` must equal the IDs that `feature_extractor` wrote into the database. So after extraction+matching, read the SQLite database (`images` and `cameras` tables) to get the real IDs, then write the triangulation input model (`cameras.txt` / `images.txt` with the fixed rig poses under those IDs, empty `points3D.txt`). The clean `rig/sparse/` model (IDs 1..4) is still used as-is for `image_undistorter`, which reads images by name and needs no database.

## Implementation notes for Cursor

- Use the shared helpers from `common.py` (`POSITION_ORDER`, `load_camera_indices`, `load_intrinsics`, `find_colmap`, `run`, `count_ply_points`, path constants). `mvs.py` must **not** import from `sfm.py` (it no longer exists).
- `--mode static` reads `mvs/static/{position}.jpg` and writes `mvs/static_fused.ply`. `--mode dynamic` reads each `mvs/frames/{n:06d}/{position}.jpg` and writes `mvs/frame_{n:06d}/fused.ply`. Both modes share the identical rig poses from `rig/sparse/`.
- The rig sparse model lives at `rig/sparse/` (not `sfm/sparse/0/` — that path no longer exists).
- Estimate the depth range once via `point_triangulator` (see section above); allow `--depth_min` / `--depth_max` to override.
- For each frame/static workspace, copy the 4 images into `.../images/` (named `{position}.jpg`) and copy `rig/sparse/` into `.../sparse/`, then run per workspace:
  ```bash
  colmap image_undistorter \
    --image_path  <workspace>/images \
    --input_path  <workspace>/sparse \
    --output_path <workspace>/dense \
    --output_type COLMAP --max_image_size 640

  colmap patch_match_stereo \
    --workspace_path <workspace>/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.max_image_size 640 \
    --PatchMatchStereo.geom_consistency 1 \
    --PatchMatchStereo.depth_min <d> \
    --PatchMatchStereo.depth_max <D>

  colmap stereo_fusion \
    --workspace_path <workspace>/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path <fused.ply>
  ```
- After processing, print a summary: frames processed, average/min/max point count, any frames that failed.
- **GPU note:** `patch_match_stereo` requires a CUDA GPU — there is no CPU fallback in COLMAP. (`feature_extractor` / matcher can fall back to CPU.)
- CLI: `--mode` (static/dynamic), `--start_frame`, `--end_frame`, `--depth_min`, `--depth_max`, `--fresh`.

---

---

---

---

# Track A-1 — Neural Point-Based Graphics++ (NPBG++)

**File:** `npbgpp.py`
**Output folder:** `track_a/npbgpp/`
**Depends on:** `mvs/static_fused.ply`, `rig/sparse/`, `mvs/static/`
**External repo:** `quasar/npbgpp/` (https://github.com/rakhimovv/npbgpp)

## What NPBG++ does

NPBG++ is the successor to the original Neural Point-Based Graphics. It keeps your raw MVS point cloud as-is and attaches a learnable N-dimensional descriptor to each point — a learned feature vector replacing RGB that encodes local geometry and view-dependent appearance. A convolutional renderer then takes a rasterized projection of those descriptors from any viewpoint and outputs a photorealistic image. Compared to the original NPBG, NPBG++ predicts the descriptors with a network (rather than optimizing them per-scene from scratch), giving faster fitting and better generalization from few views.

The key advantage for this rig is that NPBG++ is **tolerant of noisy and incomplete point clouds**. With only 4 cameras the MVS output will have holes and outliers — the neural renderer learns to fill those gaps rather than failing on them as pure geometric methods would.

## How to run

```bash
python npbgpp.py --mode train     # fit descriptors to your point cloud
python npbgpp.py --mode render    # render novel views after training
```

## Implementation notes for Cursor

- `npbgpp.py` is a preparation and launch script. Its job is to format inputs for the NPBG++ repo and invoke its training and rendering scripts via subprocess.
- Use `common.py` helpers for paths / `run` / COLMAP-pose parsing where useful.
- Load `mvs/static_fused.ply` as the point cloud input.
- Camera poses come from `rig/sparse/` — convert to the format NPBG++ expects (see the npbgpp repo's data loader for the exact convention).
- Training images are the 4 views in `mvs/static/`.
- Output trained descriptors and network checkpoint to `track_a/npbgpp/output/`.
- Render a set of novel views (interpolated camera path around the scene) to `track_a/npbgpp/renders/` for manual visual comparison against 3DGS.
- If the npbgpp repo is not found at `../npbgpp`, print a clear error with the clone instructions.

---

---

# Track A-2 — 3D Gaussian Splatting (3DGS)

**File:** `gs3d.py`
**Output folder:** `track_a/gs3d/`
**Depends on:** `mvs/static_fused.ply`, `rig/sparse/`, `mvs/static/`
**External repo:** `quasar/gaussian-splatting/`

## What 3DGS does

3DGS represents the scene as a cloud of 3D Gaussian ellipsoids — each with a position, shape (anisotropic covariance), opacity, and view-dependent color encoded as spherical harmonics. Starting from your MVS point cloud as initialization, it runs an optimization loop that alternates between:

- **Gradient-based refinement** of each Gaussian's position, shape, opacity, and color
- **Density control** — splitting large Gaussians that cover too much area, pruning transparent ones that contribute nothing

The result is a compact scene representation that renders at over 100fps at 1080p. Unlike NPBG++, the final representation is purely geometric — no neural network is needed at render time, just a fast GPU rasterizer.

The risk for this rig is that 3DGS is less forgiving of sparse initialization. If the MVS point cloud has large holes, the Gaussians may not grow to fill them well. This is exactly what the visual comparison with NPBG++ will reveal.

## How to run

```bash
python gs3d.py --mode train     # optimize Gaussians from MVS init
python gs3d.py --mode render    # render novel views after training
```

## Implementation notes for Cursor

- `gs3d.py` is a preparation and launch script.
- The 3DGS repo expects input in COLMAP format: a folder of images plus a `sparse/` model. Prepare `track_a/gs3d/input/images/` (copy the 4 static views) and `track_a/gs3d/input/sparse/` (copy from `rig/sparse/`).
- Pass the MVS point cloud `mvs/static_fused.ply` as the initialization point cloud. The 3DGS repo accepts this via `--init_pcd` or by placing it in the expected location — check the repo's README for the exact flag.
- Launch training:
  ```bash
  python ../gaussian-splatting/train.py \
    -s track_a/gs3d/input \
    -m track_a/gs3d/output \
    --iterations 30000
  ```
- After training, render a novel-view path:
  ```bash
  python ../gaussian-splatting/render.py \
    -m track_a/gs3d/output \
    --skip_train
  ```
- Copy renders to `track_a/gs3d/renders/` for manual visual comparison against NPBG++.
- If the gaussian-splatting repo is not found at `../gaussian-splatting`, print a clear error with the clone instructions.

---

---

# Track A — Viewing & Comparing Results

There is **no automated comparison script**. NPBG++ and 3DGS are compared by eye, by looking at the novel views each one renders from the same camera path.

## How to compare

1. Run both methods on the same static scene so they share the identical MVS init (`mvs/static_fused.ply`) and rig poses (`rig/sparse/`):
   ```bash
   python npbgpp.py --mode train && python npbgpp.py --mode render
   python gs3d.py  --mode train && python gs3d.py  --mode render
   ```
2. Open the two render folders side by side and inspect the same frames:
   - `track_a/npbgpp/renders/`
   - `track_a/gs3d/renders/`
3. Judge them manually on what matters for this rig: hole-filling on under-observed regions, sharpness/texture, floaters and artifacts around object boundaries, and temporal stability as the camera moves along the path.

> Render both methods over the **same novel-view camera path** so the frames line up one-to-one for an honest visual comparison. With only 4 input views, treat any quantitative score as unreliable — the deliverable here is the qualitative read on which method degrades more gracefully under sparse input.

---

---

# Track B — SC-GS (Dynamic Reconstruction)

**File:** `scgs.py`
**Output folder:** `track_b/scgs/`
**Depends on:** `mvs/frame_*/fused.ply`, `rig/sparse/`, `camera.json`
**External repo:** `quasar/SC-GS/`

## What SC-GS does and why it is a separate track

SC-GS is a **dynamic** scene method. It requires a sequence of frames — not just one — because its purpose is to model how the scene deforms over time. It learns a canonical set of 3D Gaussians (the scene at rest) plus a deformation field that warps those Gaussians to match each frame in the sequence.

This is fundamentally a different problem from Track A. Track A answers: *can we synthesize novel views of one moment?* Track B answers: *can we reconstruct a moving scene from the 4 perfectly-synchronized cameras of this rig and navigate it freely in time?*

**SC-GS works in two layers:**

**Control points (sparse skeleton).** A small set of 3D nodes (typically 512–2048) are placed throughout the scene. These learn trajectories through time. Each control point governs the motion of nearby Gaussians via ARAP (As-Rigid-As-Possible) deformation — a physics-inspired constraint preventing unrealistic tearing or stretching.

**Dense Gaussians (render primitives).** Millions of 3D Gaussian ellipsoids cover every surface. Each is skinned to nearby control points. At render time you pick a timestamp, the control points deform the Gaussians into position, and the scene is rasterized in real time.

## Setup

```bash
cd quasar/
git clone https://github.com/yihua7/SC-GS --recursive
cd SC-GS
pip install -r requirements.txt
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

## Data preparation

`scgs.py` prepares the SC-GS input from your MVS output:

1. Select a subset of frames (every Nth, controlled by `--frames_step`, default 3) to keep training tractable.
2. Copy the 4 camera images for each selected frame into `track_b/scgs/input/images/` named `{frame:06d}_{position}.jpg`.
3. Copy `rig/sparse/` into `track_b/scgs/input/sparse/`.
4. Merge all selected `fused.ply` files into a single initialization cloud at `track_b/scgs/input/init_cloud.ply` using Open3D voxel downsampling (1cm voxels). This dramatically speeds up training vs random initialization.

## Dynamic mask generation

SC-GS needs binary masks indicating which image regions are dynamic (moving) vs static background. Generate these via background subtraction:

1. Compute a median image per camera position across all frames (this approximates the static background).
2. For each frame and camera, subtract the median background and threshold to produce a binary mask.
3. Save to `track_b/scgs/input/masks/{frame:06d}_{position}.png`.
4. Pass `--gt_alpha_mask_as_dynamic_mask` to the training command.

## Training

```bash
python scgs.py --prepare    # prepare input directory and masks
python scgs.py --train      # launch SC-GS training
```

SC-GS training launches via subprocess:

```bash
python ../SC-GS/train_gui.py \
  --source_path track_b/scgs/input \
  --model_path track_b/scgs/output \
  --deform_type node \
  --node_num 512 \
  --gt_alpha_mask_as_dynamic_mask \
  --gs_with_motion_mask \
  --init_isotropic_gs_with_all_colmap_pcl \
  --W 640 \
  --H 480
```

Training runs in two automatic phases:
- **Phase 1 (~10,000 steps):** control points pre-train on the scene geometry. ~20–40 minutes.
- **Phase 2 (~30,000 steps):** dense Gaussians train jointly with the deformation field. ~1–3 hours.

## Output

```
track_b/scgs/output/
├── point_cloud/
│   └── iteration_30000/
│       └── point_cloud.ply     ← canonical Gaussians (rest pose)
├── deformation/
│   └── iteration_30000/        ← deformation field weights (MLP checkpoint)
├── cfg_args                    ← training config needed for rendering
└── cameras.json
```

The PLY alone is not the complete representation — the deformation checkpoint is required to animate it. Both files together define the full dynamic scene.

## Implementation notes for Cursor

- `scgs.py` is a preparation and launch script — it does not reimplement SC-GS internals.
- Add `--frames_step` CLI argument (default 3) and `--node_num` (default 512; increase to 1024 for scenes with multiple independently moving subjects).
- Log training time to `track_b/scgs/metrics.json`.
- If the SC-GS repo is not found at `../SC-GS`, print a clear error with the clone instructions rather than a Python traceback.

---

## Running the full pipeline

```bash
# ── One-time calibration ──────────────────────────────────────────
python intrinsics.py        # Stage 1: calibrate each camera individually
python extrinsics.py        # Stage 2: calibrate relative camera poses
python rig.py               # Stage 3: write COLMAP rig model from calibration

# ── Track A: static novel-view synthesis ─────────────────────────
python static_scene.py        # capture one synced frame from all 4 cameras
python mvs.py --mode static   # dense point cloud for the static frame
python npbgpp.py --mode train && python npbgpp.py --mode render
python gs3d.py   --mode train && python gs3d.py   --mode render
# then inspect track_a/npbgpp/renders/ vs track_a/gs3d/renders/ by eye

# ── Track B: dynamic reconstruction ──────────────────────────────
python dynamic_scene.py     # capture multi-frame sequence
python mvs.py --mode dynamic
python scgs.py --prepare && python scgs.py --train
```

---

## Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Intrinsic reprojection error > 1.0px | Poor checkerboard images | Recollect — vary angles more, avoid motion blur |
| Extrinsic translation magnitude wrong | Wrong camera.json indices | Swap device indices in camera.json |
| `rig.py` poses look wrong | World-to-camera conversion error | Verify R_colmap = R.T and t_colmap = -R.T @ t |
| MVS produces sparse or holey clouds | Low texture or bad lighting | Add surface texture, improve lighting |
| NPBG++ renders are blurry | Too few training views (only 4) | Lower learning rate, train longer, add views |
| 3DGS Gaussians don't converge | Sparse MVS init, large holes | Try denser MVS output or lower `--densification_interval` |
| SC-GS loss doesn't decrease | Bad initialization cloud | Verify init_cloud.ply covers the scene; increase node_num |
| SC-GS dynamic masks are noisy | Background subtraction failing | Manually inspect median background images; adjust threshold |