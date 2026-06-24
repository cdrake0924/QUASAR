"""
Stage 3 — Rig Model (replaces SfM) (quasar/perception)

Writes a valid COLMAP-format sparse model directly from the calibration data —
no image matching, no feature extraction, no pose optimisation. The original
plan used COLMAP SfM to recover poses; extrinsic calibration already gives a
precise world-to-camera R, t for every camera (top_left = world origin), so SfM
would be redundant and would drift away from the measured poses.

COLMAP's MVS pipeline (image_undistorter, patch_match_stereo, stereo_fusion)
only needs the camera parameters and poses from a sparse model, not an SfM
reconstruction, so this is a drop-in replacement.

Writes (TXT) to rig/sparse/:
  cameras.txt   - 4 OPENCV cameras (fx fy cx cy k1 k2 p1 p2), IDs 1..4
  images.txt    - 4 images named {position}.jpg, world-to-camera pose each
  points3D.txt  - empty (points are triangulated later by mvs.py)

The poses in extrinsics/poses.npz are ALREADY world-to-camera (COLMAP's
convention), so R and t are written straight through — no inversion.

Depends on: camera.json, intrinsics/K_*.txt, intrinsics/dist_*.txt,
            extrinsics/poses.npz.

Run:
    python rig.py
"""

import os
import shutil

import numpy as np

from common import (
    POSITION_ORDER,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    RIG_SPARSE_DIR,
    load_camera_indices,
    load_intrinsics,
    load_poses,
    rot_to_quat,
    camera_center,
)


def write_cameras_txt(intrinsics, image_size):
    """Write rig/sparse/cameras.txt with the OPENCV model, IDs 1..4."""
    w, h = image_size
    path = os.path.join(RIG_SPARSE_DIR, "cameras.txt")
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(POSITION_ORDER)}\n")
        for cam_id, position in enumerate(POSITION_ORDER, start=1):
            K, dist = intrinsics[position]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            d = np.asarray(dist).reshape(-1)
            k1 = float(d[0]) if d.size > 0 else 0.0
            k2 = float(d[1]) if d.size > 1 else 0.0
            p1 = float(d[2]) if d.size > 2 else 0.0
            p2 = float(d[3]) if d.size > 3 else 0.0
            f.write(f"{cam_id} OPENCV {w} {h} "
                    f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} "
                    f"{k1:.8f} {k2:.8f} {p1:.8f} {p2:.8f}\n")
    return path


def write_images_txt(poses):
    """
    Write rig/sparse/images.txt. Poses are world-to-camera already, so write
    R (as a quaternion) and t directly. The second (points2D) line per image is
    empty.
    """
    path = os.path.join(RIG_SPARSE_DIR, "images.txt")
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(POSITION_ORDER)}\n")
        for cam_id, position in enumerate(POSITION_ORDER, start=1):
            R, t = poses[position]
            qw, qx, qy, qz = rot_to_quat(R)
            tx, ty, tz = (float(v) for v in np.asarray(t).reshape(3))
            f.write(f"{cam_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{tx:.10f} {ty:.10f} {tz:.10f} {cam_id} {position}.jpg\n")
            f.write("\n")  # no 2D observations
    return path


def write_points3d_txt():
    """Write an empty rig/sparse/points3D.txt (header only)."""
    path = os.path.join(RIG_SPARSE_DIR, "points3D.txt")
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")
    return path


def report(poses):
    """Print camera centers and inter-camera distances for a sanity check."""
    centers = {p: camera_center(R, t) for p, (R, t) in poses.items()}

    print("\n=== Rig geometry (sanity check) ===")
    print("  Camera centers (world frame, top_left = origin):")
    for position in POSITION_ORDER:
        c = centers[position]
        print(f"    {position:10s}: [{c[0]:9.2f}, {c[1]:9.2f}, {c[2]:9.2f}]")

    print("\n  Inter-camera distances (mm):")
    for i in range(len(POSITION_ORDER)):
        for j in range(i + 1, len(POSITION_ORDER)):
            a, b = POSITION_ORDER[i], POSITION_ORDER[j]
            d = float(np.linalg.norm(centers[a] - centers[b]))
            print(f"    {a} <-> {b}: {d:.2f}")
    print("\n  Verify the 2x2 layout looks physically right (adjacent edges "
          "~180 mm, diagonals ~255 mm) before running MVS.")


def main():
    cameras = load_camera_indices()
    intrinsics = {p: load_intrinsics(idx) for p, idx in cameras}
    poses = load_poses()
    image_size = (FRAME_WIDTH, FRAME_HEIGHT)

    print("Stage 3 — Rig model (from calibration, no SfM)")
    print(f"  Output: {RIG_SPARSE_DIR}")
    print(f"  Image size: {image_size[0]}x{image_size[1]}")

    if os.path.isdir(RIG_SPARSE_DIR):
        shutil.rmtree(RIG_SPARSE_DIR)
    os.makedirs(RIG_SPARSE_DIR, exist_ok=True)

    cpath = write_cameras_txt(intrinsics, image_size)
    ipath = write_images_txt(poses)
    ppath = write_points3d_txt()
    print(f"  Wrote {os.path.basename(cpath)}, {os.path.basename(ipath)}, "
          f"{os.path.basename(ppath)}")

    report(poses)
    print(f"\nRig model ready at {RIG_SPARSE_DIR}")


if __name__ == "__main__":
    main()
