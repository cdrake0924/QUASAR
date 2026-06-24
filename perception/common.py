"""
common.py — shared helpers (quasar/perception)

Small utilities needed by more than one stage. Every stage (rig.py, mvs.py, the
Track scripts) imports from here instead of cross-importing each other or
copy-pasting. No side effects on import, no main().

See the "common.py — shared helpers" section of README.md for the full list.
"""

import json
import os
import subprocess

import numpy as np


# --- Canonical ordering ------------------------------------------------------

# Camera order used everywhere: camera IDs, 2x2 tile layout, file naming.
POSITION_ORDER = ["top_left", "top_right", "bot_left", "bot_right"]

# top_left is the world origin (identity R, zero t) by calibration convention.
REFERENCE = "top_left"

# Accepted image extensions.
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Capture resolution used throughout the project.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# --- Paths -------------------------------------------------------------------
# Derived from the location of this file so scripts work from any CWD.

HERE = os.path.dirname(os.path.abspath(__file__))

CAMERA_JSON = os.path.join(HERE, "camera.json")

INTRINSICS_DIR = os.path.join(HERE, "intrinsics")
EXTRINSICS_DIR = os.path.join(HERE, "extrinsics")
POSES_NPZ = os.path.join(EXTRINSICS_DIR, "poses.npz")

RIG_DIR = os.path.join(HERE, "rig")
RIG_SPARSE_DIR = os.path.join(RIG_DIR, "sparse")

MVS_DIR = os.path.join(HERE, "mvs")
STATIC_DIR = os.path.join(MVS_DIR, "static")
FRAMES_DIR = os.path.join(MVS_DIR, "frames")
STATIC_FUSED_PLY = os.path.join(MVS_DIR, "static_fused.ply")


# --- Loading / validation ----------------------------------------------------

def load_camera_indices():
    """Load camera.json -> ordered list of (position, device_index)."""
    if not os.path.exists(CAMERA_JSON):
        raise FileNotFoundError(
            f"camera.json not found at {CAMERA_JSON}. Create it first, e.g.:\n"
            '  {\n'
            '    "top_left":  2,\n'
            '    "top_right": 4,\n'
            '    "bot_left":  0,\n'
            '    "bot_right": 1\n'
            '  }'
        )
    with open(CAMERA_JSON, "r") as f:
        mapping = json.load(f)
    cameras = []
    for position in POSITION_ORDER:
        if position not in mapping:
            raise KeyError(
                f"camera.json is missing the '{position}' key. "
                f"Expected keys: {POSITION_ORDER}."
            )
        cameras.append((position, int(mapping[position])))
    return cameras


def load_intrinsics(camera_number):
    """Load K (3x3) and distortion coefficients (1xN) for one camera."""
    k_path = os.path.join(INTRINSICS_DIR, f"K_{camera_number}.txt")
    dist_path = os.path.join(INTRINSICS_DIR, f"dist_{camera_number}.txt")
    if not os.path.exists(k_path) or not os.path.exists(dist_path):
        raise FileNotFoundError(
            f"Missing intrinsics for camera {camera_number}. Expected "
            f"{k_path} and {dist_path}. Run intrinsics.py (Stage 1) first."
        )
    K = np.loadtxt(k_path, dtype=np.float64).reshape(3, 3)
    dist = np.loadtxt(dist_path, dtype=np.float64).reshape(1, -1)
    return K, dist


def load_poses():
    """
    Load extrinsics/poses.npz -> {position: (R, t)}.

    The stored poses are WORLD-TO-CAMERA (top_left is identity / zero), which is
    exactly COLMAP's convention. R is 3x3, t is shape (3,).
    """
    if not os.path.exists(POSES_NPZ):
        raise FileNotFoundError(
            f"{POSES_NPZ} not found. Run extrinsics.py (Stage 2) first to "
            "produce poses.npz."
        )
    data = np.load(POSES_NPZ)
    poses = {}
    for position in POSITION_ORDER:
        r_key, t_key = f"R_{position}", f"t_{position}"
        if r_key not in data or t_key not in data:
            raise KeyError(
                f"{POSES_NPZ} is missing '{r_key}' / '{t_key}'. Re-run "
                "extrinsics.py to regenerate it."
            )
        R = np.asarray(data[r_key], dtype=np.float64).reshape(3, 3)
        t = np.asarray(data[t_key], dtype=np.float64).reshape(3)
        poses[position] = (R, t)
    return poses


# --- COLMAP plumbing ---------------------------------------------------------

def find_colmap(explicit=None):
    """Resolve the colmap binary or fail with install guidance."""
    import shutil
    binary = explicit or shutil.which("colmap")
    if not binary:
        raise RuntimeError(
            "COLMAP was not found on PATH. Install it and ensure 'colmap' is "
            "runnable, or pass --colmap C:\\path\\to\\colmap.exe.\n"
            "  Windows: download from "
            "https://github.com/colmap/colmap/releases and add it to PATH.\n"
            "  Ubuntu:  sudo apt install colmap\n"
            "  Docs:    https://colmap.github.io/install.html"
        )
    return binary


def run(cmd, capture=False):
    """
    Run a subprocess command (used for all COLMAP CLI calls).

    Streams output by default and raises RuntimeError on a non-zero exit code.
    With capture=True the combined stdout/stderr is captured and returned as a
    string instead of being streamed.
    """
    print("\n$ " + " ".join(str(c) for c in cmd))
    if capture:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            print(result.stdout)
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): {cmd[1] if len(cmd) > 1 else cmd[0]}"
            )
        return result.stdout
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {cmd[1] if len(cmd) > 1 else cmd[0]}"
        )
    return None


# --- Rotation helpers --------------------------------------------------------

def quat_to_rot(qw, qx, qy, qz):
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),
         2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),
         2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
         1 - 2 * (qx * qx + qy * qy)],
    ])


def rot_to_quat(R):
    """
    3x3 rotation matrix -> COLMAP quaternion (w, x, y, z).

    Standard, numerically stable matrix-to-quaternion conversion (no scipy
    dependency). Returns a length-4 numpy array [w, x, y, z].
    """
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def camera_center(R, t):
    """World-space camera center C = -R^T @ t for a world-to-camera pose."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return (-R.T @ t).reshape(3)


# --- PLY ---------------------------------------------------------------------

def count_ply_points(path):
    """Read the vertex count from a PLY header, or 0 if unavailable."""
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
