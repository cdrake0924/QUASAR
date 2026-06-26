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


# PLY scalar type -> (struct char, byte size). Covers the COLMAP fused.ply set.
_PLY_TYPES = {
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
}


def read_ply_xyz(path):
    """
    Read just the (x, y, z) vertex coordinates from a PLY file.

    Supports `ascii` and `binary_little_endian` PLYs with arbitrary extra
    per-vertex properties (e.g. COLMAP fused.ply ships normals + rgb). Returns
    an (N, 3) float64 numpy array. Used to find a robust scene center for the
    novel-view orbit path (median of points resists MVS outliers).
    """
    import struct

    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = None
        n_vertex = 0
        props = []          # list of (name, type) in declaration order
        in_vertex = False
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            line = raw.decode("ascii", "ignore").strip()
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n_vertex = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                # property <type> <name>  (we don't expect list props here)
                props.append((tok[2], tok[1]))
            elif tok[0] == "end_header":
                break

        names = [n for n, _ in props]
        for axis in ("x", "y", "z"):
            if axis not in names:
                raise ValueError(f"PLY missing '{axis}' property: {path}")

        if fmt == "ascii":
            xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
            out = np.empty((n_vertex, 3), dtype=np.float64)
            for r in range(n_vertex):
                vals = f.readline().split()
                out[r] = (float(vals[xi]), float(vals[yi]), float(vals[zi]))
            return out

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format '{fmt}': {path}")

        # Build a struct format for one vertex; record x/y/z field positions.
        chars = "<"
        field_pos = {}
        for i, (name, typ) in enumerate(props):
            if typ not in _PLY_TYPES:
                raise ValueError(f"Unsupported PLY property type '{typ}'")
            chars += _PLY_TYPES[typ][0]
            if name in ("x", "y", "z"):
                field_pos[name] = i
        stride = struct.calcsize(chars)
        unpack = struct.Struct(chars).unpack_from
        buf = f.read(stride * n_vertex)
        out = np.empty((n_vertex, 3), dtype=np.float64)
        xi, yi, zi = field_pos["x"], field_pos["y"], field_pos["z"]
        for r in range(n_vertex):
            v = unpack(buf, r * stride)
            out[r] = (v[xi], v[yi], v[zi])
        return out


# --- GIF assembly ------------------------------------------------------------

def make_gif(frames, out_path, fps=20, loop=0):
    """
    Assemble an ordered list (or glob) of image files into an animated GIF.

    Used to turn the per-frame orbit PNGs from 3DGS / NPBG++ into a single
    looping clip for side-by-side novel-view comparison. `frames` may be a glob
    string (sorted) or an explicit ordered list of paths. Requires Pillow.
    """
    import glob as _glob
    from PIL import Image

    if isinstance(frames, str):
        frames = sorted(_glob.glob(frames))
    frames = list(frames)
    if not frames:
        raise FileNotFoundError(f"make_gif: no frames to assemble ({out_path})")

    imgs = [Image.open(p).convert("RGB") for p in frames]
    duration = max(1, int(round(1000.0 / float(fps))))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=duration, loop=loop, optimize=True, disposal=2)
    return out_path, len(imgs)
