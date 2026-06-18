"""
Quality-check viewer for the COLMAP sparse reconstruction (Stage 3).

The prebuilt COLMAP GUI on Windows is often missing its Qt `platforms`
plugin and refuses to start. This script renders the same information that
matters for a sanity check -- the 3D point cloud and the recovered camera
positions -- straight from the exported text model, with no GUI dependency.

Usage:
    python view_sparse.py                 # auto-pick largest model in sfm/sparse
    python view_sparse.py --model sfm/sparse/1
    python view_sparse.py --no-show       # just write the PNG, don't open a window
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

HERE = os.path.dirname(os.path.abspath(__file__))
SPARSE_DIR = os.path.join(HERE, "sfm", "sparse")


def quat_to_rotmat(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def find_largest_model():
    """Pick the sparse/* sub-model with the most registered images."""
    candidates = []
    for images_txt in glob.glob(os.path.join(SPARSE_DIR, "*", "images.txt")):
        model_dir = os.path.dirname(images_txt)
        n = 0
        with open(images_txt) as f:
            for line in f:
                if line.startswith("# Number of images:"):
                    n = int(line.split(":")[1].split(",")[0])
                    break
        candidates.append((model_dir, n))
    if not candidates:
        raise SystemExit(
            f"No images.txt found under {SPARSE_DIR}. Run sfm.py first "
            "(the model is exported to TXT automatically)."
        )
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def read_cameras(model_dir):
    """Return {name: camera_center_xyz} from images.txt."""
    path = os.path.join(model_dir, "images.txt")
    centers = {}
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    # Two lines per image; the second (points) line we can skip.
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        R = quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        center = -R.T @ t  # camera center in world coordinates
        centers[name] = center
    return centers


def read_points(model_dir):
    """Return (Nx3 xyz, Nx3 rgb in 0..1)."""
    path = os.path.join(model_dir, "points3D.txt")
    xyz, rgb = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
    if not xyz:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.array(xyz), np.array(rgb) / 255.0


POSITION_COLORS = {
    "top_left": "tab:red",
    "top_right": "tab:green",
    "bot_left": "tab:blue",
    "bot_right": "tab:orange",
}


def position_of(name):
    for pos in POSITION_COLORS:
        if name.startswith(pos):
            return pos
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="Path to a sparse model dir (default: largest under sfm/sparse).")
    parser.add_argument("--no-show", action="store_true",
                        help="Write the PNG without opening an interactive window.")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model) if args.model else find_largest_model()
    print(f"Reading model: {model_dir}")

    centers = read_cameras(model_dir)
    xyz, rgb = read_points(model_dir)
    print(f"  cameras: {len(centers)}   points: {len(xyz)}")

    if args.no_show:
        matplotlib.use("Agg")

    cam_xyz = np.array(list(centers.values())) if centers else np.empty((0, 3))

    # Trim point-cloud outliers for framing (a few stray triangulations can
    # blow up the axis scale and hide the real scene structure).
    inlier = xyz
    inlier_rgb = rgb
    if len(xyz) > 10:
        lo = np.percentile(xyz, 2, axis=0)
        hi = np.percentile(xyz, 98, axis=0)
        mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        inlier = xyz[mask]
        inlier_rgb = rgb[mask]

    fig = plt.figure(figsize=(15, 7))

    # --- Left: scene point cloud + cameras ---
    ax = fig.add_subplot(121, projection="3d")
    if len(inlier):
        ax.scatter(inlier[:, 0], inlier[:, 1], inlier[:, 2], c=inlier_rgb,
                   s=5, alpha=0.6, label=f"{len(inlier)} points (2-98%)")
    _plot_cameras(ax, centers, s=70, label_each=False)
    ax.set_title("Scene + cameras")
    _set_labels(ax)
    ax.legend(loc="upper right", fontsize=8)

    # --- Right: cameras only, zoomed (verify the 2x2 rig layout) ---
    ax2 = fig.add_subplot(122, projection="3d")
    _plot_cameras(ax2, centers, s=90, label_each=True)
    ax2.set_title("Cameras only (should look like your 2x2 rig)")
    _set_labels(ax2)
    if len(cam_xyz):
        _equal_aspect(ax2, cam_xyz)

    fig.suptitle(f"Sparse reconstruction: {os.path.basename(model_dir)}  "
                 f"({len(centers)} images, {len(xyz)} points)", fontsize=13)

    out_png = os.path.join(model_dir, "quality_check.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"  saved: {out_png}")

    if not args.no_show:
        plt.show()


def _plot_cameras(ax, centers, s=70, label_each=False):
    plotted = set()
    for name, c in centers.items():
        pos = position_of(name)
        color = POSITION_COLORS.get(pos, "black")
        label = pos if pos not in plotted else None
        plotted.add(pos)
        ax.scatter(c[0], c[1], c[2], c=color, s=s, marker="^",
                   edgecolors="k", linewidths=0.5, label=label)
    if label_each:
        seen = set()
        for name, c in centers.items():
            pos = position_of(name)
            if pos not in seen:
                ax.text(c[0], c[1], c[2], f"  {pos}", fontsize=9)
                seen.add(pos)
    ax.legend(loc="upper right", fontsize=8)


def _set_labels(ax):
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _equal_aspect(ax, pts):
    """Force equal aspect ratio around a set of points so layout isn't skewed."""
    center = pts.mean(axis=0)
    span = (pts.max(axis=0) - pts.min(axis=0)).max()
    r = max(span, 1e-6) * 0.6
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


if __name__ == "__main__":
    main()
