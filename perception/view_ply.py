"""
view_ply.py — quick viewer for a dense point cloud (.ply).

COLMAP's GUI needs a Qt platform plugin that this build is missing, so this is a
dependency-light matplotlib viewer for the fused cloud. Reads ascii or
binary_little_endian PLY (the format stereo_fusion writes: x y z nx ny nz
red green blue) and shows a colored 3D scatter plus a few stats.

Run:
    python view_ply.py                         # mvs/static_fused.ply
    python view_ply.py mvs/frame_000001/fused.ply
    python view_ply.py --trim 2.0              # clip outliers beyond 2 sigma
"""

import argparse
import os

import numpy as np

from common import STATIC_FUSED_PLY


_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def read_ply(path):
    """Return (xyz Nx3 float, rgb Nx3 float in 0..1 or None)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file.")
        fmt = None
        count = 0
        props = []  # (name, numpy_type_char)
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("element"):
                # Another element (e.g. face) — stop recording vertex props.
                pass
            elif line.startswith("property") and count and not line.startswith(
                    "property list"):
                _, ptype, pname = line.split()[:3]
                props.append((pname, _PLY_TYPES.get(ptype, "f4")))
            elif line == "end_header":
                break

        names = [p[0] for p in props]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=count)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            cols = {n: data[:, i] for i, n in enumerate(names)}
        else:  # binary_little_endian (COLMAP default)
            dtype = np.dtype([(n, "<" + t) for n, t in props])
            arr = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype)
            cols = {n: arr[n].astype(np.float64) for n in names}

    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1)
    rgb = None
    if {"red", "green", "blue"}.issubset(cols):
        rgb = np.stack([cols["red"], cols["green"], cols["blue"]], axis=1)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
    return xyz, rgb


def main():
    parser = argparse.ArgumentParser(description="View a dense .ply cloud.")
    parser.add_argument("ply", nargs="?", default=STATIC_FUSED_PLY,
                        help="Path to the .ply (default: mvs/static_fused.ply).")
    parser.add_argument("--trim", type=float, default=3.0,
                        help="Clip points beyond this many std from the "
                             "centroid for the view (default 3.0; 0 disables).")
    args = parser.parse_args()

    if not os.path.exists(args.ply):
        raise FileNotFoundError(f"{args.ply} not found. Run mvs.py first.")

    xyz, rgb = read_ply(args.ply)
    n = len(xyz)
    print(f"Loaded {n} points from {args.ply}")
    if n == 0:
        print("Empty cloud — nothing to show.")
        return

    centroid = xyz.mean(axis=0)
    bbox = xyz.max(axis=0) - xyz.min(axis=0)
    print(f"  centroid: {np.round(centroid, 1)}")
    print(f"  bounding box (mm): {np.round(bbox, 1)}")

    keep = np.ones(n, dtype=bool)
    if args.trim > 0 and n > 10:
        d = np.linalg.norm(xyz - centroid, axis=1)
        keep = d < (d.mean() + args.trim * d.std())
        if keep.sum() < n:
            print(f"  trimming {n - keep.sum()} outlier(s) for the view "
                  f"(>{args.trim} sigma)")

    import matplotlib.pyplot as plt  # imported late so --help is instant

    pts = xyz[keep]
    colors = rgb[keep] if rgb is not None else None
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=4, depthshade=True)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"{os.path.basename(args.ply)} — {n} points")

    # Equal aspect so the scene isn't distorted.
    c = pts.mean(axis=0)
    r = float(np.max(np.abs(pts - c))) or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
