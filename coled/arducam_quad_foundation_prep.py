"""
arducam_quad_foundation_prep.py

Run this AFTER arducam_quad_extrinsic_calibration.py.

For each non-base camera X in {TR, BL, BR}, treat (TL, X) as a stereo pair
and produce the inputs FoundationStereo expects:
  - foundation_prep/<X>/left_rect.png      (TL view, rectified for this pair)
  - foundation_prep/<X>/right_rect.png     (X  view, rectified for this pair)
  - foundation_prep/<X>/K.txt              (line 1 = flattened K, line 2 = baseline m)
  - foundation_prep/<X>/Q.npy              (disparity-to-depth matrix)

A single SPACE press in the 2x2 live preview captures one synchronized frame
from each camera, then all three pairs are rectified at once. Use
--from-images to skip live capture.

Note: For each pair, TL is rectified DIFFERENTLY (different R0, P0) because
each pair has a distinct rectified coordinate frame. That's why the same TL
frame is saved into three different output folders.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Tuple

import cv2
import numpy as np

from quad_utils import (
    BASE,
    NON_BASE,
    POSITIONS,
    load_layout,
    make_mosaic_2x2,
    open_layout_cameras,
    release_caps,
    scale_intrinsics_dict,
)

# --- Defaults ---
DEFAULT_QUAD_CALIB = "extrinsics/arducam_quad_calib.npz"
DEFAULT_OUT_DIR    = "foundation_prep"
# ---------------------


def load_quad_calib(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Quad calibration file not found: '{path}'. "
            "Run arducam_quad_extrinsic_calibration.py first."
        )
    data = np.load(path, allow_pickle=False)
    out = {
        "intrinsics": {
            "tl": {"K": data["K_tl"], "dist": data["dist_tl"]},
            "tr": {"K": data["K_tr"], "dist": data["dist_tr"]},
            "bl": {"K": data["K_bl"], "dist": data["dist_bl"]},
            "br": {"K": data["K_br"], "dist": data["dist_br"]},
        },
        "R": {
            "tr": data["R_tr"], "bl": data["R_bl"], "br": data["R_br"],
        },
        "T": {
            "tr": data["T_tr"], "bl": data["T_bl"], "br": data["T_br"],
        },
        "img_size": tuple(int(v) for v in data["img_size"].tolist()),
    }
    return out


def compute_rectification_for_pair(K0, dist0, K1, dist1, R, T,
                                   img_size: Tuple[int, int]):
    """Wrapper around stereoRectify + initUndistortRectifyMap for one pair."""
    R0, R1, P0, P1, Q, _roi0, _roi1 = cv2.stereoRectify(
        K0, dist0,
        K1, dist1,
        img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map0x, map0y = cv2.initUndistortRectifyMap(
        K0, dist0, R0, P0, img_size, cv2.CV_32FC1
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, dist1, R1, P1, img_size, cv2.CV_32FC1
    )
    return {
        "R0": R0, "R1": R1, "P0": P0, "P1": P1, "Q": Q,
        "map0x": map0x, "map0y": map0y,
        "map1x": map1x, "map1y": map1y,
    }


def write_K_txt(P0: np.ndarray, baseline_m: float, output_path: str) -> None:
    """
    Write the FoundationStereo-style intrinsic file:
      Line 1: 9 space-separated floats — flattened K (P0[:3,:3]).
      Line 2: baseline in metres.
    """
    K = P0[:3, :3]
    with open(output_path, "w") as f:
        f.write(" ".join(f"{v:.6f}" for v in K.flatten()) + "\n")
        f.write(f"{baseline_m:.6f}\n")
    print(f"  K.txt: {output_path}")
    print(f"    K =\n{K}")
    print(f"    baseline = {baseline_m*1000:.2f} mm ({baseline_m:.6f} m)")


def capture_quad_pair(caps: Dict[str, cv2.VideoCapture]
                      ) -> Dict[str, np.ndarray] | None:
    """
    Show a 2x2 preview from all four cameras. SPACE/C captures one synchronized
    frame from each. Q quits.
    """
    print("\nLive 2x2 preview — click the preview window first, "
          "then SPACE or C to capture, Q to quit.")
    while True:
        frames: Dict[str, np.ndarray | None] = {}
        ok_all = True
        for p in POSITIONS:
            ok, frame = caps[p].read()
            if not ok or frame is None:
                ok_all = False
                break
            frames[p] = frame
        if not ok_all:
            time.sleep(0.02)
            continue

        labels = {p: f"{p.upper()}" for p in POSITIONS}
        mosaic = make_mosaic_2x2(frames, scale=0.4, labels=labels)
        cv2.putText(
            mosaic, "SPACE/C = capture | Q = quit",
            (10, mosaic.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.imshow("Quad Preview", mosaic)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord(" "), ord("c"), ord("C")):
            cv2.destroyAllWindows()
            return {p: frames[p].copy() for p in POSITIONS}
        if key in (ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            return None


def rectify_pair(img_tl: np.ndarray,
                 img_x: np.ndarray,
                 maps: dict) -> Tuple[np.ndarray, np.ndarray]:
    rect_tl = cv2.remap(img_tl, maps["map0x"], maps["map0y"], cv2.INTER_LINEAR)
    rect_x  = cv2.remap(img_x,  maps["map1x"], maps["map1y"], cv2.INTER_LINEAR)
    return rect_tl, rect_x


def draw_epipolar_lines(rect0: np.ndarray, rect1: np.ndarray,
                        title: str, n_lines: int = 10) -> None:
    """Sanity-check window. Press any key to dismiss."""
    h = rect0.shape[0]
    vis0 = rect0.copy()
    vis1 = rect1.copy()
    step = h // (n_lines + 1)
    for i in range(1, n_lines + 1):
        y = i * step
        color = (0, int(255 * i / n_lines), int(255 * (1 - i / n_lines)))
        cv2.line(vis0, (0, y), (vis0.shape[1], y), color, 1)
        cv2.line(vis1, (0, y), (vis1.shape[1], y), color, 1)
    combined = np.hstack([vis0, vis1])
    combined = cv2.resize(combined, None, fx=0.4, fy=0.4)
    cv2.imshow(title, combined)


def process_pair(pair_name: str,
                 calib: dict,
                 images: Dict[str, np.ndarray],
                 img_size: Tuple[int, int],
                 out_dir: str,
                 show_epi: bool) -> dict:
    """Compute rectification, save outputs for one (TL, pair_name) pair."""
    print(f"\n--- Pair TL-{pair_name.upper()} ---")
    K_tl   = calib["intrinsics"][BASE]["K"]
    d_tl   = calib["intrinsics"][BASE]["dist"]
    K_x    = calib["intrinsics"][pair_name]["K"]
    d_x    = calib["intrinsics"][pair_name]["dist"]
    R_x    = calib["R"][pair_name]
    T_x    = calib["T"][pair_name]

    baseline_mm = float(np.linalg.norm(T_x))
    baseline_m  = baseline_mm / 1000.0
    print(f"  Baseline: {baseline_mm:.2f} mm")

    maps = compute_rectification_for_pair(
        K_tl, d_tl, K_x, d_x, R_x, T_x, img_size
    )

    rect_tl, rect_x = rectify_pair(images[BASE], images[pair_name], maps)

    pair_dir = os.path.join(out_dir, pair_name)
    os.makedirs(pair_dir, exist_ok=True)
    left_path  = os.path.join(pair_dir, "left_rect.png")
    right_path = os.path.join(pair_dir, "right_rect.png")
    cv2.imwrite(left_path,  rect_tl)
    cv2.imwrite(right_path, rect_x)
    print(f"  Saved: {left_path}")
    print(f"  Saved: {right_path}")

    write_K_txt(maps["P0"], baseline_m, os.path.join(pair_dir, "K.txt"))
    np.save(os.path.join(pair_dir, "Q.npy"), maps["Q"])
    print(f"  Q.npy: {os.path.join(pair_dir, 'Q.npy')}")

    if show_epi:
        draw_epipolar_lines(
            rect_tl, rect_x,
            f"Epipolar lines TL-{pair_name.upper()} — press any key",
        )

    return {
        "name": pair_name,
        "left": left_path,
        "right": right_path,
        "intrinsic_file": os.path.join(pair_dir, "K.txt"),
        "baseline_mm": baseline_mm,
        "out_dir": pair_dir,
    }


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FoundationStereo inputs for each of the three "
                    "TL-paired stereo views (TL-TR, TL-BL, TL-BR)."
    )
    parser.add_argument("--layout", default="camera_layout.json",
                        help="Path to camera_layout.json (default: %(default)s).")
    parser.add_argument("--quad-calib", default=DEFAULT_QUAD_CALIB,
                        help=f"Quad calibration .npz path (default: "
                             f"{DEFAULT_QUAD_CALIB}).")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Output root directory (default: {DEFAULT_OUT_DIR}/).")
    parser.add_argument("--from-images", nargs=4, metavar=("TL", "TR", "BL", "BR"),
                        default=None,
                        help="Skip live capture; provide one image per position "
                             "in TL TR BL BR order.")
    parser.add_argument("--no-epi", action="store_true",
                        help="Skip the epipolar-line sanity check windows.")
    parser.add_argument("--capture-width", type=int, default=None,
                        help="Override capture width. Intrinsics from the "
                             "quad calib are auto-scaled to match. Use this "
                             "if your USB hub can't sustain full resolution.")
    parser.add_argument("--capture-height", type=int, default=None,
                        help="Override capture height (same notes as above).")
    parser.add_argument("--capture-fps", type=int, default=None,
                        help="Override capture fps for this run only.")
    parser.add_argument("--no-strict-resolution", action="store_true",
                        help="Don't abort if a camera silently falls back to "
                             "a lower resolution. Debugging only.")
    args = parser.parse_args()

    layout = load_layout(args.layout)
    print(f"Layout loaded from '{args.layout}'.")

    print(f"Loading quad calibration: {args.quad_calib}")
    calib = load_quad_calib(args.quad_calib)
    print(f"  Image size from calibration: {calib['img_size']}")
    for p in NON_BASE:
        print(f"  Baseline TL->{p.upper()}: "
              f"{float(np.linalg.norm(calib['T'][p])):.2f} mm")

    # ---- Acquire one synchronized frame per camera ---------------------
    if args.from_images:
        order = ["tl", "tr", "bl", "br"]
        images: Dict[str, np.ndarray] = {}
        for pos, path in zip(order, args.from_images):
            img = cv2.imread(path)
            if img is None:
                print(f"Error: could not read '{path}' for position {pos}.")
                sys.exit(1)
            images[pos] = img
            print(f"  Loaded {pos.upper()}: {path}  shape={img.shape}")
    else:
        print("\nOpening cameras for live capture...")
        caps = open_layout_cameras(
            layout,
            width=args.capture_width,
            height=args.capture_height,
            fps=args.capture_fps,
            strict=not args.no_strict_resolution,
        )
        try:
            images = capture_quad_pair(caps)
        finally:
            release_caps(caps)
        if images is None:
            print("No frames captured. Exiting.")
            sys.exit(0)

    img_size = (images[BASE].shape[1], images[BASE].shape[0])
    print(f"\nUsing image size: {img_size}")

    intrinsic_size = tuple(calib["img_size"])
    if img_size != intrinsic_size:
        print(f"  Scaling intrinsics from {intrinsic_size} (calibration) "
              f"to {img_size} (capture).")
        calib["intrinsics"] = scale_intrinsics_dict(
            calib["intrinsics"], intrinsic_size, img_size
        )

    # ---- Process all 3 pairs ------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    pair_results = []
    for pair in NON_BASE:
        pair_results.append(
            process_pair(pair, calib, images, img_size, args.out_dir,
                         show_epi=not args.no_epi)
        )

    if not args.no_epi:
        print("\nEpipolar windows open. Press any key (with a window focused) "
              "to close them and continue.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ---- Print FoundationStereo commands -------------------------------
    print("\n=========== Ready for FoundationStereo ===========")
    for r in pair_results:
        out_subdir = os.path.join("output", r["name"])
        print(f"\n# Pair TL-{r['name'].upper()}  (baseline {r['baseline_mm']:.1f} mm)")
        print("python scripts/run_demo.py \\")
        print(f"  --left_file  {r['left']} \\")
        print(f"  --right_file {r['right']} \\")
        print(f"  --intrinsic_file {r['intrinsic_file']} \\")
        print(f"  --out_dir {out_subdir}/ \\")
        print(f"  --get_pc 1 --scale 1")
    print("\n==================================================")
