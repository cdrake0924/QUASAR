"""
quad_utils.py

Shared helpers for the 4-camera (top-left, top-right, bottom-left, bottom-right)
calibration pipeline. Every quad-* script reads camera_layout.json through
load_layout() so the OS camera indices and rig parameters live in one place.
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import cv2
import numpy as np

POSITIONS = ["tl", "tr", "bl", "br"]
NON_BASE = ["tr", "bl", "br"]
BASE = "tl"

DEFAULT_LAYOUT_PATH = "camera_layout.json"


def load_layout(path: str = DEFAULT_LAYOUT_PATH) -> dict:
    """
    Load and validate camera_layout.json.

    Required keys: tl, tr, bl, br  (each an int OS index, all distinct).
    Optional keys (with sensible fallbacks):
        frame_width, frame_height, fps,
        checkerboard (list[int, int]), square_size_mm (number).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"camera_layout.json not found at '{path}'. "
            "Create it with keys: tl, tr, bl, br (OS camera indices)."
        )

    with open(path, "r") as f:
        layout = json.load(f)

    missing = [p for p in POSITIONS if p not in layout]
    if missing:
        raise ValueError(
            f"camera_layout.json missing required keys: {missing}. "
            f"Need all of {POSITIONS}."
        )

    indices = [int(layout[p]) for p in POSITIONS]
    if len(set(indices)) != 4:
        raise ValueError(
            f"camera_layout.json indices must be distinct, got "
            f"tl={indices[0]}, tr={indices[1]}, bl={indices[2]}, br={indices[3]}."
        )

    layout.setdefault("frame_width", 1280)
    layout.setdefault("frame_height", 720)
    layout.setdefault("fps", 15)
    layout.setdefault("checkerboard", [8, 6])
    layout.setdefault("square_size_mm", 30)

    layout["checkerboard"] = tuple(int(v) for v in layout["checkerboard"])
    layout["square_size_mm"] = float(layout["square_size_mm"])
    layout["frame_width"] = int(layout["frame_width"])
    layout["frame_height"] = int(layout["frame_height"])
    layout["fps"] = int(layout["fps"])

    for p in POSITIONS:
        layout[p] = int(layout[p])

    return layout


_BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF",  cv2.CAP_MSMF),
]


def _try_open(index: int, backend_flag: int,
              width: int, height: int, fps: int) -> "cv2.VideoCapture | None":
    cap = cv2.VideoCapture(index, backend_flag)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def open_camera(index: int, width: int, height: int, fps: int,
                strict: bool = True) -> cv2.VideoCapture:
    """
    Open a USB camera and verify it actually streams at the requested resolution.

    Tries DirectShow first, falls back to Media Foundation if DSHOW fails to
    open / read. If `strict=True` (default), raises a clear bandwidth error
    when the camera silently falls back to a lower resolution.
    """
    last_err = ""
    cap: "cv2.VideoCapture | None" = None
    used_backend = None
    for name, flag in _BACKENDS:
        cap = _try_open(index, flag, width, height, fps)
        if cap is not None:
            used_backend = name
            break
        last_err = f"backend {name} could not open index {index}"

    if cap is None:
        raise RuntimeError(
            f"Cannot open camera at index {index}. Last attempt: {last_err}. "
            "If this came up while opening all 4 cameras, you've likely "
            "saturated USB bandwidth — try a lower --capture-width/-height "
            "or split cameras across host controllers."
        )

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Camera {index} opened via {used_backend}: "
          f"{w}x{h} @ {actual_fps:.1f} fps")

    if (w != width or h != height) and strict:
        cap.release()
        if (w * h) > 0 and (w, h) != (width, height):
            # Distinguish the two common root causes.
            #   1) Camera doesn't have a native mode at (width, height) — driver
            #      picked the closest available one. Solution: pick a native UVC
            #      resolution like 1280x720, 800x600, 640x480, 320x240, etc.
            #   2) USB bandwidth is saturated — other cameras already streaming
            #      ate the available isochronous budget on this hub/controller.
            hint = (
                "Possible causes:\n"
                "   (a) The camera doesn't expose a native mode at the "
                f"requested {width}x{height}. UVC cameras only serve a fixed "
                "list of resolutions — try a standard mode like 1280x720, "
                "1024x768, 800x600, 640x480, or 320x240.\n"
                "   (b) USB bandwidth saturation. If other cameras already "
                "opened, the hub can't allocate isochronous bandwidth for "
                "this one. Lower the resolution/fps or split cameras across "
                "host controllers."
            )
        else:
            hint = (
                "USB bandwidth saturation — lower the resolution/fps or "
                "split cameras across host controllers."
            )
        raise RuntimeError(
            f"Camera {index} delivered {w}x{h} instead of the requested "
            f"{width}x{height}.\n{hint}"
        )

    return cap


def open_layout_cameras(layout: dict,
                        width: int | None = None,
                        height: int | None = None,
                        fps: int | None = None,
                        open_delay_sec: float = 0.25,
                        strict: bool = True) -> dict:
    """
    Open every camera defined in the layout and return a dict
    {position: cv2.VideoCapture}. Caller is responsible for releasing.

    width/height/fps override the layout's frame_width/frame_height/fps for
    THIS open only — useful for testing whether a USB hub can handle a lower
    resolution without re-editing camera_layout.json. Layout values are used
    when an override is None.

    open_delay_sec gives the previous camera time to finish negotiating USB
    bandwidth before the next one starts.
    """
    import time as _time

    use_w   = int(width)  if width  is not None else int(layout["frame_width"])
    use_h   = int(height) if height is not None else int(layout["frame_height"])
    use_fps = int(fps)    if fps    is not None else int(layout["fps"])

    caps: dict = {}
    try:
        for i, p in enumerate(POSITIONS):
            if i > 0 and open_delay_sec > 0:
                _time.sleep(open_delay_sec)
            print(f"Opening {p.upper()} camera (index {layout[p]}) "
                  f"at {use_w}x{use_h}@{use_fps}...")
            caps[p] = open_camera(
                layout[p], use_w, use_h, use_fps, strict=strict
            )
    except Exception:
        for cap in caps.values():
            cap.release()
        raise
    return caps


def release_caps(caps: dict) -> None:
    for cap in caps.values():
        try:
            cap.release()
        except Exception:
            pass


def make_object_points(checkerboard: Tuple[int, int],
                       square_size_mm: float) -> np.ndarray:
    """
    Build the planar checkerboard 3D template (Z=0), in millimetres.
    Shape: (cols*rows, 3), dtype float32 — compatible with cv2.calibrate* APIs.
    """
    cols, rows = checkerboard
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_mm)
    return objp


def intrinsic_path(position: str, base_dir: str = "intrinsic") -> str:
    """Canonical path for a per-position intrinsic .npz file."""
    return os.path.join(base_dir, f"cam_{position}_intr.npz")


def scale_intrinsic_K(K: np.ndarray,
                      src_size: Tuple[int, int],
                      dst_size: Tuple[int, int]) -> np.ndarray:
    """
    Rescale a camera intrinsic matrix from one image resolution to another.

    Assumes the camera serves lower resolutions by binning/downscaling (the
    standard UVC behaviour), which preserves the lens FOV and the principal
    point in normalized image coordinates. Distortion coefficients do NOT
    need to be rescaled — they live in normalized coordinates.

    src_size, dst_size : (width, height)
    """
    sw, sh = src_size
    dw, dh = dst_size
    sx = dw / float(sw)
    sy = dh / float(sh)
    K_new = K.copy().astype(np.float64)
    K_new[0, 0] *= sx
    K_new[0, 2] *= sx
    K_new[1, 1] *= sy
    K_new[1, 2] *= sy
    return K_new


def scale_intrinsics_dict(intrinsics: dict,
                          src_size: Tuple[int, int],
                          dst_size: Tuple[int, int]) -> dict:
    """
    Apply scale_intrinsic_K to every entry in a {pos: {K, dist}} dict.

    WARNING: This is only valid when the camera UNIFORMLY DOWNSAMPLES from
    src_size to dst_size (same sensor crop, same FOV, just fewer pixels).
    Most UVC cameras change sensor crop / focal length when switching across
    aspect ratios (e.g. 16:9 -> 4:3), so the result is wrong in that case.
    The function prints a loud warning when the aspect ratios differ.
    """
    if tuple(src_size) == tuple(dst_size):
        return intrinsics

    sw, sh = src_size
    dw, dh = dst_size
    src_ar = sw / float(sh)
    dst_ar = dw / float(dh)
    if abs(src_ar - dst_ar) / max(src_ar, dst_ar) > 0.02:
        print("=" * 72)
        print(
            "  WARNING: scaling intrinsics across DIFFERENT aspect ratios:\n"
            f"    {sw}x{sh}  (aspect {src_ar:.3f})  ->  {dw}x{dh}  (aspect {dst_ar:.3f}).\n"
            "  Most UVC cameras DO NOT simply downscale across aspect-ratio\n"
            "  changes; they apply a different sensor crop, which means the\n"
            "  focal length in pixels DOES NOT scale the way this routine\n"
            "  assumes. Recovered metric distances (baselines, depths) will\n"
            "  be wrong, typically by tens of percent.\n"
            "  Fix: re-run arducam_intrinsic_calibration.py at the SAME\n"
            f"  resolution you intend to capture ({dw}x{dh}). For example:\n"
            f"      python arducam_intrinsic_calibration.py --all \\\n"
            f"          --width {dw} --height {dh}"
        )
        print("=" * 72)

    out = {}
    for p, e in intrinsics.items():
        out[p] = {
            "K": scale_intrinsic_K(e["K"], src_size, dst_size),
            "dist": e["dist"].copy(),
        }
    return out


def load_intrinsics_for_layout(base_dir: str = "intrinsic") -> dict:
    """
    Load all four cam_<position>_intr.npz files and return
        {position: {"K": ndarray, "dist": ndarray}}.
    Raises FileNotFoundError if any are missing.
    """
    out = {}
    for p in POSITIONS:
        path = intrinsic_path(p, base_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing intrinsic file for '{p}': '{path}'. "
                "Run arducam_intrinsic_calibration.py for that position first."
            )
        data = np.load(path)
        out[p] = {"K": data["mtx"], "dist": data["dist"]}
    return out


def make_mosaic_2x2(images: dict, scale: float = 0.4,
                    labels: dict | None = None) -> np.ndarray:
    """
    Compose a 2x2 mosaic from {tl, tr, bl, br} image dict for live preview.
    All four images are resized to the same shape (taken from TL) before stacking.
    """
    target = images["tl"]
    th, tw = target.shape[:2]

    def fit(img):
        if img.shape[0] != th or img.shape[1] != tw:
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        return img

    top = np.hstack([fit(images["tl"]), fit(images["tr"])])
    bot = np.hstack([fit(images["bl"]), fit(images["br"])])
    mosaic = np.vstack([top, bot])

    if labels:
        for pos, txt in labels.items():
            if pos == "tl":
                org = (10, 30)
            elif pos == "tr":
                org = (tw + 10, 30)
            elif pos == "bl":
                org = (10, th + 30)
            else:
                org = (tw + 10, th + 30)
            cv2.putText(mosaic, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)

    if scale != 1.0:
        mosaic = cv2.resize(mosaic, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)

    return mosaic
