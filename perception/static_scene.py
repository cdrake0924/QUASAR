"""
Static-scene capture helper — Track A (quasar/perception)

Captures ONE synchronized set of frames from all 4 cameras and writes them into
mvs/static/ named  {position}.jpg  (top_left.jpg, top_right.jpg, bot_left.jpg,
bot_right.jpg) — exactly the input layout `mvs.py --mode static` expects.

Keep the rig and the scene perfectly still; the 3D structure comes from the 4
different camera viewpoints. A texture-rich scene helps the later MVS / depth
triangulation (blank walls are the usual cause of holey point clouds).

Controls:
  SPACE / C  capture one synchronized set, save, and exit
  Q / ESC    exit without saving

Run:
    python static_scene.py
    python static_scene.py --fresh         # wipe mvs/static/ first
"""

import argparse
import os
import time

import cv2
import numpy as np

from common import (
    POSITION_ORDER,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    STATIC_DIR,
    IMAGE_EXTS,
    load_camera_indices,
)

FPS = 15


# --- Camera I/O --------------------------------------------------------------

def open_camera(index):
    """
    Open a webcam at the given OS index at FRAME_WIDTH x FRAME_HEIGHT.

    Tries DirectShow first (most reliable for USB UVC cameras on Windows),
    then Media Foundation, then any backend. MJPG is requested before the
    resolution because it unlocks higher modes on most USB cameras.
    """
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("DEFAULT", cv2.CAP_ANY),
    ]
    for name, flag in backends:
        cap = cv2.VideoCapture(index, flag)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        ok, _ = cap.read()
        if not ok:
            cap.release()
            continue

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Camera {index} opened via {name}: {actual_w}x{actual_h}")
        return cap

    raise RuntimeError(
        f"Cannot open camera at index {index}. Check the value in camera.json "
        "and that the device is connected and not in use by another program. "
        "Opening 4 cameras at once can also saturate USB bandwidth — try "
        "spreading them across separate USB controllers."
    )


# --- Capture -----------------------------------------------------------------

def grab_frames(caps):
    """Read one frame from every camera. Returns dict or None on any failure."""
    frames = {}
    for position, cap in caps.items():
        ok, frame = cap.read()
        if not ok:
            return None
        frames[position] = frame
    return frames


def save_set(frames):
    """Write one synchronized set to mvs/static/ as {position}.jpg."""
    for position in POSITION_ORDER:
        cv2.imwrite(os.path.join(STATIC_DIR, f"{position}.jpg"),
                    frames[position])
    print(f"  Saved {', '.join(p + '.jpg' for p in POSITION_ORDER)} "
          f"to {STATIC_DIR}")


def make_tile(frame, label):
    tile = frame.copy()
    cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    return tile


def build_preview(frames):
    tiles = {p: make_tile(frames[p], p) for p in POSITION_ORDER}
    top = np.hstack([tiles["top_left"], tiles["top_right"]])
    bottom = np.hstack([tiles["bot_left"], tiles["bot_right"]])
    grid = np.vstack([top, bottom])
    cv2.putText(grid, "[SPACE/C] capture & exit   [Q/ESC] quit",
                (8, grid.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 1)
    return grid


def capture(caps):
    """Run the capture session. Returns True if a set was saved."""
    window = "Static scene capture"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    saved = False
    try:
        while True:
            frames = grab_frames(caps)
            if frames is None:
                print("  Warning: dropped frame, retrying...")
                continue

            cv2.imshow(window, build_preview(frames))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord(" "), ord("c"), ord("C")):
                save_set(frames)
                saved = True
                break
    finally:
        cv2.destroyWindow(window)
    return saved


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Capture one static frame-set into mvs/static/ for Track A."
    )
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing images in mvs/static/ first.")
    args = parser.parse_args()

    cameras = load_camera_indices()
    os.makedirs(STATIC_DIR, exist_ok=True)

    if args.fresh:
        removed = 0
        for name in os.listdir(STATIC_DIR):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                os.remove(os.path.join(STATIC_DIR, name))
                removed += 1
        print(f"Removed {removed} existing image(s) from {STATIC_DIR}.")

    print("Static-scene capture (Track A)")
    print(f"  Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Output: {STATIC_DIR}")
    print("  Keep the rig AND the scene perfectly still. Make sure the scene "
          "is texture-rich.\n")

    caps = {}
    try:
        for position, camera_number in cameras:
            print(f"Opening '{position}' (index {camera_number})...")
            caps[position] = open_camera(camera_number)
            time.sleep(0.25)

        print("\nSPACE/C to capture & exit, Q to quit without saving.")
        saved = capture(caps)
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()

    if saved:
        print("\nDone. Next: run  python mvs.py --mode static")
    else:
        print("\nNo frames captured — nothing saved.")


if __name__ == "__main__":
    main()
