"""
Static-scene capture helper (quasar/perception)

Captures synchronized frames from all 4 cameras and writes them into
sfm/images/ named  {position}_{frame:06d}.jpg  (e.g. top_left_000001.jpg) —
exactly the layout Stage 3 (sfm.py) expects.

This is for the STATIC SfM capture: keep the rig and the scene perfectly still.
The 3D structure comes from the 4 different camera viewpoints, not from motion,
so a handful of well-exposed, synchronized sets of a TEXTURE-RICH scene is
plenty. Blank walls / low texture are the usual cause of COLMAP failing later.

Controls (interactive mode):
  SPACE / C  capture one synchronized set (all 4 cameras)
  B          capture a burst of --num sets at --interval spacing
  Q / ESC    finish and exit

Run:
    python static_scene.py                 # interactive
    python static_scene.py --num 10        # auto-capture 10 sets, then quit
    python static_scene.py --fresh         # wipe sfm/images/ first
"""

import argparse
import json
import os
import time

import cv2
import numpy as np


# --- Configuration -----------------------------------------------------------

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

POSITION_ORDER = ["top_left", "top_right", "bot_left", "bot_right"]

HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA_JSON = os.path.join(HERE, "camera.json")
OUTPUT_DIR = os.path.join(HERE, "sfm", "images")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# --- Camera I/O --------------------------------------------------------------

def load_camera_indices():
    """Load camera.json -> ordered list of (position, device_index)."""
    if not os.path.exists(CAMERA_JSON):
        raise FileNotFoundError(
            f"camera.json not found at {CAMERA_JSON}. Create it first, e.g.:\n"
            '  {\n'
            '    "top_left":  1,\n'
            '    "top_right": 2,\n'
            '    "bot_left":  3,\n'
            '    "bot_right": 4\n'
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

def next_frame_index():
    """Continue numbering after any frames already in OUTPUT_DIR."""
    if not os.path.isdir(OUTPUT_DIR):
        return 1
    highest = 0
    for name in os.listdir(OUTPUT_DIR):
        base, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        for position in POSITION_ORDER:
            prefix = position + "_"
            if base.startswith(prefix):
                digits = base[len(prefix):]
                if digits.isdigit():
                    highest = max(highest, int(digits))
    return highest + 1


def grab_frames(caps):
    """Read one frame from every camera. Returns dict or None on any failure."""
    frames = {}
    for position, cap in caps.items():
        ok, frame = cap.read()
        if not ok:
            return None
        frames[position] = frame
    return frames


def save_set(frames, frame_index):
    """Write one synchronized set to disk as {position}_{index:06d}.jpg."""
    for position in POSITION_ORDER:
        filename = f"{position}_{frame_index:06d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frames[position])
    print(f"    Saved set {frame_index:06d} "
          f"({', '.join(POSITION_ORDER)})")


def make_tile(frame, label):
    tile = frame.copy()
    cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    return tile


def build_preview(frames, saved_count):
    tiles = {p: make_tile(frames[p], p) for p in POSITION_ORDER}
    top = np.hstack([tiles["top_left"], tiles["top_right"]])
    bottom = np.hstack([tiles["bot_left"], tiles["bot_right"]])
    grid = np.vstack([top, bottom])
    cv2.putText(grid, f"Saved sets: {saved_count}   "
                "[SPACE/C] capture  [B] burst  [Q/ESC] quit",
                (8, grid.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 1)
    return grid


def capture(caps, num, interval):
    """Run the capture session. Returns the number of sets saved."""
    frame_index = next_frame_index()
    saved = 0
    window = "Static scene capture (Q to finish)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    auto_remaining = num if num and num > 0 else 0
    next_auto_time = time.time()

    while True:
        frames = grab_frames(caps)
        if frames is None:
            print("  Warning: dropped frame, retrying...")
            continue

        do_capture = False
        now = time.time()

        if auto_remaining > 0 and now >= next_auto_time:
            do_capture = True
            auto_remaining -= 1
            next_auto_time = now + interval

        cv2.imshow(window, build_preview(frames, saved))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord(" "), ord("c"), ord("C")):
            do_capture = True
        if key in (ord("b"), ord("B")) and auto_remaining == 0:
            auto_remaining = num if num and num > 0 else 10
            next_auto_time = now

        if do_capture:
            save_set(frames, frame_index)
            frame_index += 1
            saved += 1

        if num and num > 0 and auto_remaining == 0 and saved >= num:
            # Non-interactive run requested a fixed count and we're done.
            break

    cv2.destroyWindow(window)
    return saved


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Capture a static scene into sfm/images/ for Stage 3."
    )
    parser.add_argument("--num", type=int, default=0,
                        help="Auto-capture this many synchronized sets, then "
                             "quit (0 = fully interactive).")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between auto-captured sets (default 0.5).")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing frames in sfm/images/ first.")
    args = parser.parse_args()

    cameras = load_camera_indices()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.fresh:
        removed = 0
        for name in os.listdir(OUTPUT_DIR):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                os.remove(os.path.join(OUTPUT_DIR, name))
                removed += 1
        print(f"Removed {removed} existing frame(s) from {OUTPUT_DIR}.")

    print("Static-scene capture")
    print(f"  Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Output: {OUTPUT_DIR}")
    print("  Keep the rig AND the scene perfectly still. Make sure the scene "
          "is texture-rich.\n")

    caps = {}
    try:
        for position, camera_number in cameras:
            print(f"Opening '{position}' (index {camera_number})...")
            caps[position] = open_camera(camera_number)
            time.sleep(0.25)

        if args.num and args.num > 0:
            print(f"\nAuto-capturing {args.num} set(s) at "
                  f"{args.interval}s spacing...")
        else:
            print("\nInteractive: SPACE/C to capture, B for a burst, Q to "
                  "finish.")

        saved = capture(caps, args.num, args.interval)
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()

    print(f"\nDone. Saved {saved} synchronized set(s) to {OUTPUT_DIR}.")
    if saved > 0:
        print("Next: run  python sfm.py")
    else:
        print("No frames captured — nothing to run Stage 3 on yet.")


if __name__ == "__main__":
    main()
