"""
Dynamic-scene capture helper (quasar/perception)

Records synchronized frames from all 4 cameras over time and writes them into
mvs/frames/{frame:06d}/{position}.jpg  (e.g. mvs/frames/000001/top_left.jpg) —
exactly the layout Stage 4 (mvs.py) expects: one folder per synchronized moment.

This is for the DYNAMIC capture (things moving in the scene). The camera poses
were already fixed by the static SfM pass, so here we only need a clean,
well-synchronized image sequence. To keep the 4 views as close to simultaneous
as possible, every loop iteration grabs all cameras first (fast) and only then
decodes them.

Controls (interactive mode):
  R / SPACE  start / stop recording (saves a set every 1/fps while recording)
  C          capture a single synchronized set (one moment)
  Q / ESC    finish and exit

Run:
    python dynamic_scene.py                  # interactive
    python dynamic_scene.py --fps 15         # target 15 sets/sec while recording
    python dynamic_scene.py --duration 5     # auto-record 5 seconds, then quit
    python dynamic_scene.py --fresh          # wipe mvs/frames/ first
"""

import argparse
import json
import os
import shutil
import time

import cv2
import numpy as np


# --- Configuration -----------------------------------------------------------

FRAME_WIDTH = 1024
FRAME_HEIGHT = 768
FPS = 10

POSITION_ORDER = ["top_left", "top_right", "bot_left", "bot_right"]

HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA_JSON = os.path.join(HERE, "camera.json")
OUTPUT_DIR = os.path.join(HERE, "mvs", "frames")


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
        # Keep buffers small so reads return the freshest frame (less latency
        # / drift between the 4 cameras). Not honored by every backend.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
    """Continue numbering after any frame folders already in OUTPUT_DIR."""
    if not os.path.isdir(OUTPUT_DIR):
        return 1
    highest = 0
    for name in os.listdir(OUTPUT_DIR):
        if os.path.isdir(os.path.join(OUTPUT_DIR, name)) and name.isdigit():
            highest = max(highest, int(name))
    return highest + 1


def grab_frames(caps):
    """
    Read one frame from every camera with tight synchronization: grab the
    latest frame from all cameras first (cheap), then decode them. Returns a
    {position: frame} dict, or None on any failure.
    """
    for cap in caps.values():
        if not cap.grab():
            return None
    frames = {}
    for position, cap in caps.items():
        ok, frame = cap.retrieve()
        if not ok:
            return None
        frames[position] = frame
    return frames


def save_set(frames, frame_index):
    """Write one synchronized set to mvs/frames/{index:06d}/{position}.jpg."""
    folder = os.path.join(OUTPUT_DIR, f"{frame_index:06d}")
    os.makedirs(folder, exist_ok=True)
    for position in POSITION_ORDER:
        cv2.imwrite(os.path.join(folder, f"{position}.jpg"), frames[position])


def make_tile(frame, label):
    tile = frame.copy()
    cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    return tile


def build_preview(frames, saved_count, recording):
    tiles = {p: make_tile(frames[p], p) for p in POSITION_ORDER}
    top = np.hstack([tiles["top_left"], tiles["top_right"]])
    bottom = np.hstack([tiles["bot_left"], tiles["bot_right"]])
    grid = np.vstack([top, bottom])

    if recording:
        cv2.circle(grid, (grid.shape[1] - 24, 24), 10, (0, 0, 255), -1)
        cv2.putText(grid, "REC", (grid.shape[1] - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(grid, f"Frames: {saved_count}   "
                "[R/SPACE] record  [C] single  [Q/ESC] quit",
                (8, grid.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 1)
    return grid


def capture(caps, fps, duration):
    """Run the capture session. Returns the number of synchronized sets saved."""
    frame_index = next_frame_index()
    saved = 0
    interval = 1.0 / fps if fps > 0 else 0.0
    window = "Dynamic scene capture (Q to finish)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    auto = duration is not None and duration > 0
    recording = auto                      # auto mode starts recording at once
    record_start = time.time() if auto else None
    next_capture_time = time.time()

    while True:
        frames = grab_frames(caps)
        if frames is None:
            print("  Warning: dropped frame, retrying...")
            continue

        now = time.time()
        do_single = False

        cv2.imshow(window, build_preview(frames, saved, recording))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("r"), ord("R"), ord(" ")):
            recording = not recording
            if recording:
                next_capture_time = now
            print("  Recording " + ("started." if recording else "stopped."))
        if key in (ord("c"), ord("C")):
            do_single = True

        if recording and now >= next_capture_time:
            save_set(frames, frame_index)
            frame_index += 1
            saved += 1
            next_capture_time = now + interval
        elif do_single:
            save_set(frames, frame_index)
            print(f"    Saved single set {frame_index:06d}")
            frame_index += 1
            saved += 1

        if auto and (now - record_start) >= duration:
            break

    cv2.destroyWindow(window)
    return saved


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Capture a dynamic scene into mvs/frames/ for Stage 4."
    )
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Target synchronized sets per second while "
                             "recording (default 15).")
    parser.add_argument("--duration", type=float, default=None,
                        help="Auto-record this many seconds, then quit "
                             "(non-interactive). Omit for interactive mode.")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing frame folders in mvs/frames/ "
                             "first.")
    args = parser.parse_args()

    cameras = load_camera_indices()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.fresh:
        removed = 0
        for name in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, name)
            if os.path.isdir(path) and name.isdigit():
                shutil.rmtree(path)
                removed += 1
        print(f"Removed {removed} existing frame folder(s) from {OUTPUT_DIR}.")

    print("Dynamic-scene capture")
    print(f"  Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Output: {OUTPUT_DIR}")
    print("  Keep the RIG fixed (do not move the cameras) — only the scene "
          "should move.\n")

    caps = {}
    try:
        for position, camera_number in cameras:
            print(f"Opening '{position}' (index {camera_number})...")
            caps[position] = open_camera(camera_number)
            time.sleep(0.25)

        if args.duration and args.duration > 0:
            print(f"\nAuto-recording {args.duration}s at ~{args.fps} fps...")
        else:
            print("\nInteractive: R/SPACE to start/stop recording, C for a "
                  "single set, Q to finish.")

        saved = capture(caps, args.fps, args.duration)
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()

    n_folders = saved
    print(f"\nDone. Saved {n_folders} synchronized set(s) to {OUTPUT_DIR}.")
    if saved > 0:
        first = next_frame_index() - saved
        print(f"Frame folders {first:06d} .. {next_frame_index() - 1:06d}.")
        print("Next: run  python mvs.py --start_frame "
              f"{first} --end_frame {first + min(saved, 5) - 1}   "
              "(validate on a few frames first)")
    else:
        print("No frames captured — nothing to run Stage 4 on yet.")


if __name__ == "__main__":
    main()
