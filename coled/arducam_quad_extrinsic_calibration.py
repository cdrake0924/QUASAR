"""
arducam_quad_extrinsic_calibration.py

Four-camera extrinsic calibration with the TOP-LEFT (TL) camera as the
global reference frame.

Workflow
========
Phase A. Synchronized capture from all 4 cameras using a 2x2 mosaic preview.
         A frame is saved only when the checkerboard is detected in TL *and*
         at least one other camera. Frames + a visibility.json record are
         persisted so calibration can be re-run offline.

Phase B. Calibration.
         1. Per-pair stereoCalibrate (TL,TR), (TL,BL), (TL,BR) gives initial
            (R_X, T_X) for each non-base camera, using the per-camera
            intrinsics loaded from intrinsic/cam_<position>_intr.npz.
         2. Per-frame solvePnP on TL bootstraps each frame's board pose
            (rvec_i, tvec_i) in TL's coordinate frame.
         3. Joint bundle adjustment refines all 3 non-base camera poses
            and all N per-frame board poses simultaneously. scipy.optimize.
            least_squares (trust-region with Huber loss) is used, with an
            explicit sparse Jacobian structure for efficiency.
         4. Output extrinsics/arducam_quad_calib.npz.

Outputs
=======
extrinsics/quad_frames/<position>/frame_<idx>.png
extrinsics/quad_frames/visibility.json
extrinsics/arducam_quad_calib.npz   (K_*, dist_*, R_*, T_*, RMS, visibility)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from quad_utils import (
    BASE,
    NON_BASE,
    POSITIONS,
    load_intrinsics_for_layout,
    load_layout,
    make_mosaic_2x2,
    make_object_points,
    open_layout_cameras,
    release_caps,
    scale_intrinsics_dict,
)

# --- Defaults ---
DEFAULT_CAPTURE_COUNT    = 30        # Synchronized frame sets to collect
DEFAULT_CAPTURE_INTERVAL = 1.0       # Seconds between captures
DEFAULT_DW               = (11, 11)  # Sub-pixel refinement window
FRAMES_DIR               = "extrinsics/quad_frames"
VISIBILITY_FILE          = "extrinsics/quad_frames/visibility.json"
OUTPUT_FILE              = "extrinsics/arducam_quad_calib.npz"
# ---------------------


# ===========================================================================
# Phase A: synchronized 4-camera capture
# ===========================================================================
def capture_quad_frames(caps: Dict[str, cv2.VideoCapture],
                        checkerboard: Tuple[int, int],
                        n_frames: int,
                        interval_sec: float,
                        save_dir: str,
                        dw: Tuple[int, int]) -> Tuple[List[dict], dict]:
    """
    Capture synchronized 4-camera frame sets. A set is saved only if TL sees
    the checkerboard AND at least one other camera does too.

    Returns
    -------
    records : list of dicts, one per saved set:
              {idx, paths: {pos: file}, found: {pos: bool},
               corners: {pos: np.ndarray or None}}
    stats   : dict with combination counts and image shape
    """
    for pos in POSITIONS:
        os.makedirs(os.path.join(save_dir, pos), exist_ok=True)

    records: List[dict] = []
    pair_counts = {x: 0 for x in NON_BASE}
    all_four_count = 0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)
    last_capture_time = 0.0
    img_size: Tuple[int, int] | None = None

    print(f"\nCapturing up to {n_frames} synchronized frame sets.")
    print("Rule: TL must see the board AND at least one other camera must too.")
    print("Press 'q' to stop early. Aim for many frames where ALL 4 see it.\n")

    while len(records) < n_frames:
        frames: Dict[str, np.ndarray | None] = {p: None for p in POSITIONS}
        grabs_ok = True
        for pos in POSITIONS:
            ok, frame = caps[pos].read()
            if not ok or frame is None:
                grabs_ok = False
                break
            frames[pos] = frame
        if not grabs_ok:
            time.sleep(0.02)
            continue

        if img_size is None:
            h, w = frames[BASE].shape[:2]
            img_size = (w, h)

        found: Dict[str, bool] = {}
        corners: Dict[str, np.ndarray | None] = {}
        displays: Dict[str, np.ndarray] = {}
        for pos in POSITIONS:
            f = frames[pos]
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            ok, c = cv2.findChessboardCorners(gray, checkerboard, None)
            found[pos] = bool(ok)
            disp = f.copy()
            if ok:
                c = cv2.cornerSubPix(gray, c, dw, (-1, -1), criteria)
                cv2.drawChessboardCorners(disp, checkerboard, c, ok)
                corners[pos] = c
            else:
                corners[pos] = None
                cv2.putText(disp, "NOT FOUND", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            displays[pos] = disp

        now = time.time()
        tl_seen = found[BASE]
        others_seen = sum(found[x] for x in NON_BASE)
        eligible = tl_seen and others_seen >= 1

        labels = {
            p: f"{p.upper()} {'OK' if found[p] else 'X'}" for p in POSITIONS
        }

        if eligible and (now - last_capture_time) >= interval_sec:
            idx = len(records)
            paths: Dict[str, str] = {}
            for pos in POSITIONS:
                p_path = os.path.join(save_dir, pos, f"frame_{idx:03d}.png")
                cv2.imwrite(p_path, frames[pos])
                paths[pos] = p_path

            records.append({
                "idx": idx,
                "paths": paths,
                "found": dict(found),
                "corners": dict(corners),
            })
            for x in NON_BASE:
                if found[x]:
                    pair_counts[x] += 1
            if all(found[p] for p in POSITIONS):
                all_four_count += 1

            last_capture_time = now
            captured_label = f"CAPTURED set {len(records)}/{n_frames}"
            print(f"  Captured set {len(records):3d}/{n_frames}  "
                  f"TL={int(found['tl'])} TR={int(found['tr'])} "
                  f"BL={int(found['bl'])} BR={int(found['br'])}  "
                  f"(all-4: {all_four_count})")
            for p in POSITIONS:
                cv2.putText(displays[p], captured_label, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif tl_seen:
            for p in POSITIONS:
                cv2.putText(displays[p],
                            f"TL seen, {others_seen}/3 others — waiting...",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 200, 255), 2)
        else:
            for p in POSITIONS:
                cv2.putText(displays[p], "TL must see the board",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

        mosaic = make_mosaic_2x2(displays, scale=0.35, labels=labels)
        cv2.putText(mosaic,
                    f"Sets {len(records)}/{n_frames} | all-4: {all_four_count} | "
                    f"TR:{pair_counts['tr']} BL:{pair_counts['bl']} BR:{pair_counts['br']} "
                    "| 'q' to stop",
                    (10, mosaic.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Quad Extrinsic Capture", mosaic)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("  Stopped early by user.")
            break

    cv2.destroyAllWindows()

    stats = {
        "n_sets": len(records),
        "all_four": all_four_count,
        "pairs": pair_counts,
        "img_size": list(img_size) if img_size else None,
    }
    print(f"\nCapture complete: {len(records)} sets "
          f"(all-4 visible in {all_four_count}).")
    print(f"  Pairs with TL: TR={pair_counts['tr']}  "
          f"BL={pair_counts['bl']}  BR={pair_counts['br']}")
    return records, stats


def write_visibility_json(records: List[dict], stats: dict, path: str) -> None:
    payload = {
        "stats": stats,
        "frames": [
            {
                "idx": r["idx"],
                "paths": r["paths"],
                "found": r["found"],
            }
            for r in records
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Visibility map written: {path}")


def load_records_from_disk(visibility_path: str,
                           checkerboard: Tuple[int, int],
                           dw: Tuple[int, int]) -> Tuple[List[dict], dict]:
    """Re-detect corners on saved frames so we can recompute without recapture."""
    with open(visibility_path, "r") as f:
        payload = json.load(f)
    stats = payload["stats"]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

    records: List[dict] = []
    for entry in payload["frames"]:
        rec = {
            "idx": entry["idx"],
            "paths": entry["paths"],
            "found": {},
            "corners": {},
        }
        for pos in POSITIONS:
            img = cv2.imread(entry["paths"][pos])
            if img is None:
                rec["found"][pos] = False
                rec["corners"][pos] = None
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ok, corners = cv2.findChessboardCorners(gray, checkerboard, None)
            if ok:
                corners = cv2.cornerSubPix(gray, corners, dw, (-1, -1), criteria)
                rec["found"][pos] = True
                rec["corners"][pos] = corners
            else:
                rec["found"][pos] = bool(entry["found"].get(pos, False))
                rec["corners"][pos] = None
        records.append(rec)
    return records, stats


# ===========================================================================
# Phase B step 1: per-pair stereoCalibrate bootstrap
# ===========================================================================
def bootstrap_pair_extrinsics(records: List[dict],
                              non_base: str,
                              objp: np.ndarray,
                              intrinsics: dict,
                              img_size: Tuple[int, int],
                              checkerboard: Tuple[int, int]
                              ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run cv2.stereoCalibrate(TL, X) with CALIB_FIX_INTRINSIC on frames where
    both cameras saw the board. Returns (R, T, rms_pair).
    """
    objpoints = []
    img_tl, img_x = [], []
    for rec in records:
        if rec["found"][BASE] and rec["found"][non_base]:
            objpoints.append(objp)
            img_tl.append(rec["corners"][BASE])
            img_x.append(rec["corners"][non_base])

    if len(objpoints) < 4:
        raise RuntimeError(
            f"Bootstrap (TL, {non_base.upper()}): only {len(objpoints)} shared "
            "frame(s). Recapture with more overlap."
        )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv2.CALIB_FIX_INTRINSIC

    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objpoints,
        img_tl,
        img_x,
        intrinsics[BASE]["K"], intrinsics[BASE]["dist"],
        intrinsics[non_base]["K"], intrinsics[non_base]["dist"],
        img_size,
        criteria=criteria,
        flags=flags,
    )
    print(f"  Bootstrap pair TL-{non_base.upper()}: {len(objpoints)} frames, "
          f"RMS={rms:.4f} px, baseline={float(np.linalg.norm(T)):.2f} mm")
    return R, T, float(rms)


# ===========================================================================
# Phase B step 2: per-frame board-pose bootstrap from TL
# ===========================================================================
def bootstrap_board_poses(records: List[dict],
                          objp: np.ndarray,
                          K_tl: np.ndarray,
                          dist_tl: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    For every frame where TL saw the board, solvePnP gives an initial board
    pose in TL's coordinate frame. Returns list aligned with records (None for
    frames where TL did not see the board — those frames will be dropped from BA).
    """
    poses: List[Tuple[np.ndarray, np.ndarray] | None] = []
    for rec in records:
        if not rec["found"][BASE]:
            poses.append(None)
            continue
        ok, rvec, tvec = cv2.solvePnP(
            objp, rec["corners"][BASE], K_tl, dist_tl,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            poses.append(None)
        else:
            poses.append((rvec.reshape(3), tvec.reshape(3)))
    return poses


# ===========================================================================
# Phase B step 3: joint bundle adjustment
# ===========================================================================
class BAProblem:
    """
    Pack/unpack helpers + residual + sparsity for the joint optimization.

    Parameter vector layout:
        [ rvec_tr (3), tvec_tr (3),
          rvec_bl (3), tvec_bl (3),
          rvec_br (3), tvec_br (3),
          rvec_f0 (3), tvec_f0 (3),
          rvec_f1 (3), tvec_f1 (3),
          ... ]
    The first 18 entries are non-base camera poses (TR, BL, BR), in TL's frame.
    Per-frame poses follow, one (rvec, tvec) block per active frame.
    """

    CAMERA_BLOCK_SIZE = 6
    FRAME_BLOCK_SIZE = 6
    NUM_CAMERA_BLOCKS = 3  # TR, BL, BR

    def __init__(self,
                 active_records: List[dict],
                 active_board_poses: List[Tuple[np.ndarray, np.ndarray]],
                 intrinsics: dict,
                 objp: np.ndarray):
        self.records = active_records
        self.board_poses = active_board_poses
        self.intrinsics = intrinsics
        self.objp = objp.astype(np.float64)
        self.n_corners = objp.shape[0]
        self.n_frames = len(active_records)

        # Camera index lookup: TR=0, BL=1, BR=2 (TL has no camera block).
        self.cam_idx = {p: i for i, p in enumerate(NON_BASE)}

        # observations[k] = (frame_i, camera_pos, corners_Nx1x2 float64)
        self.observations: List[Tuple[int, str, np.ndarray]] = []
        for i, rec in enumerate(active_records):
            for pos in POSITIONS:
                if rec["found"][pos] and rec["corners"][pos] is not None:
                    self.observations.append(
                        (i, pos, rec["corners"][pos].astype(np.float64).reshape(-1, 2))
                    )

        self.n_obs = len(self.observations)
        self.n_residuals = self.n_obs * self.n_corners * 2
        self.n_params = (
            self.NUM_CAMERA_BLOCKS * self.CAMERA_BLOCK_SIZE
            + self.n_frames * self.FRAME_BLOCK_SIZE
        )

    # ---- packing ---------------------------------------------------------
    def pack(self,
             cam_poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
             board_poses: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        x = np.zeros(self.n_params, dtype=np.float64)
        for p in NON_BASE:
            i = self.cam_idx[p]
            r, t = cam_poses[p]
            off = i * self.CAMERA_BLOCK_SIZE
            x[off:off + 3] = np.asarray(r, dtype=np.float64).reshape(3)
            x[off + 3:off + 6] = np.asarray(t, dtype=np.float64).reshape(3)
        base = self.NUM_CAMERA_BLOCKS * self.CAMERA_BLOCK_SIZE
        for i, (r, t) in enumerate(board_poses):
            off = base + i * self.FRAME_BLOCK_SIZE
            x[off:off + 3] = np.asarray(r, dtype=np.float64).reshape(3)
            x[off + 3:off + 6] = np.asarray(t, dtype=np.float64).reshape(3)
        return x

    def unpack(self, x: np.ndarray
               ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]],
                          List[Tuple[np.ndarray, np.ndarray]]]:
        cam_poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for p in NON_BASE:
            i = self.cam_idx[p]
            off = i * self.CAMERA_BLOCK_SIZE
            cam_poses[p] = (x[off:off + 3].copy(), x[off + 3:off + 6].copy())
        base = self.NUM_CAMERA_BLOCKS * self.CAMERA_BLOCK_SIZE
        board_poses = []
        for i in range(self.n_frames):
            off = base + i * self.FRAME_BLOCK_SIZE
            board_poses.append(
                (x[off:off + 3].copy(), x[off + 3:off + 6].copy())
            )
        return cam_poses, board_poses

    # ---- residual -------------------------------------------------------
    def residuals(self, x: np.ndarray) -> np.ndarray:
        cam_poses, board_poses = self.unpack(x)
        out = np.empty(self.n_residuals, dtype=np.float64)
        cursor = 0
        block = self.n_corners * 2

        for (frame_i, pos, observed) in self.observations:
            r_b, t_b = board_poses[frame_i]
            K = self.intrinsics[pos]["K"]
            dist = self.intrinsics[pos]["dist"]

            if pos == BASE:
                rvec, tvec = r_b, t_b
            else:
                r_c, t_c = cam_poses[pos]
                rvec, tvec, *_ = cv2.composeRT(
                    r_b.reshape(3, 1), t_b.reshape(3, 1),
                    r_c.reshape(3, 1), t_c.reshape(3, 1),
                )
                rvec = rvec.reshape(3)
                tvec = tvec.reshape(3)

            projected, _ = cv2.projectPoints(
                self.objp, rvec.astype(np.float64), tvec.astype(np.float64),
                K, dist,
            )
            projected = projected.reshape(-1, 2)

            out[cursor:cursor + block] = (observed - projected).ravel()
            cursor += block

        return out

    # ---- sparsity -------------------------------------------------------
    def jacobian_sparsity(self):
        from scipy.sparse import lil_matrix
        m = lil_matrix((self.n_residuals, self.n_params), dtype=np.uint8)
        block = self.n_corners * 2
        frame_base = self.NUM_CAMERA_BLOCKS * self.CAMERA_BLOCK_SIZE
        for k, (frame_i, pos, _obs) in enumerate(self.observations):
            r0 = k * block
            r1 = r0 + block
            f_off = frame_base + frame_i * self.FRAME_BLOCK_SIZE
            m[r0:r1, f_off:f_off + self.FRAME_BLOCK_SIZE] = 1
            if pos != BASE:
                c_off = self.cam_idx[pos] * self.CAMERA_BLOCK_SIZE
                m[r0:r1, c_off:c_off + self.CAMERA_BLOCK_SIZE] = 1
        return m


def rms_from_residuals(res: np.ndarray, n_obs: int, n_corners: int) -> float:
    """RMS reprojection error in pixels."""
    if n_obs == 0:
        return float("nan")
    return float(np.sqrt(np.sum(res ** 2) / (n_obs * n_corners)))


# ===========================================================================
# Driver
# ===========================================================================
def run_calibration(records: List[dict],
                    intrinsics: dict,
                    img_size: Tuple[int, int],
                    checkerboard: Tuple[int, int],
                    square_size_mm: float,
                    huber_delta: float = 1.0
                    ) -> dict:
    """Run bootstrap + bundle adjustment. Returns dict ready to save."""
    try:
        from scipy.optimize import least_squares
    except ImportError:
        print("Error: scipy is required for bundle adjustment.")
        print("  Install with:  pip install scipy")
        raise

    objp = make_object_points(checkerboard, square_size_mm)

    print("\n--- Phase B.1: per-pair stereoCalibrate bootstrap ---")
    cam_poses_R: Dict[str, np.ndarray] = {}
    cam_poses_T: Dict[str, np.ndarray] = {}
    bootstrap_rms: Dict[str, float] = {}
    for x in NON_BASE:
        R_x, T_x, rms_x = bootstrap_pair_extrinsics(
            records, x, objp, intrinsics, img_size, checkerboard
        )
        cam_poses_R[x] = R_x
        cam_poses_T[x] = T_x
        bootstrap_rms[x] = rms_x

    print("\n--- Phase B.2: per-frame board pose bootstrap (solvePnP on TL) ---")
    all_board_poses = bootstrap_board_poses(
        records, objp, intrinsics[BASE]["K"], intrinsics[BASE]["dist"]
    )
    active_indices = [i for i, p in enumerate(all_board_poses) if p is not None]
    if not active_indices:
        raise RuntimeError("No frames where TL had a valid board pose. Cannot run BA.")
    active_records = [records[i] for i in active_indices]
    active_board_poses = [all_board_poses[i] for i in active_indices]
    print(f"  Active frames (TL pose recovered): {len(active_records)}")

    cam_poses_init: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for x in NON_BASE:
        rvec_x, _ = cv2.Rodrigues(cam_poses_R[x])
        cam_poses_init[x] = (rvec_x.reshape(3), cam_poses_T[x].reshape(3))

    print("\n--- Phase B.3: joint bundle adjustment ---")
    problem = BAProblem(active_records, active_board_poses, intrinsics, objp)
    x0 = problem.pack(cam_poses_init, active_board_poses)
    res0 = problem.residuals(x0)
    rms_initial = rms_from_residuals(res0, problem.n_obs, problem.n_corners)
    print(f"  Initial RMS (across all observations): {rms_initial:.4f} px")
    print(f"  Parameters: {problem.n_params}  |  Residuals: {problem.n_residuals}  "
          f"|  Observations: {problem.n_obs}")

    sparsity = problem.jacobian_sparsity()
    print("  Running least_squares (TRF + Huber loss)...")
    t0 = time.time()
    result = least_squares(
        problem.residuals,
        x0,
        jac_sparsity=sparsity,
        method="trf",
        loss="huber",
        f_scale=huber_delta,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=200,
        verbose=2,
    )
    dt = time.time() - t0
    rms_final = rms_from_residuals(result.fun, problem.n_obs, problem.n_corners)
    print(f"  BA finished in {dt:.1f}s — RMS: {rms_initial:.4f} -> {rms_final:.4f} px")

    cam_poses_final, board_poses_final = problem.unpack(result.x)

    per_camera_rms: Dict[str, float] = {}
    block = problem.n_corners * 2
    for pos in POSITIONS:
        sq, cnt = 0.0, 0
        for k, (_fi, p, _obs) in enumerate(problem.observations):
            if p == pos:
                sq += float(np.sum(result.fun[k * block:(k + 1) * block] ** 2))
                cnt += 1
        per_camera_rms[pos] = (
            float(np.sqrt(sq / (cnt * problem.n_corners))) if cnt else float("nan")
        )

    R_out: Dict[str, np.ndarray] = {}
    T_out: Dict[str, np.ndarray] = {}
    for pos in NON_BASE:
        rvec, tvec = cam_poses_final[pos]
        R_mat, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        R_out[pos] = R_mat
        T_out[pos] = tvec.reshape(3, 1)

    visibility = np.zeros((len(records), len(POSITIONS)), dtype=bool)
    for i, rec in enumerate(records):
        for j, p in enumerate(POSITIONS):
            visibility[i, j] = bool(rec["found"][p])

    return {
        "intrinsics": intrinsics,
        "R": R_out,
        "T": T_out,
        "rms_initial": rms_initial,
        "rms_final": rms_final,
        "bootstrap_rms": bootstrap_rms,
        "per_camera_rms": per_camera_rms,
        "n_active_frames": len(active_records),
        "n_frames_total": len(records),
        "visibility": visibility,
        "img_size": img_size,
        "checkerboard": checkerboard,
        "square_size_mm": square_size_mm,
    }


def save_quad_calibration(result: dict, path: str) -> None:
    intr = result["intrinsics"]
    save_kwargs = {
        "K_tl":   intr["tl"]["K"],   "dist_tl": intr["tl"]["dist"],
        "K_tr":   intr["tr"]["K"],   "dist_tr": intr["tr"]["dist"],
        "K_bl":   intr["bl"]["K"],   "dist_bl": intr["bl"]["dist"],
        "K_br":   intr["br"]["K"],   "dist_br": intr["br"]["dist"],
        "R_tr": result["R"]["tr"], "T_tr": result["T"]["tr"],
        "R_bl": result["R"]["bl"], "T_bl": result["T"]["bl"],
        "R_br": result["R"]["br"], "T_br": result["T"]["br"],
        "rms_initial": np.float64(result["rms_initial"]),
        "rms_final":   np.float64(result["rms_final"]),
        "per_camera_rms": np.array(
            [result["per_camera_rms"][p] for p in POSITIONS], dtype=np.float64
        ),
        "bootstrap_rms": np.array(
            [result["bootstrap_rms"][p] for p in NON_BASE], dtype=np.float64
        ),
        "visibility": result["visibility"],
        "img_size": np.array(result["img_size"], dtype=np.int32),
        "checkerboard": np.array(result["checkerboard"], dtype=np.int32),
        "square_size_mm": np.float64(result["square_size_mm"]),
        "positions": np.array(POSITIONS),
        "non_base":  np.array(NON_BASE),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **save_kwargs)
    print(f"\nQuad calibration saved to: {path}")


def print_summary(result: dict) -> None:
    print("\n=========== Quad Extrinsic Calibration Summary ===========")
    print(f"Active frames used in BA : {result['n_active_frames']} / "
          f"{result['n_frames_total']}")
    print(f"Initial RMS (px)         : {result['rms_initial']:.4f}")
    print(f"Final RMS   (px)         : {result['rms_final']:.4f}")
    if result["rms_final"] > 1.0:
        print("  Warning: final RMS > 1 px — capture more varied / closer board poses.")
    print("\nPer-camera reprojection RMS (px):")
    for p in POSITIONS:
        print(f"  {p.upper()}: {result['per_camera_rms'][p]:.4f}")

    print("\nBaselines from TL (mm | inches):")
    for p in NON_BASE:
        b_mm = float(np.linalg.norm(result["T"][p]))
        print(f"  TL -> {p.upper()}: {b_mm:8.2f} mm  ({b_mm/25.4:6.2f} in)")

    print("\nPairwise non-base baselines (mm):")
    for a in NON_BASE:
        for b in NON_BASE:
            if a >= b:
                continue
            d = float(np.linalg.norm(result["T"][a] - result["T"][b]))
            print(f"  {a.upper()} -> {b.upper()}: {d:8.2f} mm")
    print("==========================================================")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="4-camera extrinsic calibration with global bundle adjustment "
                    "(TL is the reference)."
    )
    parser.add_argument("--layout", default="camera_layout.json",
                        help="Path to camera_layout.json (default: %(default)s).")
    parser.add_argument("--intrinsic-dir", default="intrinsic",
                        help="Directory holding cam_<position>_intr.npz files.")
    parser.add_argument("--save-dir", default=FRAMES_DIR,
                        help="Where captured frames are stored (default: %(default)s).")
    parser.add_argument("--visibility-file", default=VISIBILITY_FILE,
                        help="visibility.json path (default: %(default)s).")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help="Calibration .npz output path (default: %(default)s).")
    parser.add_argument("--count", type=int, default=DEFAULT_CAPTURE_COUNT,
                        help=f"Number of frame sets to capture (default: "
                             f"{DEFAULT_CAPTURE_COUNT}).")
    parser.add_argument("--interval", type=float, default=DEFAULT_CAPTURE_INTERVAL,
                        help=f"Min seconds between captures (default: "
                             f"{DEFAULT_CAPTURE_INTERVAL}).")
    parser.add_argument("--from-saved", action="store_true",
                        help="Skip capture; re-detect corners on frames listed in "
                             "visibility.json and re-run calibration only.")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Huber loss scale in pixels (default: 1.0).")
    parser.add_argument("--capture-width", type=int, default=None,
                        help="Override capture width (layout.frame_width is "
                             "treated as the intrinsic resolution; intrinsics "
                             "are auto-scaled). Lower this if the USB hub "
                             "can't sustain 4 cameras at full resolution.")
    parser.add_argument("--capture-height", type=int, default=None,
                        help="Override capture height (same notes as "
                             "--capture-width).")
    parser.add_argument("--capture-fps", type=int, default=None,
                        help="Override capture fps for this run only.")
    parser.add_argument("--no-strict-resolution", action="store_true",
                        help="Don't abort if a camera silently falls back to "
                             "a lower resolution. Useful for debugging only.")
    args = parser.parse_args()

    layout = load_layout(args.layout)
    print(f"Layout loaded from '{args.layout}':")
    for p in POSITIONS:
        print(f"  {p.upper()}: index={layout[p]}")
    print(f"  Frame: {layout['frame_width']}x{layout['frame_height']} @ "
          f"{layout['fps']} fps   Board: {layout['checkerboard']}   "
          f"Square: {layout['square_size_mm']} mm")

    print("\nLoading intrinsics...")
    intrinsics_full = load_intrinsics_for_layout(args.intrinsic_dir)

    intrinsic_size = (layout["frame_width"], layout["frame_height"])
    capture_size = (
        args.capture_width  if args.capture_width  is not None else intrinsic_size[0],
        args.capture_height if args.capture_height is not None else intrinsic_size[1],
    )
    if capture_size != intrinsic_size:
        print(f"  Scaling intrinsics from {intrinsic_size} (calibration) "
              f"to {capture_size} (capture).")
    intrinsics = scale_intrinsics_dict(intrinsics_full, intrinsic_size, capture_size)
    for p in POSITIONS:
        K = intrinsics[p]["K"]
        print(f"  {p.upper()}: fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
              f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")

    # -------- Phase A: capture or reload --------------------------------
    if args.from_saved:
        if not os.path.exists(args.visibility_file):
            print(f"Error: --from-saved given but {args.visibility_file} not found.")
            sys.exit(1)
        print(f"\nRe-detecting corners on saved frames in {args.visibility_file}...")
        records, stats = load_records_from_disk(
            args.visibility_file, layout["checkerboard"], DEFAULT_DW
        )
        img_size = tuple(stats["img_size"])
    else:
        print("\nOpening cameras...")
        caps = open_layout_cameras(
            layout,
            width=capture_size[0],
            height=capture_size[1],
            fps=args.capture_fps,
            strict=not args.no_strict_resolution,
        )
        try:
            records, stats = capture_quad_frames(
                caps,
                layout["checkerboard"],
                args.count,
                args.interval,
                args.save_dir,
                DEFAULT_DW,
            )
        finally:
            release_caps(caps)

        if not records:
            print("Error: no frames captured. Exiting.")
            sys.exit(1)

        write_visibility_json(records, stats, args.visibility_file)
        img_size = tuple(stats["img_size"])

    if len(records) < 4:
        print(f"Error: only {len(records)} usable frame sets — need at least 4.")
        sys.exit(1)

    # -------- Phase B: calibrate ----------------------------------------
    result = run_calibration(
        records,
        intrinsics,
        img_size,
        layout["checkerboard"],
        layout["square_size_mm"],
        huber_delta=args.huber_delta,
    )

    save_quad_calibration(result, args.output)
    print_summary(result)
    print("\nNext step: python arducam_quad_foundation_prep.py")
