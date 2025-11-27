import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
from skimage import filters, morphology, measure
import matplotlib.pyplot as plt
import pandas as pd

from morphology_pipeline.segmentation.stain_dataset import (
    StainDataset,
    affine_like_skimage_no_resize,
    apply_affine_points,
)
from morphology_pipeline.pseudotime import window_segments, descriptor_from_segments, slide_windows_with_matching
from morphology_pipeline.corridor_mask import deskew_with_hull, detect_corridors_via_hull


# ---------------------------------------------------------------------
# SIMPLE & CORRECT POINT-IN-WINDOW CHECK
# ---------------------------------------------------------------------
def contains_point(cx: float, cy: float,
                   xs: np.ndarray,
                   yt: np.ndarray,
                   yb: np.ndarray) -> bool:
    """
    Check if (cx, cy) lies inside the corridor defined by (xs, yt, yb).
    Linear interpolation between the nearest two x-columns.
    """
    if cx < xs[0] or cx > xs[-1]:
        return False

    # index of first xs[i] >= cx
    i = np.searchsorted(xs, cx)

    # clamp to valid interior segment
    i = np.clip(i, 1, len(xs) - 1)

    # linear interpolation
    x0, x1 = xs[i-1], xs[i]
    t = (cx - x0) / (x1 - x0)

    y_top = yt[i-1] + t * (yt[i] -  yt[i-1])
    y_bot = yb[i-1] + t * (yb[i] -  yb[i-1])

    return y_top <= cy <= y_bot


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def run_pipeline_and_save_csvs(background: np.ndarray,
                               x0_ref: int = None,
                               cor_ref: int = None,
                               folder: str = None):

    SD = StainDataset.from_folder(folder)
    SD.add_center_eccentricity()

    # TODO make them parameters
    m, L, stride, L_min = 2, 30, 1, 8

    # deskew image
    rot_img, corr_mask, hull_mask, rot_deg = deskew_with_hull(background)

    # # visualize images and masks
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(rot_img, cmap="gray")
    # axes[0].set_title("Rotated image")
    # axes[0].axis("off")

    # axes[1].imshow(corr_mask, cmap="gray")
    # axes[1].set_title("Corridor mask")
    # axes[1].axis("off")

    # axes[2].imshow(hull_mask, cmap="gray")
    # axes[2].set_title("Hull mask")
    # axes[2].axis("off")

    # plt.tight_layout()
    # plt.show()

    # input("Press Enter to continue...")

    print(f"Detected rotation: {rot_deg:.3f} degrees")
    # detect corridors
    corridors = detect_corridors_via_hull(
        corr_mask, hull_mask,
        row_cov_thresh_rel_hull=0.07,
        min_band_height=5,
        merge_gap_px=3
    )

    print(f"Detected {len(corridors)} corridors.")

    # pick reference corridor
    centers = [0.5 * (c['y0'] + c['y1']) for c in corridors]
    mid_idx = int(np.argsort(centers)[len(centers)//2])
    mid_corr = corridors[mid_idx]

    if x0_ref is None:
        x_mid = int(0.5 * (mid_corr['x0'] + mid_corr['x1']))
        x0_ref = min(x_mid, mid_corr['x1'] - (L - 1) * m)

    # reference window
    ref_segs = window_segments(corr_mask, mid_corr, x0_ref, m, L, L_min)
    ref_desc = descriptor_from_segments(ref_segs) if ref_segs is not None else None

    # slide all windows
    all_windows = []
    for c in corridors:
        regs = slide_windows_with_matching(
            corr_mask, c, ref_desc,
            m=m, L=L, stride=stride, L_min=L_min,
            tau_abs=2, alpha=0.05, beta=0.20, S_min=0.80
        )
        all_windows.extend(regs)

    # keep matched
    matched_windows = [R for R in all_windows if R["matched"]]

    print(f"Total matched windows: {len(matched_windows)}")

    # prepare window boundaries
    window_infos = []
    for win in matched_windows:
        segs = win.get("segments") or []
        if not segs:
            continue

        xs = np.asarray([s[0] for s in segs], dtype=float)
        yt = np.asarray([s[1] for s in segs], dtype=float)
        yb = np.asarray([s[2] for s in segs], dtype=float)
        order = np.argsort(xs)
        xs, yt, yb = xs[order], yt[order], yb[order]

        window_infos.append((win, xs, yt, yb))

    # rotate all centers (IMPORTANT: converting (row,col)→(y,x)→(x,y))
    H0, W0 = background.shape[:2]
    M = affine_like_skimage_no_resize(W0, H0, rot_deg)

    if "center_rot" not in SD.dataframe.columns:
        SD.dataframe["center_rot"] = None

    # scaling to match background if dataset used different dimensions
    h_ref = float(SD.dataframe["height"].iloc[0]) if "height" in SD.dataframe.columns else H0
    w_ref = float(SD.dataframe["width"].iloc[0]) if "width" in SD.dataframe.columns else W0
    sx = W0 / w_ref if w_ref else 1.0
    sy = H0 / h_ref if h_ref else 1.0

    centers_rot_map: Dict[int, Tuple[float, float]] = {}

    for obj_id, val in SD.dataframe["center"].items():
        if val is None:
            continue

        # val = (row, col) → (y, x)
        row_raw, col_raw = val
        row_arr = np.asarray(row_raw)
        col_arr = np.asarray(col_raw)
        if row_arr.size == 0 or col_arr.size == 0:
            continue

        # scale, preserve orientation
        y = float(row_arr.reshape(-1)[0]) * sy
        x = float(col_arr.reshape(-1)[0]) * sx

        pt_rot = apply_affine_points(
            M, np.asarray([[x, y]], dtype=float)
        )[0]

        x_rot, y_rot = float(pt_rot[0]), float(pt_rot[1])
        centers_rot_map[obj_id] = (x_rot, y_rot)
        SD.dataframe.at[obj_id, "center_rot"] = (x_rot, y_rot)

    # ---------------------------------------------------------------
    # TEST ALL CENTERS AGAINST ALL WINDOWS (no central filtering!)
    # ---------------------------------------------------------------
    selected_objs = []
    mask = np.zeros(len(SD.dataframe), dtype=bool)

    # If corridor mask was cropped out of the rotated image,
    # fix coordinate offset here:
    shift_x = 0   # <-- replace later when you tell me your corr_mask crop
    shift_y = 0

    for obj_id, (cxr, cyr) in centers_rot_map.items():
        cx_loc = cxr - shift_x
        cy_loc = cyr - shift_y

        for win, xs, yt, yb in window_infos:
            if contains_point(cx_loc, cy_loc, xs, yt, yb):
                selected_objs.append({
                    "obj_id": obj_id,
                    "center_rot": (cxr, cyr),
                    "region_id": win.get("region_id"),
                    "corridor_id": win.get("corridor_id"),
                })
                mask[SD.dataframe.index.get_loc(obj_id)] = True
                break

    return SD, mask, selected_objs
