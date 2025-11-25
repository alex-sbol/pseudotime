import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
from skimage import filters, morphology, measure
import matplotlib.pyplot as plt
import pandas as pd
from bisect import bisect_left

from morphology_pipeline.segmentation.stain_dataset import (
    StainDataset,
    affine_like_skimage_no_resize,
    apply_affine_points,
)
from morphology_pipeline.pseudotime import window_segments, descriptor_from_segments, slide_windows_with_matching
from morphology_pipeline.corridor_mask import deskew_with_hull, detect_corridors_via_hull



def _contains_point_in_window(cx: float, cy: float, xs: np.ndarray, yt: np.ndarray, yb: np.ndarray, m_est: float) -> bool:
    """
    Test if (cx,cy) is inside the vertical envelope of the window, using
    linear interpolation between neighbor segments. Guard against large gaps.
    """
    if cx < xs[0] - 0.75 * m_est or cx > xs[-1] + 0.75 * m_est:
        return False

    i = bisect_left(xs, cx)
    if i == 0:
        # near left edge: accept only if close enough to first column
        if (cx - xs[0]) > 0.75 * m_est:
            return False
        y_top, y_bot = yt[0], yb[0]
    elif i == len(xs):
        # near right edge
        if (xs[-1] - cx) > 0.75 * m_est:
            return False
        y_top, y_bot = yt[-1], yb[-1]
    else:
        # between xs[i-1] and xs[i]; reject if the gap is too large
        if (xs[i] - xs[i-1]) > 1.5 * m_est:
            return False
        x0, x1 = xs[i-1], xs[i]
        t = (cx - x0) / (x1 - x0) if x1 != x0 else 0.0
        y_top = yt[i-1] + t * (yt[i] - yt[i-1])
        y_bot = yb[i-1] + t * (yb[i] - yb[i-1])

    return (y_top <= cy <= y_bot)

def _prep_window(win: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Prepare sorted arrays for a window's segments.
        Expects segments as iterable of (x, y_top, y_bot).
        """
        segs = win.get('segments') or []
        if len(segs) == 0:
            return None
        xs = np.asarray([s[0] for s in segs], dtype=float)
        yt = np.asarray([s[1] for s in segs], dtype=float)
        yb = np.asarray([s[2] for s in segs], dtype=float)
        order = np.argsort(xs)
        xs, yt, yb = xs[order], yt[order], yb[order]
        # estimate step (m) from median dx if available
        m_est = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
        return xs, yt, yb, m_est


def run_pipeline_and_save_csvs(background: np.ndarray,
                               x0_ref: int = None,
                               cor_ref: int = None,
                               folder: str = None,
                               ):

    SD = StainDataset.from_folder(folder)
    SD.add_center_eccentricity()
    #TODO make them parameters
    m, L, stride, L_min = 2, 10, 1, 8  # spacing, length, stride, min len

    rot_img, corr_mask, hull_mask, rot_deg = deskew_with_hull(background)

    #one corridor is a {id, y0, y1, x0, x1} dict
    corridors = detect_corridors_via_hull(corr_mask, hull_mask,
                                        row_cov_thresh_rel_hull=0.07,
                                        min_band_height=5, merge_gap_px=3)

    #SD.add_corridor_ids(corridors)
    #TODO make a logic in case the starting point is specified
    if x0_ref is None:
        centers = [0.5 * (c['y0'] + c['y1']) for c in corridors]
        mid_idx = int(np.argsort(centers)[len(centers) // 2])
        mid_corr = corridors[mid_idx]
        x_mid = int(0.5 * (mid_corr['x0'] + mid_corr['x1']))
        x0_ref = min(x_mid, mid_corr['x1'] - (L - 1) * m)  # start at middle, clamp if needed
    #TODO select a corridor by the x0_ref and cor_cef when specified

    ref_segs = window_segments(corr_mask, mid_corr, x0_ref, m, L, L_min)
    ref_desc = descriptor_from_segments(ref_segs) if ref_segs is not None else None

    # slide and match on all corridors
    all_windows = []

    for c in corridors:
        regs = slide_windows_with_matching(corr_mask, c, ref_desc,
                                        m=m, L=L, stride=stride, L_min=L_min,
                                        tau_abs=2, alpha=0.05, beta=0.20, S_min=0.80)
        all_windows.extend(regs)

    matched_windows = [R for R in all_windows if R['matched']]

    res_dir = Path("res")
    res_dir.mkdir(parents=True, exist_ok=True)

    window_infos = []
    for win in matched_windows:
        prep = _prep_window(win)
        if prep is None:
            continue
        segs = win.get("segments") or []
        cx = float(np.mean(prep[0]))
        cy = float(np.mean([0.5 * (s[1] + s[2]) for s in segs])) if segs else 0.0
        window_infos.append((win, prep, (cx, cy)))

    selected_objs = []
    print(window_infos)
    if window_infos:
        wc = np.asarray([w[2] for w in window_infos], dtype=float)
        x_low, x_high = np.percentile(wc[:, 0], [15, 85])
        y_low, y_high = np.percentile(wc[:, 1], [15, 85])
        central_windows = [
            w for w in window_infos
            if x_low <= w[2][0] <= x_high and y_low <= w[2][1] <= y_high
        ]

        H, W = background.shape[:2]
        M = affine_like_skimage_no_resize(W, H, rot_deg)

        valid_idxs, centers_raw = [], []
        for obj_id, val in SD.dataframe["center"].items():
            if val is None:
                continue
            cx, cy = val
            if cx.size < 1 or cy.size < 1:
                print(f"No centre for obj_id {obj_id}")
                continue
            valid_idxs.append(obj_id)
            centers_raw.append((float(cx), float(cy)))

        mask = np.zeros(len(SD.dataframe), dtype=bool)
        if centers_raw and central_windows:
            centers_rot = apply_affine_points(M, np.asarray(centers_raw, dtype=float))
            for obj_id, (cxr, cyr) in zip(valid_idxs, centers_rot):
                if not (x_low <= cxr <= x_high and y_low <= cyr <= y_high):
                    continue
                for win, prep, _ in central_windows:
                    xs, yt, yb, m_est = prep
                    if _contains_point_in_window(cxr, cyr, xs, yt, yb, m_est):
                        print(f"Selected obj_id {obj_id} in corridor {win.get('corridor_id')} region {win.get('region_id')}")
                        selected_objs.append({
                            "obj_id": obj_id,
                            "center_rot": (float(cxr), float(cyr)),
                            "region_id": win.get("region_id"),
                            "corridor_id": win.get("corridor_id"),
                        })
                        mask[SD.dataframe.index.get_loc(obj_id)] = True
                        break

        return SD, mask, selected_objs # selected_objs
    
