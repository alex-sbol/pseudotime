import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List, Union

from morphology_pipeline.segmentation.stain_dataset import (
    StainDataset,
    affine_like_skimage_no_resize,
    apply_affine_points,
)
from morphology_pipeline.pseudotime import (
    window_segments,
    descriptor_from_segments,
    slide_windows_with_matching,
)
from morphology_pipeline.corridor_mask import (
    deskew_with_hull,
    detect_corridors_via_hull,
)


# ---------------------------------------------------------------------
# utility: correct point-in-window check
# ---------------------------------------------------------------------
def contains_point(
    cx: float,
    cy: float,
    xs: np.ndarray,
    yt: np.ndarray,
    yb: np.ndarray,
) -> bool:
    """
    Check if point (cx, cy) lies inside a corridor band defined by
    x-coordinates xs and top/bottom boundaries yt(x), yb(x).
    Linear interpolation between neighbouring xs.
    """
    if cx < xs[0] or cx > xs[-1]:
        return False

    i = np.searchsorted(xs, cx)
    i = int(np.clip(i, 1, len(xs) - 1))

    x0, x1 = xs[i - 1], xs[i]
    if x1 == x0:
        t = 0.0
    else:
        t = (cx - x0) / (x1 - x0)

    y_top = yt[i - 1] + t * (yt[i] - yt[i - 1])
    y_bot = yb[i - 1] + t * (yb[i] - yb[i - 1])

    return y_top <= cy <= y_bot


def _as_scalar(x):
    """Convert arbitrary stored value (possibly array) to scalar float."""
    arr = np.asarray(x)
    if arr.size == 0:
        return np.nan
    return float(arr.ravel()[0])


# ---------------------------------------------------------------------
# MAIN PIPELINE: detection, corridors, centers, corridor mask etc.
# ---------------------------------------------------------------------
def run_pipeline_and_save_csvs(
    background: np.ndarray,
    x0_ref: int = None,
    cor_ref: Union[int, str, None] = None,
    folder: str = None,
):
    """
    Full TopoChip corridor pipeline.

    Parameters
    ----------
    background : np.ndarray
        Raw background image (TopoChip overview).
    x0_ref : int or None
        X-position (in pixels) along the chosen reference corridor where
        the reference window starts. If None, use corridor center.
    cor_ref : int, str or None
        Which corridor to use as reference:
        - None      : middle corridor by vertical position (default)
        - int k     : use corridors[k]
        - "top"     : corridor with smallest vertical center
        - "bottom"  : corridor with largest vertical center
    folder : str
        Path to folder with DAPI/YAP/actin images for StainDataset.

    Returns
    -------
    SD : StainDataset
        Dataset with all cell features.

    mask : np.ndarray[bool]
        Boolean mask over SD.dataframe rows: True = cell lies inside
        one of the selected corridor windows.

    selected_objs : list of dict
        Info about each selected cell:
        - obj_id, center_rot, region_id, corridor_id

    window_infos : list
        Each element is (win_dict, xs, yt, yb) describing one matched window.

    centers_rot_map : dict
        Mapping obj_id -> (x_rot, y_rot) in rotated image coordinates.

    rot_img : np.ndarray
        Rotated background image.
    """

    # ------------------------------------------------------------------
    # load dataset & basic shape features
    # ------------------------------------------------------------------
    SD = StainDataset.from_folder(folder)
    SD.add_center_eccentricity()   # adds 'center' and 'eccentricity' columns

    # TODO: make parameters configurable
    m, L, stride, L_min = 2, 30, 1, 8

    # ------------------------------------------------------------------
    # deskew image & detect corridors via hull
    # ------------------------------------------------------------------
    rot_img, corr_mask, hull_mask, rot_deg = deskew_with_hull(background)

    # visualize basic masks
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rot_img, cmap="gray")
    axes[0].set_title("Rotated image")
    axes[0].axis("off")

    axes[1].imshow(corr_mask, cmap="gray")
    axes[1].set_title("Corridor mask")
    axes[1].axis("off")

    axes[2].imshow(hull_mask, cmap="gray")
    axes[2].set_title("Hull mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Detected rotation: {rot_deg:.3f} degrees")

    corridors = detect_corridors_via_hull(
        corr_mask,
        hull_mask,
        row_cov_thresh_rel_hull=0.07,
        min_band_height=5,
        merge_gap_px=3,
    )

    print(f"Detected {len(corridors)} corridors.")

    if len(corridors) == 0:
        raise RuntimeError(
            "No corridors detected in this background image. "
            "Check that the image really contains TopoChip corridors "
            "and/or lower row_cov_thresh_rel_hull in run_pipeline_and_save_csvs."
        )

    # ------------------------------------------------------------------
    # pick reference corridor + reference x-position (sector control)
    # ------------------------------------------------------------------
    centers_y = [0.5 * (c["y0"] + c["y1"]) for c in corridors]

    # 1) choose which corridor is the reference
    if cor_ref is None:
        # default: middle corridor by vertical position
        ref_idx = int(np.argsort(centers_y)[len(centers_y) // 2])
    elif isinstance(cor_ref, int):
        ref_idx = int(np.clip(cor_ref, 0, len(corridors) - 1))
    elif isinstance(cor_ref, str):
        c_low = cor_ref.lower()
        if c_low == "top":
            ref_idx = int(np.argmin(centers_y))
        elif c_low == "bottom":
            ref_idx = int(np.argmax(centers_y))
        else:
            raise ValueError(
                f"Unknown cor_ref='{cor_ref}'. Use None, int, 'top' or 'bottom'."
            )
    else:
        raise ValueError(
            f"Unsupported type for cor_ref={type(cor_ref)}. "
            "Use None, int, or 'top'/'bottom'."
        )

    ref_corr = corridors[ref_idx]
    print(f"Using corridor #{ref_idx} as reference corridor.")

    # 2) choose x0_ref on this corridor
    if x0_ref is None:
        x_mid = int(0.5 * (ref_corr["x0"] + ref_corr["x1"]))
        x0_ref = min(x_mid, ref_corr["x1"] - (L - 1) * m)
    else:
        x0_ref = int(np.clip(x0_ref, ref_corr["x0"], ref_corr["x1"] - (L - 1) * m))

    print(f"Reference corridor index: {ref_idx}, x0_ref: {x0_ref}")

    # reference window on reference corridor
    ref_segs = window_segments(corr_mask, ref_corr, x0_ref, m, L, L_min)
    ref_desc = descriptor_from_segments(ref_segs) if ref_segs is not None else None

    # ------------------------------------------------------------------
    # slide windows in ALL corridors and keep only matched ones
    # ------------------------------------------------------------------
    all_windows = []
    for c in corridors:
        regs = slide_windows_with_matching(
            corr_mask,
            c,
            ref_desc,
            m=m,
            L=L,
            stride=stride,
            L_min=L_min,
            tau_abs=2,
            alpha=0.05,
            beta=0.20,
            S_min=0.80,
        )
        all_windows.extend(regs)

    matched_windows = [R for R in all_windows if R["matched"]]
    print(f"Total matched windows: {len(matched_windows)}")

    # prepare window boundary arrays
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

    # ------------------------------------------------------------------
    # rotate all centers into same frame as rot_img
    # ------------------------------------------------------------------
    H0, W0 = background.shape[:2]
    M = affine_like_skimage_no_resize(W0, H0, rot_deg)

    if "center_rot" not in SD.dataframe.columns:
        SD.dataframe["center_rot"] = None

    h_ref = float(SD.dataframe["height"].iloc[0]) if "height" in SD.dataframe.columns else H0
    w_ref = float(SD.dataframe["width"].iloc[0]) if "width" in SD.dataframe.columns else W0
    sx = W0 / w_ref if w_ref else 1.0
    sy = H0 / h_ref if h_ref else 1.0

    centers_rot_map: Dict[int, Tuple[float, float]] = {}

    for obj_id, val in SD.dataframe["center"].items():
        if val is None:
            continue

        row_raw, col_raw = val  # (row, col)
        row_arr = np.asarray(row_raw)
        col_arr = np.asarray(col_raw)
        if row_arr.size == 0 or col_arr.size == 0:
            continue

        y = float(row_arr.reshape(-1)[0]) * sy
        x = float(col_arr.reshape(-1)[0]) * sx

        pt_rot = apply_affine_points(
            M,
            np.asarray([[x, y]], dtype=float),
        )[0]

        x_rot, y_rot = float(pt_rot[0]), float(pt_rot[1])
        centers_rot_map[obj_id] = (x_rot, y_rot)
        SD.dataframe.at[obj_id, "center_rot"] = (x_rot, y_rot)

    # ------------------------------------------------------------------
    # select cells whose centers lie inside ANY matched window
    # ------------------------------------------------------------------
    selected_objs = []
    mask = np.zeros(len(SD.dataframe), dtype=bool)

    shift_x = 0.0  # if corr_mask is cropped from rot_img, adjust here
    shift_y = 0.0

    for obj_id, (cxr, cyr) in centers_rot_map.items():
        cx_loc = cxr - shift_x
        cy_loc = cyr - shift_y

        for win, xs, yt, yb in window_infos:
            if contains_point(cx_loc, cy_loc, xs, yt, yb):
                selected_objs.append(
                    {
                        "obj_id": obj_id,
                        "center_rot": (cxr, cyr),
                        "region_id": win.get("region_id"),
                        "corridor_id": win.get("corridor_id"),
                    }
                )
                mask[SD.dataframe.index.get_loc(obj_id)] = True
                break

    # ------------------------------------------------------------------
    # visual summary: whole chip + corridor region + cells
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rot_img, cmap="gray")

    for _, xs, yt, yb in window_infos:
        ax.fill_between(xs, yt, yb, alpha=0.25)

    for obj_id, (cxr, cyr) in centers_rot_map.items():
        idx = SD.dataframe.index.get_loc(obj_id)
        if mask[idx]:
            ax.plot(cxr, cyr, "ro", markersize=4)
        else:
            ax.plot(cxr, cyr, "bo", markersize=2, alpha=0.3)

    ax.set_title("Corridor region where we search for cells")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # zoom into a single representative window & corridor stats
    # ------------------------------------------------------------------
    if window_infos:
        win0, xs0, yt0, yb0 = window_infos[0]

        xs0 = np.asarray(xs0, dtype=float)
        yt0 = np.asarray(yt0, dtype=float)
        yb0 = np.asarray(yb0, dtype=float)

        y_mid = 0.5 * (yt0 + yb0)
        dx = np.diff(xs0)
        dy = np.diff(y_mid)
        seg_lengths = np.sqrt(dx**2 + dy**2)
        length_px = float(seg_lengths.sum())

        widths = yb0 - yt0
        mean_width = float(widths.mean())
        min_width = float(widths.min())
        max_width = float(widths.max())

        n_steps = len(xs0)
        step_px = float(length_px / (n_steps - 1)) if n_steps > 1 else np.nan

        total_cells = len(SD.dataframe)
        cells_in_corridor = int(mask.sum())
        frac_cells = (cells_in_corridor / total_cells) if total_cells > 0 else 0.0

        # eccentricity stats
        ecc_line = "Eccentricity: column 'eccentricity' not found"
        if "eccentricity" in SD.dataframe.columns:
            ecc_corr_raw = SD.dataframe.loc[mask, "eccentricity"].dropna()
            ecc_all_raw = SD.dataframe["eccentricity"].dropna()

            ecc_corr = ecc_corr_raw.apply(_as_scalar).dropna()
            ecc_all = ecc_all_raw.apply(_as_scalar).dropna()

            if len(ecc_corr) > 0:
                mean_corr = float(ecc_corr.mean())
                std_corr = float(ecc_corr.std())
                mean_all = float(ecc_all.mean()) if len(ecc_all) > 0 else float("nan")
                ecc_line = (
                    f"Mean eccentricity (corridor): {mean_corr:.3f} ± {std_corr:.3f} "
                    f"(all cells: {mean_all:.3f})"
                )
            else:
                ecc_line = "Mean eccentricity (corridor): n/a (no values)"

        x0 = int(max(0, np.floor(xs0.min())))
        x1 = int(min(rot_img.shape[1], np.ceil(xs0.max())))
        y0 = int(max(0, np.floor(yt0.min())))
        y1 = int(min(rot_img.shape[0], np.ceil(yb0.max())))

        crop = rot_img[y0:y1, x0:x1]

        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        # top: window with corridor cells
        ax0 = axes[0]
        ax0.imshow(crop, cmap="gray")
        ax0.fill_between(xs0 - x0, yt0 - y0, yb0 - y0, alpha=0.25)
        ax0.plot(xs0 - x0, yt0 - y0, linewidth=1)
        ax0.plot(xs0 - x0, yb0 - y0, linewidth=1)

        for obj_id, (cxr, cyr) in centers_rot_map.items():
            idx = SD.dataframe.index.get_loc(obj_id)
            if not mask[idx]:
                continue
            if (x0 <= cxr <= x1) and (y0 <= cyr <= y1):
                ax0.plot(cxr - x0, cyr - y0, "ro", markersize=4)

        ax0.set_title("Single matched window (corridor cells)")
        ax0.axis("off")

        # bottom: same window + text stats
        ax1 = axes[1]
        ax1.imshow(crop, cmap="gray")
        ax1.fill_between(xs0 - x0, yt0 - y0, yb0 - y0, alpha=0.15)
        ax1.plot(xs0 - x0, yt0 - y0, linewidth=1)
        ax1.plot(xs0 - x0, yb0 - y0, linewidth=1)
        ax1.axis("off")

        lines = [
            f"Length along corridor: {length_px:.1f} px",
            f"Width (mean / min / max): {mean_width:.1f} / {min_width:.1f} / {max_width:.1f} px",
            f"n steps: {n_steps}   pseudotime step ≈ {step_px:.2f} px",
            f"Cells in CORRIDOR region: {cells_in_corridor} of {total_cells} ({frac_cells*100:.1f}%)",
            ecc_line,
        ]
        for i, text in enumerate(lines):
            ax1.text(
                0.02,
                0.98 - 0.13 * i,
                text,
                transform=ax1.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        plt.tight_layout()
        plt.show()

    # also return window_infos, centers_rot_map and the rotated background
    return SD, mask, selected_objs, window_infos, centers_rot_map, rot_img


# ---------------------------------------------------------------------
# Build cells_corridor with arbitrary feature columns
# ---------------------------------------------------------------------
def build_cells_corridor(
    SD,
    centers_rot_map,
    window_infos,
    mask,
    feature_cols,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
):
    """
    Flatten all corridor windows into a single 1D 'global pseudotime'
    axis and collect scalar features for cells inside the corridor.

    Parameters
    ----------
    SD : StainDataset
    centers_rot_map : dict
        obj_id -> (x_rot, y_rot)
    window_infos : list
        (win, xs, yt, yb) as returned by run_pipeline_and_save_csvs
    mask : np.ndarray[bool]
        corridor-cell mask over SD.dataframe
    feature_cols : list of str
        Names of columns in SD.dataframe to extract (each entry can
        be an array; _as_scalar() will reduce it to a float).
    shift_x, shift_y : float
        Optional offsets if corridor mask is cropped from rot_img.

    Returns
    -------
    cells_corridor : list of dict
        Each dict has:
        - 's_global' : float
        - one key per feature in feature_cols
    total_len : float
        Total corridor length (sum over all matched windows).
    """

    # 1) order windows along corridor(s)
    ordered = sorted(
        window_infos,
        key=lambda tup: (tup[0].get("corridor_id", 0), tup[0].get("region_id", 0)),
    )

    window_data = []
    offset = 0.0

    for win, xs, yt, yb in ordered:
        xs = np.asarray(xs, float)
        yt = np.asarray(yt, float)
        yb = np.asarray(yb, float)

        y_mid = 0.5 * (yt + yb)
        dx = np.diff(xs)
        dy = np.diff(y_mid)
        seg_lengths = np.sqrt(dx**2 + dy**2)
        s_local = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        length = float(s_local[-1])

        window_data.append(
            {
                "win": win,
                "xs": xs,
                "yt": yt,
                "yb": yb,
                "s_local": s_local,
                "offset": offset,
            }
        )
        offset += length

    total_len = offset

    # 2) scalar features
    feature_series = {}
    for col in feature_cols:
        if col in SD.dataframe.columns:
            feature_series[col] = SD.dataframe[col].apply(_as_scalar)
        else:
            print(f"[build_cells_corridor] WARNING: column '{col}' not found in SD.dataframe; skipping.")
    if not feature_series:
        raise ValueError("No valid feature columns found in SD.dataframe.")

    # 3) assign global s to each corridor cell
    cells_corridor = []
    corridor_idx = np.where(mask)[0]

    for idx in corridor_idx:
        obj_id = SD.dataframe.index[idx]
        if obj_id not in centers_rot_map:
            continue

        cxr, cyr = centers_rot_map[obj_id]
        cx_loc = cxr - shift_x
        cy_loc = cyr - shift_y

        s_global = None

        for wd in window_data:
            xs = wd["xs"]
            yt = wd["yt"]
            yb = wd["yb"]
            s_local = wd["s_local"]
            offset_win = wd["offset"]

            if not contains_point(cx_loc, cy_loc, xs, yt, yb):
                continue

            i = np.searchsorted(xs, cx_loc)
            i = int(np.clip(i, 1, len(xs) - 1))

            x0, x1 = xs[i - 1], xs[i]
            if x1 == x0:
                t = 0.0
            else:
                t = (cx_loc - x0) / (x1 - x0)

            s_cell = s_local[i - 1] + t * (s_local[i] - s_local[i - 1])
            s_global = offset_win + s_cell
            break

        if s_global is None:
            continue

        entry = {"s_global": float(s_global)}
        for col, series in feature_series.items():
            val = series.iloc[idx]
            entry[col] = float(val) if np.isfinite(val) else np.nan

        cells_corridor.append(entry)

    return cells_corridor, total_len


# ---------------------------------------------------------------------
# Generic sliding-window profile for any scalar feature
# ---------------------------------------------------------------------
def corridor_feature_profile_sliding(
    cells,
    total_len: float,
    feature_key: str,
    window_width_px: float = 1000.0,
    step_px: float = 500.0,
) -> pd.DataFrame:
    """
    Sliding-window 1D profile for an arbitrary scalar feature along the
    global corridor pseudotime axis.

    cells : list of dict (output of build_cells_corridor)
        Each dict must contain 's_global' and feature_key.

    Returns
    -------
    df_slide : DataFrame
        center_s, s_min, s_max, mean_val, std_val, n_cells
    """
    s_vals = np.array([c["s_global"] for c in cells], dtype=float)
    feat_vals = np.array([c[feature_key] for c in cells], dtype=float)

    centers = np.arange(
        window_width_px / 2,
        total_len - window_width_px / 2 + 1,
        step_px,
    )

    rows = []
    for c_s in centers:
        s_min = c_s - window_width_px / 2
        s_max = c_s + window_width_px / 2

        m = (s_vals >= s_min) & (s_vals <= s_max)
        vals_win = feat_vals[m]

        if vals_win.size > 0:
            mean_val = float(np.nanmean(vals_win))
            std_val = float(np.nanstd(vals_win))
            n_cells = int(np.isfinite(vals_win).sum())
        else:
            mean_val = np.nan
            std_val = np.nan
            n_cells = 0

        rows.append(
            dict(
                center_s=c_s,
                s_min=s_min,
                s_max=s_max,
                mean_val=mean_val,
                std_val=std_val,
                n_cells=n_cells,
            )
        )

    return pd.DataFrame(rows)


# =====================================================================
# HELPER: collect cells + local s for a single window
# =====================================================================

def _cells_for_window(
    window_idx: int,
    SD,
    mask: np.ndarray,
    centers_rot_map: Dict[int, Tuple[float, float]],
    window_infos,
    ecc_col: str = "eccentricity",
) -> pd.DataFrame:
    """
    For a given window index, return a DataFrame with:
    - obj_id
    - s_local  : arc-length position along this window (px)
    - ecc      : scalar eccentricity
    Only cells with mask == True and lying inside this window are used.
    """

    win, xs, yt, yb = window_infos[window_idx]

    xs = np.asarray(xs, dtype=float)
    yt = np.asarray(yt, dtype=float)
    yb = np.asarray(yb, dtype=float)

    # geometry: arc-length coordinate s along the midline
    y_mid = 0.5 * (yt + yb)
    dx = np.diff(xs)
    dy = np.diff(y_mid)
    seg_lengths = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0.0], np.cumsum(seg_lengths)))  # s[i] at xs[i]

    # precompute scalar eccentricity
    if ecc_col not in SD.dataframe.columns:
        raise KeyError(f"Column '{ecc_col}' not found in SD.dataframe")
    ecc_series = SD.dataframe[ecc_col].apply(_as_scalar)

    rows = []
    corridor_idx = np.where(mask)[0]

    for idx in corridor_idx:
        obj_id = SD.dataframe.index[idx]
        if obj_id not in centers_rot_map:
            continue

        cxr, cyr = centers_rot_map[obj_id]

        # keep only if the center is inside this window
        if not contains_point(cxr, cyr, xs, yt, yb):
            continue

        # local arc-length s for this cell
        i = np.searchsorted(xs, cxr)
        i = int(np.clip(i, 1, len(xs) - 1))
        x0, x1 = xs[i - 1], xs[i]
        if x1 == x0:
            t = 0.0
        else:
            t = (cxr - x0) / (x1 - x0)
        s_cell = s[i - 1] + t * (s[i] - s[i - 1])

        ecc_val = ecc_series.iloc[idx]
        if np.isnan(ecc_val):
            continue

        rows.append(
            {
                "obj_id": obj_id,
                "s_local": float(s_cell),
                "ecc": float(ecc_val),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# HELPER: per-window profile ecc vs position (like your slide plots)
# =====================================================================

def ecc_profile_for_single_window(
    window_idx: int,
    SD,
    mask: np.ndarray,
    centers_rot_map: Dict[int, Tuple[float, float]],
    window_infos,
    bin_size_px: float = 1.0,
) -> pd.DataFrame:
    """
    Build 'eccentricity vs position along window' profile
    for one window, by binning local s positions.

    Returns DataFrame with columns:
    - pos_px     : mean position in this bin (px)
    - median_ecc : median eccentricity in the bin
    - std_ecc    : std of eccentricity in the bin
    - n_cells    : number of cells in the bin
    """

    df_cells = _cells_for_window(window_idx, SD, mask, centers_rot_map, window_infos)
    if df_cells.empty:
        return pd.DataFrame(columns=["pos_px", "median_ecc", "std_ecc", "n_cells"])

    s_vals = df_cells["s_local"].values
    ecc_vals = df_cells["ecc"].values

    length_px = s_vals.max()  # window length approx.
    bins = np.arange(0.0, length_px + bin_size_px, bin_size_px)

    df = pd.DataFrame({"s": s_vals, "ecc": ecc_vals})
    df["bin"] = np.digitize(df["s"], bins) - 1   # 0-based

    grouped = (
        df.groupby("bin")
        .agg(
            pos_px=("s", "mean"),
            median_ecc=("ecc", "median"),
            std_ecc=("ecc", "std"),
            n_cells=("ecc", "size"),
        )
        .dropna(subset=["pos_px"])
    )

    # replace NaN std (only 1 cell) by 0 for plotting
    grouped["std_ecc"] = grouped["std_ecc"].fillna(0.0)
    return grouped


# =====================================================================
# HELPER: scan all windows and find extremes in median ecc
# =====================================================================

def find_extreme_windows_by_median(
    SD,
    mask: np.ndarray,
    centers_rot_map: Dict[int, Tuple[float, float]],
    window_infos,
    min_cells: int = 5,
) -> pd.DataFrame:
    """
    For each window, compute median eccentricity of its cells.
    Returns a DataFrame with one row per window:
      - win_idx
      - n_cells
      - median_ecc
    You can then choose the highest/lowest median windows as A/B.
    """

    records = []
    for k in range(len(window_infos)):
        df_cells = _cells_for_window(k, SD, mask, centers_rot_map, window_infos)
        n = len(df_cells)
        if n < min_cells:
            continue
        med = float(df_cells["ecc"].median())
        records.append({"win_idx": k, "n_cells": n, "median_ecc": med})

    if not records:
        return pd.DataFrame(columns=["win_idx", "n_cells", "median_ecc"])

    df_wins = pd.DataFrame(records).sort_values("median_ecc")
    return df_wins

def find_representative_window(
    window_infos,
    centers_rot_map,
    SD,
    mask,
    mode: str = "max_cells",
):
    """
    Pick ONE window that acts as the 'smallest repeating unit' of the corridor.

    The corridor is constructed by repeating these matched windows along x,
    so each (win, xs, yt, yb) in window_infos is one candidate unit.
    We choose a representative window according to the chosen scoring mode.

    Parameters
    ----------
    window_infos : list
        Output of run_pipeline_and_save_csvs; list of
        (win_dict, xs, yt, yb) for each matched window.

    centers_rot_map : dict
        obj_id -> (x_rot, y_rot) in the rotated image coordinates.

    SD : StainDataset
        Dataset; SD.dataframe used to map row index ↔ obj_id.

    mask : np.ndarray[bool]
        Boolean mask over SD.dataframe: True for cells inside SOME corridor
        window (global corridor mask, like we already computed).

    mode : {"max_cells", "median_x", "longest"}
        Strategy for picking the representative window:
        - "max_cells": choose the window that contains the largest number of
          *corridor* cells (recommended).
        - "median_x":  choose the window whose center in x is closest to the
          median of all window centers (geometric middle).
        - "longest":   choose the window with the largest length along the
          midline (should all be similar if created with fixed L).

    Returns
    -------
    rep_idx : int
        Index into window_infos of the chosen window.

    rep_win : dict
        The win_dict of that window.

    xs0, yt0, yb0 : np.ndarray
        The boundary arrays describing that window.

    length_px : float
        Length of the window along its midline in pixels.
    """

    if not window_infos:
        raise ValueError("window_infos is empty – no windows to choose from.")

    # --- precompute geometry and basic stats for each window ------------
    lengths = []
    x_centers = []
    cell_counts = []

    for w_idx, (win, xs, yt, yb) in enumerate(window_infos):
        xs = np.asarray(xs, dtype=float)
        yt = np.asarray(yt, dtype=float)
        yb = np.asarray(yb, dtype=float)

        # geometry
        y_mid = 0.5 * (yt + yb)
        dx = np.diff(xs)
        dy = np.diff(y_mid)
        seg_lengths = np.sqrt(dx**2 + dy**2)
        length_px = float(seg_lengths.sum())
        x_center = float(xs.mean()) if xs.size > 0 else np.nan

        lengths.append(length_px)
        x_centers.append(x_center)

        # how many *corridor* cells fall in this window?
        count = 0
        for obj_id, (cxr, cyr) in centers_rot_map.items():
            idx = SD.dataframe.index.get_loc(obj_id)
            if not mask[idx]:
                continue  # ignore cells outside global corridor mask
            if contains_point(cxr, cyr, xs, yt, yb):
                count += 1
        cell_counts.append(count)

    lengths   = np.asarray(lengths, dtype=float)
    x_centers = np.asarray(x_centers, dtype=float)
    cell_counts = np.asarray(cell_counts, dtype=int)

    # --- choose representative window according to mode -----------------
    if mode == "max_cells":
        rep_idx = int(np.argmax(cell_counts))
    elif mode == "median_x":
        # center-of-corridor window (in x)
        median_x = np.nanmedian(x_centers)
        rep_idx = int(np.nanargmin(np.abs(x_centers - median_x)))
    elif mode == "longest":
        rep_idx = int(np.nanargmax(lengths))
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'max_cells', 'median_x', or 'longest'.")

    rep_win, xs0, yt0, yb0 = window_infos[rep_idx]
    length_px = float(lengths[rep_idx])

    return rep_idx, rep_win, np.asarray(xs0, float), np.asarray(yt0, float), np.asarray(yb0, float), length_px


# ---------------------------------------------------------------------
# Generic binned-median + IQR plot for any feature
# ---------------------------------------------------------------------
def plot_binned_median_profile(
    df_slide: pd.DataFrame,
    total_len: float,
    feature_label: str = "feature",
    N_MIN: int = 5,
    N_BINS: int = 18,
):
    """
    2-panel plot: top = binned median feature along pseudotime with IQR
    error bars; bottom = n_cells per sliding window.
    """

    df = df_slide.copy()
    df["pseudo01"] = df["center_s"] / total_len
    df = df.sort_values("pseudo01")

    df_good = df[df["n_cells"] >= N_MIN].copy()

    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    df_good["bin"] = pd.cut(
        df_good["pseudo01"],
        bins=bins,
        labels=False,
        include_lowest=True,
    )

    prof = (
        df_good.groupby("bin")
        .agg(
            pseudo01=("pseudo01", "mean"),
            median_val=("mean_val", "median"),
            q25=("mean_val", lambda x: x.quantile(0.25)),
            q75=("mean_val", lambda x: x.quantile(0.75)),
            n_cells=("n_cells", "sum"),
            n_windows=("mean_val", "size"),
        )
        .dropna(subset=["pseudo01"])
    )

    err_low = prof["median_val"] - prof["q25"]
    err_high = prof["q75"] - prof["median_val"]
    yerr = np.vstack([err_low.values, err_high.values])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    # top: binned median + IQR
    ax1.errorbar(
        prof["pseudo01"],
        prof["median_val"],
        yerr=yerr,
        marker="o",
        linestyle="-",
        linewidth=1.5,
    )
    ax1.set_ylabel(f"Median {feature_label}")
    ax1.set_title(f"Median {feature_label} along corridor (binned)")
    ax1.grid(True, alpha=0.3)

    y_min = prof["q25"].min()
    y_max = prof["q75"].max()
    if np.isfinite(y_min) and np.isfinite(y_max):
        span = y_max - y_min or 0.01
        margin = 0.1 * span
        ax1.set_ylim(y_min - margin, y_max + margin)

    # bottom: n_cells per window
    ax2.plot(
        df["pseudo01"],
        df["n_cells"],
        marker="x",
        linestyle="-",
        alpha=0.5,
        label="cells per window",
    )
    ax2.axhline(
        N_MIN,
        color="red",
        linestyle="--",
        alpha=0.4,
        label=f"N_MIN = {N_MIN}",
    )
    ax2.set_xlabel("Normalized corridor position (pseudotime, 0–1)")
    ax2.set_ylabel("n_cells in window")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()

    return prof
