import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from skimage import filters, morphology, measure
import pandas as pd

from pseudotime import window_segments, descriptor_from_segments, slide_windows_with_matching
from corridor_mask import deskew_with_hull, detect_corridors_via_hull

def run_pipeline_and_save_csvs( cp_df, out_prefix: str,
                                img: np.ndarray,
                               x0_ref: int = None):
    #TODO: look into pseudo.ipynb

    #TODO make them parameters
    m, L, stride, L_min = 2, 10, 1, 8 # spacing, length, stride, min len

    rot_img, corr_mask, hull_mask, rot_deg = deskew_with_hull(img)
    corridors = detect_corridors_via_hull(corr_mask, hull_mask,
                                        row_cov_thresh_rel_hull=0.07,
                                        min_band_height=5, merge_gap_px=3)

    # Here the example segment is calculated
    

    #TODO make a logic in case the starting point is specified
    if x0_ref is None:
        centers = [0.5*(c['y0'] + c['y1']) for c in corridors]
        mid_idx = int(np.argsort(centers)[len(centers)//2])
        mid_corr = corridors[mid_idx]
        x_mid = int(0.5 * (mid_corr['x0'] + mid_corr['x1']))
        x0_ref = min(x_mid, mid_corr['x1'] - (L-1)*m)  # start at middle, clamp if needed
    else:
        #Implement
        x0_ref = x0_ref
    
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

    #TODO find all DAPI strains such that centre if mass is inside matched windows
    #TODO get like 20 of them



    # Build the 4 CSVs
    # 6a) pseudotime_lines.csv — each used vertical line 
    rows_lines = []
    for R in merged:
        for j, (x, y_top, y_end) in enumerate(R['segments']):
            rows_lines.append(dict(
                region_id=R['region_id'], corridor_id=R['corridor_id'],
                x=int(x), y_top=int(y_top), y_end=int(y_end), line_index=j
            ))
    df_lines = pd.DataFrame(rows_lines)
    df_lines.to_csv(f"{out_prefix}_pseudotime_lines.csv", index=False)

    # 6b) regions.csv — one row per (possibly merged) region
    rows_regions = []
    for R in merged:
        xs = [s[0] for s in R['segments']]
        rows_regions.append(dict(
            region_id=R['region_id'], corridor_id=R['corridor_id'],
            x0=int(min(xs)), x1=int(max(xs)), m=R['m'],
            L_used=len(R['segments']),
            n_cells=len(R['cell_ids']),
            first_interrupt_at_line=R.get('first_interrupt_at', None)
        ))
    df_regions = pd.DataFrame(rows_regions)
    df_regions.to_csv(f"{out_prefix}_regions.csv", index=False)

    # 6c) cells_in_areas.csv — CP join
    rows_cells = []
    for R in merged:
        for idx in R['cell_ids']:
            rows_cells.append(dict(
                region_id=R['region_id'],
                image_id=int(cp_df.iloc[idx]['ImageNumber']),
                object_id=int(cp_df.iloc[idx]['ObjectNumber']),
                cx=float(cp_df.iloc[idx]['AreaShape_Center_X']),
                cy=float(cp_df.iloc[idx]['AreaShape_Center_Y']),
                eccentricity=float(cp_df.iloc[idx]['AreaShape_Eccentricity'])
            ))
    df_cells = pd.DataFrame(rows_cells)
    df_cells.to_csv(f"{out_prefix}_cells_in_areas.csv", index=False)

    # 6d) ecc_summary.csv — per region
    rows_ecc = []
    for R in merged:
        ecc = cp_df.iloc[R['cell_ids']]['AreaShape_Eccentricity'].to_numpy(dtype=float)
        if ecc.size:
            rows_ecc.append(dict(
                region_id=R['region_id'], n=len(ecc),
                ecc_mean=float(np.mean(ecc)),
                ecc_median=float(np.median(ecc)),
                ecc_q25=float(np.quantile(ecc, 0.25)),
                ecc_q75=float(np.quantile(ecc, 0.75)),
            ))
        else:
            rows_ecc.append(dict(region_id=R['region_id'], n=0,
                                 ecc_mean=np.nan, ecc_median=np.nan, ecc_q25=np.nan, ecc_q75=np.nan))
    df_ecc = pd.DataFrame(rows_ecc)
    df_ecc.to_csv(f"{out_prefix}_ecc_summary.csv", index=False)
