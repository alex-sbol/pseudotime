import numpy as np


def gen_lines(x0: int, m: int, L: int) -> np.ndarray:
    return x0 + m * np.arange(L, dtype=int)

def segment_outer_envelope(mask_white: np.ndarray, x: int, y0: int, y1: int):
    col = mask_white[y0:y1+1, x]
    ys = np.flatnonzero(col)
    if ys.size == 0:
        return None
    y_top = y0 + int(ys[0])
    y_bot = y0 + int(ys[-1])
    if y_bot < y_top:
        return None
    return (x, y_top, y_bot)

def window_segments(mask_white: np.ndarray, corridor: dict,
                    x0: int, m: int, L: int, L_min: int):
    xs = gen_lines(x0, m, L)
    segs = []
    for x in xs:
        if x < corridor['x0'] or x > corridor['x1']:
            break
        seg = segment_outer_envelope(mask_white, x, corridor['y0'], corridor['y1'])
        if seg is None:
            break  # interrupt at first empty column
        segs.append(seg)
    if len(segs) < L_min:
        return None
    return segs

def descriptor_from_segments(segments):
    xs = [s[0] for s in segments]
    y_top = [s[1] for s in segments]
    y_bot = [s[2] for s in segments]
    w = [b - t for t, b in zip(y_top, y_bot)]
    order = np.argsort(xs)
    return {
        "x": np.asarray(xs, int)[order],
        "y_top": np.asarray(y_top, int)[order],
        "y_bot": np.asarray(y_bot, int)[order],
        "w": np.asarray(w, int)[order],
    }

def match_segments_to_descriptor(segments, ref_desc,
                                 tau_abs=2, alpha=0.05, beta=0.20,
                                 L_min=12, S_min=0.80):
    if segments is None or len(segments) == 0 or ref_desc is None:
        return {"matched": False, "dy": 0, "score": 0.0}
    obs = descriptor_from_segments(segments)
    n = min(len(obs["x"]), len(ref_desc["x"]))
    if n == 0:
        return {"matched": False, "dy": 0, "score": 0.0}
    dy = int(np.median(obs["y_top"][:n] - ref_desc["y_top"][:n]))
    ref_w = ref_desc["w"][:n]
    tau_pos = np.maximum(tau_abs, (alpha * np.maximum(1, ref_w)))
    tau_w   = np.maximum(1, (beta * np.maximum(1, ref_w)))
    top_err = np.abs(obs["y_top"][:n] - (ref_desc["y_top"][:n] + dy))
    bot_err = np.abs(obs["y_bot"][:n] - (ref_desc["y_bot"][:n] + dy))
    w_err   = np.abs(obs["w"][:n]     -  ref_w)
    accept = (top_err <= tau_pos) & (bot_err <= tau_pos) & (w_err <= tau_w)
    score = float(accept.mean()) if n > 0 else 0.0
    matched = (accept.sum() >= L_min) and (score >= S_min)
    return {"matched": matched, "dy": dy, "score": score}

def slide_windows_with_matching(mask_white: np.ndarray, corridor: dict,
                                ref_desc, m=4, L=20, stride=4, L_min=12,
                                tau_abs=2, alpha=0.05, beta=0.20, S_min=0.80):
    regions = []
    x_start = corridor['x0']
    x_last  = corridor['x1'] - (L - 1) * m
    rid = 0
    while x_start <= x_last:
        segs = window_segments(mask_white, corridor, x_start, m, L, L_min)
        if segs is not None:
            res = match_segments_to_descriptor(segs, ref_desc,
                                               tau_abs=tau_abs, alpha=alpha, beta=beta,
                                               L_min=L_min, S_min=S_min)
            regions.append({
                'region_id': rid,
                'corridor_id': corridor['id'],
                'x0': x_start,
                'x_last': x_start + (L - 1) * m,
                'segments': segs,
                'matched': res["matched"],
                'dy': res["dy"],
                'score': res["score"],
            })
            rid += 1
        x_start += stride
    return regions



if __name__ == "__main__":
    from corridor_mask import deskew_with_hull, detect_corridors_via_hull
    from skimage import io
    img_path = r"C:\Users\sbsas\Documents\uni\Projects\Nikita PhD\res\rotated_best.png"
    img = io.imread(img_path)

    rot_img, corr_mask, hull_mask, rot_deg = deskew_with_hull(img)
    corridors = detect_corridors_via_hull(corr_mask, hull_mask,
                                        row_cov_thresh_rel_hull=0.07,
                                        min_band_height=5, merge_gap_px=3)

    # pick the middle corridor by y-center
    centers = [0.5*(c['y0'] + c['y1']) for c in corridors]
    mid_idx = int(np.argsort(centers)[len(centers)//2])
    mid_corr = corridors[mid_idx]

    m, L, stride, L_min = 2, 10, 1, 8 # spacing, length, stride, min len


    x_mid = int(0.5 * (mid_corr['x0'] + mid_corr['x1']))
    x0_ref = min(x_mid, mid_corr['x1'] - (L-1)*m)  # start at middle, clamp if needed
    ref_segs = window_segments(corr_mask, mid_corr, x0_ref, m, L, L_min)
    ref_desc = descriptor_from_segments(ref_segs) if ref_segs is not None else None

    # slide and match on all corridors
    all_windows = []
    for c in corridors:
        regs = slide_windows_with_matching(corr_mask, c, ref_desc,
                                        m=m, L=L, stride=stride, L_min=L_min,
                                        tau_abs=2, alpha=0.05, beta=0.20, S_min=0.80)
        all_windows.extend(regs)

    matched_count = sum(r['matched'] for r in all_windows)

    print(f"Deskew rotation: {rot_deg:.2f}°")
    print(f"Detected corridors: {len(corridors)}")
    print(f"Reference corridor id: {mid_corr['id']}, x0_ref={x0_ref}, m={m}, L={L}")
    print(f"Total windows evaluated: {len(all_windows)}, matched: {matched_count}")