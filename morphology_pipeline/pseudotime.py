import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import find_peaks
from typing import Dict, List, Tuple

def process_corridors(
    mask_white,
    corridor_bbox,
    m,
    min_period,
    max_period,
):
    """
    Docstring for process_corridors
    
    :param mask_white: Rotated binary mask of the corridors
    :param corridor_bbox: list of dict with keys 'x0', 'x1', 'y0', 'y1' defining the corridor bounding box

    :param min_period: minimum possible period
    :param max_period: maximum possible period
    """
    signals = collect_corridors(mask_white, corridor_bbox, m)

    T0 = estimate_period(signals, min_period, max_period)

    candidates = range(
        max(min_period, T0 - 2),
        min(max_period, T0 + 3),
    )
    best_T = T0
    best_score = -np.inf

    for Tc in candidates:
        score = alignment_score(signals, Tc)
        if score > best_score:
            best_score = score
            best_T = Tc

    T = best_T

    peak_list = [detect_cycles(s, min_dist=T//2) for s in signals]
    signal = max(signals, key=len)  # pick longest signal
    #initial_template = fold_signal(signal, T)
    initial_template = fold_cycles(signal, peak_list[signals.index(signal)], L=T)

    
    # offsets = []
    # for s in signals:
    #     p = fold_signal(s, T)
    #     o = estimate_offset(p, initial_template)
    #     offsets.append(o)

    # template = refined_template(signals, offsets, T)

    results = []
    for i, s in enumerate(signals):
        line_id = assign_line_ids(len(s), T, offsets[i])
        confidence = compute_confidence(s, template, line_id)
        line_id = apply_missing_policy(line_id, confidence, conf_threshold=0.4)
        
        results.append({
            "signal": s,
            "line_id": line_id,
            "confidence": confidence,
            "offset": offsets[i],
        })

    return {
        "period": T,
        "template": template,
        "corridors": results,
    }

def collect_corridors(mask_white, corridor_bbox, m) -> List[np.ndarray]:
    signals = []
    for bbox in corridor_bbox:
        x0, x1 = bbox['x0'], bbox['x1']
        y0, y1 = bbox['y0'], bbox['y1']
        signal = []
        for x in range(x0, x1 + 1, m):
            col = mask_white[y0:y1 + 1, x]
            ys = np.flatnonzero(col)
            if ys.size == 0:
                signal.append(0)
            else:
                w = ys[-1] - ys[0] + 1
                signal.append(w)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        signals.append(np.array(signal, dtype=float))
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8) # normalize
    return signals


def estimate_period(signals: List[np.ndarray],
                    min_period: int,
                    max_period: int) -> int:
    """
    Robust period estimate using mean autocorrelation.
    """
    min_len = min(len(s) for s in signals)
    X = np.stack([s[:min_len] for s in signals])
    mean_signal = X.mean(axis=0)

    acf = np.correlate(mean_signal, mean_signal, mode="full")
    acf = acf[len(acf)//2:]

    peaks, _ = find_peaks(acf[min_period:max_period])
    if len(peaks) == 0:
        raise RuntimeError("No period peak found")

    T = peaks[0] + min_period
    return T

def fold_signal(signal: np.ndarray, T: int) -> np.ndarray:
    acc = np.zeros(T)
    cnt = np.zeros(T)

    for t, v in enumerate(signal):
        k = t % T
        acc[k] += v
        cnt[k] += 1

    return acc / np.maximum(cnt, 1)
 
def detect_cycles(signal, min_dist):
    peaks, _ = find_peaks(signal, distance=min_dist)
    return peaks

def fold_cycles(signal, peaks, L):
    # L is median cycle length
    acc = np.zeros(L)
    cnt = np.zeros(L)

    for a, b in zip(peaks[:-1], peaks[1:]):
        seg = signal[a:b]
        xs = np.linspace(0, 1, len(seg))
        xq = np.linspace(0, 1, L)
        seg_rs = np.interp(xq, xs, seg)

        acc += seg_rs
        cnt += 1

    return acc / np.maximum(cnt, 1)

def estimate_offset(signal: np.ndarray, template: np.ndarray) -> int:
    """
    Returns offset o such that (t + o) % T aligns signal to template.
    """
    f_sig = fft(signal)
    f_tmp = fft(template)
    corr = np.real(ifft(f_sig * np.conj(f_tmp)))
    return int(np.argmax(corr))


def refined_template(signals, offsets, T):
    acc = np.zeros(T)
    cnt = np.zeros(T)

    for s, o in zip(signals, offsets):
        for t, v in enumerate(s):
            k = (t + o) % T
            acc[k] += v
            cnt[k] += 1

    return acc / np.maximum(cnt, 1)

def assign_line_ids(length: int, T: int, offset: int) -> np.ndarray:
    return (np.arange(length) + offset) % T

def compute_confidence(signal, template, line_id, window=2):
    conf = np.zeros(len(signal))

    for t in range(len(signal)):
        errs = []
        for k in range(-window, window + 1):
            tt = t + k
            if 0 <= tt < len(signal):
                ref = template[line_id[tt]]
                errs.append((signal[tt] - ref) ** 2)

        mse = np.mean(errs) if errs else np.inf
        conf[t] = np.exp(-mse)

    return conf


def apply_missing_policy(
    line_id: np.ndarray,
    conf: np.ndarray,
    conf_threshold: float = 0.4
) -> np.ndarray:
    lid = line_id.copy()
    #lid[conf < conf_threshold] = -1
    return lid

def alignment_score(signals, T):
    # use longest signal as reference
    ref = max(signals, key=len)
    tmpl0 = fold_signal(ref, T)

    offsets = []
    for s in signals:
        p = fold_signal(s, T)
        o = estimate_offset(p, tmpl0)
        offsets.append(o)

    tmpl = refined_template(signals, offsets, T)

    # score = mean correlation between aligned folds and template
    scores = []
    for s, o in zip(signals, offsets):
        p = fold_signal(s, T)
        p_aligned = np.roll(p, -o)

        a = p_aligned - p_aligned.mean()
        b = tmpl - tmpl.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
        scores.append(np.dot(a, b) / denom)

    return float(np.mean(scores))


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