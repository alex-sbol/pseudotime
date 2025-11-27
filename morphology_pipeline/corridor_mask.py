# Corridor BBoxes via **black convex hull** (white corridors inside the hull).
#   - thresholds image -> black mask
#   - convex hull of black -> content boundary
#   - corridors = (~black) & hull  (white strips bounded by black)
#   - deskews using corridors mask
#   - computes corridor BBoxes **using the hull for extents**
#   - visualizes BBoxes on the deskewed image
#
# You can adjust `row_cov_thresh_rel_hull` if corridors are thinner/thicker.

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, transform, feature
from morphology_pipeline.align_corridors import estimate_corridor_angle

# ---------------- helpers ----------------
def black_mask(img: np.ndarray):
    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
    th = filters.threshold_otsu(gray)
    black = gray < th
    return black

def corridors_from_black_hull(img: np.ndarray, tiny_white_px: int = 16):
    blk = black_mask(img)
    hull = morphology.convex_hull_image(blk)
    corridors = (~blk) & hull  # white within hull only
    # clean: remove tiny white specks; fill tiny holes
    corridors = morphology.remove_small_objects(corridors, min_size=tiny_white_px)
    corridors = morphology.remove_small_holes(corridors, area_threshold=64)
    return corridors.astype(np.uint8), hull.astype(np.uint8)


def deskew_with_hull(img: np.ndarray):
    corr0, hull0 = corridors_from_black_hull(img)
    rot_deg, _ = estimate_corridor_angle(corr0)
    img_rot  = transform.rotate(img,  rot_deg, order=1, preserve_range=True).astype(img.dtype)
    corr_rot = transform.rotate(corr0, rot_deg, order=0, preserve_range=True).astype(np.uint8)
    hull_rot = transform.rotate(hull0, rot_deg, order=0, preserve_range=True).astype(np.uint8)
    return img_rot, corr_rot, hull_rot, rot_deg

def detect_corridors_via_hull(corr_mask: np.ndarray,
                              hull_mask: np.ndarray,
                              row_cov_thresh_rel_hull: float = 0.08,
                              min_band_height: int = 5,
                              merge_gap_px: int = 3):
    """
    Corridor y-bands are decided by the fraction of WHITE (corridor) pixels
    relative to the HULL width on that row. This keeps peaks/corners because
    x-extent is taken from the hull, not the corridor mask.
    Returns list of bbox dicts {id, y0, y1, x0, x1}.
    """
    H, W = corr_mask.shape
    hull_width = hull_mask.sum(axis=1).astype(float)
    white_counts = corr_mask.sum(axis=1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        coverage = np.where(hull_width > 0, white_counts / hull_width, 0.0)

    good = (coverage >= row_cov_thresh_rel_hull).astype(np.uint8)
    if merge_gap_px > 1:
        k = np.ones(merge_gap_px, dtype=int)
        good = (np.convolve(good, k, mode='same') > 0)

    idx = np.flatnonzero(good)
    bboxes = []
    if idx.size:
        starts = [idx[0]]
        ends = []
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                ends.append(a)
                starts.append(b)
        ends.append(idx[-1])

        cid = 0
        for y0, y1 in zip(starts, ends):
            if (y1 - y0 + 1) < min_band_height:
                continue
            # x-extent from the HULL across this band (outer envelope)
            band = hull_mask[y0:y1+1, :]
            x_any = band.any(axis=0)
            if not np.any(x_any):
                continue
            x0 = int(np.argmax(x_any))
            x1 = int(len(x_any) - 1 - np.argmax(x_any[::-1]))
            bboxes.append({'id': cid, 'y0': int(y0), 'y1': int(y1), 'x0': int(x0), 'x1': int(x1)})
            cid += 1
    return bboxes

if __name__ == "__main__":
    img_path = r"C:\Users\sbsas\Documents\uni\Projects\Nikita PhD\res\rotated_best.png"
    img = io.imread(img_path)

    rot_img, corr_mask, hull_mask, rot_deg = deskew_with_hull(img)
    bboxes = detect_corridors_via_hull(
        corr_mask, hull_mask,
        row_cov_thresh_rel_hull=0.07,  # relative to hull width
        min_band_height=5,
        merge_gap_px=3
    )

    fig = plt.figure(figsize=(7,7))
    plt.imshow(rot_img, cmap='gray')

    for b in bboxes:
        x0, x1 = b['x0'], b['x1']
        y0, y1 = b['y0'], b['y1']
        # draw rectangle
        plt.plot([x0, x1], [y0, y0], color='red', linewidth=1.5)
        plt.plot([x0, x1], [y1, y1], color='red', linewidth=1.5)
        plt.plot([x0, x0], [y0, y1], color='red', linewidth=1.0)
        plt.plot([x1, x1], [y0, y1], color='red', linewidth=1.0)

    plt.title(f"Corridor BBoxes via BLACK HULL (deskew={rot_deg:.2f}°), N={len(bboxes)}")
    plt.axis('off')
    plt.show()
    bboxes
