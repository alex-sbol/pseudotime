"""
Cellpose-derived corridor processing with ROI-based demonstration panels.

What changed in v5
------------------
- Panel A shows the full Cellpose image with a red ROI box.
- Panels B, C, and D show cropped views of that ROI through the workflow.
- Panel C still uses the corridor-support visualization: only the measured
  first-to-last accessible-pixel span at each x-position is shown in white.
  Ignored accessible pixels are shown in black.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage import color, filters, morphology, transform
from tifffile import imread


def _load_image(image_or_path):
    if image_or_path is None:
        return None
    if isinstance(image_or_path, (str, Path)):
        path = Path(image_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path.resolve()}")
        return imread(path)
    return np.asarray(image_or_path)


def black_mask_exact(image):
    image = np.asarray(image)
    if image.ndim == 3:
        gray = color.rgb2gray(image[..., :3])
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0

    threshold = filters.threshold_otsu(gray)
    black = gray < threshold
    return gray, black, float(threshold)


def true_runs(values):
    values = np.asarray(values, dtype=bool)
    if values.size == 0:
        return []

    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts, ends))


def rows_to_mask(row_flags, width):
    row_flags = np.asarray(row_flags, dtype=bool)
    return np.repeat(row_flags[:, None], int(width), axis=1)


def resolve_roi(image_shape, roi=None, roi_rel=None):
    """
    Resolve ROI into integer pixel coordinates (x, y, w, h).

    Parameters
    ----------
    image_shape : tuple
        Shape of the source image, either (H, W) or (H, W, C).
    roi : tuple or None
        Explicit ROI in pixels: (x, y, w, h).
    roi_rel : tuple or None
        Relative ROI in fractions of width/height: (x_rel, y_rel, w_rel, h_rel).

    Returns
    -------
    tuple[int, int, int, int]
    """
    h, w = image_shape[:2]

    if roi is not None:
        x, y, rw, rh = [int(round(v)) for v in roi]
    elif roi_rel is not None:
        x = int(round(roi_rel[0] * w))
        y = int(round(roi_rel[1] * h))
        rw = int(round(roi_rel[2] * w))
        rh = int(round(roi_rel[3] * h))
    else:
        # A sensible central default if nothing is provided.
        x = int(round(0.18 * w))
        y = int(round(0.22 * h))
        rw = int(round(0.36 * w))
        rh = int(round(0.36 * h))

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return x, y, rw, rh


def rotate_roi_to_bbox(image_shape, roi, angle_deg, margin_px=8):
    """
    Map an ROI from the original image into the resized rotated canvas.

    This is done robustly by rotating a binary ROI mask with the same skimage
    settings as the actual processing pipeline, then extracting the bbox.
    """
    h, w = image_shape[:2]
    x, y, rw, rh = [int(v) for v in roi]

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[y:y + rh, x:x + rw] = 1

    roi_rot = transform.rotate(
        roi_mask,
        angle_deg,
        order=0,
        preserve_range=True,
        resize=True,
    ).astype(bool)

    yy, xx = np.where(roi_rot)
    if len(xx) == 0 or len(yy) == 0:
        return None

    x0 = max(0, int(xx.min()) - margin_px)
    x1 = min(roi_rot.shape[1], int(xx.max()) + 1 + margin_px)
    y0 = max(0, int(yy.min()) - margin_px)
    y1 = min(roi_rot.shape[0], int(yy.max()) + 1 + margin_px)
    return x0, x1, y0, y1


def crop_xy(image, bbox):
    x0, x1, y0, y1 = bbox
    return image[y0:y1, x0:x1]


def process_cellpose_corridors(
    cellpose_image,
    rotation_deg,
    reference_image=None,
    invert_cellpose=False,
    row_cov_thresh_rel_hull=0.07,
    tiny_white_px=16,
    hole_area_threshold=64,
    merge_gap_px=3,
    min_band_height=5,
    scan_step_px=1,
    pixel_size_um=0.56,
):
    cellpose_raw = _load_image(cellpose_image)
    reference_raw = _load_image(reference_image)

    cellpose_used = np.asarray(cellpose_raw)
    if invert_cellpose:
        cellpose_used = cellpose_used.max() - cellpose_used

    gray, pillars, threshold = black_mask_exact(cellpose_used)

    hull = morphology.convex_hull_image(pillars)
    candidate = (~pillars) & hull

    after_small_objects = morphology.remove_small_objects(
        candidate,
        min_size=tiny_white_px,
    )
    cleaned = morphology.remove_small_holes(
        after_small_objects,
        area_threshold=hole_area_threshold,
    )

    removed_small_objects = candidate & ~after_small_objects
    filled_small_holes = cleaned & ~after_small_objects

    corridor_rot = transform.rotate(
        cleaned,
        rotation_deg,
        order=0,
        preserve_range=True,
        resize=True,
    ).astype(np.uint8)

    hull_rot = transform.rotate(
        hull,
        rotation_deg,
        order=0,
        preserve_range=True,
        resize=True,
    ).astype(np.uint8)

    if reference_raw is not None:
        reference_rot = transform.rotate(
            reference_raw,
            rotation_deg,
            order=1,
            preserve_range=True,
            resize=True,
        ).astype(reference_raw.dtype)
    else:
        reference_rot = None

    hull_width = hull_rot.sum(axis=1).astype(float)
    white_counts = corridor_rot.sum(axis=1).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.where(hull_width > 0, white_counts / hull_width, 0.0)

    retained_before_smoothing = coverage >= row_cov_thresh_rel_hull

    if merge_gap_px > 1:
        kernel = np.ones(merge_gap_px, dtype=int)
        retained_after_smoothing = (
            np.convolve(
                retained_before_smoothing.astype(np.uint8),
                kernel,
                mode="same",
            )
            > 0
        )
    else:
        retained_after_smoothing = retained_before_smoothing.copy()

    retained_row_mask_2d = rows_to_mask(
        retained_after_smoothing,
        corridor_rot.shape[1],
    )

    band_filtered_corridor_rot = (
        corridor_rot.astype(bool) & retained_row_mask_2d
    )

    records = []
    for y0, y1 in true_runs(retained_after_smoothing):
        band_height = int(y1 - y0 + 1)
        if band_height < min_band_height:
            continue

        band = hull_rot[y0 : y1 + 1, :]
        x_any = band.any(axis=0)
        if not np.any(x_any):
            continue

        x0 = int(np.argmax(x_any))
        x1 = int(len(x_any) - 1 - np.argmax(x_any[::-1]))

        used_for_width_profile = not (
            y1 < corridor_rot.shape[0] * 0.10
            or y0 > corridor_rot.shape[0] * 0.90
        )

        records.append(
            {
                "corridor_id": len(records),
                "x0": x0,
                "x1": x1,
                "y0": int(y0),
                "y1": int(y1),
                "height_px": band_height,
                "used_for_width_profile": used_for_width_profile,
            }
        )

    corridors = pd.DataFrame(records)
    if corridors.empty:
        raise RuntimeError(
            "No corridors detected. Check the surface, inversion setting, "
            "Cellpose file, and rotation angle."
        )

    profile_tables = {}
    for row in corridors.itertuples(index=False):
        values = []
        for x in range(row.x0, row.x1 + 1, scan_step_px):
            col = corridor_rot[row.y0 : row.y1 + 1, x]
            ys = np.flatnonzero(col)

            if ys.size == 0:
                y_top = np.nan
                y_bottom = np.nan
                width_px = 0
            else:
                y_top = int(row.y0 + ys[0])
                y_bottom = int(row.y0 + ys[-1])
                width_px = int(ys[-1] - ys[0] + 1)

            values.append(
                {
                    "corridor_id": int(row.corridor_id),
                    "x_px": int(x),
                    "x_relative_px": int(x - row.x0),
                    "x_relative_um": (x - row.x0) * pixel_size_um,
                    "y_top_px": y_top,
                    "y_bottom_px": y_bottom,
                    "width_px": width_px,
                    "width_um": width_px * pixel_size_um,
                    "used_for_width_profile": bool(row.used_for_width_profile),
                }
            )
        profile_tables[int(row.corridor_id)] = pd.DataFrame(values)

    corridor_support_mask = np.zeros_like(corridor_rot, dtype=bool)
    for row in corridors.itertuples(index=False):
        profile = profile_tables[int(row.corridor_id)]
        for point in profile.itertuples(index=False):
            if point.width_px <= 0:
                continue
            y_top = int(point.y_top_px)
            y_bottom = int(point.y_bottom_px)
            x = int(point.x_px)
            corridor_support_mask[y_top : y_bottom + 1, x] = True

    return {
        "cellpose_raw": cellpose_raw,
        "cellpose_used": cellpose_used,
        "gray": gray,
        "threshold": threshold,
        "pillars": pillars,
        "hull": hull,
        "candidate": candidate,
        "after_small_objects": after_small_objects,
        "cleaned": cleaned,
        "removed_small_objects": removed_small_objects,
        "filled_small_holes": filled_small_holes,
        "corridor_rot": corridor_rot,
        "hull_rot": hull_rot,
        "reference_raw": reference_raw,
        "reference_rot": reference_rot,
        "coverage": coverage,
        "retained_before_smoothing": retained_before_smoothing,
        "retained_after_smoothing": retained_after_smoothing,
        "retained_row_mask_2d": retained_row_mask_2d,
        "band_filtered_corridor_rot": band_filtered_corridor_rot,
        "corridor_support_mask": corridor_support_mask,
        "corridors": corridors,
        "profiles": profile_tables,
        "rotation_deg": float(rotation_deg),
        "scan_step_px": int(scan_step_px),
        "pixel_size_um": float(pixel_size_um),
        "row_cov_thresh_rel_hull": float(row_cov_thresh_rel_hull),
        "invert_cellpose": bool(invert_cellpose),
    }


def _display_mask(accessible, hull, outside_hull_value=0.70):
    accessible = np.asarray(accessible, dtype=bool)
    hull = np.asarray(hull, dtype=bool)
    if accessible.shape != hull.shape:
        raise ValueError(
            "accessible and hull must have the same shape, "
            f"got {accessible.shape} and {hull.shape}"
        )

    display = np.full(accessible.shape, float(outside_hull_value), dtype=float)
    display[hull] = 0.0
    display[accessible & hull] = 1.0
    return display


def _draw_box(ax, row, linewidth=0.7, linestyle="-", color=None):
    x0, x1, y0, y1 = row.x0, row.x1, row.y0, row.y1
    ax.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
    )


def create_exact_processing_schematic(
    result,
    surface_name,
    output_dir,
    selected_corridor_id=None,
    save_pdf=True,
    roi=None,
    roi_rel=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corridors = result["corridors"]
    eligible = corridors.loc[corridors["used_for_width_profile"]]
    selection_pool = eligible if not eligible.empty else corridors

    if selected_corridor_id is None:
        centre = result["corridor_rot"].shape[0] / 2
        centers = (
            selection_pool["y0"].to_numpy()
            + selection_pool["y1"].to_numpy()
        ) / 2
        selected = selection_pool.iloc[int(np.argmin(np.abs(centers - centre)))]
    else:
        matches = corridors.loc[
            corridors["corridor_id"] == int(selected_corridor_id)
        ]
        if matches.empty:
            raise ValueError(f"Corridor ID {selected_corridor_id} does not exist.")
        selected = matches.iloc[0]

    selected_id = int(selected["corridor_id"])

    roi_px = resolve_roi(result["cellpose_used"].shape, roi=roi, roi_rel=roi_rel)
    rot_roi_bbox = rotate_roi_to_bbox(
        result["cellpose_used"].shape,
        roi_px,
        result["rotation_deg"],
        margin_px=8,
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    # A. Full image with ROI box.
    axes[0].imshow(result["cellpose_used"], cmap="gray")
    x, y, rw, rh = roi_px
    axes[0].add_patch(
        Rectangle(
            (x, y),
            rw,
            rh,
            linewidth=2.0,
            edgecolor="red",
            facecolor="none",
        )
    )
    axes[0].set_title(
        "A  Cellpose flow input"
        + (" after inversion" if result["invert_cellpose"] else "")
        + "\nRed box: ROI shown in panels B–D"
    )

    # B. Cropped unrotated mask + hull.
    pillars_crop = result["pillars"][y:y + rh, x:x + rw]
    hull_crop = result["hull"][y:y + rh, x:x + rw]
    axes[1].imshow(pillars_crop, cmap="gray_r", interpolation="nearest")
    axes[1].contour(
        hull_crop.astype(float),
        levels=[0.5],
        linestyles="--",
        linewidths=0.8,
    )
    axes[1].set_title(
        f"B  Otsu black-pixel mask in ROI\nthreshold = {result['threshold']:.4g}"
    )

    # C. Cropped rotated corridor-support representation.
    support_display = _display_mask(
        result["corridor_support_mask"],
        result["hull_rot"].astype(bool),
    )
    rotated_display = _display_mask(
        result["corridor_rot"].astype(bool),
        result["hull_rot"].astype(bool),
    )

    if rot_roi_bbox is not None:
        c_img = crop_xy(support_display, rot_roi_bbox)
        d_img = crop_xy(rotated_display, rot_roi_bbox)
    else:
        c_img = support_display
        d_img = rotated_display

    axes[2].imshow(
        c_img,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[2].set_title(
        "C  Horizontal corridor support in rotated ROI\n"
        "Ignored accessible pixels are black"
    )

    axes[3].imshow(
        d_img,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[3].set_title(
        f"D  Rotated cleaned corridor mask in ROI ({result['rotation_deg']:+.2f}°)\n"
        "Gray: outside hull and excluded"
    )

    # E. Full rotated view with bands.
    filtered_display = _display_mask(
        result["band_filtered_corridor_rot"],
        result["hull_rot"].astype(bool),
    )

    axes[4].imshow(
        filtered_display,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    image_width = filtered_display.shape[1]
    y_axis = np.arange(result["coverage"].size)
    coverage_x = result["coverage"] * image_width * 0.30
    threshold_x = result["row_cov_thresh_rel_hull"] * image_width * 0.30

    axes[4].plot(coverage_x, y_axis, linewidth=0.8)
    axes[4].axvline(threshold_x, linestyle="--", linewidth=0.8)

    for row in corridors.itertuples(index=False):
        style = "-" if row.used_for_width_profile else ":"
        _draw_box(axes[4], row, linewidth=0.6, linestyle=style)

    # Draw rotated ROI bbox on panel E as a red rectangle for context.
    if rot_roi_bbox is not None:
        rx0, rx1, ry0, ry1 = rot_roi_bbox
        axes[4].add_patch(
            Rectangle(
                (rx0, ry0),
                rx1 - rx0,
                ry1 - ry0,
                linewidth=1.5,
                edgecolor="red",
                facecolor="none",
            )
        )

    axes[4].set_title(
        "E  Row coverage, 3-row smoothing and corridor bands\n"
        "Dotted boxes: excluded by outer 10% positional rule"
    )

    # F. Width extraction zoom for selected corridor.
    axes[5].imshow(
        support_display,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    x0, x1 = int(selected["x0"]), int(selected["x1"])
    y0, y1 = int(selected["y0"]), int(selected["y1"])
    axes[5].set_xlim(max(0, x0 - 3), min(support_display.shape[1] - 1, x1 + 3))
    axes[5].set_ylim(min(support_display.shape[0] - 1, y1 + 3), max(0, y0 - 3))

    _draw_box(
        axes[5],
        type("Row", (), dict(x0=x0, x1=x1, y0=y0, y1=y1))(),
        linewidth=1.0,
    )

    profile = result["profiles"][selected_id]
    valid = profile.loc[profile["width_px"] > 0]

    if not valid.empty:
        n_lines = min(18, len(valid))
        shown = valid.iloc[np.linspace(0, len(valid) - 1, n_lines, dtype=int)]
        for point in shown.itertuples(index=False):
            axes[5].plot(
                [point.x_px, point.x_px],
                [point.y_top_px, point.y_bottom_px],
                linewidth=0.7,
            )

        midpoint = valid.iloc[len(valid) // 2]
        axes[5].annotate(
            "",
            xy=(midpoint["x_px"], midpoint["y_top_px"]),
            xytext=(midpoint["x_px"], midpoint["y_bottom_px"]),
            arrowprops={"arrowstyle": "<->", "linewidth": 1.0},
        )
        axes[5].text(
            midpoint["x_px"] + 2,
            (midpoint["y_top_px"] + midpoint["y_bottom_px"]) / 2,
            r"$W(x)$",
            va="center",
        )

    profile_ax = inset_axes(
        axes[5],
        width="80%",
        height="27%",
        loc="lower center",
        borderpad=0.8,
    )
    profile_ax.plot(profile["x_relative_um"], profile["width_um"], linewidth=0.8)
    profile_ax.set_xlabel("Position (µm)", fontsize=7)
    profile_ax.set_ylabel("W (µm)", fontsize=7)
    profile_ax.tick_params(labelsize=6)
    profile_ax.spines[["top", "right"]].set_visible(False)

    axes[5].set_title(
        f"F  Width from first to last accessible pixel + 1\n"
        f"corridor {selected_id}, step = {result['scan_step_px']} px"
    )

    for ax in axes:
        ax.set_axis_off()

    fig.suptitle(
        f"Cellpose-derived corridor processing | {surface_name}",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        bottom=0.05,
        top=0.91,
        wspace=0.08,
        hspace=0.18,
    )

    png_path = output_dir / "cellpose_corridor_processing_schematic.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(
            output_dir / "cellpose_corridor_processing_schematic.pdf",
            bbox_inches="tight",
        )

    plt.show()

    corridors.to_csv(output_dir / "corridor_bounding_boxes.csv", index=False)
    pd.concat(result["profiles"].values(), ignore_index=True).to_csv(
        output_dir / "corridor_width_profiles.csv",
        index=False,
    )

    print(f"Detected bands: {len(corridors)}")
    print(
        "Bands retained for width profiling after outer 10% rule: "
        f"{int(corridors['used_for_width_profile'].sum())}"
    )
    print(f"Selected corridor: {selected_id}")
    print(f"ROI (x, y, w, h): {roi_px}")
    print(f"Saved: {png_path.resolve()}")
