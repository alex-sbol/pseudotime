
"""
Cellpose-derived corridor processing with a publication-oriented ROI schematic.

Figure logic
------------
A. Full Cellpose-derived input with a red ROI.
B. Mask cleanup and smoothing within the original ROI.
C. The same physical ROI after rotation and cleanup.
D. The same rotated ROI after ignored accessible pixels are set to black.
E. Corridor acceptance and exclusion criteria with a separate coverage axis.
F. Local width extraction from the selected corridor segment overlapping the ROI.

The analysis itself is unchanged. The ROI and support masks affect only the
schematic visualization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from skimage import color, filters, measure, morphology, transform
from tifffile import imread


# =============================================================================
# BASIC HELPERS
# =============================================================================

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
    """
    Reproduce the thresholding logic used in the original workflow.
    """
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
    Return an ROI as integer pixel coordinates: (x, y, width, height).
    """
    image_height, image_width = image_shape[:2]

    if roi is not None:
        x, y, width, height = [int(round(value)) for value in roi]

    elif roi_rel is not None:
        x = int(round(roi_rel[0] * image_width))
        y = int(round(roi_rel[1] * image_height))
        width = int(round(roi_rel[2] * image_width))
        height = int(round(roi_rel[3] * image_height))

    else:
        x = int(round(0.20 * image_width))
        y = int(round(0.24 * image_height))
        width = int(round(0.25 * image_width))
        height = int(round(0.25 * image_height))

    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    width = max(1, min(width, image_width - x))
    height = max(1, min(height, image_height - y))

    return x, y, width, height


def rotate_roi_mask(image_shape, roi, angle_deg):
    """
    Rotate a binary ROI mask using exactly the same geometry as the image mask.
    """
    image_height, image_width = image_shape[:2]
    x, y, width, height = roi

    roi_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    roi_mask[y:y + height, x:x + width] = 1

    return transform.rotate(
        roi_mask,
        angle_deg,
        order=0,
        preserve_range=True,
        resize=True,
    ).astype(bool)


def bbox_from_mask(mask, margin_px=4):
    """
    Return (x0, x1, y0, y1) around a non-empty binary mask.
    """
    y_coords, x_coords = np.where(mask)

    if x_coords.size == 0:
        return None

    x0 = max(0, int(x_coords.min()) - margin_px)
    x1 = min(mask.shape[1], int(x_coords.max()) + 1 + margin_px)
    y0 = max(0, int(y_coords.min()) - margin_px)
    y1 = min(mask.shape[0], int(y_coords.max()) + 1 + margin_px)

    return x0, x1, y0, y1


def crop_by_bbox(array, bbox):
    x0, x1, y0, y1 = bbox
    return array[y0:y1, x0:x1]


def display_mask(accessible, hull, outside_value=0.72):
    """
    Convert binary masks into a publication-friendly grayscale image.

    0.00 = black: non-accessible or ignored pixels inside the hull
    1.00 = white: accessible pixels retained for the displayed stage
    0.72 = gray: outside the convex hull
    """
    accessible = np.asarray(accessible, dtype=bool)
    hull = np.asarray(hull, dtype=bool)

    if accessible.shape != hull.shape:
        raise ValueError(
            f"Mask-shape mismatch: {accessible.shape} versus {hull.shape}"
        )

    displayed = np.full(accessible.shape, float(outside_value), dtype=float)
    displayed[hull] = 0.0
    displayed[accessible & hull] = 1.0

    return displayed


def mask_rotated_roi_display(displayed_image, rotated_roi_mask, outside_value=0.72):
    """
    Keep only the exact transformed ROI. Everything outside it becomes gray.
    """
    output = np.full_like(displayed_image, float(outside_value), dtype=float)
    output[rotated_roi_mask] = displayed_image[rotated_roi_mask]
    return output


def draw_box(ax, x0, x1, y0, y1, *, linewidth=0.8, linestyle="-", color=None):
    rectangle = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=linewidth,
        linestyle=linestyle,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rectangle)


# =============================================================================
# CORE PROCESSING
# =============================================================================

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
    """
    Run the Cellpose-derived corridor-processing workflow.
    """
    cellpose_raw = _load_image(cellpose_image)
    reference_raw = _load_image(reference_image)

    cellpose_used = np.asarray(cellpose_raw)

    if invert_cellpose:
        cellpose_used = cellpose_used.max() - cellpose_used

    gray, pillars, threshold = black_mask_exact(cellpose_used)

    hull = morphology.convex_hull_image(pillars)
    candidate_accessible = (~pillars) & hull

    after_small_objects = morphology.remove_small_objects(
        candidate_accessible,
        min_size=tiny_white_px,
    )

    cleaned_accessible = morphology.remove_small_holes(
        after_small_objects,
        area_threshold=hole_area_threshold,
    )

    removed_small_objects = candidate_accessible & ~after_small_objects
    filled_small_holes = cleaned_accessible & ~after_small_objects

    corridor_rot = transform.rotate(
        cleaned_accessible,
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
    accessible_count = corridor_rot.sum(axis=1).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.where(
            hull_width > 0,
            accessible_count / hull_width,
            0.0,
        )

    retained_before_smoothing = coverage >= row_cov_thresh_rel_hull

    if merge_gap_px > 1:
        retained_after_smoothing = (
            np.convolve(
                retained_before_smoothing.astype(np.uint8),
                np.ones(merge_gap_px, dtype=int),
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

    corridor_records = []
    rejected_short_band_records = []

    for y0, y1 in true_runs(retained_after_smoothing):
        band_height = int(y1 - y0 + 1)

        hull_band = hull_rot[y0:y1 + 1, :]
        x_any = hull_band.any(axis=0)

        if not np.any(x_any):
            continue

        x0 = int(np.argmax(x_any))
        x1 = int(len(x_any) - 1 - np.argmax(x_any[::-1]))

        if band_height < min_band_height:
            rejected_short_band_records.append(
                {
                    "x0": x0,
                    "x1": x1,
                    "y0": int(y0),
                    "y1": int(y1),
                    "height_px": band_height,
                    "rejection_reason": (
                        f"band height < {min_band_height} px"
                    ),
                }
            )
            continue

        used_for_width_profile = not (
            y1 < corridor_rot.shape[0] * 0.10
            or y0 > corridor_rot.shape[0] * 0.90
        )

        corridor_records.append(
            {
                "corridor_id": len(corridor_records),
                "x0": x0,
                "x1": x1,
                "y0": int(y0),
                "y1": int(y1),
                "height_px": band_height,
                "used_for_width_profile": used_for_width_profile,
            }
        )

    corridors = pd.DataFrame(corridor_records)
    rejected_short_bands = pd.DataFrame(rejected_short_band_records)

    if corridors.empty:
        raise RuntimeError(
            "No corridors were detected. Check the Cellpose file, inversion, "
            "rotation angle, and row-coverage threshold."
        )

    profile_tables = {}

    for row in corridors.itertuples(index=False):
        profile_rows = []

        for x in range(row.x0, row.x1 + 1, scan_step_px):
            column = corridor_rot[row.y0:row.y1 + 1, x]
            y_positions = np.flatnonzero(column)

            if y_positions.size == 0:
                y_top = np.nan
                y_bottom = np.nan
                width_px = 0

            else:
                y_top = int(row.y0 + y_positions[0])
                y_bottom = int(row.y0 + y_positions[-1])
                width_px = int(y_positions[-1] - y_positions[0] + 1)

            profile_rows.append(
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

        profile_tables[int(row.corridor_id)] = pd.DataFrame(profile_rows)

    # Visualization-only corridor-support mask.
    corridor_support_mask = np.zeros_like(corridor_rot, dtype=bool)

    for corridor_id, profile in profile_tables.items():
        valid = profile.loc[profile["width_px"] > 0]

        for point in valid.itertuples(index=False):
            x = int(point.x_px)
            y_top = int(point.y_top_px)
            y_bottom = int(point.y_bottom_px)
            corridor_support_mask[y_top:y_bottom + 1, x] = True

    return {
        "cellpose_raw": cellpose_raw,
        "cellpose_used": cellpose_used,
        "gray": gray,
        "threshold": threshold,
        "pillars": pillars,
        "hull": hull,
        "candidate_accessible": candidate_accessible,
        "after_small_objects": after_small_objects,
        "cleaned_accessible": cleaned_accessible,
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
        "rejected_short_bands": rejected_short_bands,
        "profiles": profile_tables,
        "rotation_deg": float(rotation_deg),
        "scan_step_px": int(scan_step_px),
        "pixel_size_um": float(pixel_size_um),
        "row_cov_thresh_rel_hull": float(row_cov_thresh_rel_hull),
        "min_band_height": int(min_band_height),
        "outer_exclusion_fraction": 0.10,
        "invert_cellpose": bool(invert_cellpose),
    }


# =============================================================================
# FIGURE HELPERS
# =============================================================================

def choose_corridor_for_roi(corridors, rotated_roi_mask, selected_corridor_id=None):
    """
    Select a corridor that intersects the rotated ROI and is closest to its centre.
    """
    if selected_corridor_id is not None:
        matches = corridors.loc[
            corridors["corridor_id"] == int(selected_corridor_id)
        ]

        if matches.empty:
            raise ValueError(
                f"Corridor ID {selected_corridor_id} does not exist."
            )

        return matches.iloc[0]

    roi_y, roi_x = np.where(rotated_roi_mask)

    if roi_x.size == 0:
        eligible = corridors.loc[corridors["used_for_width_profile"]]
        pool = eligible if not eligible.empty else corridors
        return pool.iloc[len(pool) // 2]

    roi_x0 = int(roi_x.min())
    roi_x1 = int(roi_x.max())
    roi_y0 = int(roi_y.min())
    roi_y1 = int(roi_y.max())
    roi_center_y = (roi_y0 + roi_y1) / 2

    candidates = corridors.loc[
        (corridors["x1"] >= roi_x0)
        & (corridors["x0"] <= roi_x1)
        & (corridors["y1"] >= roi_y0)
        & (corridors["y0"] <= roi_y1)
        & corridors["used_for_width_profile"]
    ].copy()

    if candidates.empty:
        candidates = corridors.loc[corridors["used_for_width_profile"]].copy()

    if candidates.empty:
        candidates = corridors.copy()

    candidates["distance_to_roi_center"] = np.abs(
        (candidates["y0"] + candidates["y1"]) / 2 - roi_center_y
    )

    return candidates.sort_values("distance_to_roi_center").iloc[0]


def find_ignored_connection_points(
    cleaned_rot,
    support_mask,
    rotated_roi_mask,
    max_points=3,
):
    """
    Find representative ignored components for arrow annotations in panel C.
    """
    ignored = (
        cleaned_rot.astype(bool)
        & ~support_mask.astype(bool)
        & rotated_roi_mask.astype(bool)
    )

    labeled = measure.label(ignored, connectivity=2)
    regions = sorted(
        measure.regionprops(labeled),
        key=lambda region: region.area,
        reverse=True,
    )

    points = []

    for region in regions:
        if region.area < 2:
            continue

        y, x = region.centroid
        points.append((float(x), float(y)))

        if len(points) >= max_points:
            break

    return points


def local_corridor_segment(
    selected,
    profile,
    rotated_roi_mask,
    max_width_px=220,
):
    """
    Choose a local selected-corridor segment, preferably the ROI intersection.
    """
    roi_y, roi_x = np.where(rotated_roi_mask)

    if roi_x.size:
        roi_x0 = int(roi_x.min())
        roi_x1 = int(roi_x.max())

        segment_x0 = max(int(selected["x0"]), roi_x0)
        segment_x1 = min(int(selected["x1"]), roi_x1)

    else:
        segment_x0 = int(selected["x0"])
        segment_x1 = int(selected["x1"])

    if segment_x1 <= segment_x0:
        center_x = int((selected["x0"] + selected["x1"]) / 2)
        half_width = max_width_px // 2
        segment_x0 = max(int(selected["x0"]), center_x - half_width)
        segment_x1 = min(int(selected["x1"]), center_x + half_width)

    if segment_x1 - segment_x0 > max_width_px:
        center_x = int((segment_x0 + segment_x1) / 2)
        half_width = max_width_px // 2
        segment_x0 = center_x - half_width
        segment_x1 = center_x + half_width

    segment_profile = profile.loc[
        (profile["x_px"] >= segment_x0)
        & (profile["x_px"] <= segment_x1)
    ].copy()

    return segment_x0, segment_x1, segment_profile


# =============================================================================
# PUBLICATION SCHEMATIC
# =============================================================================

def create_exact_processing_schematic(
    result,
    surface_name,
    output_dir,
    selected_corridor_id=None,
    save_pdf=True,
    save_svg=True,
    roi=None,
    roi_rel=None,
    max_local_width_px=220,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_px = resolve_roi(
        result["cellpose_used"].shape,
        roi=roi,
        roi_rel=roi_rel,
    )

    rotated_roi_mask = rotate_roi_mask(
        result["cellpose_used"].shape,
        roi_px,
        result["rotation_deg"],
    )

    rotated_roi_bbox = bbox_from_mask(rotated_roi_mask, margin_px=3)

    if rotated_roi_bbox is None:
        raise RuntimeError("The transformed ROI is empty.")

    selected = choose_corridor_for_roi(
        result["corridors"],
        rotated_roi_mask,
        selected_corridor_id=selected_corridor_id,
    )

    selected_id = int(selected["corridor_id"])
    selected_profile = result["profiles"][selected_id]

    segment_x0, segment_x1, segment_profile = local_corridor_segment(
        selected,
        selected_profile,
        rotated_roi_mask,
        max_width_px=max_local_width_px,
    )

    selected_y0 = int(selected["y0"])
    selected_y1 = int(selected["y1"])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    figure = plt.figure(figsize=(14, 9))
    outer = figure.add_gridspec(
        2,
        3,
        left=0.04,
        right=0.985,
        bottom=0.06,
        top=0.90,
        wspace=0.22,
        hspace=0.24,
    )

    ax_a = figure.add_subplot(outer[0, 0])
    ax_b = figure.add_subplot(outer[0, 1])
    ax_c = figure.add_subplot(outer[0, 2])
    ax_d = figure.add_subplot(outer[1, 0])

    e_grid = outer[1, 1].subgridspec(
        1,
        2,
        width_ratios=[4.8, 1.15],
        wspace=0.06,
    )
    ax_e_image = figure.add_subplot(e_grid[0, 0])
    ax_e_coverage = figure.add_subplot(e_grid[0, 1])

    f_grid = outer[1, 2].subgridspec(
        2,
        1,
        height_ratios=[3.2, 1.25],
        hspace=0.18,
    )
    ax_f_image = figure.add_subplot(f_grid[0, 0])
    ax_f_profile = figure.add_subplot(f_grid[1, 0])

    # -------------------------------------------------------------------------
    # A. Full input with ROI
    # -------------------------------------------------------------------------
    ax_a.imshow(result["cellpose_used"], cmap="gray")

    roi_x, roi_y, roi_width, roi_height = roi_px

    ax_a.add_patch(
        Rectangle(
            (roi_x, roi_y),
            roi_width,
            roi_height,
            linewidth=2.0,
            edgecolor="red",
            facecolor="none",
        )
    )

    ax_a.set_title("A  Cellpose-derived topography mask")
    ax_a.set_axis_off()

    # -------------------------------------------------------------------------
    # B. Mask cleanup and smoothing in the original ROI
    # -------------------------------------------------------------------------
    cleaned_unrotated_display = display_mask(
        result["cleaned_accessible"],
        result["hull"],
    )

    cleaned_unrotated_crop = cleaned_unrotated_display[
        roi_y:roi_y + roi_height,
        roi_x:roi_x + roi_width,
    ]

    ax_b.imshow(
        cleaned_unrotated_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax_b.set_title("B  Mask cleanup and smoothing")
    ax_b.set_axis_off()

    # -------------------------------------------------------------------------
    # C. Same ROI after rotation and cleanup
    # -------------------------------------------------------------------------
    cleaned_display = display_mask(
        result["corridor_rot"].astype(bool),
        result["hull_rot"].astype(bool),
    )

    cleaned_roi_display = mask_rotated_roi_display(
        cleaned_display,
        rotated_roi_mask,
    )

    cleaned_roi_crop = crop_by_bbox(
        cleaned_roi_display,
        rotated_roi_bbox,
    )

    ax_c.imshow(
        cleaned_roi_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax_c.set_title("C  Rotated accessible-space mask")
    ax_c.set_axis_off()

    # Add arrows to representative pixels that will be ignored in D.
    ignored_points = find_ignored_connection_points(
        result["corridor_rot"],
        result["corridor_support_mask"],
        rotated_roi_mask,
        max_points=3,
    )

    bbox_x0, _, bbox_y0, _ = rotated_roi_bbox

    for point_index, (point_x, point_y) in enumerate(ignored_points):
        local_x = point_x - bbox_x0
        local_y = point_y - bbox_y0

        text_x = local_x + 18
        text_y = local_y - 12 - point_index * 4

        ax_c.annotate(
            "",
            xy=(local_x, local_y),
            xytext=(text_x, text_y),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.1,
                "color": "red",
            },
        )

    # -------------------------------------------------------------------------
    # D. Same rotated ROI after ignored pixels become black
    # -------------------------------------------------------------------------
    support_display = display_mask(
        result["corridor_support_mask"],
        result["hull_rot"].astype(bool),
    )

    support_roi_display = mask_rotated_roi_display(
        support_display,
        rotated_roi_mask,
    )

    support_roi_crop = crop_by_bbox(
        support_roi_display,
        rotated_roi_bbox,
    )

    ax_d.imshow(
        support_roi_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax_d.set_title("D  Isolated horizontal corridor support")
    ax_d.set_axis_off()

    # -------------------------------------------------------------------------
    # E. Corridor acceptance and exclusion criteria
    # -------------------------------------------------------------------------
    band_display = display_mask(
        result["band_filtered_corridor_rot"],
        result["hull_rot"].astype(bool),
    )

    ax_e_image.imshow(
        band_display,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    image_height = band_display.shape[0]
    outer_fraction = result["outer_exclusion_fraction"]
    upper_limit = image_height * outer_fraction
    lower_limit = image_height * (1.0 - outer_fraction)

    # Criterion 3: the outermost 10% of the rotated image is excluded.
    ax_e_image.axhspan(
        0,
        upper_limit,
        facecolor="lightgray",
        alpha=0.75,
        hatch="///",
        edgecolor="gray",
        linewidth=0.0,
        zorder=2,
    )
    ax_e_image.axhspan(
        lower_limit,
        image_height,
        facecolor="lightgray",
        alpha=0.75,
        hatch="///",
        edgecolor="gray",
        linewidth=0.0,
        zorder=2,
    )
    ax_e_image.axhline(
        upper_limit,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        zorder=4,
    )
    ax_e_image.axhline(
        lower_limit,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        zorder=4,
    )

    ax_e_image.text(
        0.02,
        0.055,
        "Excluded: outer 10%",
        transform=ax_e_image.transAxes,
        fontsize=7,
        fontweight="bold",
        va="center",
        ha="left",
        bbox={
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.85,
            "pad": 1.5,
        },
        zorder=6,
    )
    ax_e_image.text(
        0.02,
        0.945,
        "Excluded: outer 10%",
        transform=ax_e_image.transAxes,
        fontsize=7,
        fontweight="bold",
        va="center",
        ha="left",
        bbox={
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.85,
            "pad": 1.5,
        },
        zorder=6,
    )

    # Criterion 2: candidate bands must be at least min_band_height pixels high.
    rejected_short_bands = result["rejected_short_bands"]

    if not rejected_short_bands.empty:
        for row in rejected_short_bands.itertuples(index=False):
            # Highlight short rejected bands in the same region-based style as the
            # outer exclusion criterion: translucent fill + hatch + boundary.
            short_band_patch = Rectangle(
                (row.x0, row.y0),
                row.x1 - row.x0,
                row.y1 - row.y0,
                facecolor="mistyrose",
                edgecolor="red",
                linewidth=0.9,
                linestyle="--",
                hatch="xxx",
                alpha=0.60,
                zorder=5,
            )
            ax_e_image.add_patch(short_band_patch)

        # Add one compact label pointing to the short-band exclusion group.
        first_short_band = rejected_short_bands.iloc[0]
        short_label_x = (first_short_band["x0"] + first_short_band["x1"]) / 2
        short_label_y = max(10, first_short_band["y0"] - 18)

        ax_e_image.annotate(
            f"Excluded: band height < {result['min_band_height']} px",
            xy=(short_label_x, (first_short_band["y0"] + first_short_band["y1"]) / 2),
            xytext=(short_label_x + 55, short_label_y),
            textcoords="data",
            fontsize=7,
            fontweight="bold",
            color="red",
            bbox={
                "facecolor": "white",
                "edgecolor": "red",
                "alpha": 0.90,
                "pad": 1.5,
            },
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.0,
                "color": "red",
            },
            zorder=8,
        )

    # Accepted bands, outer-position exclusions, and the selected example.
    for row in result["corridors"].itertuples(index=False):
        if int(row.corridor_id) == selected_id:
            draw_box(
                ax_e_image,
                row.x0,
                row.x1,
                row.y0,
                row.y1,
                linewidth=1.7,
                linestyle="-",
                color="red",
            )

        elif row.used_for_width_profile:
            draw_box(
                ax_e_image,
                row.x0,
                row.x1,
                row.y0,
                row.y1,
                linewidth=0.65,
                linestyle="-",
                color="black",
            )

        else:
            draw_box(
                ax_e_image,
                row.x0,
                row.x1,
                row.y0,
                row.y1,
                linewidth=1.0,
                linestyle="--",
                color="gray",
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.0,
            linestyle="-",
            label="Accepted band",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=1.7,
            linestyle="-",
            label="Selected example",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.0,
            linestyle="--",
            label="Excluded: outer 10%",
        ),
        Patch(
            facecolor="mistyrose",
            edgecolor="red",
            hatch="xxx",
            label=(
                f"Excluded: height < "
                f"{result['min_band_height']} px"
            ),
        ),
    ]

    ax_e_image.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        fontsize=6.5,
        frameon=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.9,
        columnspacing=0.8,
        handlelength=1.8,
    )

    ax_e_image.set_title("E  Corridor acceptance and exclusion criteria")
    ax_e_image.set_axis_off()

    # Criterion 1: row coverage must meet the 0.07 threshold.
    y_values = np.arange(result["coverage"].size)
    threshold = result["row_cov_thresh_rel_hull"]
    coverage_max = max(
        threshold * 2.2,
        float(np.nanmax(result["coverage"])) * 1.05,
    )

    ax_e_coverage.axvspan(
        0,
        threshold,
        facecolor="lightgray",
        alpha=0.65,
        zorder=0,
    )

    ax_e_coverage.plot(
        result["coverage"],
        y_values,
        linewidth=1.0,
        color="black",
        zorder=2,
    )

    ax_e_coverage.axvline(
        threshold,
        linestyle="--",
        linewidth=1.0,
        color="red",
        zorder=3,
    )

    ax_e_coverage.text(
        threshold,
        0.02,
        f"  threshold = {threshold:.2f}",
        transform=ax_e_coverage.get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=6.5,
        color="red",
    )

    ax_e_coverage.text(
        threshold * 0.50,
        0.98,
        "Rejected",
        transform=ax_e_coverage.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6.5,
        fontweight="bold",
    )

    ax_e_coverage.set_ylim(result["coverage"].size - 1, 0)
    ax_e_coverage.set_xlim(0, coverage_max)
    ax_e_coverage.set_xlabel(r"Row coverage $C(y)$", fontsize=8)
    ax_e_coverage.set_ylabel("Image row", fontsize=8)
    ax_e_coverage.tick_params(labelsize=7)
    ax_e_coverage.spines[["top", "right"]].set_visible(False)

    # -------------------------------------------------------------------------
    # F. Local width extraction
    # -------------------------------------------------------------------------
    local_margin_y = 4
    local_y0 = max(0, selected_y0 - local_margin_y)
    local_y1 = min(
        support_display.shape[0],
        selected_y1 + 1 + local_margin_y,
    )

    local_support = support_display[
        local_y0:local_y1,
        segment_x0:segment_x1 + 1,
    ]

    ax_f_image.imshow(
        local_support,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )

    valid_segment = segment_profile.loc[
        segment_profile["width_px"] > 0
    ]

    if not valid_segment.empty:
        number_of_lines = min(8, len(valid_segment))

        line_rows = valid_segment.iloc[
            np.linspace(
                0,
                len(valid_segment) - 1,
                number_of_lines,
                dtype=int,
            )
        ]

        for point in line_rows.itertuples(index=False):
            local_x = point.x_px - segment_x0
            local_top = point.y_top_px - local_y0
            local_bottom = point.y_bottom_px - local_y0

            ax_f_image.plot(
                [local_x, local_x],
                [local_top, local_bottom],
                linewidth=0.9,
                color="gray",
            )

        midpoint = valid_segment.iloc[len(valid_segment) // 2]
        midpoint_x = midpoint["x_px"] - segment_x0
        midpoint_top = midpoint["y_top_px"] - local_y0
        midpoint_bottom = midpoint["y_bottom_px"] - local_y0

        ax_f_image.annotate(
            "",
            xy=(midpoint_x, midpoint_top),
            xytext=(midpoint_x, midpoint_bottom),
            arrowprops={
                "arrowstyle": "<->",
                "linewidth": 1.2,
                "color": "red",
            },
        )

        ax_f_image.text(
            midpoint_x + 3,
            (midpoint_top + midpoint_bottom) / 2,
            r"$W(x)$",
            va="center",
            fontsize=10,
        )

    ax_f_image.set_title("F  Local confinement-width extraction")
    ax_f_image.set_axis_off()

    if segment_profile.empty:
        ax_f_profile.text(
            0.5,
            0.5,
            "No width values in selected segment",
            ha="center",
            va="center",
            transform=ax_f_profile.transAxes,
        )

    else:
        x_um = (
            segment_profile["x_px"].to_numpy() - segment_x0
        ) * result["pixel_size_um"]

        ax_f_profile.plot(
            x_um,
            segment_profile["width_um"],
            linewidth=1.0,
            color="black",
        )

        ax_f_profile.set_xlabel("Position within displayed segment (µm)")
        ax_f_profile.set_ylabel(r"$W(x)$ (µm)")
        ax_f_profile.spines[["top", "right"]].set_visible(False)
        ax_f_profile.tick_params(labelsize=7)

    figure.suptitle(
        f"Cellpose-derived corridor processing | {surface_name}",
        fontsize=14,
        fontweight="bold",
    )

    png_path = output_dir / "cellpose_corridor_processing_schematic.png"
    pdf_path = output_dir / "cellpose_corridor_processing_schematic.pdf"
    svg_path = output_dir / "cellpose_corridor_processing_schematic.svg"

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    if save_pdf:
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )

    if save_svg:
        figure.savefig(
            svg_path,
            bbox_inches="tight",
        )

    plt.show()

    result["corridors"].to_csv(
        output_dir / "corridor_bounding_boxes.csv",
        index=False,
    )

    result["rejected_short_bands"].to_csv(
        output_dir / "rejected_short_corridor_bands.csv",
        index=False,
    )

    pd.concat(
        result["profiles"].values(),
        ignore_index=True,
    ).to_csv(
        output_dir / "corridor_width_profiles.csv",
        index=False,
    )

    print(f"Detected bands: {len(result['corridors'])}")
    print(
        "Bands retained after outer 10% rule: "
        f"{int(result['corridors']['used_for_width_profile'].sum())}"
    )
    print(f"Selected corridor: {selected_id}")
    print(
        "Rejected short bands: "
        f"{len(result['rejected_short_bands'])}"
    )
    print(f"ROI in original image (x, y, w, h): {roi_px}")
    print(f"Displayed corridor segment: x={segment_x0} to x={segment_x1}")
    print(f"Saved: {png_path.resolve()}")
