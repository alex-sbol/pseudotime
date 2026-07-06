"""
Cellpose-derived corridor processing with an explicit "gap-ignoring" display.

Key display change
------------------
Panel C shows the first stage at which ignored vertical gaps / ignored rows are
excluded from the visualization. Everything ignored at that step is drawn as
black inside the convex hull.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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

    # This is the first "gap-ignoring" visual stage requested by the user:
    # keep accessible pixels only on retained rows; ignored rows/gaps become black.
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


def _draw_box(ax, row, linewidth=0.7, linestyle="-"):
    x0, x1, y0, y1 = row.x0, row.x1, row.y0, row.y1
    ax.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        linewidth=linewidth,
        linestyle=linestyle,
    )


def create_exact_processing_schematic(
    result,
    surface_name,
    output_dir,
    selected_corridor_id=None,
    save_pdf=True,
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

    axes[0].imshow(result["cellpose_used"], cmap="gray")
    axes[0].set_title(
        "A  Cellpose flow input"
        + (" after inversion" if result["invert_cellpose"] else "")
    )

    axes[1].imshow(result["pillars"], cmap="gray_r", interpolation="nearest")
    axes[1].contour(
        result["hull"].astype(float),
        levels=[0.5],
        linestyles="--",
        linewidths=0.8,
    )
    axes[1].set_title(
        f"B  Otsu black-pixel mask and convex hull\nthreshold = {result['threshold']:.4g}"
    )

    filtered_display = _display_mask(
        result["band_filtered_corridor_rot"],
        result["hull_rot"].astype(bool),
    )
    axes[2].imshow(
        filtered_display,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[2].set_title(
        "C  First gap-ignoring step after row selection\nIgnored gaps/rows are shown in black"
    )

    rotated_display = _display_mask(
        result["corridor_rot"].astype(bool),
        result["hull_rot"].astype(bool),
    )
    axes[3].imshow(
        rotated_display,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[3].set_title(
        f"D  Rotated cleaned corridor mask ({result['rotation_deg']:+.2f}°)\nGray: outside hull and excluded"
    )

    axes[4].imshow(
        filtered_display,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    image_width = filtered_display.shape[1]
    y = np.arange(result["coverage"].size)
    coverage_x = result["coverage"] * image_width * 0.30
    threshold_x = result["row_cov_thresh_rel_hull"] * image_width * 0.30

    axes[4].plot(coverage_x, y, linewidth=0.8)
    axes[4].axvline(threshold_x, linestyle="--", linewidth=0.8)

    for row in corridors.itertuples(index=False):
        style = "-" if row.used_for_width_profile else ":"
        _draw_box(axes[4], row, linewidth=0.6, linestyle=style)

    axes[4].set_title(
        "E  Row coverage, 3-row smoothing and corridor bands\nDotted boxes: excluded by outer 10% positional rule"
    )

    axes[5].imshow(
        filtered_display,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    x0, x1 = int(selected["x0"]), int(selected["x1"])
    y0, y1 = int(selected["y0"]), int(selected["y1"])
    axes[5].set_xlim(max(0, x0 - 3), min(filtered_display.shape[1] - 1, x1 + 3))
    axes[5].set_ylim(min(filtered_display.shape[0] - 1, y1 + 3), max(0, y0 - 3))

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
        f"F  Width from first to last accessible pixel + 1\ncorridor {selected_id}, step = {result['scan_step_px']} px"
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
    print(f"Saved: {png_path.resolve()}")
