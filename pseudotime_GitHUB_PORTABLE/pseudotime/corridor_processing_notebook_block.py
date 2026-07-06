# Complete notebook block: real-image corridor isolation and schematic generation

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.morphology import convex_hull_image
from skimage.segmentation import clear_border
from skimage.transform import rotate


# =============================================================================
# SETTINGS
# =============================================================================

# Use the image already loaded in your notebook:
# INPUT_IMAGE = img
# Or load it directly from disk:
INPUT_IMAGE = Path(r"..\FixedPattern_16x16_1462_binary_filled.tif")

SURFACE_NAME = "TopoChip feature 1462"
ROTATION_DEG = 28.34       # 1462: +28.34; 1476: -13.47
PIXEL_SIZE_UM = 0.56
COVERAGE_THRESHOLD = 0.07
MIN_ACCESSIBLE_AREA = 16
MAX_FILLED_HOLE_AREA = 64
ROW_KERNEL_SIZE = 3
MIN_BAND_HEIGHT = 5
MANUAL_THRESHOLD = None    # normalized 0-1 threshold; None = automatic
SELECTED_CORRIDOR_ID = None
OUTPUT_DIR = Path("corridor_processing_output")
SAVE_PDF = False           # first inspect PNG; enable after the layout is final


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def to_gray01(image):
    image = np.squeeze(np.asarray(image))
    if image.ndim == 3:
        image = rgb2gray(image[..., :3]) if image.shape[-1] >= 3 else image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D or RGB image, got {image.shape}")
    image = image.astype(np.float32)
    lo, hi = np.nanmin(image), np.nanmax(image)
    return (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)


def rotate_mask(mask, angle):
    return rotate(
        mask.astype(np.uint8),
        angle=angle,
        resize=True,
        order=0,
        mode="constant",
        cval=0,
        preserve_range=True,
    ) > 0.5


def true_runs(values):
    values = np.asarray(values, dtype=bool)
    changes = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts, ends))


def remove_components_smaller_than(mask, min_area):
    labels, _ = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


def fill_enclosed_holes_smaller_than(mask, max_area):
    all_enclosed = ndi.binary_fill_holes(mask) & ~mask
    labels, _ = ndi.label(all_enclosed, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    fill = sizes < max_area
    fill[0] = False
    return mask | fill[labels]


def process_corridors(
    image_or_path,
    rotation_deg,
    pixel_size_um=0.56,
    coverage_threshold=0.07,
    min_accessible_area=16,
    max_filled_hole_area=64,
    row_kernel_size=3,
    min_band_height=5,
    manual_threshold=None,
):
    if isinstance(image_or_path, (str, Path)):
        image_or_path = Path(image_or_path)
        if not image_or_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_or_path.resolve()}\n"
                "Set INPUT_IMAGE to img or to the correct TIFF path."
            )
        raw = imread(image_or_path)
    else:
        raw = np.asarray(image_or_path)

    gray = to_gray01(raw)
    threshold = (
        float(manual_threshold)
        if manual_threshold is not None
        else float(np.unique(gray).mean())
        if np.unique(gray).size <= 2
        else float(threshold_otsu(gray))
    )

    # Black = pillars, white = accessible space.
    dark_pixels = gray <= threshold

    # Remove dark padding/background connected to the image border.
    pillars = clear_border(dark_pixels)
    if pillars.sum() < max(100, int(0.001 * pillars.size)):
        pillars = dark_pixels.copy()

    # Convex hull of pillars defines the patterned region.
    hull = convex_hull_image(pillars)

    # Non-pillar pixels inside the hull are candidate corridor space.
    candidate = hull & ~pillars

    # Remove accessible components with area <16 pixels.
    after_small_object_removal = remove_components_smaller_than(
        candidate, min_accessible_area
    )
    removed_accessible = candidate & ~after_small_object_removal

    # Fill enclosed non-accessible clusters with area <64 pixels.
    cleaned = fill_enclosed_holes_smaller_than(
        after_small_object_removal, max_filled_hole_area
    ) & hull
    filled_holes = cleaned & ~after_small_object_removal

    # Rotate the cleaned corridor mask and hull into one coordinate system.
    cleaned_rot = rotate_mask(cleaned, rotation_deg)
    hull_rot = rotate_mask(hull, rotation_deg)
    pillars_rot = rotate_mask(pillars, rotation_deg)
    cleaned_rot &= hull_rot
    pillars_rot &= hull_rot

    # Row-wise accessible-space coverage inside the hull.
    hull_count = hull_rot.sum(axis=1).astype(float)
    accessible_count = cleaned_rot.sum(axis=1).astype(float)
    coverage = np.divide(
        accessible_count,
        hull_count,
        out=np.zeros_like(accessible_count),
        where=hull_count > 0,
    )
    retained_raw = (hull_count > 0) & (coverage >= coverage_threshold)

    # Extend each retained row by one row above and below with a 3-row kernel.
    retained_dilated = ndi.binary_dilation(
        retained_raw, structure=np.ones(row_kernel_size, dtype=bool)
    )
    retained_dilated &= hull_count > 0

    # Group consecutive rows and discard bands shorter than five rows.
    retained_final = np.zeros_like(retained_dilated)
    corridor_records = []

    for y0, y1 in true_runs(retained_dilated):
        height = int(y1 - y0 + 1)
        if height < min_band_height:
            continue

        x_present = np.flatnonzero(np.any(hull_rot[y0:y1 + 1], axis=0))
        if x_present.size == 0:
            continue

        x0, x1 = int(x_present[0]), int(x_present[-1])
        retained_final[y0:y1 + 1] = True
        corridor_records.append(
            dict(
                corridor_id=len(corridor_records),
                x0=x0,
                x1=x1,
                y0=int(y0),
                y1=int(y1),
                height_px=height,
                centre_y_px=(y0 + y1) / 2,
            )
        )

    corridors = pd.DataFrame(corridor_records)
    if corridors.empty:
        raise RuntimeError(
            "No corridors were detected. First check image polarity and ROTATION_DEG."
        )

    # At every x pixel, find the uppermost and lowermost accessible pixels.
    profiles = {}
    for corridor in corridors.itertuples(index=False):
        profile = []
        for x in range(corridor.x0, corridor.x1 + 1):
            local_y = np.flatnonzero(cleaned_rot[corridor.y0:corridor.y1 + 1, x])
            if local_y.size:
                y_top = int(corridor.y0 + local_y[0])
                y_bottom = int(corridor.y0 + local_y[-1])
                width_px = y_bottom - y_top
                width_um = width_px * pixel_size_um
            else:
                y_top = y_bottom = np.nan
                width_px = width_um = np.nan

            profile.append(
                dict(
                    corridor_id=corridor.corridor_id,
                    x_px=x,
                    x_um=(x - corridor.x0) * pixel_size_um,
                    y_top_px=y_top,
                    y_bottom_px=y_bottom,
                    width_px=width_px,
                    width_um=width_um,
                )
            )
        profiles[corridor.corridor_id] = pd.DataFrame(profile)

    return dict(
        gray=gray,
        threshold=threshold,
        pillars=pillars,
        hull=hull,
        candidate=candidate,
        cleaned=cleaned,
        removed_accessible=removed_accessible,
        filled_holes=filled_holes,
        pillars_rot=pillars_rot,
        hull_rot=hull_rot,
        cleaned_rot=cleaned_rot,
        coverage=coverage,
        retained_raw=retained_raw,
        retained_dilated=retained_dilated,
        retained_final=retained_final,
        corridors=corridors,
        profiles=profiles,
    )


# =============================================================================
# FIGURE FUNCTION
# =============================================================================

def mask_for_display(accessible, hull):
    # 0 = outside hull, 1 = non-accessible/pillars, 2 = accessible space
    display = np.zeros(accessible.shape, dtype=np.uint8)
    display[hull] = 1
    display[accessible] = 2
    return display


def draw_box(ax, row, color, linewidth=0.7, alpha=0.6):
    x0, x1, y0, y1 = row.x0, row.x1, row.y0, row.y1
    ax.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )


def create_processing_schematic(
    result,
    surface_name,
    rotation_deg,
    pixel_size_um,
    coverage_threshold,
    output_dir,
    selected_corridor_id=None,
    save_pdf=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corridors = result["corridors"]
    if selected_corridor_id is None:
        centre_y = result["cleaned_rot"].shape[0] / 2
        selected = corridors.iloc[
            np.argmin(np.abs(corridors.centre_y_px.to_numpy() - centre_y))
        ]
    else:
        selected = corridors.loc[corridors.corridor_id == selected_corridor_id].iloc[0]
    selected_id = int(selected.corridor_id)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "pdf.fonttype": 42,
        }
    )

    cmap = ListedColormap(["#eeeeee", "#111111", "#ffffff"])
    blue = "#377eb8"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.ravel()

    # A: binary mask and hull
    axes[0].imshow(
        mask_for_display(result["candidate"], result["hull"]),
        cmap=cmap,
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    axes[0].contour(
        result["hull"].astype(float),
        levels=[0.5],
        colors=["0.45"],
        linestyles="--",
        linewidths=1,
    )
    axes[0].set_title("A  Binary mask and convex hull")

    # B: actual cleanup result
    axes[1].imshow(
        mask_for_display(result["cleaned"], result["hull"]),
        cmap=cmap,
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    ry, rx = np.where(result["removed_accessible"])
    fy, fx = np.where(result["filled_holes"])
    if rx.size:
        axes[1].scatter(rx, ry, s=2, marker="s", color="#d95f02", label="Removed <16 px")
    if fx.size:
        axes[1].scatter(fx, fy, s=2, marker="s", color="#1b9e77", label="Filled <64 px")
    if rx.size or fx.size:
        axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8)
    axes[1].set_title("B  Candidate corridor space after cleanup")

    # C: rotated masks
    rotated_display = mask_for_display(result["cleaned_rot"], result["hull_rot"])
    axes[2].imshow(rotated_display, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axes[2].contour(
        result["hull_rot"].astype(float),
        levels=[0.5],
        colors=["0.45"],
        linestyles="--",
        linewidths=1,
    )
    axes[2].text(
        0.02,
        0.02,
        f"Rotation = {rotation_deg:+.2f}°",
        transform=axes[2].transAxes,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
    )
    axes[2].set_title("C  Rotation to horizontal corridors")

    # D: row coverage and band formation
    axes[3].imshow(rotated_display, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    image_width = rotated_display.shape[1]
    coverage_x = result["coverage"] * image_width * 0.32
    y = np.arange(result["coverage"].size)
    axes[3].plot(coverage_x, y, color="black", linewidth=0.7)
    axes[3].axvline(
        coverage_threshold * image_width * 0.32,
        color=blue,
        linestyle="--",
        linewidth=1,
    )
    for y0, y1 in true_runs(result["retained_raw"]):
        axes[3].axhspan(y0 - 0.5, y1 + 0.5, color=blue, alpha=0.08, linewidth=0)
    for row in corridors.itertuples(index=False):
        axes[3].axhline(row.y0, color=blue, linewidth=0.7)
        axes[3].axhline(row.y1, color=blue, linewidth=0.7)
    axes[3].set_title("D  Row coverage, smoothing and retained bands")

    # E: bounding boxes used only for localization
    axes[4].imshow(rotated_display, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    for row in corridors.itertuples(index=False):
        chosen = row.corridor_id == selected_id
        draw_box(
            axes[4],
            row,
            blue if chosen else "0.45",
            linewidth=1.6 if chosen else 0.5,
            alpha=1 if chosen else 0.55,
        )
    axes[4].text(
        0.02,
        0.02,
        "Boxes localize corridors; widths are measured from the mask",
        transform=axes[4].transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
    )
    axes[4].set_title("E  Individual corridor localization")

    # F: local confinement width and full width profile
    row = selected
    x0, x1, y0, y1 = map(int, [row.x0, row.x1, row.y0, row.y1])
    profile = result["profiles"][selected_id]
    valid = profile.dropna(subset=["y_top_px", "y_bottom_px"])

    axes[5].imshow(rotated_display, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axes[5].set_xlim(max(0, x0 - 5), min(rotated_display.shape[1] - 1, x1 + 5))
    axes[5].set_ylim(min(rotated_display.shape[0] - 1, y1 + 5), max(0, y0 - 5))
    draw_box(axes[5], row, blue, linewidth=1.2, alpha=1)

    shown = valid.iloc[np.linspace(0, len(valid) - 1, min(15, len(valid)), dtype=int)]
    for point in shown.itertuples(index=False):
        axes[5].plot(
            [point.x_px, point.x_px],
            [point.y_top_px, point.y_bottom_px],
            color="0.4",
            linewidth=0.8,
        )
        axes[5].scatter(
            [point.x_px, point.x_px],
            [point.y_top_px, point.y_bottom_px],
            s=7,
            color=blue,
            zorder=3,
        )

    middle = valid.iloc[len(valid) // 2]
    axes[5].annotate(
        "",
        xy=(middle.x_px, middle.y_top_px),
        xytext=(middle.x_px, middle.y_bottom_px),
        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2),
    )
    axes[5].text(
        middle.x_px + 3,
        (middle.y_top_px + middle.y_bottom_px) / 2,
        "$W(x)$",
        va="center",
    )
    axes[5].set_title(f"F  Local width measurement, corridor {selected_id}")

    profile_ax = inset_axes(axes[5], width="82%", height="27%", loc="lower center", borderpad=0.9)
    profile_ax.plot(profile.x_um, profile.width_um, color="black", linewidth=0.9)
    profile_ax.set_xlabel("Position (µm)", fontsize=7)
    profile_ax.set_ylabel("W (µm)", fontsize=7)
    profile_ax.tick_params(labelsize=6)
    profile_ax.spines[["top", "right"]].set_visible(False)

    for ax in axes:
        ax.set_axis_off()

    fig.suptitle(
        f"Isolation and registration of horizontal corridors | {surface_name}",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.045, top=0.91, wspace=0.08, hspace=0.16)

    png_path = output_dir / "corridor_processing_schematic.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / "corridor_processing_schematic.pdf", bbox_inches="tight")
    plt.show()

    corridors.to_csv(output_dir / "corridor_bounding_boxes.csv", index=False)
    pd.concat(result["profiles"].values(), ignore_index=True).to_csv(
        output_dir / "corridor_width_profiles.csv", index=False
    )

    print(f"Detected corridors: {len(corridors)}")
    print(f"Selected corridor in panel F: {selected_id}")
    print(f"Binary threshold: {result['threshold']:.4f}")
    print(f"Saved figure: {png_path.resolve()}")


# =============================================================================
# RUN
# =============================================================================

result = process_corridors(
    image_or_path=INPUT_IMAGE,
    rotation_deg=ROTATION_DEG,
    pixel_size_um=PIXEL_SIZE_UM,
    coverage_threshold=COVERAGE_THRESHOLD,
    min_accessible_area=MIN_ACCESSIBLE_AREA,
    max_filled_hole_area=MAX_FILLED_HOLE_AREA,
    row_kernel_size=ROW_KERNEL_SIZE,
    min_band_height=MIN_BAND_HEIGHT,
    manual_threshold=MANUAL_THRESHOLD,
)

create_processing_schematic(
    result=result,
    surface_name=SURFACE_NAME,
    rotation_deg=ROTATION_DEG,
    pixel_size_um=PIXEL_SIZE_UM,
    coverage_threshold=COVERAGE_THRESHOLD,
    output_dir=OUTPUT_DIR,
    selected_corridor_id=SELECTED_CORRIDOR_ID,
    save_pdf=SAVE_PDF,
)
