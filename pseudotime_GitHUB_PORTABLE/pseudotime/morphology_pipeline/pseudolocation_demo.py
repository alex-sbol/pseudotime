from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import FancyArrowPatch, Polygon
from skimage import transform

from morphology_pipeline.corridor_mask import corridors_from_black_hull, detect_corridors_via_hull
from morphology_pipeline.pseudotime import segment_outer_envelope


PASTEL_UNIT_COLORS = (
    (0.83, 0.91, 0.98, 0.35),
    (0.97, 0.88, 0.89, 0.35),
    (0.88, 0.95, 0.88, 0.35),
)
NUCLEUS_RGB = np.array((0.12, 0.28, 0.95), dtype=float)


@dataclass
class DemoConfig:
    csv_path: Path
    output_dir: Path
    background_path: Path | None = None
    rotation_deg: float = -13.47
    period_bins: int | None = None
    bin_stride_px: float | None = None
    steps_per_unit: int = 3
    n_units: int = 3
    corridor_padding_px: int = 18
    dapi_threshold: float = 0.30
    dapi_crop_pad_px: int = 4
    nucleus_scale: float = 0.55
    surface_label: str = ""


def parse_vec(value, dtype=float) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype)
    if isinstance(value, list):
        return np.asarray(value, dtype=dtype)
    if pd.isna(value):
        return np.array([], dtype=dtype)

    text = str(value).replace("[", " ").replace("]", " ").replace(",", " ").strip()
    if not text:
        return np.array([], dtype=dtype)

    return np.asarray([part for part in text.split() if part], dtype=dtype)


def resolve_existing_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.exists():
        return path

    fallback = Path(path.name)
    if fallback.exists():
        return fallback

    return path


def derive_dapi_path(background_path: str | Path) -> Path:
    background_path = resolve_existing_path(background_path)

    name = background_path.name
    candidates = [
        name.replace("stainbackground", "staindapi"),
        name.replace("background", "dapi"),
        name.replace("Background", "DAPI"),
    ]

    for candidate in candidates:
        if candidate == name:
            continue
        candidate_path = background_path.with_name(candidate)
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(f"Could not derive DAPI path from background path: {background_path}")


def build_unique_key(row: pd.Series) -> str:
    image_number = pd.to_numeric(row.get("dapi_ImageNumber"), errors="coerce")
    object_number = pd.to_numeric(row.get("ObjectNumber"), errors="coerce")

    if pd.notna(image_number) and pd.notna(object_number):
        return f"img{int(image_number):05d}_obj{int(object_number):05d}"

    return Path(str(row["path_background"])).stem


def load_demo_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if "path_background" not in df.columns:
        raise KeyError(f"{csv_path} does not contain a 'path_background' column.")

    if "path_dapi" in df.columns:
        df["path_dapi"] = df["path_dapi"].map(resolve_existing_path)
    else:
        df["path_dapi"] = df["path_background"].map(derive_dapi_path)

    df["path_background"] = df["path_background"].map(resolve_existing_path)
    df["center_bin"] = pd.to_numeric(df["pseudotime_center_bin"], errors="coerce")
    df = df.loc[df["center_bin"].notna()].copy()
    df["center_bin"] = df["center_bin"].astype(int)
    df["area"] = pd.to_numeric(df.get("dapi_AreaShape_Area"), errors="coerce")
    df["pseudotime_arr"] = df.get("pseudotime", pd.Series(index=df.index, dtype=object)).apply(
        lambda value: parse_vec(value, dtype=int)
    )
    df["pseudotime_widths_arr"] = df.get(
        "pseudotime_widths", pd.Series(index=df.index, dtype=object)
    ).apply(lambda value: parse_vec(value, dtype=float))
    df["unique_key"] = df.apply(build_unique_key, axis=1)
    return df


def build_target_bins(period_bins: int, steps_per_unit: int) -> np.ndarray:
    if period_bins <= 0:
        raise ValueError("period_bins must be positive.")
    if steps_per_unit <= 0:
        raise ValueError("steps_per_unit must be positive.")
    if steps_per_unit == 1:
        return np.array([period_bins // 2], dtype=int)

    return np.unique(np.round(np.linspace(0, period_bins - 1, steps_per_unit)).astype(int))


def rank_cells_for_targets(
    df: pd.DataFrame,
    target_bins: Iterable[int],
    n_units: int,
) -> pd.DataFrame:
    target_bins = list(int(value) for value in target_bins)
    if len(df) < len(target_bins) * n_units:
        raise ValueError("Not enough nuclei available to build a unique directional sequence.")

    selections: list[dict] = []
    used_indices: set[int] = set()
    previous_area = np.nan
    rank = 1

    for unit_idx in range(n_units):
        for target_bin in target_bins:
            available = df.loc[~df.index.isin(used_indices)].copy()
            available["bin_error"] = (available["center_bin"] - target_bin).abs()

            if pd.isna(previous_area) or available["area"].isna().all():
                available["area_error"] = 0.0
            else:
                area_fallback = float(available["area"].median()) if available["area"].notna().any() else 0.0
                available["area_error"] = (available["area"] - previous_area).abs().fillna(area_fallback)

            chosen = available.sort_values(
                ["bin_error", "area_error", "center_bin", "unique_key"],
                ascending=[True, True, True, True],
            ).iloc[0]

            selections.append(
                {
                    "rank": rank,
                    "unit_idx": unit_idx,
                    "target_bin": int(target_bin),
                    "row_idx": int(chosen.name),
                    "unique_key": chosen["unique_key"],
                    "center_bin": int(chosen["center_bin"]),
                    "bin_error": int(chosen["bin_error"]),
                    "area": float(chosen["area"]) if pd.notna(chosen["area"]) else np.nan,
                    "path_background": str(chosen["path_background"]),
                    "path_dapi": str(chosen["path_dapi"]),
                }
            )
            used_indices.add(int(chosen.name))
            if pd.notna(chosen["area"]):
                previous_area = float(chosen["area"])
            rank += 1

    return pd.DataFrame(selections)


def to_2d(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    img = np.squeeze(img)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[0] <= 4:
            return img[0]
        return img[..., 0]
    raise ValueError(f"Unsupported image shape: {img.shape}")


def read_image(path_like: str | Path) -> np.ndarray:
    return to_2d(tifffile.imread(str(resolve_existing_path(path_like))))


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    arr = arr - arr.min()
    max_value = float(arr.max())
    if max_value <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip(arr / max_value * 255.0, 0, 255).astype(np.uint8)


def extract_dapi_rgba(
    path_like: str | Path,
    threshold: float = 0.30,
    crop_pad_px: int = 4,
) -> np.ndarray:
    dapi = read_image(path_like).astype(np.float32)
    dapi = dapi / max(float(dapi.max()), 1.0)

    mask = dapi > max(threshold, float(dapi.max()) * threshold)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        crop = dapi
    else:
        y0 = max(int(ys.min()) - crop_pad_px, 0)
        y1 = min(int(ys.max()) + crop_pad_px + 1, dapi.shape[0])
        x0 = max(int(xs.min()) - crop_pad_px, 0)
        x1 = min(int(xs.max()) + crop_pad_px + 1, dapi.shape[1])
        crop = dapi[y0:y1, x0:x1]

    alpha = np.clip((crop - threshold * 0.25) / max(1.0 - threshold * 0.25, 1e-6), 0, 1)
    rgba = np.zeros((*crop.shape, 4), dtype=float)
    rgba[..., :3] = NUCLEUS_RGB
    rgba[..., 3] = np.power(alpha, 0.85) * 0.92
    return rgba


def choose_reference_background(
    df: pd.DataFrame,
    period_bins: int,
    background_path: Path | None = None,
) -> Path:
    if background_path is not None:
        return resolve_existing_path(background_path)

    target_center = period_bins // 2
    ranked = df.assign(center_distance=(df["center_bin"] - target_center).abs())
    ranked = ranked.sort_values(["center_distance", "area"], ascending=[True, False])
    return resolve_existing_path(ranked.iloc[0]["path_background"])


def extract_corridor_envelope(mask: np.ndarray, bbox: dict, x_values: np.ndarray) -> pd.DataFrame:
    known_x = []
    y_top = []
    y_bottom = []

    for x in range(int(bbox["x0"]), int(bbox["x1"]) + 1):
        seg = segment_outer_envelope(mask, x, int(bbox["y0"]), int(bbox["y1"]))
        if seg is None:
            continue
        known_x.append(seg[0])
        y_top.append(seg[1])
        y_bottom.append(seg[2])

    if len(known_x) < 2:
        raise RuntimeError("Could not extract a continuous corridor envelope from the rotated mask.")

    x_values = np.asarray(x_values, dtype=float)
    return pd.DataFrame(
        {
            "x": x_values,
            "y_top": np.interp(x_values, known_x, y_top),
            "y_bottom": np.interp(x_values, known_x, y_bottom),
        }
    )


def prepare_corridor_context(
    background_path: Path,
    period_bins: int,
    rotation_deg: float,
    bin_stride_px: float | None,
    n_units: int,
    corridor_padding_px: int,
) -> dict:
    background = normalize_to_uint8(read_image(background_path))
    corridor_mask, hull_mask = corridors_from_black_hull(background, background)

    rot_bg = transform.rotate(background, rotation_deg, order=1, preserve_range=True, resize=True).astype(np.uint8)
    rot_corr = transform.rotate(corridor_mask, rotation_deg, order=0, preserve_range=True, resize=True).astype(
        np.uint8
    )
    rot_hull = transform.rotate(hull_mask, rotation_deg, order=0, preserve_range=True, resize=True).astype(np.uint8)
    rot_bg[rot_hull == 0] = 255

    corridors = detect_corridors_via_hull(
        rot_corr,
        rot_hull,
        row_cov_thresh_rel_hull=0.07,
        min_band_height=5,
        merge_gap_px=3,
    )
    if not corridors:
        raise RuntimeError("No corridor envelope was detected in the reference background.")

    corridor_bbox = sorted(corridors, key=lambda item: 0.5 * (item["y0"] + item["y1"]))[len(corridors) // 2]
    corridor_width = float(corridor_bbox["x1"] - corridor_bbox["x0"])
    max_unit_width = corridor_width / max(n_units, 1)
    if bin_stride_px is None:
        unit_width_px = max_unit_width
    else:
        desired_unit_width = float(period_bins) * float(bin_stride_px)
        unit_width_px = min(desired_unit_width, max_unit_width)
    span_width_px = unit_width_px * float(n_units)

    center_x = 0.5 * (corridor_bbox["x0"] + corridor_bbox["x1"])
    min_start = float(corridor_bbox["x0"])
    max_start = float(corridor_bbox["x1"]) - span_width_px
    x_start = float(np.clip(center_x - span_width_px / 2.0, min_start, max_start))
    x_end = x_start + span_width_px

    x_values = np.arange(int(np.floor(x_start)), int(np.ceil(x_end)) + 1)
    envelope = extract_corridor_envelope(rot_corr, corridor_bbox, x_values)

    crop_x0 = max(0, int(np.floor(x_start - corridor_padding_px)))
    crop_x1 = min(rot_bg.shape[1], int(np.ceil(x_end + corridor_padding_px)))
    crop_y0 = max(0, int(np.floor(envelope["y_top"].min() - corridor_padding_px)))
    crop_y1 = min(rot_bg.shape[0], int(np.ceil(envelope["y_bottom"].max() + corridor_padding_px)))

    envelope["x_local"] = envelope["x"] - crop_x0
    envelope["y_top_local"] = envelope["y_top"] - crop_y0
    envelope["y_bottom_local"] = envelope["y_bottom"] - crop_y0

    return {
        "background_crop": rot_bg[crop_y0:crop_y1, crop_x0:crop_x1],
        "x_start_local": x_start - crop_x0,
        "unit_width_px": unit_width_px,
        "period_bins": period_bins,
        "envelope": envelope,
        "rotation_deg": rotation_deg,
    }


def build_unit_polygons(context: dict, n_units: int) -> list[np.ndarray]:
    envelope = context["envelope"]
    x_start_local = float(context["x_start_local"])
    unit_width_px = float(context["unit_width_px"])
    polygons = []

    for unit_idx in range(n_units):
        x0 = x_start_local + unit_idx * unit_width_px
        x1 = x0 + unit_width_px
        mask = (envelope["x_local"] >= x0) & (envelope["x_local"] <= x1)
        unit = envelope.loc[mask]
        x_poly = np.concatenate([unit["x_local"].to_numpy(), unit["x_local"].to_numpy()[::-1]])
        y_poly = np.concatenate([unit["y_top_local"].to_numpy(), unit["y_bottom_local"].to_numpy()[::-1]])
        polygons.append(np.column_stack([x_poly, y_poly]))

    return polygons


def add_render_positions(selection_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    selection_df = selection_df.copy()
    x_start_local = float(context["x_start_local"])
    unit_width_px = float(context["unit_width_px"])
    period_bins = float(context["period_bins"])
    envelope = context["envelope"]
    center_line = 0.5 * (envelope["y_top_local"].to_numpy() + envelope["y_bottom_local"].to_numpy())
    inner_margin_px = unit_width_px * 0.12

    x_positions = []
    y_positions = []
    for row in selection_df.itertuples(index=False):
        unit_x0 = x_start_local + row.unit_idx * unit_width_px
        if period_bins <= 1:
            x_pos = unit_x0 + unit_width_px / 2.0
        else:
            frac = float(row.target_bin) / max(period_bins - 1.0, 1.0)
            x_pos = unit_x0 + inner_margin_px + frac * max(unit_width_px - 2.0 * inner_margin_px, 1.0)
        y_pos = float(np.interp(x_pos, envelope["x_local"].to_numpy(), center_line))
        x_positions.append(x_pos)
        y_positions.append(y_pos)

    selection_df["x_local"] = x_positions
    selection_df["y_local"] = y_positions
    return selection_df


def render_demo_canvas(
    selection_df: pd.DataFrame,
    context: dict,
    output_path: Path,
    dapi_threshold: float = 0.30,
    dapi_crop_pad_px: int = 4,
    nucleus_scale: float = 0.55,
    show_order_labels: bool = True,
) -> None:
    background = context["background_crop"]
    polygons = build_unit_polygons(context, n_units=int(selection_df["unit_idx"].max()) + 1)
    span_width = float(context["unit_width_px"]) * len(polygons)
    x_start_local = float(context["x_start_local"])
    envelope = context["envelope"]

    fig_width = max(7.0, background.shape[1] / 28.0)
    fig_height = max(4.5, background.shape[0] / 28.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(background, cmap="gray", vmin=0, vmax=255)

    for unit_idx, polygon in enumerate(polygons):
        color = PASTEL_UNIT_COLORS[unit_idx % len(PASTEL_UNIT_COLORS)]
        ax.add_patch(Polygon(polygon, closed=True, facecolor=color, edgecolor=(1, 1, 1, 0.55), linewidth=1.0))

    for unit_idx in range(1, len(polygons)):
        divider_x = x_start_local + unit_idx * float(context["unit_width_px"])
        ax.axvline(divider_x, color=(1, 1, 1, 0.60), linewidth=0.9)

    arrow_y = max(8.0, float(envelope["y_top_local"].min()) - 10.0)
    ax.add_patch(
        FancyArrowPatch(
            (x_start_local, arrow_y),
            (x_start_local + span_width, arrow_y),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.5,
            color=(0.18, 0.18, 0.18, 0.95),
        )
    )
    ax.text(
        x_start_local + span_width / 2.0,
        arrow_y - 3.0,
        "Pseudolocation rank",
        ha="center",
        va="bottom",
        fontsize=9,
        color=(0.18, 0.18, 0.18, 0.95),
    )

    max_nucleus_span = float(context["unit_width_px"]) * nucleus_scale
    for row in selection_df.itertuples(index=False):
        patch = extract_dapi_rgba(
            row.path_dapi,
            threshold=dapi_threshold,
            crop_pad_px=dapi_crop_pad_px,
        )
        patch_h, patch_w = patch.shape[:2]
        scale = min(1.0, max_nucleus_span / max(patch_h, patch_w))
        render_h = patch_h * scale
        render_w = patch_w * scale
        ax.imshow(
            patch,
            extent=(
                row.x_local - render_w / 2.0,
                row.x_local + render_w / 2.0,
                row.y_local + render_h / 2.0,
                row.y_local - render_h / 2.0,
            ),
            interpolation="bilinear",
            zorder=4,
        )
        if show_order_labels:
            ax.text(
                row.x_local,
                row.y_local - render_h / 2.0 - 2.0,
                str(row.rank),
                ha="center",
                va="bottom",
                fontsize=8,
                color=(0.05, 0.10, 0.45, 0.95),
                fontweight="bold",
                zorder=5,
            )

    ax.set_xlim(0, background.shape[1])
    ax.set_ylim(background.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_contact_sheet(
    selection_df: pd.DataFrame,
    output_path: Path,
    dapi_threshold: float = 0.30,
    dapi_crop_pad_px: int = 4,
) -> None:
    n_cols = len(selection_df)
    fig, axes = plt.subplots(1, n_cols, figsize=(1.6 * n_cols, 2.6))
    axes = np.atleast_1d(axes)

    for ax, row in zip(axes, selection_df.itertuples(index=False)):
        ax.imshow(np.ones((24, 24, 3), dtype=float))
        ax.imshow(
            extract_dapi_rgba(row.path_dapi, threshold=dapi_threshold, crop_pad_px=dapi_crop_pad_px),
            interpolation="bilinear",
        )
        ax.set_title(f"#{row.rank}\nbin {row.center_bin}", fontsize=8)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def run_pseudolocation_demo(config: DemoConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_demo_table(config.csv_path)
    period_bins = config.period_bins or int(df["center_bin"].max() + 1)
    target_bins = build_target_bins(period_bins, config.steps_per_unit)
    selection_df = rank_cells_for_targets(df, target_bins=target_bins, n_units=config.n_units)

    reference_background = choose_reference_background(
        df,
        period_bins=period_bins,
        background_path=config.background_path,
    )
    context = prepare_corridor_context(
        background_path=reference_background,
        period_bins=period_bins,
        rotation_deg=config.rotation_deg,
        bin_stride_px=config.bin_stride_px,
        n_units=config.n_units,
        corridor_padding_px=config.corridor_padding_px,
    )
    selection_df = add_render_positions(selection_df, context)

    stem = config.surface_label or config.csv_path.stem
    canvas_path = config.output_dir / f"{stem}_pseudolocation_demo.png"
    contact_path = config.output_dir / f"{stem}_pseudolocation_contact_sheet.png"
    metadata_path = config.output_dir / f"{stem}_pseudolocation_selection.csv"

    render_demo_canvas(
        selection_df,
        context,
        canvas_path,
        dapi_threshold=config.dapi_threshold,
        dapi_crop_pad_px=config.dapi_crop_pad_px,
        nucleus_scale=config.nucleus_scale,
    )
    render_contact_sheet(
        selection_df,
        contact_path,
        dapi_threshold=config.dapi_threshold,
        dapi_crop_pad_px=config.dapi_crop_pad_px,
    )

    selection_df = selection_df.copy()
    selection_df["reference_background_path"] = str(reference_background)
    selection_df["rotation_deg"] = float(config.rotation_deg)
    selection_df.to_csv(metadata_path, index=False)

    return {
        "canvas_path": canvas_path,
        "contact_path": contact_path,
        "metadata_path": metadata_path,
        "reference_background_path": reference_background,
        "period_bins": period_bins,
        "target_bins": target_bins.tolist(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a static pseudolocation demo with a 3-unit pastel pseudocorridor."
    )
    parser.add_argument("--csv", dest="csv_path", type=Path, required=True, help="Input metadata CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the rendered outputs.")
    parser.add_argument(
        "--background-path",
        type=Path,
        default=None,
        help="Optional background TIFF to use as the static canvas.",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=-13.47,
        help="Manual corridor rotation angle used to align the corridor direction left-to-right.",
    )
    parser.add_argument("--period-bins", type=int, default=None, help="Override the inferred pseudotime period.")
    parser.add_argument(
        "--bin-stride-px",
        type=float,
        default=None,
        help="Optional pixel stride per pseudotime bin. Leave unset to span the full 3-unit corridor width.",
    )
    parser.add_argument(
        "--steps-per-unit",
        type=int,
        default=3,
        help="How many ranked nuclei to place in each confinement unit.",
    )
    parser.add_argument("--units", dest="n_units", type=int, default=3, help="Number of confinement units to show.")
    parser.add_argument(
        "--surface-label",
        default="",
        help="Optional label used in the output filenames.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run_pseudolocation_demo(
        DemoConfig(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            background_path=args.background_path,
            rotation_deg=args.rotation_deg,
            period_bins=args.period_bins,
            bin_stride_px=args.bin_stride_px,
            steps_per_unit=args.steps_per_unit,
            n_units=args.n_units,
            surface_label=args.surface_label,
        )
    )

    print(f"Saved demo canvas to: {outputs['canvas_path']}")
    print(f"Saved contact sheet to: {outputs['contact_path']}")
    print(f"Saved selection metadata to: {outputs['metadata_path']}")
    print(f"Reference background: {outputs['reference_background_path']}")
    print(f"Period bins: {outputs['period_bins']}")
    print(f"Target bins: {outputs['target_bins']}")


if __name__ == "__main__":
    main()
