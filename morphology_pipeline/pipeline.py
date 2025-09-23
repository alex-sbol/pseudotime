"""Pipeline orchestration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from .align import CorridorAligner
from .analysis import compare_environments, compute_variability, detect_regions, records_to_dataframe
from .config import PipelineConfig
from .data_models import CellRecord, CorridorSignals
from .io import discover_indices, iter_bundles
from .model_integration import create_shape_generator
from .period import estimate_period
from .pseudotime import assign_pseudotime
from .segmentation import get_segmenter
from .shapes import extract_cell_records
from .signals import compute_signals
from .utils import ensure_dir
from .visualization import plot_comparison, plot_corridor_signals, plot_feature_trends


@dataclass
class EnvironmentResult:
    """Container for per-environment intermediate results."""
    signals: CorridorSignals
    records: List[CellRecord]
    backgrounds: List[np.ndarray]


class MorphologyPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.aligner = CorridorAligner()
        self.segmenter = get_segmenter(config)
        self.generator = create_shape_generator(config.model)

    def _process_environment(self, env_id: str, env_name: str, root: Path) -> EnvironmentResult:
        # Discover available indices on disk, optionally narrowed by config.
        indices = discover_indices(root, self.config)
        if self.config.indices is not None:
            indices = [idx for idx in indices if idx in self.config.indices]
        width_stack: List[np.ndarray] = []
        density_stack: List[np.ndarray] = []
        y_stack: List[np.ndarray] = []
        records: List[CellRecord] = []
        backgrounds: List[np.ndarray] = []
        collect_backgrounds = self.generator is not None and self.config.model.max_backgrounds > 0

        for bundle in iter_bundles(root, self.config, indices):
            # Align, segment, and extract features for each image sequentially.(root, self.config, indices):
            aligned = self.aligner.align(bundle)
            segmentation = self.segmenter.segment(aligned.channels)
            background = aligned.channels.get("background") or aligned.channels.get("actin")
            if background is None:
                raise ValueError("Generator requires a background or actin channel for conditioning")
            signals = compute_signals(
                background,
                segmentation,
                threshold=self.config.signals.corridor_threshold,
                density_window=self.config.signals.density_window,
            )
            width_stack.append(signals.corridor_width)
            density_stack.append(signals.cell_density)
            y_stack.append(signals.y_coordinates)
            env_records = extract_cell_records(env_id, bundle.index, segmentation, aligned.channels)
            records.extend(env_records)
            if collect_backgrounds and len(backgrounds) < self.config.model.max_backgrounds:
                backgrounds.append(np.asarray(background, dtype=np.float32))

        if not width_stack:
            raise RuntimeError(f"No images found for environment {env_name} at {root}")
        min_len = min(arr.size for arr in width_stack)
        widths = np.vstack([arr[:min_len] for arr in width_stack])
        densities = np.vstack([arr[:min_len] for arr in density_stack])
        y_coords = y_stack[0][:min_len]
        mean_width = widths.mean(axis=0)
        mean_density = densities.mean(axis=0)
        signals = CorridorSignals(corridor_width=mean_width, cell_density=mean_density, y_coordinates=y_coords)
        return EnvironmentResult(signals=signals, records=records, backgrounds=backgrounds)

    def run(self) -> Dict[str, EnvironmentResult]:
        ensure_dir(self.config.output_dir)
        tables_dir = self.config.output_dir / "tables"
        figures_dir = self.config.output_dir / "figures"
        reports_dir = self.config.output_dir / "reports"
        for directory in (tables_dir, figures_dir, reports_dir):
            ensure_dir(directory)

        env_results: Dict[str, EnvironmentResult] = {}  # Persist results for downstream aggregation.
        periods: Dict[str, float] = {}
        phase_offsets: Dict[str, float] = {}

        for env_id, env_cfg in self.config.environments.items():
            result = self._process_environment(env_id, env_cfg.name, env_cfg.root)
            env_results[env_id] = result
            period_estimate = estimate_period(result.signals, self.config.period)
            periods[env_id] = period_estimate.period
            phase_offsets[env_id] = period_estimate.phase_offset
            for record in result.records:
                record.pseudotime = (
                    (record.centroid_y - period_estimate.phase_offset) % period_estimate.period
                ) / period_estimate.period
            if self.config.visualization.enabled:
                plot_corridor_signals(result.signals, env_cfg.name, figures_dir)

        all_records: List[CellRecord] = [record for result in env_results.values() for record in result.records]
        df = records_to_dataframe(all_records)
        df.to_csv(tables_dir / "cell_features.csv", index=False)

        stats_df = compute_variability(df, self.config.analysis, self.config.pseudotime.bins)
        stats_df.to_csv(tables_dir / "pseudotime_stats.csv", index=False)

        regions_df = detect_regions(stats_df, self.config.analysis)
        regions_df.to_csv(tables_dir / "detected_regions.csv", index=False)
        with (reports_dir / "detected_regions.json").open("w", encoding="utf-8") as fh:
            json.dump(regions_df.to_dict(orient="records"), fh, indent=2)

        if self.config.visualization.enabled:
            for env_id, env_cfg in self.config.environments.items():
                env_stats = stats_df[stats_df["env_id"] == env_id]
                plot_feature_trends(env_stats, env_cfg.name, figures_dir)

        report = compare_environments(stats_df)
        with (reports_dir / "comparison.json").open("w", encoding="utf-8") as fh:
            json.dump({
                "feature_differences": {k: list(map(float, v)) for k, v in report.feature_differences.items()},
                "morphogenic_capacity": report.morphogenic_capacity,
            }, fh, indent=2)
        if self.config.visualization.enabled:
            plot_comparison(report, figures_dir)

        periods_out = {
            env_id: {"period": periods[env_id], "phase_offset": phase_offsets[env_id]}
            for env_id in periods
        }
        with (reports_dir / "periods.json").open("w", encoding="utf-8") as fh:
            json.dump(periods_out, fh, indent=2)

        if self.generator is not None:
            generated_dir = self.config.output_dir / "generated"
            ensure_dir(generated_dir)
            generated_summary: Dict[str, List[Dict[str, str]]] = {}
            for env_id, env_cfg in self.config.environments.items():
                result = env_results[env_id]
                if not result.backgrounds:
                    continue
                sample_count = min(len(result.backgrounds), self.config.model.samples_per_env)
                backgrounds = np.stack(result.backgrounds[:sample_count], axis=0)
                generated = self.generator.generate(env_id, backgrounds)
                env_summary: List[Dict[str, str]] = []
                for idx, sample in enumerate(generated):
                    rgb = np.clip((sample.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
                    output_path = generated_dir / (
                        f"{env_cfg.name.replace(' ', '_').lower()}_sample_{idx:02d}.{self.config.model.output_format}"
                    )
                    Image.fromarray(rgb).save(output_path)
                    env_summary.append(
                        {
                            "index": str(idx),
                            "file": str(output_path.relative_to(self.config.output_dir)),
                        }
                    )
                generated_summary[env_id] = env_summary
            if generated_summary:
                with (reports_dir / "generated_samples.json").open("w", encoding="utf-8") as fh:
                    json.dump(generated_summary, fh, indent=2)

        return env_results
