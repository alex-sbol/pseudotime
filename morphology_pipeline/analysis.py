"""Higher level analysis routines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import AnalysisConfig
from .data_models import ComparisonReport, CellRecord


def records_to_dataframe(records: List[CellRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        # Core metadata that every record contributes.
        row = {
            "env_id": record.env_id,
            "image_index": record.image_index,
            "cell_id": record.cell_id,
            "centroid_y": record.centroid_y,
            "pseudotime": record.pseudotime,
        }
        row.update(record.features)
        rows.append(row)
    return pd.DataFrame(rows)


def _dispersion(values: pd.Series, metric: str) -> float:
    if metric == "mad":
        return float(stats.median_abs_deviation(values, nan_policy="omit"))
    if metric == "iqr":
        return float(stats.iqr(values, nan_policy="omit"))
    return float(values.std())


def compute_variability(df: pd.DataFrame, config: AnalysisConfig, bins: int) -> pd.DataFrame:
    df = df.copy()
    df["bin"] = pd.cut(df["pseudotime"], bins=bins, labels=False, include_lowest=True)
    metrics: Dict[str, List[float]] = {}
    for column in df.columns:
        if column in {"env_id", "image_index", "cell_id", "centroid_y", "pseudotime", "bin"}:
            continue
        # Aggregate statistics per environment/pseudotime bin for each feature.
        grouped = df.groupby(["env_id", "bin"])[column]
        stats_df = grouped.agg([
            ("median", "median"),
            ("mean", "mean"),
            (config.variability_metric, lambda s: _dispersion(s, config.variability_metric)),
        ])
        stats_df = stats_df.reset_index()
        stats_df["feature"] = column
        metrics[column] = stats_df
    return pd.concat(metrics.values(), ignore_index=True)


def detect_regions(stats_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    results = []
    for env_id, group in stats_df.groupby("env_id"):
        for feature, feature_df in group.groupby("feature"):
            threshold = feature_df[config.variability_metric].quantile(config.detect_threshold_quantile)
            feature_df = feature_df.copy()
            feature_df["is_driver"] = feature_df[config.variability_metric] >= threshold
            feature_df["is_stable"] = feature_df[config.variability_metric] <= threshold / 2
            feature_df["feature"] = feature
            results.append(feature_df.assign(env_id=env_id))
    return pd.concat(results, ignore_index=True)


def compare_environments(stats_df: pd.DataFrame) -> ComparisonReport:
    # Pivot so we can subtract environment medians directly per feature/bin.
    pivot = stats_df.pivot_table(
        index=["bin", "feature"], columns="env_id", values="median"
    )
    environments = list(stats_df["env_id"].unique())
    if len(environments) != 2:
        raise ValueError("Comparison requires exactly two environments")
    env_a, env_b = environments
    diff = pivot[env_a] - pivot[env_b]
    diff_dict = {
        feature: diff.xs(feature, level="feature").to_numpy()
        for feature in pivot.index.get_level_values("feature").unique()
    }
    morphogenic_capacity = {
        feature: float(np.abs(values).sum())
        for feature, values in diff_dict.items()
    }
    return ComparisonReport(feature_differences=diff_dict, morphogenic_capacity=morphogenic_capacity)
