"""Convenience launcher for the pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from morphology_pipeline.config import load_config
from morphology_pipeline.pipeline import MorphologyPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the morphology pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = MorphologyPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
