"""Command-line interface for the morphology pipeline."""
from __future__ import annotations

from pathlib import Path

import typer

from .config import load_config
from .pipeline import MorphologyPipeline

app = typer.Typer(add_completion=False, help="Run the modular morphology analysis pipeline.")


@app.command()
def run(config: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to YAML configuration.")) -> None:
    """Execute the pipeline end to end."""
    cfg = load_config(config)
    pipeline = MorphologyPipeline(cfg)
    pipeline.run()
    typer.echo("Analysis complete. Results saved to the configured output directory.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
