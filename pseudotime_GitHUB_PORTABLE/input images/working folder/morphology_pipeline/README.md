# Modular Cell Morphology Pipeline (Method A)

This project processes corridor microscopy datasets exported from CellProfiler or raw TIFF stacks,
computes confinement signals, assigns spatial pseudotime, extracts detailed cell morphology
measurements, and compares two corridor environments. The default configuration targets:

- Environment 1: Chip_1462_1_real
- Environment 2: Chip_1476_1_real

All functionality is exposed through a single command-line interface designed for biologists. No
Python scripting is required.

## Quick Start

1. **Activate Python 3.9**: %.venv\Scripts\activate%
2. **Install dependencies**: pip install -r requirements.txt
3. **Prepare configuration**: copy configs/default_config.yaml and edit paths if needed.
4. **Run the pipeline**:
   `ash
   python -m morphology_pipeline --config configs/default_config.yaml
   `
5. Results (CSV tables, figures, JSON reports) appear in the esults/ directory defined in the config.

See docs/USAGE.md (to be completed) for detailed explanations of outputs.

## Repository Layout

- morphology_pipeline/ – Python package with modular processing stages
- configs/ – YAML configuration templates for environment paths and parameters
- 	ests/ – pytest-based unit and integration checks
- scripts/ – helper entry points (CLI wrappers, utilities)
- docs/ – supplementary documentation and diagrams (optional)

## Support

Create an issue or contact the pipeline maintainer with dataset details if you need help.
