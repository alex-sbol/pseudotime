# Usage Guide

## 1. Overview
The morphology pipeline ingests two corridor environments exported from CellProfiler (or equivalent
TIFF stacks), aligns the corridors, segments cells, calculates confinement signals, maps cells to spatial
pseudotime, and reports morphology trends. An optional TopoChip flow-matching model can generate
synthetic shapes conditioned on corridor backgrounds.

## 2. Prerequisites
- Python 3.9
- Recommended: Windows PowerShell or a UNIX-like shell
- At least 8 GB RAM (more for full datasets or model inference)
- GPU is optional; CPU inference works for the pretrained model

## 3. Environment Setup
`powershell
# from the repository root
env python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
`
If you maintain multiple Python installations, replace env python with py -3.9 (Windows) or python3.9 (macOS/Linux).

## 4. Configuration File
Copy configs/default_config.yaml to a working file (e.g. configs/run.yaml). Key sections:
- output_dir: destination for tables, plots, generated assets
- channels: channel suffix mapping; adjust if file names differ
- environments: absolute or relative paths to the two corridor folders
- segmentation, signals, period, pseudotime, nalysis: analysis parameters
- indices: optional list of image indices (useful for quick dry runs)
- model: settings for the TopoChip generator (see §7)

All paths are resolved relative to the configuration file unless made absolute.

## 5. Running the Pipeline
`powershell
python -m morphology_pipeline --config configs/run.yaml
`
Optional flags:
- Restrict to a few images by setting indices: [12, 48, 103] inside the YAML
- Disable plot generation with isualization.enabled: false

Progress and timing information is printed to the console. Rerun the command after editing the config to regenerate outputs.

## 6. Outputs
The pipeline writes to output_dir:
- 	ables/ – CSV summaries (cell_features.csv, pseudotime_stats.csv, detected_regions.csv)
- igures/ – corridor signals and morphology trend plots (PNG by default)
- eports/ – JSON exports of detected regions, environment comparisons, period estimates
- generated/ – synthetic samples when the model is enabled (§7)

## 7. Enabling the TopoChip Generator (Optional)
1. Ensure 	orch, 	orchcfm, and dependencies are installed (already listed in equirements.txt).
2. Set the following in your config:
   `yaml
   model:
     enabled: true
     weights_path: ../topochip-flow-matching/model_weights/otcfm_topo_weights_step_145000.pt
     num_cond_channels: 2   # required for classifier-free guidance checkpoints
     use_cfg: true
     cfg_strength: 1.0
     samples_per_env: 4
     max_backgrounds: 32
   `
3. Run the pipeline; generated images appear under generated/ with metadata in eports/generated_samples.json.

CPU inference works but is slower; to force CPU, set model.device: cpu. For GPU, leave device unset (the code auto-detects CUDA).

## 8. Troubleshooting
- **No images found**: verify ile_pattern matches the TIFF naming convention.
- **Segmentation too aggressive or lenient**: adjust segmentation.cell_min_area, 
ucleus_min_area, or gaussian_sigma.
- **Period detection fails**: widen period.min_period_pixels/max_period_pixels or inspect igures/*signals.png.
- **Model load errors**: confirm 
um_cond_channels matches the checkpoint and the weights were produced with classifier-free guidance (use_cfg: true).
- **Windows NumPy warnings**: the project pins NumPy <2.0; if you upgrade, rebuild native wheels (torch, opencv) before running.

## 9. Further Customisation
- Extend feature extraction in morphology_pipeline/shapes.py.
- Add statistics or visualisations in morphology_pipeline/analysis.py and isualization.py.
- Integrate alternate generative models by implementing a new generator and updating model_integration.create_shape_generator.

