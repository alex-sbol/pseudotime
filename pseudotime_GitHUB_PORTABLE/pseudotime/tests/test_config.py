import pathlib

from morphology_pipeline.config import load_config


def test_load_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output_dir: ./out
channels:
  actin: "Actin"
  nucleus: "DAPI"
  background: null
  auxiliary: {}
file_pattern: "obj{index:04d}_stain{channel}_real.tiff"
environments:
  env1:
    name: "Env1"
    root: ./env1
  env2:
    name: "Env2"
    root: ./env2
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert "env1" in config.environments
    assert config.channels.nucleus == "DAPI"
