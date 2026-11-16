from pathlib import Path

from dslm.config import load_config, resolve_path

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_selects_smollm2():
    cfg = load_config(ROOT / "configs/smollm2-135m-instruct.yaml")
    assert cfg["model"]["name"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert resolve_path(cfg, cfg["data"]["train_file"]).exists()
    assert "Harborline" in cfg["query"]["system_prompt"]
