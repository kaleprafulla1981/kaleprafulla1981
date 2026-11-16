"""YAML config loading. Paths inside a config are relative to the repo root."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "model": {
        "name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "torch_dtype": "float32",
        "trust_remote_code": False,
        "load_in_4bit": False,
    },
    "data": {
        "train_file": "data/sample/train.jsonl",
        "eval_file": "data/sample/eval.jsonl",
        "max_length": 512,
    },
    "lora": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": "all-linear",
    },
    "train": {
        "output_dir": "runs/smollm2-harborline",
        "num_train_epochs": 1,
        "max_steps": -1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2.0e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "gradient_checkpointing": False,
        "seed": 7,
    },
    "query": {
        "max_new_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
        "system_prompt": (
            "You are Harborline, the policy assistant for Harborline Logistics. "
            "Answer only from Harborline policy. If a rule is not in the policy, say so."
        ),
    },
}


def project_root(config_path: Path) -> Path:
    path = config_path.resolve()
    if path.parent.name == "configs":
        return path.parent.parent
    return path.parent


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    cfg = _merge(DEFAULTS, raw)
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(project_root(config_path))
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(cfg["_root"]) / path
