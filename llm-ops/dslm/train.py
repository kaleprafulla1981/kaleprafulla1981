"""LoRA supervised fine-tuning. Loss is applied only to assistant tokens."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from dslm.config import resolve_path
from dslm.data import DataError, load_jsonl
from dslm.masking import mask_labels
from dslm.template import render_messages, sample_training_text


class EncodedDataset(Dataset):
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class PadCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown torch_dtype {name!r}") from exc


def load_tokenizer(model_name: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def encode_rows(tokenizer: Any, rows: list[list[dict[str, str]]], max_length: int) -> list[dict[str, list[int]]]:
    def encode(messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
        if hasattr(encoded, "input_ids"):
            return list(encoded.input_ids)
        return list(encoded)

    prepared: list[dict[str, list[int]]] = []
    skipped = 0
    for messages in rows:
        item = mask_labels(messages, encode, max_length)
        if item is None:
            skipped += 1
            continue
        prepared.append(item)
    if not prepared:
        raise DataError(
            "Every example was dropped. Raise data.max_length so the assistant answer fits."
        )
    if skipped:
        print(f"Skipped {skipped} example(s) with no supervised assistant tokens inside max_length.")
    return prepared


def _build_model(cfg: dict[str, Any]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_cfg = cfg["model"]
    dtype = _dtype(model_cfg["torch_dtype"])
    quant = None
    if model_cfg.get("load_in_4bit"):
        if not torch.cuda.is_available():
            raise RuntimeError("model.load_in_4bit needs a CUDA GPU and bitsandbytes. Use float32 on CPU.")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        dtype=dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        quantization_config=quant,
    )
    if quant is not None:
        model = prepare_model_for_kbit_training(model)
    lora_cfg = cfg["lora"]
    adapter = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, adapter)
    if cfg["train"].get("gradient_checkpointing"):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model


def train(cfg: dict[str, Any]) -> Path:
    from transformers import Trainer, TrainingArguments

    model_name = cfg["model"]["name"]
    train_rows = load_jsonl(resolve_path(cfg, cfg["data"]["train_file"]))
    eval_path = cfg["data"].get("eval_file")
    eval_rows = load_jsonl(resolve_path(cfg, eval_path)) if eval_path else []

    tokenizer = load_tokenizer(model_name, cfg["model"].get("trust_remote_code", False))
    max_length = int(cfg["data"]["max_length"])
    train_encoded = encode_rows(tokenizer, train_rows, max_length)
    eval_encoded = encode_rows(tokenizer, eval_rows, max_length) if eval_rows else []

    model = _build_model(cfg)
    train_cfg = cfg["train"]
    output_dir = resolve_path(cfg, train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    use_eval = bool(eval_encoded) and train_cfg.get("eval_strategy", "no") != "no"
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=int(train_cfg["max_steps"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        eval_strategy="epoch" if use_eval else "no",
        seed=int(train_cfg.get("seed", 7)),
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=torch.cuda.is_available() and _dtype(cfg["model"]["torch_dtype"]) == torch.bfloat16,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=EncodedDataset(train_encoded),
        eval_dataset=EncodedDataset(eval_encoded) if use_eval else None,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    result = trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    sample = sample_training_text(model_name, tokenizer)
    (output_dir / "chat_template_sample.txt").write_text(sample["rendered"], encoding="utf-8")

    losses = [entry.get("loss") for entry in trainer.state.log_history if "loss" in entry]
    manifest = {
        "format": "dslm.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "adapter_dir": "adapter",
        "train_examples": len(train_encoded),
        "eval_examples": len(eval_encoded),
        "max_length": max_length,
        "train_loss": float(result.training_loss),
        "logged_losses": [float(loss) for loss in losses],
        "lora": cfg["lora"],
        "query": cfg["query"],
        "chat_template_source": sample["source"],
        "chat_template_sample": sample["rendered"],
        "generation_stop": sample["generation_stop"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved adapter to {adapter_dir}")
    print(f"Train loss: {result.training_loss:.4f}")
    return output_dir


def validate(cfg: dict[str, Any]) -> dict[str, Any]:
    """Check the dataset and print one rendered training string. Does not load weights."""
    train_rows = load_jsonl(resolve_path(cfg, cfg["data"]["train_file"]))
    eval_file = cfg["data"].get("eval_file")
    eval_count = 0
    if eval_file:
        eval_path = resolve_path(cfg, eval_file)
        if eval_path.exists():
            eval_count = len(load_jsonl(eval_path))
    sample = sample_training_text(cfg["model"]["name"])
    summary = {
        "model_name": cfg["model"]["name"],
        "train_examples": len(train_rows),
        "eval_examples": eval_count,
        "template_source": sample["source"],
        "sample": sample["rendered"],
    }
    print(f"Model: {summary['model_name']}")
    print(f"Train rows: {summary['train_examples']}  Eval rows: {summary['eval_examples']}")
    print(f"Chat template source: {summary['template_source']}")
    print("--- rendered training example ---")
    print(summary["sample"].rstrip("\n"))
    print("--- end ---")
    return summary
