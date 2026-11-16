"""Load a saved adapter, or the base model, and answer a chat."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from dslm.config import resolve_path
from dslm.template import render_messages
from dslm.train import load_tokenizer


class ModelBundle:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        run_dir: Path | None,
        using_adapter: bool,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.run_dir = run_dir
        self.using_adapter = using_adapter
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop

    def render(self, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
        text, _source = render_messages(
            self.model_name,
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenizer=self.tokenizer,
        )
        return text

    def prepare_messages(
        self,
        messages: list[dict[str, str]] | None = None,
        question: str | None = None,
        system: str | None = None,
    ) -> list[dict[str, str]]:
        if messages:
            prepared = [{"role": item["role"], "content": item["content"]} for item in messages]
        elif question:
            prepared = [{"role": "user", "content": question}]
        else:
            raise ValueError("Pass messages or a question.")
        if prepared[0]["role"] != "system":
            prepared.insert(0, {"role": "system", "content": system or self.system_prompt})
        if prepared[-1]["role"] != "user":
            raise ValueError("The last message must be a user turn.")
        return prepared

    @torch.inference_mode()
    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        prompt = self.render(messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
        output = self.model.generate(**inputs, **generate_kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        if self.stop and self.stop in answer:
            answer = answer.split(self.stop, 1)[0]
        answer = answer.replace("<|im_end|>", "").replace("<|eot_id|>", "").strip()
        return {"prompt": prompt, "answer": answer, "messages": messages}


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {run_dir}. Train a run first.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_bundle(
    cfg: dict[str, Any] | None = None,
    run_dir: str | Path | None = None,
) -> ModelBundle:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    run_path = Path(run_dir).resolve() if run_dir else None
    manifest: dict[str, Any] = {}
    if run_path is not None:
        manifest = load_manifest(run_path)

    if cfg is None and not manifest:
        raise ValueError("Pass a config or a trained run directory.")

    model_name = manifest.get("model_name") or cfg["model"]["name"]  # type: ignore[index]
    query_cfg = (cfg or {}).get("query") or manifest.get("query") or {}
    trust = bool((cfg or {}).get("model", {}).get("trust_remote_code", False))
    dtype_name = (cfg or {}).get("model", {}).get("torch_dtype", "float32")
    from dslm.train import _dtype

    dtype = _dtype(dtype_name)
    adapter_dir = None
    if run_path is not None:
        candidate = run_path / manifest.get("adapter_dir", "adapter")
        if (candidate / "adapter_config.json").exists():
            adapter_dir = candidate

    tokenizer_source = adapter_dir if adapter_dir is not None else model_name
    tokenizer = load_tokenizer(str(tokenizer_source), trust)
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=trust)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base, adapter_dir)
        using_adapter = True
    else:
        model = base
        using_adapter = False
    model.eval()
    stop = manifest.get("generation_stop") or "<|im_end|>"
    if "Llama-3" in model_name or "llama-3" in model_name.lower():
        stop = "<|eot_id|>"
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        run_dir=run_path,
        using_adapter=using_adapter,
        system_prompt=query_cfg.get(
            "system_prompt",
            "You are a careful domain assistant. Answer only from the material you were trained on.",
        ),
        max_new_tokens=int(query_cfg.get("max_new_tokens", 128)),
        temperature=float(query_cfg.get("temperature", 0.2)),
        top_p=float(query_cfg.get("top_p", 0.9)),
        stop=str(stop),
    )


def resolve_run(cfg: dict[str, Any], run: str | None) -> Path | None:
    if run:
        return Path(run).resolve()
    output = cfg["train"]["output_dir"]
    path = resolve_path(cfg, output)
    if (path / "manifest.json").exists():
        return path
    return None
