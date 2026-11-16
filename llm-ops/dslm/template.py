"""Render a model's chat template, offline from a preset or from a tokenizer."""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, StrictUndefined

from dslm.presets import SAMPLE_MESSAGES, preset_for

_ENV = Environment(undefined=StrictUndefined, autoescape=False)


def render_template(
    template: str,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    special_tokens: dict[str, Any] | None = None,
) -> str:
    compiled = _ENV.from_string(template)
    values: dict[str, Any] = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "tools": None,
    }
    if special_tokens:
        values.update(special_tokens)
    return compiled.render(**values)


def template_source(model_name: str, tokenizer: Any | None = None) -> tuple[str, str, dict[str, Any]]:
    """Return (jinja, source_label, special_tokens)."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        special: dict[str, Any] = {}
        bos = getattr(tokenizer, "bos_token", None)
        if isinstance(bos, str):
            special["bos_token"] = bos
        eos = getattr(tokenizer, "eos_token", None)
        if isinstance(eos, str):
            special["eos_token"] = eos
        return tokenizer.chat_template, "tokenizer", special
    preset = preset_for(model_name)
    if preset is None:
        known = ", ".join(sorted(preset_for(name) and name for name in _known()))
        raise KeyError(
            f"No chat template is bundled for {model_name}. "
            f"Download the tokenizer, or use one of: {known}"
        )
    return str(preset["chat_template"]), "preset", dict(preset["special_tokens"])  # type: ignore[arg-type]


def _known() -> list[str]:
    from dslm.presets import PRESETS

    return list(PRESETS)


def render_messages(
    model_name: str,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    tokenizer: Any | None = None,
) -> tuple[str, str]:
    template, source, special = template_source(model_name, tokenizer)
    text = render_template(
        template,
        messages,
        add_generation_prompt=add_generation_prompt,
        special_tokens=special,
    )
    return text, source


def sample_training_text(model_name: str, tokenizer: Any | None = None) -> dict[str, Any]:
    """The filled chat string for one training row, plus the generation prefix."""
    full, source = render_messages(
        model_name, SAMPLE_MESSAGES, add_generation_prompt=False, tokenizer=tokenizer
    )
    prompt, _ = render_messages(
        model_name,
        SAMPLE_MESSAGES[:-1],
        add_generation_prompt=True,
        tokenizer=tokenizer,
    )
    preset = preset_for(model_name) or {}
    return {
        "model_name": model_name,
        "source": source,
        "messages": SAMPLE_MESSAGES,
        "rendered": full,
        "generation_prefix": prompt,
        "notes": preset.get("notes", ""),
        "generation_stop": preset.get("generation_stop", ""),
    }
