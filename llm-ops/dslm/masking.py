"""Mask every token that is not part of an assistant answer.

`encode(messages, add_generation_prompt)` must be prefix-stable for ChatML and
Llama 3 templates: encoding a longer conversation starts with the encoding of
the shorter one. When a tokenizer is not prefix-stable, labels fall back to the
final assistant turn only.
"""

from __future__ import annotations

from collections.abc import Callable

EncodeFn = Callable[[list[dict[str, str]], bool], list[int]]


def mask_labels(
    messages: list[dict[str, str]],
    encode: EncodeFn,
    max_length: int,
) -> dict[str, list[int]] | None:
    full_ids = encode(messages, False)
    labels = [-100] * len(full_ids)
    supervised = 0

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt_ids = encode(messages[:index], True)
        turn_ids = encode(messages[: index + 1], False)
        if full_ids[: len(turn_ids)] != turn_ids:
            continue
        start = len(prompt_ids)
        end = len(turn_ids)
        if start >= end:
            continue
        for position in range(start, end):
            labels[position] = full_ids[position]
            supervised += 1

    if supervised == 0 and messages and messages[-1]["role"] == "assistant":
        prompt_ids = encode(messages[:-1], True)
        if full_ids[: len(prompt_ids)] == prompt_ids and len(prompt_ids) < len(full_ids):
            for position in range(len(prompt_ids), len(full_ids)):
                labels[position] = full_ids[position]
                supervised += 1

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
        supervised = sum(1 for label in labels if label != -100)

    if supervised == 0:
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
