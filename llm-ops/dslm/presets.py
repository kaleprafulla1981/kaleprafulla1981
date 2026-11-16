"""Offline chat templates for the models this toolkit ships configs for.

The strings are the Jinja templates published with each tokenizer. Training and
query both prefer the template attached to the loaded tokenizer when it is
available, and fall back to these copies when the tokenizer has not been
downloaded yet.
"""

from __future__ import annotations

SELECTED_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Harborline Logistics is sample domain text so the template has a concrete row.
SAMPLE_MESSAGES: list[dict[str, str]] = [
    {
        "role": "system",
        "content": (
            "You are Harborline, the policy assistant for Harborline Logistics. "
            "Answer only from Harborline policy. If a rule is not in the policy, say so."
        ),
    },
    {
        "role": "user",
        "content": "What is the cutoff for same-day dispatch from the Pune hub?",
    },
    {
        "role": "assistant",
        "content": (
            "Same-day dispatch from the Pune hub closes at 14:30 IST. "
            "Orders released after 14:30 IST leave on the next business day."
        ),
    },
]

SMOLLM2_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\\n' }}"
    "{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# No-tools path of the Qwen2.5-Instruct template. Tool-call branches are omitted
# here; the live tokenizer template is used once the model is downloaded.
QWEN2_5_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\") %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)

LLAMA32_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{%- endif %}"
)

PRESETS: dict[str, dict[str, object]] = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": {
        "family": "smollm2",
        "chat_template": SMOLLM2_TEMPLATE,
        "special_tokens": {},
        "generation_stop": "<|im_end|>",
        "notes": (
            "ChatML. Each turn is <|im_start|>role\\ncontent<|im_end|>. "
            "Generation continues after <|im_start|>assistant\\n."
        ),
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "family": "qwen2.5",
        "chat_template": QWEN2_5_TEMPLATE,
        "special_tokens": {},
        "generation_stop": "<|im_end|>",
        "notes": (
            "ChatML, same visible shape as SmolLM2. If you omit a system message, "
            "Qwen inserts its own default system line. The tokenizer's full template "
            "also covers tool calls; this preset is the plain chat path."
        ),
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "family": "llama3",
        "chat_template": LLAMA32_TEMPLATE,
        "special_tokens": {"bos_token": "<|begin_of_text|>"},
        "generation_stop": "<|eot_id|>",
        "notes": (
            "Llama 3 header tags. A turn is <|start_header_id|>role<|end_header_id|>\\n\\n"
            "content<|eot_id|>, and the string starts with <|begin_of_text|>."
        ),
    },
}


def preset_for(model_name: str) -> dict[str, object] | None:
    return PRESETS.get(model_name)
