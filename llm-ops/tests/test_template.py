from dslm.presets import PRESETS, SAMPLE_MESSAGES, SELECTED_MODEL
from dslm.template import render_messages, sample_training_text


def test_selected_model_is_smollm2():
    assert SELECTED_MODEL == "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_smollm2_sample_is_chatml():
    sample = sample_training_text(SELECTED_MODEL)
    text = sample["rendered"]
    assert text.startswith("<|im_start|>system\n")
    assert "<|im_start|>user\nWhat is the cutoff" in text
    assert text.rstrip("\n").endswith("<|im_end|>")
    assert "14:30 IST" in text
    prefix = sample["generation_prefix"]
    assert prefix.endswith("<|im_start|>assistant\n")
    assert "14:30 IST" not in prefix


def test_qwen_inserts_default_system_when_missing():
    text, source = render_messages(
        "Qwen/Qwen2.5-0.5B-Instruct",
        [{"role": "user", "content": "Hello"}],
        add_generation_prompt=True,
    )
    assert source == "preset"
    assert text.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.")
    assert text.endswith("<|im_start|>assistant\n")


def test_qwen_keeps_caller_system_once():
    text, _ = render_messages(
        "Qwen/Qwen2.5-0.5B-Instruct",
        SAMPLE_MESSAGES,
        add_generation_prompt=False,
    )
    assert text.count("<|im_start|>system\n") == 1
    assert "You are Harborline" in text


def test_llama32_headers():
    text, _ = render_messages(
        "meta-llama/Llama-3.2-1B-Instruct",
        SAMPLE_MESSAGES[:2],
        add_generation_prompt=True,
    )
    assert text.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
    assert "<|start_header_id|>user<|end_header_id|>" in text
    assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert "<|eot_id|>" in text


def test_every_preset_renders():
    for name in PRESETS:
        rendered, source = render_messages(name, SAMPLE_MESSAGES, add_generation_prompt=False)
        assert source == "preset"
        assert "Pune hub" in rendered
        assert "14:30 IST" in rendered
