from dslm.masking import mask_labels


def encode(messages, add_generation_prompt: bool) -> list[int]:
    """Prefix-stable stand-in: one id per role boundary, then content code points."""
    ids = [1]
    for message in messages:
        role_id = {"system": 10, "user": 11, "assistant": 12}[message["role"]]
        ids.append(role_id)
        ids.extend((ord(char) % 40) + 20 for char in message["content"])
        ids.append(2)
    if add_generation_prompt:
        ids.append(12)
    return ids


def test_only_assistant_tokens_are_labeled():
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Cutoff?"},
        {"role": "assistant", "content": "14:30 IST."},
    ]
    item = mask_labels(messages, encode, max_length=200)
    assert item is not None
    full = encode(messages, False)
    prompt = encode(messages[:-1], True)
    assert item["input_ids"] == full
    assert item["labels"][: len(prompt)] == [-100] * len(prompt)
    assert item["labels"][len(prompt) :] == full[len(prompt) :]
    assert any(label != -100 for label in item["labels"])


def test_drops_example_when_answer_is_truncated_away():
    messages = [
        {"role": "user", "content": "A long question about docks and detention rules."},
        {"role": "assistant", "content": "Three hours free."},
    ]
    assert mask_labels(messages, encode, max_length=4) is None
