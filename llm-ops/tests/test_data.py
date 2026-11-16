from pathlib import Path

import pytest

from dslm.data import DataError, load_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_load_chat_rows():
    rows = load_jsonl(ROOT / "data/sample/train.jsonl")
    assert len(rows) == 12
    assert rows[0][0]["role"] == "system"
    assert rows[0][-1]["role"] == "assistant"


def test_instruction_format_becomes_messages():
    rows = load_jsonl(ROOT / "data/sample/instruction.jsonl")
    assert [message["role"] for message in rows[0]] == ["system", "user", "assistant"]
    assert "14:30 IST" in rows[0][-1]["content"]


def test_rejects_bad_role(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"messages": [{"role": "tool", "content": "x"}, {"role": "assistant", "content": "y"}]}\n',
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="role"):
        load_jsonl(path)


def test_rejects_missing_assistant(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"messages": [{"role": "user", "content": "hi"}]}\n',
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="assistant"):
        load_jsonl(path)
