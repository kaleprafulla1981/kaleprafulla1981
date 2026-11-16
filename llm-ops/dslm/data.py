"""JSONL readers for chat rows and the older instruction/input/output shape."""

from __future__ import annotations

import json
from pathlib import Path

ALLOWED_ROLES = {"system", "user", "assistant"}


class DataError(ValueError):
    pass


def _as_messages(row: dict, line_no: int, path: Path) -> list[dict[str, str]]:
    if "messages" in row:
        messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            raise DataError(f"{path}:{line_no}: messages must be a non-empty list")
        cleaned: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise DataError(f"{path}:{line_no}: message {index} is not an object")
            role = message.get("role")
            content = message.get("content")
            if role not in ALLOWED_ROLES:
                raise DataError(
                    f"{path}:{line_no}: role must be system, user, or assistant (got {role!r})"
                )
            if not isinstance(content, str) or not content.strip():
                raise DataError(f"{path}:{line_no}: message {index} content must be a non-empty string")
            cleaned.append({"role": role, "content": content})
        return cleaned

    if "instruction" in row and "output" in row:
        instruction = row["instruction"]
        output = row["output"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise DataError(f"{path}:{line_no}: instruction must be a non-empty string")
        if not isinstance(output, str) or not output.strip():
            raise DataError(f"{path}:{line_no}: output must be a non-empty string")
        extra = row.get("input") or ""
        if extra and not isinstance(extra, str):
            raise DataError(f"{path}:{line_no}: input must be a string")
        user = instruction if not extra else f"{instruction}\n\n{extra}"
        messages = []
        system = row.get("system")
        if isinstance(system, str) and system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": output})
        return messages

    raise DataError(
        f"{path}:{line_no}: expected a messages list or instruction/output fields"
    )


def _check_turns(messages: list[dict[str, str]], line_no: int, path: Path) -> None:
    roles = [message["role"] for message in messages]
    if "user" not in roles or "assistant" not in roles:
        raise DataError(f"{path}:{line_no}: need at least one user turn and one assistant turn")
    if roles[-1] != "assistant":
        raise DataError(f"{path}:{line_no}: the last message must be the assistant answer")


def load_jsonl(path: str | Path) -> list[list[dict[str, str]]]:
    file_path = Path(path)
    if not file_path.exists():
        raise DataError(f"Dataset not found: {file_path}")
    rows: list[list[dict[str, str]]] = []
    for line_no, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DataError(f"{file_path}:{line_no}: invalid JSON ({exc})") from exc
        if not isinstance(payload, dict):
            raise DataError(f"{file_path}:{line_no}: each line must be a JSON object")
        messages = _as_messages(payload, line_no, file_path)
        _check_turns(messages, line_no, file_path)
        rows.append(messages)
    if not rows:
        raise DataError(f"{file_path}: no examples found")
    return rows
