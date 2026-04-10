from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {i} in {p}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_sample(item: dict[str, Any]) -> tuple[str, str | None, list[dict[str, Any]] | None]:
    sample_id = str(item.get("id", "sample"))
    messages = item.get("messages")
    if isinstance(messages, list) and messages:
        norm: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {
                "role": str(m.get("role", "user")),
                "content": str(m.get("content", "")),
            }
            if m.get("tool_calls"):
                entry["tool_calls"] = m["tool_calls"]
            if m.get("tool_call_id") is not None:
                entry["tool_call_id"] = str(m["tool_call_id"])
            if m.get("reasoning_content") is not None:
                entry["reasoning_content"] = str(m["reasoning_content"])
            norm.append(entry)
        return sample_id, None, norm

    for key in ("prompt", "instruction", "question", "input"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return sample_id, v, None

    return sample_id, None, None
