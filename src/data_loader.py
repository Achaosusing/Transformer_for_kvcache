from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_jsonl


class JsonlDataset:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows = read_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def build_dataset(path: str | Path) -> JsonlDataset:
    return JsonlDataset(path)
