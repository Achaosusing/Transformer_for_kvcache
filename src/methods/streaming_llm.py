from __future__ import annotations

import numpy as np


class StreamingLLMPolicy:
    name = "streamingllm"

    def __init__(self, sink_size: int, local_window_size: int) -> None:
        if sink_size < 0:
            raise ValueError("sink_size must be >= 0")
        if local_window_size <= 0:
            raise ValueError("local_window_size must be > 0")
        self.sink_size = sink_size
        self.local_window_size = local_window_size

    def select_keep_indices(
        self,
        total_tokens: int,
        cumulative_scores: np.ndarray,
    ) -> list[int]:
        if total_tokens <= 0:
            return []

        sink_end = min(self.sink_size, total_tokens)
        tail_start = max(sink_end, total_tokens - self.local_window_size)
        keep = set(range(sink_end))
        keep.update(range(tail_start, total_tokens))
        keep.add(total_tokens - 1)
        return sorted(keep)
