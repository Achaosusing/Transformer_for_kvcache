from __future__ import annotations

import numpy as np
import torch


class H2OPolicy:
    name = "h2o"

    def __init__(
        self,
        sink_size: int,
        local_window_size: int,
        heavy_hitter_size: int,
    ) -> None:
        if sink_size < 0:
            raise ValueError("sink_size must be >= 0")
        if local_window_size <= 0:
            raise ValueError("local_window_size must be > 0")
        if heavy_hitter_size < 0:
            raise ValueError("heavy_hitter_size must be >= 0")

        self.sink_size = sink_size
        self.local_window_size = local_window_size
        self.heavy_hitter_size = heavy_hitter_size
        self.cache_budget = sink_size + local_window_size + heavy_hitter_size

    def select_streaming_keep_indices(self, total_tokens: int) -> list[int]:
        if total_tokens <= 0:
            return []

        sink_end = min(self.sink_size, total_tokens)
        recent_start = max(sink_end, total_tokens - self.local_window_size)
        keep = set(range(sink_end))
        keep.update(range(recent_start, total_tokens))
        keep.add(total_tokens - 1)
        return sorted(keep)

    @staticmethod
    def _topk_with_recent_tiebreak_torch(scores: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or scores.numel() == 0:
            return torch.empty(0, device=scores.device, dtype=torch.long)
        if k >= scores.numel():
            return torch.arange(scores.numel(), device=scores.device, dtype=torch.long)

        tie_break = torch.arange(scores.numel(), device=scores.device, dtype=scores.dtype)
        tie_break = tie_break / max(scores.numel(), 1)
        adjusted_scores = scores.to(torch.float32) + tie_break * 1e-6
        return torch.topk(adjusted_scores, k=k, largest=True, sorted=False).indices

    @staticmethod
    def _topk_with_recent_tiebreak_numpy(scores: np.ndarray, k: int) -> np.ndarray:
        if k <= 0 or scores.size == 0:
            return np.empty(0, dtype=np.int64)
        if k >= scores.size:
            return np.arange(scores.size, dtype=np.int64)

        tie_break = np.arange(scores.size, dtype=np.float64) / max(scores.size, 1)
        adjusted_scores = scores.astype(np.float64, copy=False) + tie_break * 1e-6
        return np.argpartition(adjusted_scores, -k)[-k:]

    def _compute_keep_layout(
        self, total_tokens: int
    ) -> tuple[int, int, int, int]:
        """Return (sink_end, recent_start, heavy_budget, candidate_count)."""
        sink_end = min(self.sink_size, total_tokens)
        recent_start = max(sink_end, total_tokens - self.local_window_size)
        protected_tokens = sink_end + max(0, total_tokens - recent_start)
        heavy_budget = max(0, self.cache_budget - protected_tokens)
        candidate_count = max(0, recent_start - sink_end)
        return sink_end, recent_start, heavy_budget, candidate_count

    def select_keep_tensor(
        self,
        total_tokens: int,
        cumulative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """GPU-resident version that avoids list round-trips."""
        device = cumulative_scores.device
        if total_tokens <= 0:
            return torch.empty(0, device=device, dtype=torch.long)
        if total_tokens <= self.cache_budget:
            return torch.arange(total_tokens, device=device, dtype=torch.long)

        sink_end, recent_start, heavy_budget, candidate_count = (
            self._compute_keep_layout(total_tokens)
        )

        parts: list[torch.Tensor] = []
        if sink_end > 0:
            parts.append(torch.arange(sink_end, device=device, dtype=torch.long))
        if total_tokens > recent_start:
            parts.append(
                torch.arange(recent_start, total_tokens, device=device, dtype=torch.long)
            )
        if candidate_count > 0 and heavy_budget > 0:
            k = min(heavy_budget, candidate_count)
            if k >= candidate_count:
                parts.append(
                    torch.arange(sink_end, recent_start, device=device, dtype=torch.long)
                )
            else:
                mid_scores = cumulative_scores[sink_end:recent_start]
                topk_rel = self._topk_with_recent_tiebreak_torch(mid_scores, k)
                parts.append(topk_rel.to(torch.long) + sink_end)

        # Parts (sink / heavy-hitters / window) are disjoint — sort is enough.
        return torch.cat(parts).sort().values

    def select_keep_indices(
        self,
        total_tokens: int,
        cumulative_scores: np.ndarray | torch.Tensor,
    ) -> list[int]:
        if total_tokens <= 0:
            return []
        if total_tokens <= self.cache_budget:
            return list(range(total_tokens))

        if torch.is_tensor(cumulative_scores):
            return self.select_keep_tensor(total_tokens, cumulative_scores).tolist()

        sink_end, recent_start, heavy_budget, candidate_count = (
            self._compute_keep_layout(total_tokens)
        )

        keep = set(range(sink_end))
        keep.update(range(recent_start, total_tokens))

        if candidate_count > 0 and heavy_budget > 0:
            k = min(heavy_budget, candidate_count)
            if k >= candidate_count:
                topk_rel = np.arange(candidate_count, dtype=np.int64)
            else:
                mid_scores = cumulative_scores[sink_end:recent_start]
                topk_rel = self._topk_with_recent_tiebreak_numpy(mid_scores, k)
            keep.update((topk_rel + sink_end).tolist())

        return sorted(keep)
