from .baseline import BaselineFullAttentionPolicy
from .h2o import H2OPolicy
from .streaming_llm import StreamingLLMPolicy

METHODS = {
    "baseline": BaselineFullAttentionPolicy,
    "streamingllm": StreamingLLMPolicy,
    "streaming_llm": StreamingLLMPolicy,
    "h2o": H2OPolicy,
}

__all__ = [
    "BaselineFullAttentionPolicy",
    "StreamingLLMPolicy",
    "H2OPolicy",
    "METHODS",
]


def prune_streaming_prompt(full_ids: list[int], policy: StreamingLLMPolicy) -> list[int]:
    """Apply StreamingLLM sink+window pruning to prompt token ids."""
    n = len(full_ids)
    if n <= policy.cache_budget:
        return full_ids
    sink_end = min(policy.sink_size, n)
    tail_start = max(sink_end, n - policy.local_window_size)
    return full_ids[:sink_end] + full_ids[tail_start:]
