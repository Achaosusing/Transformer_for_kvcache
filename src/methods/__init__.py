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
