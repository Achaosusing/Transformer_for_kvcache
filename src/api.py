from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .methods import METHODS, prune_streaming_prompt
from .model import LocalTransformerModel, StepTrace
from .utils import normalize_sample


@dataclass
class EvalResult:
    sample_id: str
    method: str
    output_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    elapsed_sec: float


class OracleKVProjectAPI:
    """Unified API for separated baseline/streamingLLM/h2o evaluation."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        allow_remote_files: bool = False,
    ) -> None:
        self.model = LocalTransformerModel(
            model_path=model_path,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            allow_remote_files=allow_remote_files,
        )

    def _build_policy(self, method: str, method_cfg: dict[str, Any]):
        key = method.lower()
        if key not in METHODS:
            raise ValueError(f"Unsupported method: {method}")

        if key in ("streamingllm", "streaming_llm"):
            return METHODS[key](
                sink_size=int(method_cfg.get("sink_size", 4)),
                local_window_size=int(method_cfg.get("local_window_size", 256)),
            )
        if key == "h2o":
            return METHODS[key](
                sink_size=int(method_cfg.get("sink_size", 4)),
                local_window_size=int(method_cfg.get("local_window_size", 256)),
                heavy_hitter_size=int(method_cfg.get("heavy_hitter_size", 128)),
            )
        return METHODS[key]()

    @staticmethod
    def _accumulate_h2o_scores(score_counters: torch.Tensor, attention_scores: torch.Tensor) -> None:
        if score_counters.numel() == 0:
            return
        if attention_scores.shape[0] != score_counters.shape[0]:
            raise RuntimeError("Attention score length does not match H2O cache length")
        score_counters.add_(attention_scores.to(device=score_counters.device, dtype=score_counters.dtype))

    def _generate_with_manual_cache(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_on_eos: bool,
        save_step_trace: bool,
        full_context_tokens: int,
    ) -> tuple[list[int], list[StepTrace]]:
        cached_logits, past_key_values = self.model.prefill_next_token_logits(token_ids)

        generated_ids: list[int] = []
        traces: list[StepTrace] = []
        active_context_tokens = len(token_ids)
        eos_token_id = self.model.tokenizer.eos_token_id

        for step in range(max_new_tokens):
            if save_step_trace:
                traces.append(
                    StepTrace(
                        step=step,
                        full_context_tokens=full_context_tokens + len(generated_ids),
                        kept_tokens=active_context_tokens,
                        kept_ratio=active_context_tokens / max(full_context_tokens + len(generated_ids), 1),
                    )
                )

            next_id = self.model.sample_next_token(cached_logits, temperature, top_p)
            generated_ids.append(next_id)

            if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                break

            cached_logits, past_key_values = self.model.next_token_logits_from_cache(
                next_id,
                past_key_values,
            )
            active_context_tokens += 1

        return generated_ids, traces

    def _generate_with_streaming_cache(
        self,
        token_ids: list[int],
        policy: Any,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_on_eos: bool,
        save_step_trace: bool,
        prompt_len: int,
        evict_period: int = 1,
    ) -> tuple[list[int], list[StepTrace]]:
        sink_size = policy.sink_size
        cache_budget = policy.cache_budget
        device = self.model.model.device

        cached_logits, past_key_values = self.model.prefill_next_token_logits(token_ids)

        # Pre-allocate sink indices tensor (reused every eviction step).
        sink_indices = torch.arange(sink_size, device=device, dtype=torch.long)

        generated_ids: list[int] = []
        traces: list[StepTrace] = []
        active_token_count = len(token_ids)
        eos_token_id = self.model.tokenizer.eos_token_id

        for step in range(max_new_tokens):
            if save_step_trace:
                traces.append(
                    StepTrace(
                        step=step,
                        full_context_tokens=prompt_len + len(generated_ids),
                        kept_tokens=active_token_count,
                        kept_ratio=active_token_count / max(prompt_len + len(generated_ids), 1),
                    )
                )

            next_id = self.model.sample_next_token(cached_logits, temperature, top_p)
            generated_ids.append(next_id)

            if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                break

            # StreamingLLM eviction: sink tokens are fixed, window slides.
            # When over budget, directly drop the oldest non-sink tokens
            # (at positions sink_size .. sink_size+excess-1).
            next_total = active_token_count + 1
            if next_total > cache_budget + evict_period - 1:
                excess = next_total - cache_budget
                keep = torch.cat([
                    sink_indices,
                    torch.arange(sink_size + excess, active_token_count,
                                 device=device, dtype=torch.long),
                ])
                past_key_values = self.model.prune_past_key_values(past_key_values, keep)
                active_token_count -= excess

            cached_logits, past_key_values = self.model.next_token_logits_from_cache(
                next_id,
                past_key_values,
            )
            active_token_count += 1

        return generated_ids, traces

    def _generate_with_h2o(
        self,
        token_ids: list[int],
        policy: Any,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_on_eos: bool,
        save_step_trace: bool,
        prompt_len: int,
        evict_period: int = 1,
    ) -> tuple[list[int], list[StepTrace]]:
        # H2O = StreamingLLM (sink + window) + heavy-hitter (topk from middle).
        # Prefill with ALL prompt tokens so attention scores cover the full
        # prompt — heavy-hitters are selected from actual attention, not zeros.
        cached_logits, past_key_values, attention_scores = (
            self.model.prefill_next_token_logits_with_attention(token_ids)
        )

        device = attention_scores.device
        score_counters = attention_scores.clone()
        active_token_count = len(token_ids)
        _zero = torch.zeros(1, dtype=score_counters.dtype, device=device)

        # Initial pruning: use H2O policy (sink + topk heavy-hitters + window)
        # based on real attention scores from the full prompt.
        if active_token_count > policy.cache_budget:
            keep_tensor = policy.select_keep_tensor(active_token_count, score_counters)
            score_counters = score_counters[keep_tensor]
            past_key_values = self.model.prune_past_key_values(past_key_values, keep_tensor)
            active_token_count = keep_tensor.numel()

        generated_ids: list[int] = []
        traces: list[StepTrace] = []
        eos_token_id = self.model.tokenizer.eos_token_id

        for step in range(max_new_tokens):
            if save_step_trace:
                traces.append(
                    StepTrace(
                        step=step,
                        full_context_tokens=prompt_len + len(generated_ids),
                        kept_tokens=active_token_count,
                        kept_ratio=active_token_count / max(prompt_len + len(generated_ids), 1),
                    )
                )

            next_id = self.model.sample_next_token(cached_logits, temperature, top_p)
            generated_ids.append(next_id)

            if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                break

            next_total_tokens = active_token_count + 1
            need_evict = next_total_tokens > policy.cache_budget + evict_period - 1
            if need_evict:
                extended_scores = torch.cat([score_counters, _zero])
                keep_tensor = policy.select_keep_tensor(next_total_tokens, extended_scores)
                # Filter out the not-yet-inserted new-token index.
                keep_tensor = keep_tensor[keep_tensor < active_token_count]
                score_counters = score_counters[keep_tensor]
                past_key_values = self.model.prune_past_key_values(past_key_values, keep_tensor)
                active_token_count = keep_tensor.numel()

            score_counters = torch.cat([score_counters, _zero])
            active_token_count += 1

            if evict_period <= 1 or need_evict:
                cached_logits, past_key_values, attention_scores = self.model.next_token_logits_from_cache_with_attention(
                    next_id,
                    past_key_values,
                    expected_tokens=active_token_count,
                )
                self._accumulate_h2o_scores(score_counters, attention_scores)
            else:
                cached_logits, past_key_values = self.model.next_token_logits_from_cache(
                    next_id,
                    past_key_values,
                )

        return generated_ids, traces

    def evaluate(
        self,
        samples: list[dict[str, Any]],
        methods: list[str],
        method_configs: dict[str, dict[str, Any]] | None = None,
        *,
        max_new_tokens: int = 512,
        evict_period: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        max_input_tokens: int | None = None,
        save_step_trace: bool = False,
    ) -> dict[str, Any]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        cfg = method_configs or {}
        results_by_method: dict[str, list[dict[str, Any]]] = {}
        flat_results: list[dict[str, Any]] = []

        for method in methods:
            policy = self._build_policy(method, cfg.get(method, {}))
            method_key = method.lower()
            rows: list[dict[str, Any]] = []

            for idx, item in enumerate(samples):
                sample_id, prompt, messages = normalize_sample(item)
                if prompt is None and messages is None:
                    continue

                full_ids = self.model.format_prompt_ids(prompt, messages, max_input_tokens)
                prompt_len = len(full_ids)
                generated_ids: list[int] = []
                traces: list[StepTrace] = []

                if method_key in ("baseline", "streamingllm", "streaming_llm"):
                    t0 = time.perf_counter()
                    if method_key == "baseline":
                        pruned_ids = full_ids
                        generated_ids, traces = self._generate_with_manual_cache(
                            token_ids=pruned_ids,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stop_on_eos=stop_on_eos,
                            save_step_trace=save_step_trace,
                            full_context_tokens=len(full_ids),
                        )
                    else:
                        pruned_ids = prune_streaming_prompt(full_ids, policy)
                        generated_ids, traces = self._generate_with_streaming_cache(
                            token_ids=pruned_ids,
                            policy=policy,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stop_on_eos=stop_on_eos,
                            save_step_trace=save_step_trace,
                            prompt_len=prompt_len,
                            evict_period=evict_period,
                        )
                    elapsed = time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    generated_ids, traces = self._generate_with_h2o(
                        token_ids=full_ids,
                        policy=policy,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop_on_eos=stop_on_eos,
                        save_step_trace=save_step_trace,
                        prompt_len=prompt_len,
                        evict_period=evict_period,
                    )
                    elapsed = time.perf_counter() - t0
                output_text = self.model.tokenizer.decode(generated_ids, skip_special_tokens=True)

                rec = EvalResult(
                    sample_id=sample_id if sample_id != "sample" else f"sample_{idx}",
                    method=method_key,
                    output_text=output_text,
                    prompt_tokens=prompt_len,
                    generated_tokens=len(generated_ids),
                    total_tokens=prompt_len + len(generated_ids),
                    elapsed_sec=elapsed,
                )
                row = asdict(rec)
                if save_step_trace:
                    row["step_trace"] = [asdict(t) for t in traces]
                rows.append(row)
                flat_results.append(row)

            results_by_method[method_key] = rows

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "methods": [m.lower() for m in methods],
            "results_by_method": results_by_method,
            "flat_results": flat_results,
        }
