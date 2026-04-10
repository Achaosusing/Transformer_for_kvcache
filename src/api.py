from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .chat_format import ROLE_GENERATED
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


@dataclass
class H2ORuntimeState:
    past_key_values: Any
    score_counters: torch.Tensor
    active_token_count: int
    steps_since_collect: int = 0
    role_tags: torch.Tensor | None = None  # per-position role id (ROLE_*)


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
        attn_implementation: str = "eager",
    ) -> None:
        self.model = LocalTransformerModel(
            model_path=model_path,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            allow_remote_files=allow_remote_files,
            attn_implementation=attn_implementation,
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

    def initialize_h2o_state(
        self,
        token_ids: list[int],
        policy: Any,
        role_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, H2ORuntimeState]:
        device = self.model.model.device
        active_token_count = len(token_ids)

        # --- Scheme D: collect prefill attention scores via SDPA capture ---
        # Instead of leaving scores at zero and relying on recency tie-break
        # for the initial eviction, we extract the last-query attention from
        # the prefill forward pass so that the first prune is informed.
        from .model import SDPAAttentionCapture

        if self.model.attn_implementation == "sdpa":
            with torch.no_grad(), SDPAAttentionCapture(active_token_count, device) as cap:
                input_ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
                out = self.model.model(
                    input_ids=input_ids_t,
                    use_cache=True,
                )
            cached_logits = out.logits[0, -1, :].float()
            past_key_values = out.past_key_values
            score_counters = cap.get_scores()
        else:
            # Eager fallback: use output_attentions for the prefill.
            cached_logits, past_key_values, score_counters = (
                self.model.prefill_next_token_logits_with_attention(token_ids)
            )

        # Build role_tags tensor.
        if role_ids is not None:
            role_tags = torch.tensor(role_ids[:active_token_count], dtype=torch.int8, device=device)
            if role_tags.numel() < active_token_count:
                pad = torch.full(
                    (active_token_count - role_tags.numel(),),
                    ROLE_GENERATED, dtype=torch.int8, device=device,
                )
                role_tags = torch.cat([role_tags, pad])
        else:
            role_tags = None

        # Initial pruning – now informed by prefill attention scores.
        if active_token_count > policy.cache_budget:
            keep_tensor = policy.select_keep_tensor(active_token_count, score_counters)
            score_counters = score_counters[keep_tensor]
            past_key_values = self.model.prune_past_key_values(past_key_values, keep_tensor)
            if role_tags is not None:
                role_tags = role_tags[keep_tensor]
            active_token_count = keep_tensor.numel()

        return cached_logits, H2ORuntimeState(
            past_key_values=past_key_values,
            score_counters=score_counters,
            active_token_count=active_token_count,
            role_tags=role_tags,
        )

    def continue_h2o_state(
        self,
        token_ids: list[int],
        policy: Any,
        state: H2ORuntimeState,
        evict_period: int = 1,
        collect_period: int = 1,
    ) -> tuple[torch.Tensor | None, H2ORuntimeState]:
        cached_logits: torch.Tensor | None = None
        for token_id in token_ids:
            cached_logits, state = self._advance_h2o_state_with_token(
                token_id=token_id,
                policy=policy,
                state=state,
                evict_period=evict_period,
                collect_period=collect_period,
            )
        return cached_logits, state

    def generate_from_h2o_state(
        self,
        cached_logits: torch.Tensor,
        state: H2ORuntimeState,
        policy: Any,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_on_eos: bool,
        save_step_trace: bool,
        prompt_len: int,
        evict_period: int = 1,
        collect_period: int = 1,
    ) -> tuple[list[int], list[StepTrace], torch.Tensor, H2ORuntimeState]:
        generated_ids: list[int] = []
        traces: list[StepTrace] = []
        eos_token_id = self.model.tokenizer.eos_token_id

        for step in range(max_new_tokens):
            if save_step_trace:
                traces.append(
                    StepTrace(
                        step=step,
                        full_context_tokens=prompt_len + len(generated_ids),
                        kept_tokens=state.active_token_count,
                        kept_ratio=state.active_token_count / max(prompt_len + len(generated_ids), 1),
                    )
                )

            next_id = self.model.sample_next_token(cached_logits, temperature, top_p)
            generated_ids.append(next_id)

            if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                break

            cached_logits, state = self._advance_h2o_state_with_token(
                token_id=next_id,
                policy=policy,
                state=state,
                evict_period=evict_period,
                collect_period=collect_period,
            )

        return generated_ids, traces, cached_logits, state

    def clone_h2o_state(
        self,
        state: H2ORuntimeState,
        device: str | torch.device,
    ) -> H2ORuntimeState:
        return H2ORuntimeState(
            past_key_values=self.model.clone_past_key_values(state.past_key_values, device),
            score_counters=state.score_counters.detach().to(device=device).clone(),
            active_token_count=state.active_token_count,
            steps_since_collect=state.steps_since_collect,
            role_tags=state.role_tags.detach().to(device=device).clone() if state.role_tags is not None else None,
        )

    @staticmethod
    def apply_max_normalized_h2o_decay(
        score_counters: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        if not (0.0 <= alpha):
            raise ValueError("alpha must be >= 0")
        if score_counters.numel() == 0:
            return score_counters.clone()

        max_score = torch.max(score_counters)
        if max_score.item() <= 0:
            normalized = torch.zeros_like(score_counters)
        else:
            normalized = score_counters / max_score
        return normalized * alpha

    @staticmethod
    def apply_role_aware_h2o_decay(
        score_counters: torch.Tensor,
        role_tags: torch.Tensor,
        role_alphas: dict[int, float],
        default_alpha: float = 0.5,
    ) -> torch.Tensor:
        """Per-role exponential decay: each role type can have its own alpha.

        Higher alpha means the role's tokens retain more importance across
        turns.  Recommended defaults:
            ROLE_SYSTEM=0.9, ROLE_TOOL=0.7, ROLE_USER=0.3, ROLE_ASSISTANT=0.3
        """
        if score_counters.numel() == 0:
            return score_counters.clone()

        decayed = score_counters.clone()
        for role_id, alpha in role_alphas.items():
            mask = role_tags == role_id
            if not mask.any():
                continue
            segment = decayed[mask]
            max_s = segment.max()
            if max_s.item() <= 0:
                decayed[mask] = 0.0
            else:
                decayed[mask] = (segment / max_s) * alpha

        # Tokens with roles not in role_alphas get default_alpha.
        covered = torch.zeros_like(role_tags, dtype=torch.bool)
        for role_id in role_alphas:
            covered |= (role_tags == role_id)
        if not covered.all():
            uncovered = ~covered
            segment = decayed[uncovered]
            max_s = segment.max()
            if max_s.item() > 0:
                decayed[uncovered] = (segment / max_s) * default_alpha
            else:
                decayed[uncovered] = 0.0

        return decayed

    def restore_h2o_state(
        self,
        state: H2ORuntimeState,
        *,
        alpha: float,
        role_alphas: dict[int, float] | None = None,
    ) -> H2ORuntimeState:
        restored = self.clone_h2o_state(state, self.model.model.device)

        # --- Scheme C: role-aware decay when role_tags are available ---
        if restored.role_tags is not None and role_alphas is not None:
            restored.score_counters = self.apply_role_aware_h2o_decay(
                restored.score_counters,
                restored.role_tags,
                role_alphas,
                default_alpha=alpha,
            )
        else:
            # Fallback to the original uniform decay.
            restored.score_counters = self.apply_max_normalized_h2o_decay(
                restored.score_counters,
                alpha,
            )
        restored.steps_since_collect = 0
        return restored

    def trim_h2o_state_tail(
        self,
        state: H2ORuntimeState,
        trim_tokens: int,
    ) -> H2ORuntimeState:
        if trim_tokens <= 0:
            return state
        if trim_tokens >= state.active_token_count:
            raise ValueError("trim_tokens must be smaller than active_token_count")

        keep_count = state.active_token_count - trim_tokens
        keep_tensor = torch.arange(
            keep_count,
            device=self.model.model.device,
            dtype=torch.long,
        )
        state.score_counters = state.score_counters[:keep_count]
        if state.role_tags is not None:
            state.role_tags = state.role_tags[:keep_count]
        state.past_key_values = self.model.prune_past_key_values(state.past_key_values, keep_tensor)
        state.active_token_count = keep_count
        state.steps_since_collect = 0
        return state

    def _advance_h2o_state_with_token(
        self,
        token_id: int,
        policy: Any,
        state: H2ORuntimeState,
        evict_period: int = 1,
        collect_period: int = 1,
    ) -> tuple[torch.Tensor, H2ORuntimeState]:
        zero = torch.zeros(1, dtype=state.score_counters.dtype, device=state.score_counters.device)

        next_total_tokens = state.active_token_count + 1
        need_evict = next_total_tokens > policy.cache_budget + evict_period - 1
        if need_evict:
            extended_scores = torch.cat([state.score_counters, zero])
            keep_tensor = policy.select_keep_tensor(next_total_tokens, extended_scores)
            keep_tensor = keep_tensor[keep_tensor < state.active_token_count]
            state.score_counters = state.score_counters[keep_tensor]
            if state.role_tags is not None:
                state.role_tags = state.role_tags[keep_tensor]
            state.past_key_values = self.model.prune_past_key_values(state.past_key_values, keep_tensor)
            state.active_token_count = keep_tensor.numel()

        state.score_counters = torch.cat([state.score_counters, zero])
        if state.role_tags is not None:
            new_tag = torch.tensor([ROLE_GENERATED], dtype=state.role_tags.dtype, device=state.role_tags.device)
            state.role_tags = torch.cat([state.role_tags, new_tag])
        state.active_token_count += 1
        state.steps_since_collect += 1

        need_collect = need_evict or state.steps_since_collect >= collect_period
        if need_collect:
            cached_logits, state.past_key_values, attention_scores = self.model.next_token_logits_from_cache_with_attention(
                token_id,
                state.past_key_values,
                expected_tokens=state.active_token_count,
            )
            self._accumulate_h2o_scores(state.score_counters, attention_scores)
            state.steps_since_collect = 0
        else:
            cached_logits, state.past_key_values = self.model.next_token_logits_from_cache(
                token_id,
                state.past_key_values,
            )

        return cached_logits, state

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
        collect_period: int = 1,
    ) -> tuple[list[int], list[StepTrace]]:
        cached_logits, state = self.initialize_h2o_state(token_ids, policy)
        generated_ids, traces, _, _ = self.generate_from_h2o_state(
            cached_logits=cached_logits,
            state=state,
            policy=policy,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_on_eos=stop_on_eos,
            save_step_trace=save_step_trace,
            prompt_len=prompt_len,
            evict_period=evict_period,
            collect_period=collect_period,
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
        collect_period: int = 1,
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
                        collect_period=collect_period,
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
