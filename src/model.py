from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class StepTrace:
    step: int
    full_context_tokens: int
    kept_tokens: int
    kept_ratio: float


class LocalTransformerModel:
    def __init__(
        self,
        model_path: str,
        device: str,
        gpu_memory_utilization: float,
        dtype: str,
        trust_remote_code: bool,
        allow_remote_files: bool,
        attn_implementation: str = "eager",
    ) -> None:
        self.device = self._resolve_device(device)
        self._configure_gpu_memory_utilization(self.device, gpu_memory_utilization)
        self.dtype = self._resolve_dtype(dtype, self.device)

        local_files_only = not allow_remote_files
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.attn_implementation = attn_implementation

        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return device

    @staticmethod
    def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
        if dtype == "auto":
            if device == "cuda":
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping[dtype]

    @staticmethod
    def _configure_gpu_memory_utilization(device: str, gpu_memory_utilization: float) -> None:
        if not (0.0 < gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if device != "cuda":
            return
        torch.cuda.set_per_process_memory_fraction(gpu_memory_utilization)

    def format_prompt_ids(
        self,
        prompt: str | None,
        messages: list[dict[str, str]] | None,
        max_input_tokens: int | None,
    ) -> list[int]:
        if messages:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
                ) + "\nassistant:"
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        elif prompt is not None:
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            raise ValueError("Either prompt or messages must be provided")

        if max_input_tokens is not None and len(ids) > max_input_tokens:
            ids = ids[-max_input_tokens:]
        if not ids:
            raise ValueError("Empty input after tokenization")
        return ids

    @torch.no_grad()
    def generate_new_tokens(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_on_eos: bool,
    ) -> list[int]:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if stop_on_eos and self.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        if temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(temperature, 1e-6)
            gen_kwargs["top_p"] = top_p

        out = self.model.generate(input_ids=input_ids, **gen_kwargs)
        return out[0, input_ids.shape[1] :].tolist()

    @torch.no_grad()
    def next_token_logits(self, token_ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits[0, -1, :].float()

    @torch.no_grad()
    def prefill_next_token_logits(self, token_ids: list[int]) -> tuple[torch.Tensor, Any]:
        if not token_ids:
            raise ValueError("token_ids must not be empty")

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        out = self.model(input_ids=input_ids, use_cache=True)
        return out.logits[0, -1, :].float(), out.past_key_values

    @staticmethod
    def _aggregate_last_query_attention(
        attentions: Any,
        expected_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        if expected_tokens <= 0 or not attentions:
            return torch.zeros(expected_tokens, dtype=torch.float32, device=device)

        acc = torch.zeros(expected_tokens, dtype=torch.float32, device=device)
        valid_layers = 0
        for layer_attn in attentions:
            if layer_attn is None or layer_attn.ndim != 4:
                continue
            vec = layer_attn[0, :, -1, :].mean(dim=0).float()
            if vec.numel() == 0:
                continue
            if vec.numel() == expected_tokens:
                acc.add_(vec)
            elif vec.numel() < expected_tokens:
                acc[-vec.numel() :].add_(vec)
            else:
                acc.add_(vec[-expected_tokens:])
            valid_layers += 1

        if valid_layers == 0:
            return torch.zeros(expected_tokens, dtype=torch.float32, device=device)
        return acc / valid_layers

    @torch.no_grad()
    def prefill_next_token_logits_with_attention(
        self,
        token_ids: list[int],
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        if not token_ids:
            raise ValueError("token_ids must not be empty")

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        out = self.model(
            input_ids=input_ids,
            use_cache=True,
            output_attentions=True,
        )
        attention_scores = self._aggregate_last_query_attention(
            out.attentions,
            expected_tokens=len(token_ids),
            device=self.model.device,
        )
        return out.logits[0, -1, :].float(), out.past_key_values, attention_scores

    @torch.no_grad()
    def next_token_logits_from_cache(
        self,
        token_id: int,
        past_key_values: Any,
    ) -> tuple[torch.Tensor, Any]:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.model.device)
        out = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return out.logits[0, -1, :].float(), out.past_key_values

    @torch.no_grad()
    def next_token_logits_from_cache_with_attention(
        self,
        token_id: int,
        past_key_values: Any,
        expected_tokens: int,
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.model.device)
        out = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
        )
        attention_scores = self._aggregate_last_query_attention(
            out.attentions,
            expected_tokens=expected_tokens,
            device=self.model.device,
        )
        return out.logits[0, -1, :].float(), out.past_key_values, attention_scores

    @staticmethod
    def _prune_cache_tensor(cache_tensor: Any, keep_indices: torch.Tensor) -> Any:
        # Some model backends leave selected cache slots as None.
        if cache_tensor is None:
            return None
        if not torch.is_tensor(cache_tensor):
            raise TypeError(f"Unsupported cache tensor type: {type(cache_tensor)}")
        if cache_tensor.ndim < 3:
            raise ValueError("Unsupported cache tensor rank")

        seq_dim = cache_tensor.ndim - 2
        keep = keep_indices.to(device=cache_tensor.device, dtype=torch.long)
        if keep.numel() > 0:
            keep = keep[(keep >= 0) & (keep < cache_tensor.shape[seq_dim])]
        return cache_tensor.index_select(seq_dim, keep)

    def prune_past_key_values(
        self,
        past_key_values: Any,
        keep_indices: list[int] | torch.Tensor,
    ) -> Any:
        keep_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=self.model.device)

        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx] = self._prune_cache_tensor(
                    past_key_values.key_cache[layer_idx],
                    keep_tensor,
                )
                past_key_values.value_cache[layer_idx] = self._prune_cache_tensor(
                    past_key_values.value_cache[layer_idx],
                    keep_tensor,
                )
            return past_key_values

        if hasattr(past_key_values, "layers"):
            for layer in past_key_values.layers:
                if hasattr(layer, "keys"):
                    layer.keys = self._prune_cache_tensor(layer.keys, keep_tensor)
                if hasattr(layer, "values"):
                    layer.values = self._prune_cache_tensor(layer.values, keep_tensor)
            return past_key_values

        legacy_cache = past_key_values.to_legacy_cache() if hasattr(past_key_values, "to_legacy_cache") else past_key_values
        pruned_cache = []
        for layer_cache in legacy_cache:
            if not isinstance(layer_cache, tuple) or len(layer_cache) < 2:
                raise ValueError("Unsupported past_key_values format")
            pruned_layer = (
                self._prune_cache_tensor(layer_cache[0], keep_tensor),
                self._prune_cache_tensor(layer_cache[1], keep_tensor),
                *layer_cache[2:],
            )
            pruned_cache.append(pruned_layer)

        cache_type = type(past_key_values)
        if hasattr(cache_type, "from_legacy_cache"):
            return cache_type.from_legacy_cache(tuple(pruned_cache))
        return tuple(pruned_cache)

    @staticmethod
    def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
        if temperature <= 0:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            csum = torch.cumsum(sorted_probs, dim=-1)
            mask = csum > top_p
            mask[0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            denom = sorted_probs.sum()
            if denom.item() <= 0:
                return int(sorted_idx[0].item())
            sorted_probs = sorted_probs / denom
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_idx[sampled].item())
        return int(torch.multinomial(probs, num_samples=1).item())
