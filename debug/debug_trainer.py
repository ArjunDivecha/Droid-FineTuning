"""Comprehensive debugging wrapper for MLX GSPO/GRPO training.

This module instruments the stock ``mlx_lm_lora`` training pipeline with
structured logging, deterministic replay hooks, and crash forensics.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import traceback
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx import utils as mx_utils
    from mlx.nn.utils import average_gradients
except Exception as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "The MLX runtime is required to use the debug trainer."
    ) from exc

from mlx_lm_lora.trainer import grpo_trainer
from mlx_lm_lora.trainer.grpo_trainer import (
    GRPOTrainingArgs,
    evaluate_grpo,
    iterate_grpo_batches,
)
from mlx_lm_lora.trainer.datasets import (
    load_dataset as base_load_dataset,
    GRPODataset,
    CacheDataset,
)
import mlx_lm_lora.trainer.datasets as datasets_module
from mlx_lm.models import cache as mlx_cache
from mlx_lm.generate import generate as mlx_generate, make_sampler as mlx_make_sampler
from mlx_lm_lora.trainer.grpo_reward_functions import (
    RewardFunctions,
    get_default_reward_functions,
    get_reward_function,
    list_available_reward_functions,
)
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm_lora.train import (
    CONFIG_DEFAULTS,
    build_parser as build_base_parser,
    run as base_run,
)


# ---------------------------------------------------------------------------
# Debug configuration and context containers
# ---------------------------------------------------------------------------


@dataclass
class DebugConfig:
    """User-configurable switches for the debug trainer."""

    log_root: Path
    preflight_limit: int = 2000
    keep_recent_steps: int = 32
    fallback_sequence_on_empty: bool = True
    disable_eval: bool = False
    disable_checkpoints: bool = False
    min_new_tokens: int = 1
    detect_nans: bool = True


@dataclass
class DebugContext:
    """Per-step context captured for crash forensics."""

    current_step: int = 0
    token_counts: Optional[List[int]] = None
    empty_indices: Optional[List[int]] = None
    batch_indices: Optional[List[int]] = None
    prompts_preview: Optional[List[str]] = None
    answers_preview: Optional[List[str]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    raw_loss: Optional[float] = None
    grad_norm: Optional[float] = None
    grad_nan_count: int = 0
    lr: Optional[float] = None
    val_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "token_counts": self.token_counts,
            "empty_indices": self.empty_indices,
            "batch_indices": self.batch_indices,
            "prompts_preview": self.prompts_preview,
            "answers_preview": self.answers_preview,
            "metrics": self.metrics,
            "raw_loss": self.raw_loss,
            "grad_norm": self.grad_norm,
            "grad_nan_count": self.grad_nan_count,
            "learning_rate": self.lr,
            "val_loss": self.val_loss,
        }


@dataclass
class DebugState:
    """Process-wide mutable state shared between patched functions."""

    logger: "DebugLogger"
    config: DebugConfig
    context: DebugContext = field(default_factory=DebugContext)
    run_metadata: Dict[str, Any] = field(default_factory=dict)


CURRENT_DEBUG_STATE: Optional[DebugState] = None


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


class DebugLogger:
    """Structured JSON logger with crash bundles."""

    def __init__(self, run_dir: Path, keep_recent: int = 32) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._steps_path = self.run_dir / "steps.jsonl"
        self._events_path = self.run_dir / "events.jsonl"
        self._steps_handle = self._steps_path.open("a", encoding="utf-8")
        self._events_handle = self._events_path.open("a", encoding="utf-8")
        self._recent_steps: List[Dict[str, Any]] = []
        self._recent_limit = keep_recent

    # ------------------------------------------------------------------
    def _serialize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, mx.array):
                np_value = np.array(value)
                if np_value.size == 1:
                    return float(np_value)
                return np_value.tolist()
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return float(value.reshape(-1)[0])
                return value.tolist()
            if isinstance(value, (np.floating, np.integer)):
                return float(value)
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value

        return {k: convert(v) for k, v in payload.items()}

    # ------------------------------------------------------------------
    def log_step(self, payload: Dict[str, Any]) -> None:
        record = self._serialize(payload)
        record.setdefault("timestamp", time.time())
        json.dump(record, self._steps_handle, ensure_ascii=False)
        self._steps_handle.write("\n")
        self._steps_handle.flush()

        self._recent_steps.append(record)
        if len(self._recent_steps) > self._recent_limit:
            self._recent_steps.pop(0)

    # ------------------------------------------------------------------
    def log_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None, level: str = "info") -> None:
        data = {
            "timestamp": time.time(),
            "event": event_type,
            "level": level,
        }
        if payload:
            data.update(self._serialize(payload))
        json.dump(data, self._events_handle, ensure_ascii=False)
        self._events_handle.write("\n")
        self._events_handle.flush()

    # ------------------------------------------------------------------
    def capture_crash(
        self,
        exc: BaseException,
        context: DebugContext,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        crash_dir = self.run_dir / "failures" / f"{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-step-{context.current_step:04d}"
        crash_dir.mkdir(parents=True, exist_ok=True)

        # Exception details
        with (crash_dir / "exception.txt").open("w", encoding="utf-8") as fh:
            fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

        # Context snapshot
        with (crash_dir / "context.json").open("w", encoding="utf-8") as fh:
            json.dump(self._serialize(context.to_dict()), fh, indent=2)

        # Recent steps buffer
        with (crash_dir / "recent_steps.jsonl").open("w", encoding="utf-8") as fh:
            for record in self._recent_steps:
                json.dump(record, fh)
                fh.write("\n")

        if extra:
            with (crash_dir / "extra.json").open("w", encoding="utf-8") as fh:
                json.dump(self._serialize(extra), fh, indent=2)

        self.log_event("crash", {"message": str(exc), "crash_dir": crash_dir}, level="error")

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._steps_handle.close()
        self._events_handle.close()


# ---------------------------------------------------------------------------
# Pre-flight dataset inspection
# ---------------------------------------------------------------------------


def _summarize_lengths(lengths: List[int]) -> Dict[str, Any]:
    if not lengths:
        return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    arr = np.array(lengths)
    return {
        "count": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def run_preflight_scan(
    dataset: Any,
    split_name: str,
    logger: DebugLogger,
    limit: int,
) -> Dict[str, Any]:
    """Inspect the dataset for empty prompts/answers and token length stats."""
    if not isinstance(dataset, GRPODataset):
        logger.log_event(
            "preflight_skipped",
            {"split": split_name, "reason": "unsupported_dataset_type"},
            level="warning",
        )
        return {"split": split_name, "skipped": True}

    prompt_token_lengths: List[int] = []
    answer_token_lengths: List[int] = []
    empty_answers: List[int] = []
    empty_prompts: List[int] = []
    examples_sample: List[Dict[str, Any]] = []

    for idx, (prompt_tokens, answer_tokens, prompt_text, answer_text, _type) in enumerate(dataset._data):
        if limit and idx >= limit:
            break
        prompt_len = len(prompt_tokens)
        answer_len = len(answer_tokens)
        prompt_token_lengths.append(prompt_len)
        answer_token_lengths.append(answer_len)
        if prompt_len == 0:
            empty_prompts.append(idx)
        if answer_len == 0:
            empty_answers.append(idx)
        if len(examples_sample) < 5:
            examples_sample.append(
                {
                    "index": idx,
                    "prompt_preview": prompt_text[:120],
                    "answer_preview": answer_text[:120],
                    "prompt_tokens": prompt_len,
                    "answer_tokens": answer_len,
                }
            )

    summary = {
        "split": split_name,
        "inspected": len(prompt_token_lengths),
        "prompt_token_stats": _summarize_lengths(prompt_token_lengths),
        "answer_token_stats": _summarize_lengths(answer_token_lengths),
        "empty_prompt_indices": empty_prompts[:20],
        "empty_answer_indices": empty_answers[:20],
        "example_previews": examples_sample,
    }

    logger.log_event("preflight_summary", summary)
    return summary


# ---------------------------------------------------------------------------
# Safe token log probability computation with instrumentation
# ---------------------------------------------------------------------------


def safe_get_per_token_logps(model: nn.Module, inputs, lengths):
    """Variant of ``get_per_token_logps`` that tolerates empty sequences."""
    logits = model(inputs).astype(mx.float16)
    logits = logits[:, :-1, :]
    targets = inputs[:, 1:]

    lengths_np = np.array(lengths, dtype=np.int32)
    per_token_logps = []
    empty_indices: List[int] = []

    for i in range(logits.shape[0]):
        seq_len = int(lengths_np[i]) - 1
        if seq_len <= 0:
            per_token_logps.append(mx.zeros((0,), dtype=mx.float32))
            empty_indices.append(i)
            continue

        seq_logits = logits[i, :seq_len]
        seq_targets = targets[i, :seq_len]
        log_probs = nn.log_softmax(seq_logits, axis=-1)
        token_log_probs = mx.take_along_axis(
            log_probs, seq_targets.reshape(seq_len, 1), axis=-1
        ).squeeze(-1)
        per_token_logps.append(token_log_probs.astype(mx.float32))

    mx.eval(logits)

    state = CURRENT_DEBUG_STATE
    if state is not None:
        state.context.token_counts = (lengths_np - 1).clip(min=0).tolist()
        state.context.empty_indices = empty_indices
        if empty_indices:
            preview = None
            if state.context.prompts_preview:
                preview = [state.context.prompts_preview[idx][:160] for idx in empty_indices]
            state.logger.log_event(
                "empty_completion_tokens",
                {
                    "step": state.context.current_step,
                    "indices": empty_indices,
                    "token_counts": state.context.token_counts,
                    "prompt_preview": preview,
                },
                level="warning",
            )
        else:
            state.context.empty_indices = []

    return per_token_logps


# ---------------------------------------------------------------------------
# Instrumented GRPO loss (guarding against zero-token divisions)
# ---------------------------------------------------------------------------


def debug_grpo_loss(
    model,
    ref_model,
    tokenizer,
    batch,
    completions=None,
    completion_texts=None,
    batch_indices=None,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    beta: float = 0.1,
    group_size: int = 4,
    epsilon: float = 1e-4,
    epsilon_high: float = None,
    max_tokens: int = 64,
    temperature: float = 0.8,
    reward_weights: Optional[List[float]] = None,
    batch_size: int = 1,
    importance_sampling_level: str = "token",
    grpo_loss_type: str = "grpo",
):
    state = CURRENT_DEBUG_STATE
    context = state.context if state else DebugContext()

    prompt_tokens, _, prompt_text, answer_text, type_info = batch
    context.prompts_preview = [p[:200] for p in prompt_text]
    context.answers_preview = [a[:200] for a in answer_text]

    if (
        completions is not None
        and completion_texts is not None
        and batch_indices is not None
    ):
        all_completions = completions
        all_completion_texts = completion_texts
        batch_indices = batch_indices
    else:
        # Custom generator that enforces min tokens per completion
        def _debug_generate_grpo(prompt_tokens_local):
            all_comps = []
            all_texts = []
            all_idx = []
            total = len(prompt_tokens_local)
            for i in range(0, total, batch_size):
                cur_bs = min(batch_size, total - i)
                batch_prompts = prompt_tokens_local[i : i + cur_bs]
                for j, prompt in enumerate(batch_prompts):
                    prompt_text = tokenizer.decode(prompt)
                    for k in range(group_size):
                        sampler = mlx_make_sampler(
                            temperature,
                            top_p=1.0,
                            min_p=0.0,
                            min_tokens_to_keep=max(1, state.config.min_new_tokens if state else 1),
                            top_k=0,
                            xtc_probability=0.0,
                        )
                        prompt_cache = mlx_cache.make_prompt_cache(model)
                        completion = mlx_generate(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt_text,
                            max_tokens=max_tokens,
                            verbose=False,
                            sampler=sampler,
                            prompt_cache=prompt_cache,
                        )
                        if isinstance(completion, str):
                            comp_ids = tokenizer.encode(completion)
                        else:
                            comp_ids = completion
                        comp_arr = mx.array(comp_ids)
                        all_comps.append(mx.stop_gradient(comp_arr))
                        all_texts.append(completion)
                        all_idx.append(i + j)
            return all_comps, all_texts, all_idx

        all_completions, all_completion_texts, batch_indices = _debug_generate_grpo(prompt_tokens)

    if not all_completions:
        raise ValueError("No completions were generated. Please check your model and inputs.")

    expanded_answers: List[str] = []
    expanded_prompts: List[str] = []
    expanded_types: List[Optional[str]] = []
    unique_prompt_indices = sorted(set(batch_indices))
    grouped_completions = {idx: [] for idx in unique_prompt_indices}

    for i, completion_idx in enumerate(batch_indices):
        grouped_completions[completion_idx].append(i)

    ordered_completions = []
    ordered_completion_texts = []
    ordered_batch_indices = []

    for prompt_idx in unique_prompt_indices:
        completion_indices = grouped_completions[prompt_idx]
        for idx in completion_indices:
            ordered_completions.append(all_completions[idx])
            ordered_completion_texts.append(all_completion_texts[idx])
            ordered_batch_indices.append(prompt_idx)
            expanded_answers.append(answer_text[prompt_idx])
            expanded_prompts.append(prompt_text[prompt_idx])
            expanded_types.append(type_info[prompt_idx] if type_info is not None else None)

    all_completions = ordered_completions
    all_completion_texts = ordered_completion_texts
    batch_indices = ordered_batch_indices
    max_length = max(ids.shape[0] for ids in all_completions) if all_completions else 0
    padded_completions = []
    attention_masks = []

    for completion_ids in all_completions:
        completion_tensor = mx.array(completion_ids.tolist())
        padding_length = max(0, max_length - completion_tensor.shape[0])
        if padding_length > 0:
            padding = mx.zeros((padding_length,), dtype=completion_tensor.dtype)
            padded_ids = mx.concatenate([completion_tensor, padding])
            mask = mx.concatenate([mx.ones_like(completion_tensor), mx.zeros_like(padding)])
        else:
            padded_ids = completion_tensor
            mask = mx.ones_like(completion_tensor)
        padded_completions.append(padded_ids)
        attention_masks.append(mask)

    inputs = mx.stack(padded_completions) if padded_completions else mx.zeros((1, 1), dtype=mx.int32)
    attention_mask = mx.stack(attention_masks) if attention_masks else mx.zeros((1, 1), dtype=mx.int32)
    lengths = attention_mask.sum(axis=1)

    token_log_probs = safe_get_per_token_logps(model, inputs, lengths)
    mx.eval(token_log_probs)

    if ref_model is None:
        ref_token_log_probs = token_log_probs
    else:
        ref_token_log_probs = safe_get_per_token_logps(ref_model, inputs, lengths)
        mx.eval(ref_token_log_probs)

    max_len = max((x.shape[0] for x in token_log_probs), default=0)
    padded_log_probs = []
    padded_ref_log_probs = []

    for i in range(len(token_log_probs)):
        seq_len = token_log_probs[i].shape[0]
        if seq_len == 0:
            padding = mx.zeros((max_len,), dtype=mx.float32)
            padded_log_probs.append(padding)
            padded_ref_log_probs.append(padding)
            continue
        padding_needed = max(0, max_len - seq_len)
        padding = mx.zeros((padding_needed,), dtype=mx.float32)
        padded_log_probs.append(mx.concatenate([token_log_probs[i], padding]))
        padded_ref_log_probs.append(mx.concatenate([ref_token_log_probs[i], padding]))

    token_log_probs = mx.stack(padded_log_probs) if padded_log_probs else mx.zeros((1, 1), dtype=mx.float32)
    ref_token_log_probs = mx.stack(padded_ref_log_probs) if padded_ref_log_probs else mx.zeros((1, 1), dtype=mx.float32)

    all_func_rewards = []
    for reward_func in reward_funcs:
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
            types=expanded_types,
        )
        if raw_rewards is None:
            processed_rewards = [float("nan")] * len(all_completion_texts)
        else:
            processed_rewards = [
                float(r) if r is not None else float("nan") for r in raw_rewards
            ]
        func_rewards = mx.array(processed_rewards, dtype=mx.float32)
        all_func_rewards.append(func_rewards)

    rewards = mx.stack(all_func_rewards, axis=1) if all_func_rewards else mx.zeros((len(all_completion_texts), 1), dtype=mx.float32)

    if reward_weights is not None:
        if len(reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Number of reward weights ({len(reward_weights)}) must match number of reward functions ({len(reward_funcs)})"
            )
        reward_weights = mx.array(reward_weights, dtype=mx.float32)
    else:
        reward_weights = mx.ones(len(reward_funcs), dtype=mx.float32)

    valid_reward_mask = ~mx.isnan(rewards)
    rewards_no_nan = mx.where(valid_reward_mask, rewards, mx.zeros_like(rewards))
    rewards = (rewards_no_nan * mx.expand_dims(reward_weights, 0)).sum(axis=1)

    num_unique_prompts = len(unique_prompt_indices)
    rewards_by_prompt = [[] for _ in range(num_unique_prompts)]
    for i, prompt_idx in enumerate(batch_indices):
        prompt_position = unique_prompt_indices.index(prompt_idx)
        rewards_by_prompt[prompt_position].append(rewards[i])

    advantages = mx.zeros_like(rewards)
    for i, prompt_rewards in enumerate(rewards_by_prompt):
        if len(prompt_rewards) > 1:
            prompt_rewards = mx.array(prompt_rewards)
            mean_reward = mx.mean(prompt_rewards)
            std_reward = mx.std(prompt_rewards)
            indices = [
                j for j, idx in enumerate(batch_indices) if idx == unique_prompt_indices[i]
            ]
            for j, idx in enumerate(indices):
                advantages[idx] = (prompt_rewards[j] - mean_reward) / (std_reward + 1e-4)
        else:
            idx = batch_indices.index(unique_prompt_indices[i])
            advantages[idx] = 0.0

    kl_div = mx.exp(ref_token_log_probs - token_log_probs) - (
        ref_token_log_probs - token_log_probs
    ) - 1

    length_mask = mx.arange(token_log_probs.shape[1])[None, :] < mx.maximum(lengths[:, None] - 1, 0)
    valid_token_counts = mx.maximum(length_mask.sum(axis=1), 1.0)
    total_valid_tokens = mx.maximum(length_mask.sum(), 1.0)

    log_ratio = token_log_probs - mx.stop_gradient(ref_token_log_probs)
    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        sequence_log_ratio = (log_ratio * length_mask).sum(axis=1) / valid_token_counts
        log_importance_weights = mx.expand_dims(sequence_log_ratio, axis=1)
    elif importance_sampling_level in (None, "none"):
        log_importance_weights = mx.zeros_like(log_ratio)
    else:
        raise ValueError(
            f"Unknown importance sampling level: {importance_sampling_level}. "
            "Choose 'token', 'sequence', or None."
        )

    coef_1 = mx.exp(log_importance_weights)
    epsilon_high = epsilon_high if epsilon_high is not None else epsilon
    coef_2 = mx.clip(coef_1, 1 - epsilon, 1 + epsilon_high)

    is_low_clipped = (coef_1 < 1 - epsilon) & (advantages.reshape(-1, 1) < 0)
    is_high_clipped = (coef_1 > 1 + epsilon_high) & (advantages.reshape(-1, 1) > 0)
    is_region_clipped = is_low_clipped | is_high_clipped

    unclipped_obj = coef_1 * advantages.reshape(-1, 1)
    clipped_obj = coef_2 * advantages.reshape(-1, 1)

    per_token_loss = -mx.minimum(unclipped_obj, clipped_obj)
    if beta != 0.0:
        per_token_loss = per_token_loss + beta * kl_div

    if grpo_loss_type == "grpo":
        loss = (per_token_loss * length_mask).sum() / total_valid_tokens
    elif grpo_loss_type == "bnpo":
        loss = (per_token_loss * length_mask).sum() / total_valid_tokens
    elif grpo_loss_type == "dr_grpo":
        loss = (per_token_loss * length_mask).sum() / (
            per_token_loss.shape[0] * max_tokens if max_tokens else 1
        )
    else:
        raise ValueError(f"Unknown loss type: {grpo_loss_type}")

    denom = mx.maximum(valid_token_counts, 1.0)
    mean_kl = ((kl_div * length_mask).sum(axis=1) / denom).mean()

    reward_metrics: Dict[str, Any] = {}
    for i, reward_func in enumerate(reward_funcs):
        func_name = reward_func.__name__
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
        )
        raw_rewards_arr = [r for r in raw_rewards if r is not None]
        if raw_rewards_arr:
            arr = np.array(raw_rewards_arr, dtype=np.float32)
            reward_metrics[f"{func_name}_mean"] = float(arr.mean())
            reward_metrics[f"{func_name}_std"] = float(arr.std()) if arr.size > 1 else 0.0
            reward_metrics[f"{func_name}_coverage"] = float(len(raw_rewards_arr) / len(raw_rewards))
        else:
            reward_metrics[f"{func_name}_mean"] = float("nan")
            reward_metrics[f"{func_name}_std"] = float("nan")
            reward_metrics[f"{func_name}_coverage"] = 0.0

    grouped_rewards_mean = [mx.mean(mx.array(rewards)) for rewards in rewards_by_prompt]
    grouped_rewards_std = [
        mx.std(mx.array(rewards)) if len(rewards) > 1 else mx.zeros(1)
        for rewards in rewards_by_prompt
    ]

    metrics_mx = {
        "total_rewards_mean": mx.mean(rewards),
        "total_rewards_std": mx.std(rewards) if rewards.size else mx.zeros(1),
        "grouped_rewards_mean": mx.mean(mx.array(grouped_rewards_mean))
        if grouped_rewards_mean
        else mx.zeros(1),
        "grouped_rewards_std": mx.mean(mx.array(grouped_rewards_std))
        if grouped_rewards_std
        else mx.zeros(1),
        "kl": mean_kl,
        "average_generated_tokens": float(max_tokens),
        "clip_ratio_low": (is_low_clipped * length_mask).sum() / total_valid_tokens,
        "clip_ratio_high": (is_high_clipped * length_mask).sum() / total_valid_tokens,
        "clip_ratio_total": (is_region_clipped * length_mask).sum() / total_valid_tokens,
        **{k: mx.array(v) if not isinstance(v, mx.array) else v for k, v in reward_metrics.items()},
    }

    metrics_py = {k: float(np.array(v)) for k, v in metrics_mx.items()}

    if state is not None:
        context.metrics = metrics_py
        context.batch_indices = list(batch_indices)
        if context.empty_indices is None:
            context.empty_indices = []
        state.context = context

    mx.clear_cache()

    return loss, length_mask.sum(axis=1).sum(), metrics_mx


# ---------------------------------------------------------------------------
# Gradient statistics helpers
# ---------------------------------------------------------------------------


def _flatten_arrays(tree: Any) -> Iterable[np.ndarray]:
    if tree is None:
        return []
    flat = mx_utils.tree_flatten(tree)
    arrays: List[np.ndarray] = []
    for arr in flat:
        if arr is None:
            continue
        try:
            np_arr = np.array(arr, dtype=np.float64)
        except Exception:
            continue
        arrays.append(np_arr)
    return arrays


def _grad_stats(grad_tree: Any) -> Tuple[float, int]:
    arrays = _flatten_arrays(grad_tree)
    if not arrays:
        return 0.0, 0
    sq_sum = 0.0
    nan_count = 0
    for arr in arrays:
        nan_mask = np.isnan(arr)
        nan_count += int(nan_mask.sum())
        if nan_mask.any():
            arr = np.nan_to_num(arr, nan=0.0)
        sq_sum += float(np.square(arr).sum())
    return float(np.sqrt(sq_sum)), nan_count


def _param_norm(model) -> float:
    arrays = _flatten_arrays(model.trainable_parameters())
    if not arrays:
        return 0.0
    sq_sum = sum(float(np.square(arr).sum()) for arr in arrays)
    return float(np.sqrt(sq_sum))


# ---------------------------------------------------------------------------
# Instrumented GRPO training loop
# ---------------------------------------------------------------------------


def debug_train_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    args: GRPOTrainingArgs = GRPOTrainingArgs(),
    loss_fn: callable = debug_grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
    training_callback: TrainingCallback = None,
):
    state = CURRENT_DEBUG_STATE
    if state is None:
        raise RuntimeError("Debug state not initialised")

    logger = state.logger
    cfg = state.config
    context = state.context

    if reward_funcs is None:
        reward_funcs = get_default_reward_functions()

    world = mx.distributed.init()
    rank = world.rank()
    world_size = world.size()

    logger.log_event(
        "train_start",
        {
            "iters": args.iters,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "importance_sampling_level": args.importance_sampling_level,
            "beta": args.beta,
            "epsilon": args.epsilon,
        },
    )

    if cfg.disable_eval:
        logger.log_event("debug_flag", {"disable_eval": True}, level="warning")
    if cfg.disable_checkpoints:
        logger.log_event("debug_flag", {"disable_checkpoints": True}, level="warning")

    batch_iterator = iterate_batches(
        dataset=train_dataset,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        train=True,
    )

    losses = mx.array(0.0)
    n_tokens = mx.array(0.0)
    steps_accumulated = 0
    trained_tokens = 0
    grad_buffer = None
    accum_count = 0
    accumulated_metrics: Dict[str, float] = {}
    for reward_func in reward_funcs or []:
        name = reward_func.__name__
        accumulated_metrics[f"{name}_mean"] = 0.0
        accumulated_metrics[f"{name}_std"] = 0.0
        accumulated_metrics[f"{name}_coverage"] = 0.0
    for key in [
        "total_rewards_mean",
        "total_rewards_std",
        "grouped_rewards_mean",
        "grouped_rewards_std",
        "kl",
        "average_generated_tokens",
        "clip_ratio_low",
        "clip_ratio_high",
        "clip_ratio_total",
    ]:
        accumulated_metrics.setdefault(key, 0.0)

    param_norm = _param_norm(model)

    def _prepare_batch() -> Tuple[Any, ...]:
        nonlocal batch_iterator
        try:
            return next(batch_iterator)
        except StopIteration:
            batch_iterator = iterate_batches(
                dataset=train_dataset,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                train=True,
            )
            return next(batch_iterator)

    def _average_gradients(grad_tree):
        return average_gradients(grad_tree)

    start_wall = time.perf_counter()
    tokens_since_log = 0

    for it in range(1, args.iters + 1):
        context.current_step = it
        batch = _prepare_batch()
        context.batch_indices = None  # will be populated by loss
        context.val_loss = None
        context.prompts_preview = None
        context.answers_preview = None

        # Optional evaluation
        if not cfg.disable_eval and (
            it == 1 or it % args.steps_per_eval == 0 or it == args.iters
        ):
            eval_start = time.perf_counter()
            val_loss, val_tokens, val_metrics = evaluate_grpo(
                model=model,
                ref_model=ref_model,
                dataset=val_dataset,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                beta=args.beta,
                epsilon=args.epsilon,
                epsilon_high=args.epsilon_high,
                group_size=args.group_size,
                max_seq_length=args.max_seq_length,
                max_tokens=args.max_completion_length,
                temperature=args.temperature,
                reward_funcs=reward_funcs,
                loss_fn=loss_fn,
                iterate_batches=iterate_batches,
                grpo_loss_type=args.grpo_loss_type,
                importance_sampling_level=args.importance_sampling_level,
            )
            eval_time = time.perf_counter() - eval_start
            context.val_loss = float(val_loss)
            logger.log_event(
                "eval",
                {
                    "step": it,
                    "val_loss": float(val_loss),
                    "val_tokens": int(val_tokens),
                    "metrics": {k: float(np.array(v)) for k, v in val_metrics.items()},
                    "elapsed_sec": eval_time,
                },
            )
            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {
                        "iteration": it,
                        "val_loss": float(val_loss),
                        "val_time": eval_time,
                    }
                )

        step_start = time.perf_counter()
        (loss_value, toks, metrics), grad = nn.value_and_grad(model, loss_fn)(
            model,
            tokenizer=tokenizer,
            batch=batch,
            reward_funcs=reward_funcs,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            epsilon_high=args.epsilon_high,
            ref_model=ref_model,
            grpo_loss_type=args.grpo_loss_type,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            reward_weights=args.reward_weights,
            batch_size=args.batch_size,
            importance_sampling_level=args.importance_sampling_level,
        )

        losses = losses + loss_value
        n_tokens = n_tokens + toks
        steps_accumulated += 1
        tokens_since_log += float(np.array(toks))

        metrics_py = {k: float(np.array(v)) for k, v in metrics.items()}
        for k in accumulated_metrics.keys():
            accumulated_metrics[k] += metrics_py.get(k, 0.0)
        context.metrics = metrics_py
        context.raw_loss = float(np.array(loss_value))

        grad_buffer = (
            grad if grad_buffer is None else mx_utils.tree_map(lambda a, b: a + b, grad_buffer, grad)
        )
        accum_count += 1

        grad_norm_step, grad_nan_count = _grad_stats(grad_buffer)
        context.grad_norm = grad_norm_step
        context.grad_nan_count = grad_nan_count

        update_performed = False
        if accum_count >= args.gradient_accumulation_steps:
            grad_avg = mx_utils.tree_map(
                lambda g: g / args.gradient_accumulation_steps, grad_buffer
            )
            grad_avg = _average_gradients(grad_avg)
            grad_norm_step, grad_nan_count = _grad_stats(grad_avg)
            context.grad_norm = grad_norm_step
            context.grad_nan_count = grad_nan_count

            if grad_nan_count and cfg.detect_nans:
                logger.log_event(
                    "nan_gradient",
                    {
                        "step": it,
                        "nan_count": grad_nan_count,
                        "grad_norm": grad_norm_step,
                    },
                    level="error",
                )
                raise FloatingPointError(
                    f"Detected NaNs in gradients at step {it}"
                )

            optimizer.update(model, grad_avg)
            update_performed = True
            grad_buffer = None
            accum_count = 0
            param_norm = _param_norm(model)

        mx.eval(losses, n_tokens)

        # Reporting at same cadence as base trainer
        if it % args.steps_per_report == 0 or it == args.iters:
            wall = time.perf_counter() - start_wall
            train_loss = float(mx.distributed.all_sum(losses).item()) / (steps_accumulated * max(world_size, 1))
            total_tokens = float(mx.distributed.all_sum(n_tokens).item())
            learning_rate = float(optimizer.learning_rate.item())
            it_per_sec = steps_accumulated / max(wall, 1e-6)
            tokens_per_sec = total_tokens / max(wall, 1e-6)
            trained_tokens += int(total_tokens)
            peak_mem = mx.get_peak_memory() / 1e9

            avg_metrics = {
                k: v / max(steps_accumulated, 1) for k, v in accumulated_metrics.items()
            }

            step_record = {
                "step": it,
                "train_loss": train_loss,
                "step_loss": context.raw_loss,
                "grad_norm": context.grad_norm,
                "grad_nan_count": context.grad_nan_count,
                "param_norm": param_norm,
                "update_performed": update_performed,
                "lr": learning_rate,
                "it_per_sec": it_per_sec,
                "tokens_per_sec": tokens_per_sec,
                "trained_tokens": trained_tokens,
                "peak_memory_gb": peak_mem,
                "metrics": avg_metrics,
                "token_count_summary": {
                    "min": min(context.token_counts or [0]),
                    "max": max(context.token_counts or [0]),
                    "mean": float(np.mean(context.token_counts)) if context.token_counts else 0.0,
                    "empty_count": len(context.empty_indices or []),
                },
            }
            logger.log_step(step_record)

            if training_callback is not None:
                training_callback.on_train_loss_report(
                    {
                        "iteration": it,
                        "train_loss": train_loss,
                        **{f"train_{k}": v for k, v in avg_metrics.items()},
                        "learning_rate": learning_rate,
                        "iterations_per_second": it_per_sec,
                        "tokens_per_second": tokens_per_sec,
                        "trained_tokens": trained_tokens,
                        "peak_memory": peak_mem,
                    }
                )

            losses = mx.array(0.0)
            n_tokens = mx.array(0.0)
            steps_accumulated = 0
            accumulated_metrics = {k: 0.0 for k in accumulated_metrics}
            start_wall = time.perf_counter()
            tokens_since_log = 0

        if not cfg.disable_checkpoints and (it % args.steps_per_save == 0 or it == args.iters):
            adapter_weights = dict(mx_utils.tree_flatten(model.trainable_parameters()))
            path = Path(args.adapter_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(path), adapter_weights)
            checkpoint = path.parent / f"{it:07d}_adapters.safetensors"
            mx.save_safetensors(str(checkpoint), adapter_weights)
            logger.log_event(
                "checkpoint",
                {"step": it, "adapter": str(path), "checkpoint": str(checkpoint)},
            )

    if not cfg.disable_checkpoints:
        adapter_weights = dict(mx_utils.tree_flatten(model.trainable_parameters()))
        path = Path(args.adapter_file)
        mx.save_safetensors(str(path), adapter_weights)
        logger.log_event("checkpoint", {"step": args.iters, "adapter": str(path), "final": True})


# ---------------------------------------------------------------------------
# Training callback to mirror metrics into debug logs
# ---------------------------------------------------------------------------


class DebugTrainingCallback(TrainingCallback):
    def __init__(self, logger: DebugLogger) -> None:
        self.logger = logger

    def on_train_loss_report(self, train_info: dict):
        payload = {k: (float(np.array(v)) if hasattr(v, "__array__") else v) for k, v in train_info.items()}
        self.logger.log_event("train_report", payload)

    def on_val_loss_report(self, val_info: dict):
        payload = {k: (float(np.array(v)) if hasattr(v, "__array__") else v) for k, v in val_info.items()}
        self.logger.log_event("val_report", payload)


# ---------------------------------------------------------------------------
# Dataset loader patch (preflight)
# ---------------------------------------------------------------------------


def patched_load_dataset(args, tokenizer):
    train, valid, test = base_load_dataset(args, tokenizer)
    state = CURRENT_DEBUG_STATE
    if state is not None:
        summaries = []
        try:
            summaries.append(run_preflight_scan(train, "train", state.logger, state.config.preflight_limit))
            summaries.append(run_preflight_scan(valid, "valid", state.logger, state.config.preflight_limit))
        except Exception as exc:  # pragma: no cover - defensive logging
            state.logger.log_event(
                "preflight_error",
                {"error": str(exc)},
                level="error",
            )
        state.run_metadata["preflight"] = summaries
    return train, valid, test


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_debug_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.add_argument(
        "--debug-log-root",
        type=str,
        default=None,
        help="Directory to store debug artifacts. Defaults to <adapter_path>/debug_runs.",
    )
    parser.add_argument(
        "--debug-preflight-limit",
        type=int,
        default=2000,
        help="Maximum examples to inspect per split during preflight validation.",
    )
    parser.add_argument(
        "--debug-keep-steps",
        type=int,
        default=32,
        help="Recent step records to retain for crash bundles.",
    )
    parser.add_argument(
        "--debug-no-sequence-fallback",
        action="store_true",
        help="Disable automatic fallback to sequence-level importance sampling when empty token batches are detected.",
    )
    parser.add_argument(
        "--debug-disable-eval",
        action="store_true",
        help="Skip validation passes (useful when bisecting eval-related crashes).",
    )
    parser.add_argument(
        "--debug-disable-checkpoints",
        action="store_true",
        help="Do not write periodic adapter checkpoints during debug runs.",
    )
    parser.add_argument(
        "--debug-min-new-tokens",
        type=int,
        default=1,
        help="Minimum tokens to sample per completion when debugging (safeguard against immediate EOS).",
    )
    parser.add_argument(
        "--debug-skip-nan-check",
        action="store_true",
        help="Do not raise when NaNs are detected in gradients (default: raise).",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_debug_training(argv: Optional[List[str]] = None) -> None:
    parser = build_debug_parser()
    args = parser.parse_args(argv)

    def _json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        if isinstance(value, dict):
            return {k: _json_safe(v) for k, v in value.items()}
        if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
            try:
                return _json_safe(vars(value))
            except Exception:
                return str(value)
        return value

    debug_cfg = DebugConfig(
        log_root=Path(args.debug_log_root) if args.debug_log_root else None,
        preflight_limit=args.debug_preflight_limit,
        keep_recent_steps=args.debug_keep_steps,
        fallback_sequence_on_empty=not args.debug_no_sequence_fallback,
        disable_eval=args.debug_disable_eval,
        disable_checkpoints=args.debug_disable_checkpoints,
        min_new_tokens=args.debug_min_new_tokens,
        detect_nans=not args.debug_skip_nan_check,
    )

    debug_keys = {
        "debug_log_root",
        "debug_preflight_limit",
        "debug_keep_steps",
        "debug_no_sequence_fallback",
        "debug_disable_eval",
        "debug_disable_checkpoints",
        "debug_min_new_tokens",
        "debug_skip_nan_check",
    }

    base_args_dict = {k: v for k, v in vars(args).items() if k not in debug_keys}
    base_args = types.SimpleNamespace(**base_args_dict)

    # Merge CONFIG_DEFAULTS similar to mlx_lm_lora.train.main
    for key, value in CONFIG_DEFAULTS.items():
        if getattr(base_args, key, None) is None:
            setattr(base_args, key, value)

    log_root = debug_cfg.log_root or Path(base_args.adapter_path) / "debug_runs"
    run_dir = log_root / _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = DebugLogger(run_dir, keep_recent=debug_cfg.keep_recent_steps)

    metadata = {
        "argv": sys.argv,
        "timestamp": _dt.datetime.utcnow().isoformat(),
        "workdir": str(Path.cwd()),
        "python": sys.version,
        "device": _json_safe(mx.metal.device_info()) if mx.metal.is_available() else "cpu",
        "config": {k: _json_safe(getattr(base_args, k)) for k in CONFIG_DEFAULTS.keys()},
        "debug": _json_safe(asdict(debug_cfg)),
    }

    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    global CURRENT_DEBUG_STATE
    CURRENT_DEBUG_STATE = DebugState(logger=logger, config=debug_cfg, run_metadata=metadata)

    # Patch functions
    original_train_grpo = grpo_trainer.train_grpo
    original_get_per_token_logps = grpo_trainer.get_per_token_logps
    original_load_dataset = datasets_module.load_dataset

    grpo_trainer.train_grpo = lambda *a, **k: debug_train_grpo(*a, **k)
    grpo_trainer.get_per_token_logps = safe_get_per_token_logps
    datasets_module.load_dataset = patched_load_dataset
    try:
        debug_callback = DebugTrainingCallback(logger)
        base_run(base_args, training_callback=debug_callback)
    except Exception as exc:
        logger.capture_crash(exc, CURRENT_DEBUG_STATE.context)
        raise
    finally:
        grpo_trainer.train_grpo = original_train_grpo
        grpo_trainer.get_per_token_logps = original_get_per_token_logps
        datasets_module.load_dataset = original_load_dataset
        logger.close()
        CURRENT_DEBUG_STATE = None


def main() -> None:  # pragma: no cover - CLI entry
    run_debug_training()


if __name__ == "__main__":  # pragma: no cover
    main()
