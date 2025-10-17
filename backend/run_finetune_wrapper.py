#!/usr/bin/env python3
"""
Lightweight wrapper for MLX LoRA finetuning used by the standard Setup flow.

Reads a YAML config (produced by backend/main.py), builds the mlx-lm lora
command, and streams output while supporting optional early stopping.

Adds support for gradient accumulation via the `grad_accumulation_steps`
field in the config.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from typing import Any, List, Tuple

import yaml


def _parse_version(version_str: str) -> Tuple[int, ...]:
    parts = []
    for token in version_str.split("."):
        num = ''.join(ch for ch in token if ch.isdigit())
        if num:
            parts.append(int(num))
        else:
            break
    return tuple(parts)


def _supports_grad_accumulation() -> Tuple[bool, str]:
    try:
        import mlx_lm

        version_str = getattr(mlx_lm, "__version__", "0.0.0")
        return _parse_version(version_str) >= (0, 28, 3), version_str
    except Exception:
        return False, "unknown"


def _run(cmd: List[str]) -> subprocess.Popen:
    print("Running:", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid,
    )


def _build_lora_config(cfg: dict[str, Any]) -> str:
    """Write a temporary LoRA YAML config and return its path."""
    if isinstance(cfg.get("lora_parameters"), dict):
        lora_config = {
            "num_layers": cfg.get("num_layers", -1),
            "lora_parameters": cfg["lora_parameters"],
        }
    else:
        # Fallback: attention-only or full depending on provided hint
        lora_layers_setting = cfg.get("lora_layers", "all")
        if lora_layers_setting == "all":
            lora_keys = [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]
        else:
            lora_keys = ["self_attn.q_proj", "self_attn.v_proj"]
        lora_config = {
            "num_layers": cfg.get("num_layers", -1),
            "lora_parameters": {
                "rank": cfg.get("lora_rank", 8),
                "scale": cfg.get("lora_scale", 20.0),
                "dropout": cfg.get("lora_dropout", 0.0),
                "keys": lora_keys,
            },
        }

    path = "/tmp/lora_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(lora_config, f, default_flow_style=False)
    return path


def _build_cmd(
    cfg: dict[str, Any],
    data_dir: str,
    adapter_dir: str,
    lora_config_path: str,
    supports_grad_accum: bool,
) -> List[str]:
    python = cfg["venv_python"]
    cmd: List[str] = [
        python,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        cfg["base_model_dir"],
        "--train",
        "--data",
        data_dir,
        "--config",
        lora_config_path,
        "--fine-tune-type",
        str(cfg.get("fine_tune_type", "lora")),
        "--optimizer",
        str(cfg.get("optimizer", "adamw")),
        "--batch-size",
        str(cfg.get("batch_size", 1)),
        "--iters",
        str(cfg.get("iters", 100)),
        "--val-batches",
        str(cfg.get("val_batches", -1)),
        "--learning-rate",
        str(cfg.get("learning_rate", 1e-4)),
        "--steps-per-report",
        str(cfg.get("steps_per_report", 25)),
        "--steps-per-eval",
        str(cfg.get("steps_per_eval", 25)),
        "--max-seq-length",
        str(cfg.get("max_seq_length", 2048)),
        "--adapter-path",
        adapter_dir,
        "--save-every",
        str(cfg.get("save_every", 100)),
    ]

    # Optional flags
    if "num_layers" in cfg:
        cmd += ["--num-layers", str(cfg["num_layers"])]
    if bool(cfg.get("grad_checkpoint", True)):
        cmd.append("--grad-checkpoint")
    if bool(cfg.get("mask_prompt", False)):
        cmd.append("--mask-prompt")
    if cfg.get("resume_adapter_file"):
        cmd += ["--resume-adapter-file", str(cfg["resume_adapter_file"])]

    # New: gradient accumulation for effective larger batch size
    gas = int(cfg.get("grad_accumulation_steps", 1) or 1)
    if gas > 1:
        if supports_grad_accum:
            cmd += ["--grad-accumulation-steps", str(gas)]
        else:
            print(
                "⚠️  Gradient accumulation requested but current mlx-lm does not support "
                "--grad-accumulation-steps. Update to >=0.28.3. Falling back to 1."
            )
            gas = 1
    cfg["grad_accumulation_steps"] = gas

    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Run MLX LoRA finetune (with grad accumulation support)")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # Resolve paths
    data_dir = cfg.get("prepared_data_dir") or cfg.get("out_data_dir") or cfg.get("data_dir")
    if not data_dir:
        print("ERROR: prepared_data_dir not found in config")
        sys.exit(2)
    adapter_dir = os.path.join(cfg["adapter_output_dir"], cfg["adapter_name"])
    os.makedirs(adapter_dir, exist_ok=True)

    # LoRA config and command
    lora_cfg = _build_lora_config(cfg)
    supports_grad_accum, version_str = _supports_grad_accumulation()
    if not supports_grad_accum and int(cfg.get("grad_accumulation_steps", 1) or 1) > 1:
        print(
            f"⚠️  Detected mlx-lm {version_str}; gradient accumulation requires >=0.28.3. "
            "Proceeding without accumulation."
        )
    cmd = _build_cmd(cfg, data_dir, adapter_dir, lora_cfg, supports_grad_accum)

    # Early stopping parameters
    enable_es = bool(cfg.get("enable_early_stop", False))
    patience = int(cfg.get("no_improve_patience_evals", 3))
    min_delta = float(cfg.get("early_stop_min_delta", 0.0) or 0.0)
    val_re = re.compile(r"Val loss\s+([0-9.]+)")
    best_val = float("inf")
    no_improve = 0
    early_stopped = False
    last_step = None

    if enable_es:
        print(f"[EarlyStop] enabled (patience={patience}, min_delta={min_delta})")
    else:
        print("[EarlyStop] disabled")

    log_path = cfg.get("train_log", "/tmp/train_one_step.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def stream_process(process: subprocess.Popen) -> int:
        nonlocal no_improve, early_stopped
        try:
            assert process.stdout is not None
            with open(log_path, "a", encoding="utf-8") as logf:
                for line in process.stdout:
                    print(line, end="")
                    logf.write(line)
                    logf.flush()
                    if enable_es:
                        m = val_re.search(line)
                        if m:
                            try:
                                val = float(m.group(1))
                                step_match = re.search(r"Iter\s+(\d+)", line)
                                step = int(step_match.group(1)) if step_match else None

                                if step is not None:
                                    if last_step is not None and step <= last_step:
                                        # Skip duplicate/out-of-order metrics
                                        continue
                                    last_step = step

                                if val < best_val - min_delta:
                                    best_val = val
                                    no_improve = 0
                                else:
                                    no_improve += 1
                                    if no_improve >= patience:
                                        early_stopped = True
                                        trigger_step = step if step is not None else "unknown"
                                        print(
                                            f"Early stop: no val improvement for {no_improve} evals (best={best_val:.3f} at step {last_step})."
                                        )
                                        try:
                                            os.killpg(process.pid, signal.SIGINT)
                                            process.wait(timeout=60)
                                        except Exception:
                                            try:
                                                os.killpg(process.pid, signal.SIGTERM)
                                                process.wait(timeout=30)
                                            except Exception:
                                                try:
                                                    os.killpg(process.pid, signal.SIGKILL)
                                                except Exception:
                                                    pass
                                        break
                            except Exception:
                                pass
        finally:
            rc = process.poll()
            if rc is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except Exception:
                        pass
        return process.returncode if process.returncode is not None else 1

    proc = _run(cmd)
    exit_code = stream_process(proc)

    # Retry without grad accumulation if it failed and we attempted it
    if exit_code != 0 and int(cfg.get("grad_accumulation_steps", 1) or 1) > 1:
        print(
            f"⚠️  Training exited with code {exit_code}. Retrying with grad accumulation disabled."
        )
        cfg["grad_accumulation_steps"] = 1
        retry_cmd = _build_cmd(cfg, data_dir, adapter_dir, lora_cfg, False)
        proc = _run(retry_cmd)
        exit_code = stream_process(proc)

    if early_stopped:
        print("Training completed successfully via early stopping.")
        sys.exit(0)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
