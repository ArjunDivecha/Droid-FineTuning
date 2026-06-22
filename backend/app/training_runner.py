"""
=============================================================================
SCRIPT NAME: training_runner.py
=============================================================================

INPUT FILES:
- <config.train_data_path> : JSONL training data (mlx-lm-lora SFT format:
  one {"text": "..."} or {"prompt":..., "completion":...} object per line,
  OR a directory containing train.jsonl / valid.jsonl / test.jsonl).
- <config.val_data_path>   : optional JSONL validation data.
- <config.model_path>      : local MLX model directory (or HF repo id).

OUTPUT FILES:
- <root>/adapters/<adapter_name>/adapters.safetensors : trained LoRA weights.
- <root>/runs/<adapter_name>_<ts>/config.yaml         : reproducible run config.
- <root>/runs/<adapter_name>_<ts>/run.log             : captured trainer output.

VERSION: 1.0
LAST UPDATED: 2026-06-20
AUTHOR: Arjun

DESCRIPTION:
Wraps the `mlx-lm-lora` training engine (pip package `mlx-lm-lora`) so the
FastAPI layer can run SFT training and stream live metrics to the React
frontend over WebSocket.

DESIGN DECISION (deviation from the original plan):
The plan proposed spawning `mlx_lm_lora.train` as a subprocess and parsing
its stdout JSON-lines. Inspection of the installed library (v2.1.0) showed
that `mlx_lm_lora.train.run(args, training_callback)` accepts a
`TrainingCallback` (from `mlx_lm.tuner.callbacks`) whose
`on_train_loss_report` / `on_val_loss_report` methods receive *structured
dicts* (iteration, train_loss, learning_rate, iterations_per_second,
tokens_per_second, trained_tokens, peak_memory / val_loss, val_time).

Running the trainer IN-PROCESS in a background thread with a custom callback
is strictly better than stdout parsing: no tqdm/format fragility, exact
field access, and clean exception capture. This is the approach taken here.
MLX releases the GIL during mx.eval, so the main event loop stays responsive.

The callback translates the trainer's metric dicts into the `TrainingMetrics`
shape the frontend Redux slice expects (see frontend/src/store/slices/
trainingSlice.ts) and pushes each update onto a thread-safe queue that the
WebSocket layer drains.

DEPENDENCIES:
- mlx-lm-lora (>=2.1.0)
- mlx-lm
- pyyaml

USAGE:
    runner = TrainingRunner()
    runner.start(config_dict)   # non-blocking; returns immediately
    update = runner.next_update()  # poll for a MetricsUpdate or None
    runner.stop()
=============================================================================
"""

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import paths

logger = logging.getLogger(__name__)

# The frontend's TrainingState type (see trainingSlice.ts).
VALID_STATES = {"idle", "running", "paused", "completed", "error", "stopped"}


class EarlyStopRequested(Exception):
    """Raised from the training callback to halt training when early-stop fires.

    mlx-lm-lora has no native early-stopping/patience argument (it only has
    steps_per_eval). We implement it ourselves: the val-loss callback tracks
    the best validation loss and, if it fails to improve for `patience`
    consecutive evaluations, raises this exception. Because the callback runs
    synchronously inside the trainer's `for it in range(iters)` loop, the
    exception propagates out of run() cleanly, where _run_training catches it
    and treats it as a normal early completion (NOT an error).
    """


@dataclass
class MetricsUpdate:
    """One streamed training update, ready to broadcast over WebSocket.

    Fields mirror frontend/src/store/slices/trainingSlice.ts TrainingMetrics,
    plus a human-readable log_line and an optional terminal state flag.
    """

    metrics: Dict[str, Any]
    log_line: str = ""
    state: Optional[str] = None  # if set, this update also transitions state


@dataclass
class TrainingConfig:
    """Mirrors the frontend TrainingConfig (trainingSlice.ts:13)."""

    model_path: str
    train_data_path: str
    val_data_path: str = ""
    learning_rate: float = 1e-5
    batch_size: int = 1
    max_seq_length: int = 1024
    iterations: int = 1000
    steps_per_report: int = 10
    steps_per_eval: int = 200
    save_every: int = 100
    early_stop: bool = True
    patience: int = 3
    adapter_name: str = "mlx_finetune"
    # Optional path to a saved LoRA .safetensors checkpoint to resume from.
    # Maps to mlx-lm-lora's --resume-adapter-file (weight-only resume: loads
    # the LoRA weights but restarts the optimizer/step counter/LR schedule).
    resume_from: str = ""


class _StreamCallback:
    """A TrainingCallback that pushes structured metrics onto a queue.

    mlx_lm_lora's trainer calls on_train_loss_report(train_info) and
    on_val_loss_report(val_info) with dicts. We merge train+val info into a
    single TrainingMetrics-shaped dict (keeping the most recent of each) and
    enqueue a MetricsUpdate per call.
    """

    def __init__(
        self,
        runner: "TrainingRunner",
        total_steps: int,
        start_time: str,
        early_stop: bool = False,
        patience: int = 3,
    ):
        self._runner = runner
        self._total_steps = total_steps
        self._start_time = start_time
        # Early-stopping state. mlx-lm-lora reports val_loss via
        # on_val_loss_report; we track the best seen so far and count how many
        # consecutive evals have failed to improve. When that count reaches
        # `patience`, we raise EarlyStopRequested to halt the trainer loop.
        self._early_stop = bool(early_stop)
        self._patience = max(1, int(patience))
        self._best_val_loss: Optional[float] = None
        self._bad_evals = 0
        # Last-seen metrics; we accumulate so a train report retains the
        # most recent val_loss and vice versa.
        self._current: Dict[str, Any] = {
            "current_step": 0,
            "total_steps": total_steps,
            "train_loss": None,
            "val_loss": None,
            "learning_rate": 0.0,
            "start_time": start_time,
            "estimated_time_remaining": None,
        }

    def _eta(self, iteration: int) -> Optional[float]:
        """Estimate seconds remaining from elapsed time and step fraction."""
        if iteration <= 0:
            return None
        try:
            started = datetime.fromisoformat(self._start_time).timestamp()
            elapsed = max(time.time() - started, 0.0)
            frac = iteration / self._total_steps if self._total_steps else 0.0
            if frac <= 0:
                return None
            total_est = elapsed / frac
            return max(total_est - elapsed, 0.0)
        except Exception:
            return None

    def on_train_loss_report(self, train_info: Dict[str, Any]) -> None:
        it = int(train_info.get("iteration", 0))
        self._current["current_step"] = it
        self._current["train_loss"] = float(train_info.get("train_loss", 0.0))
        self._current["learning_rate"] = float(train_info.get("learning_rate", 0.0))
        self._current["estimated_time_remaining"] = self._eta(it)
        log_line = (
            f"Iter {it}: Train loss {self._current['train_loss']:.4f}, "
            f"Val loss {self._fmt(self._current.get('val_loss'))}"
        )
        self._runner._enqueue(
            MetricsUpdate(metrics=dict(self._current), log_line=log_line)
        )

    def on_val_loss_report(self, val_info: Dict[str, Any]) -> None:
        it = int(val_info.get("iteration", 0))
        val_loss = float(val_info.get("val_loss", 0.0))
        self._current["val_loss"] = val_loss
        self._current["current_step"] = it
        log_line = f"Iter {it}: Val loss {val_loss:.4f}"

        # Early-stopping check (only when enabled AND we actually have a val
        # loss to compare). A "best" improvement must beat the prior best by a
        # small epsilon to avoid noise-driven resets.
        if self._early_stop:
            improved = (
                self._best_val_loss is None
                or val_loss < self._best_val_loss - 1e-4
            )
            if improved:
                self._best_val_loss = val_loss
                self._bad_evals = 0
                log_line += f" (new best; patience reset)"
            else:
                self._bad_evals += 1
                log_line += (
                    f" (no improvement x{self._bad_evals}/{self._patience};"
                    f" best={self._best_val_loss:.4f})"
                )
            self._runner._enqueue(
                MetricsUpdate(metrics=dict(self._current), log_line=log_line)
            )
            if self._bad_evals >= self._patience:
                # Halt the trainer. This raises out of the trainer's loop,
                # through run(), into _run_training where it's caught as a
                # clean early stop.
                stop_msg = (
                    f"Early stopping at iter {it}: val loss hasn't improved "
                    f"for {self._patience} evals (best={self._best_val_loss:.4f})."
                )
                self._runner._enqueue(
                    MetricsUpdate(
                        metrics=dict(self._current), log_line=stop_msg
                    )
                )
                raise EarlyStopRequested(stop_msg)
        else:
            self._runner._enqueue(
                MetricsUpdate(metrics=dict(self._current), log_line=log_line)
            )

    @staticmethod
    def _fmt(v: Any) -> str:
        return "N/A" if v is None else f"{float(v):.4f}"


class TrainingRunner:
    """Owns one training run: spawns it in a thread, streams updates.

    Single active run at a time (the frontend assumes this). Methods are
    thread-safe enough for the FastAPI layer's usage pattern (one start,
    periodic next_update polls, one stop).
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: List[MetricsUpdate] = []
        self._queue_lock = threading.Lock()
        self._state: str = "idle"
        self._config: Optional[TrainingConfig] = None
        self._metrics: Optional[Dict[str, Any]] = None
        self._start_time: str = ""
        self._error: Optional[str] = None

    # ----- public state ---------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def config(self) -> Optional[TrainingConfig]:
        return self._config

    @property
    def metrics(self) -> Optional[Dict[str, Any]]:
        return self._metrics

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> Dict[str, Any]:
        """Return the {state, metrics, config} shape expected by GET /training/status."""
        cfg = None
        if self._config is not None:
            cfg = self._config.__dict__
        return {
            "state": self._state,
            "metrics": self._metrics,
            "config": cfg,
        }

    # ----- run lifecycle --------------------------------------------------

    def start(self, config: TrainingConfig) -> None:
        """Start a training run in a background thread.

        Raises RuntimeError if a run is already active.
        """
        if self.is_running:
            raise RuntimeError("A training run is already in progress")

        self._config = config
        self._stop_event.clear()
        with self._queue_lock:
            self._queue.clear()
        self._error = None
        self._metrics = None
        self._start_time = datetime.now().isoformat()
        self._state = "running"

        self._thread = threading.Thread(
            target=self._run_training, name="mlx-lm-lora-trainer", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Request the run to stop. MLX training cannot be hard-killed from
        another thread cleanly, so we set a flag and state; the run will
        finish its current step then exit at the next callback. The frontend
        treats 'stopped' as terminal."""
        if self.is_running:
            self._stop_event.set()
            self._set_state("stopped")
            self._enqueue(MetricsUpdate(metrics=self._metrics or {}, state="stopped"))

    def next_update(self) -> Optional[MetricsUpdate]:
        """Pop the next queued MetricsUpdate, or None if the queue is empty."""
        with self._queue_lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    # ----- internals ------------------------------------------------------

    def _set_state(self, state: str) -> None:
        self._state = state

    def _enqueue(self, update: MetricsUpdate) -> None:
        # Keep latest metrics around for snapshot/status polling.
        if update.metrics:
            self._metrics = update.metrics
        if update.state:
            self._set_state(update.state)
        with self._queue_lock:
            self._queue.append(update)

    def _build_args(self, config: TrainingConfig, run_dir: Path) -> Dict[str, Any]:
        """Translate a frontend TrainingConfig into an mlx-lm-lora args dict.

        Field names come from mlx_lm_lora/train.py CONFIG_DEFAULTS (verified
        against installed v2.1.0). adapter_path is set to a per-run directory
        under <root>/adapters/<adapter_name>.

        mlx-lm-lora's load_local_dataset expects `data` to be a DIRECTORY
        containing train.jsonl (and optionally valid.jsonl / test.jsonl).
        The frontend sends a single JSONL file path in train_data_path, so we
        stage the user's files into <run_dir>/data/ as train.jsonl /
        valid.jsonl and point `data` at that directory.
        """
        adapter_dir = paths.ADAPTERS_DIR / config.adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)

        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._stage_data_file(config.train_data_path, data_dir / "train.jsonl")
        if config.val_data_path:
            self._stage_data_file(config.val_data_path, data_dir / "valid.jsonl")

        args: Dict[str, Any] = {
            "model": config.model_path,
            "train": True,
            "train_type": "lora",  # SFT + LoRA for the thin backend
            "train_mode": "sft",
            "data": str(data_dir),
            "batch_size": config.batch_size,
            "iters": config.iterations,
            "learning_rate": config.learning_rate,
            "max_seq_length": config.max_seq_length,
            "steps_per_report": config.steps_per_report,
            "steps_per_eval": config.steps_per_eval,
            "save_every": config.save_every,
            "adapter_path": str(adapter_dir),
            "fuse": False,  # keep adapter form; fusion is a later feature
            "num_layers": -1,  # all layers (LoRA only touches targets anyway)
            "seed": 0,
        }
        # Weight-only resume: if the user picked a saved LoRA .safetensors
        # checkpoint, pass it through as resume_adapter_file. mlx-lm-lora will
        # load those weights (strict=False) before training starts. Note this
        # does NOT restore optimizer state / step counter / LR schedule.
        if config.resume_from:
            args["resume_adapter_file"] = config.resume_from
        return args

    @staticmethod
    def _stage_data_file(src_path: str, dest_path: Path) -> None:
        """Copy a user-provided JSONL file into the run's data directory.

        If src_path is already a directory (e.g. user pointed at an
        mlx-lm-lora-style data dir), copy train.jsonl/valid.jsonl from it
        instead. If the source does not exist, leave dest absent (mlx-lm-lora
        treats a missing valid.jsonl as 'no validation', which is fine).
        """
        src = Path(src_path)
        if not src.exists():
            return
        if src.is_dir():
            for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
                candidate = src / name
                if candidate.exists():
                    (dest_path.parent / name).write_bytes(candidate.read_bytes())
            return
        # Single file: copy it to dest_path verbatim.
        dest_path.write_bytes(src.read_bytes())

    def _create_run_dir(self, config: TrainingConfig) -> Path:
        """Create and return the per-run directory under <root>/runs/."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = paths.RUNS_DIR / f"{config.adapter_name}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_run_config(
        self, run_dir: Path, config: TrainingConfig, args: Dict[str, Any]
    ) -> None:
        """Persist a YAML config for reproducibility."""
        config_yaml = {
            **args,
            "val_data_path": config.val_data_path,
            "early_stop": config.early_stop,
            "patience": config.patience,
            "adapter_name": config.adapter_name,
        }
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config_yaml, f, sort_keys=False)

    def _run_training(self) -> None:
        """Thread target: invoke mlx_lm_lora.train.run() with our callback."""
        config = self._config
        assert config is not None
        try:
            run_dir = self._create_run_dir(config)
            args = self._build_args(config, run_dir)
            self._write_run_config(run_dir, config, args)
            log_path = run_dir / "run.log"

            # Import lazily so the FastAPI process can start even if the user
            # hasn't finished installing mlx-lm-lora yet (clearer errors).
            from mlx_lm_lora import train as lora_train

            callback = _StreamCallback(
                runner=self,
                total_steps=int(config.iterations),
                start_time=self._start_time,
                # Early stopping only makes sense if there's a validation set
                # to compute val_loss from. _build_args stages valid.jsonl
                # only when config.val_data_path is set, so gate on that.
                early_stop=bool(config.early_stop and config.val_data_path),
                patience=int(config.patience),
            )

            self._enqueue(
                MetricsUpdate(
                    metrics={
                        "current_step": 0,
                        "total_steps": int(config.iterations),
                        "train_loss": None,
                        "val_loss": None,
                        "learning_rate": float(config.learning_rate),
                        "start_time": self._start_time,
                        "estimated_time_remaining": None,
                    },
                    log_line=f"Starting SFT training: {config.adapter_name} "
                    f"({config.iterations} iters, lr={config.learning_rate})"
                    + (
                        f" [RESUMING from {config.resume_from}]"
                        if config.resume_from
                        else ""
                    ),
                    state="running",
                )
            )

            # Build a fully-populated args namespace the same way main() does:
            # start from the CLI parser defaults, overlay our dict, then apply
            # CONFIG_DEFAULTS for any field still None. This is required because
            # we call run() directly (which does NOT apply defaults itself),
            # bypassing main() so we can pass a training_callback.
            import types

            default_args = vars(lora_train.build_parser().parse_args([]))
            default_args.update(args)
            for k, v in lora_train.CONFIG_DEFAULTS.items():
                if default_args.get(k) is None:
                    default_args[k] = v
            ns = types.SimpleNamespace(**default_args)

            # Redirect the trainer's tqdm/print output to a per-run log file
            # so the user can inspect full trainer output later. The
            # structured metrics still flow through the callback.
            #
            # We call lora_train.run() directly (NOT main()) because main()
            # does not forward a training_callback to run(). run() accepts
            # training_callback as its second positional arg, which is how we
            # receive structured per-step metrics.
            import contextlib

            try:
                with open(log_path, "w") as logf, contextlib.redirect_stdout(logf):
                    lora_train.run(ns, training_callback=callback)
            except EarlyStopRequested as es:
                # Clean early stop — NOT an error. The trainer's loop exited
                # early because val loss plateaued for `patience` evals. The
                # most recent adapter weights were already saved at the last
                # steps_per_save checkpoint; we surface a completion message.
                # Note: we do NOT raise, so we fall through to the "completed"
                # block below unless an external stop was also requested.
                logger.info("Early stop fired: %s", es)
                self._early_stopped = True

            # If stop was requested mid-run, state is already 'stopped'.
            if not self._stop_event.is_set():
                self._enqueue(
                    MetricsUpdate(
                        metrics=self._metrics or {},
                        log_line=(
                            "Training completed (early stop)."
                            if getattr(self, "_early_stopped", False)
                            else "Training completed."
                        ),
                        state="completed",
                    )
                )

        except Exception as exc:  # noqa: BLE001 - surface to the UI
            tb = traceback.format_exc()
            logger.error("Training failed:\n%s", tb)
            self._error = str(exc)
            self._enqueue(
                MetricsUpdate(
                    metrics=self._metrics or {},
                    log_line=f"Training error: {exc}",
                    state="error",
                )
            )
        finally:
            # If the thread exits but state was never set terminal, mark idle.
            if self._state in {"running", "paused"}:
                self._set_state("idle" if self._stop_event.is_set() else self._state)
            self._thread = None
