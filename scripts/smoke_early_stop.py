"""
=============================================================================
SCRIPT NAME: smoke_early_stop.py
=============================================================================

INPUT FILES:
- data/sft_b_train_split/train.jsonl : SFT training samples
- data/sft_b_train_split/valid.jsonl : SFT validation samples (REQUIRED for early stop)
- /Users/arjundivecha/.lmstudio/models/.../Qwen2.5-0.5B-Instruct-MLX-4bit : base model

OUTPUT FILES:
- adapters/_es_smoke/adapters.safetensors : trained LoRA weights (test artifact)
- runs/_es_smoke_<ts>/config.yaml, run.log : run records (test artifacts)

VERSION: 1.0
LAST UPDATED: 2026-06-21
AUTHOR: Droid-FineTuning

DESCRIPTION:
Verifies that early stopping actually halts training when validation loss
plateaus. Uses a tiny config: 60 iters, eval every 5 steps, patience=2, so
early stop should fire well before iter 60 if val loss doesn't keep
improving. The script asserts that the final state is 'completed' (early
stop is treated as a clean completion, NOT an error) and that an
early-stop log line was emitted.

USAGE:
    .venv/bin/python scripts/smoke_early_stop.py
=============================================================================
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.app.training_runner import TrainingConfig, TrainingRunner  # noqa: E402

MODEL = "/Users/arjundivecha/.lmstudio/models/lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-4bit"
TRAIN = str(REPO / "data" / "sft_b_train_split" / "train.jsonl")
VALID = str(REPO / "data" / "sft_b_train_split" / "valid.jsonl")

cfg = TrainingConfig(
    model_path=MODEL,
    train_data_path=TRAIN,
    val_data_path=VALID,
    learning_rate=1e-4,
    iterations=60,
    steps_per_report=5,
    steps_per_eval=5,   # evaluate often so patience is exercised quickly
    save_every=10,
    early_stop=True,
    patience=2,
    adapter_name="_es_smoke",
)

runner = TrainingRunner()
runner.start(cfg)
print(f"Started early-stop smoke test: {cfg.iterations} iters, patience={cfg.patience}, eval every {cfg.steps_per_eval}")

saw_early_stop_line = False
final_state = None
last_step = 0
t0 = time.time()
while True:
    upd = runner.next_update()
    if upd is not None:
        if upd.log_line:
            print(f"  [log] {upd.log_line}")
        if "early stop" in (upd.log_line or "").lower() or "no improvement" in (upd.log_line or "").lower():
            saw_early_stop_line = True
        if upd.metrics and upd.metrics.get("current_step") is not None:
            last_step = max(last_step, int(upd.metrics.get("current_step", 0)))
        if upd.state:
            final_state = upd.state
        continue
    # No queued update: if the thread is still running, wait; else we're done.
    if runner.is_running:
        time.sleep(0.2)
    else:
        break

print()
print(f"=== RESULT ===")
print(f"elapsed: {time.time()-t0:.1f}s")
print(f"final state: {final_state}")
print(f"last step reached: {last_step} / {cfg.iterations}")
print(f"saw early-stop log line: {saw_early_stop_line}")
print(f"runner.error: {runner.error}")

ok = (
    final_state == "completed"
    and runner.error is None
    and last_step < cfg.iterations  # halted before full iters
    and saw_early_stop_line
)
print(f"PASS={ok}")
sys.exit(0 if ok else 1)
