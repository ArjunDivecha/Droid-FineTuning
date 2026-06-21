"""
Smoke test: run a tiny SFT via mlx-lm-lora's in-process main(args=...) path
with a TrainingCallback, to confirm the runner mechanism works end-to-end
before wiring it into Electron.

Uses the cached gpt2 model and the generated sample JSONL.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.training_runner import TrainingConfig, TrainingRunner  # noqa: E402

# Use the cached gpt2 model (HF cache) and the sample data we generated.
GPT2 = "openai-community/gpt2"  # mlx-lm will resolve from HF cache

# Find a generated sample file, or make one.
data_dir = ROOT / "data"
sample = next(data_dir.glob("sample_*.jsonl"), None)
if sample is None:
    # generate inline
    sample = data_dir / "sample_smoke.jsonl"
    lines = [json.dumps({"text": f"User: Say hi.\nAssistant: Hi there!"}) for _ in range(8)]
    sample.write_text("\n".join(lines) + "\n")

cfg = TrainingConfig(
    model_path=GPT2,
    train_data_path=str(sample),
    learning_rate=1e-4,
    batch_size=1,
    max_seq_length=256,
    iterations=4,
    steps_per_report=1,
    steps_per_eval=4,
    save_every=4,
    adapter_name="smoke_test_adapter",
)

runner = TrainingRunner()
print(f"Starting smoke SFT: model={GPT2} data={sample} iters=4")
runner.start(cfg)

import time  # noqa: E402

# Drain updates as they arrive until the run terminates.
last_state = None
seen_train_loss = False
timeout = time.time() + 180  # 3 min ceiling
while time.time() < timeout:
    upd = runner.next_update()
    if upd is not None:
        if upd.log_line:
            print(f"[log] {upd.log_line}")
        if upd.metrics and upd.metrics.get("train_loss") is not None:
            seen_train_loss = True
        if upd.state and upd.state != last_state:
            print(f"[state] {upd.state}")
            last_state = upd.state
        if upd.state in {"completed", "error", "stopped"}:
            break
    else:
        if not runner.is_running:
            # thread finished; drain any remaining queued updates
            if runner.next_update() is None:
                break
        time.sleep(0.05)

print(f"Final state: {runner.state}")
print(f"Final metrics: {runner.metrics}")
print(f"Error: {runner.error}")
print(f"Callback fired with train_loss: {seen_train_loss}")
if runner.state == "completed":
    print("SMOKE TEST PASSED")
    sys.exit(0)
else:
    print("SMOKE TEST FAILED")
    sys.exit(1)
