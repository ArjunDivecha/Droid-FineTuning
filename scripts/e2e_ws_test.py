"""
End-to-end test: POST /training/start, then connect to /ws and stream
training_progress frames. Verifies the full v2 stack (FastAPI -> runner ->
mlx-lm-lora -> callback -> WS broadcast) end to end.
"""
import asyncio
import json
import urllib.request

import websockets

BASE = "http://127.0.0.1:8000"
WS = "ws://127.0.0.1:8000/ws"

# Find a generated sample file.
import glob
import os
from pathlib import Path

samples = sorted(glob.glob("/Users/arjundivecha/Dropbox/AAA Backup/A Working/Droid Fine Tuning/Droid-FineTuning/data/sample_*.jsonl"))
data_path = samples[-1] if samples else None
if not data_path:
    raise SystemExit("No sample data file found")

config = {
    "model_path": "openai-community/gpt2",
    "train_data_path": data_path,
    "val_data_path": "",
    "learning_rate": 1e-4,
    "batch_size": 1,
    "max_seq_length": 256,
    "iterations": 6,
    "steps_per_report": 1,
    "steps_per_eval": 100,
    "save_every": 100,
    "early_stop": False,
    "patience": 3,
    "adapter_name": "e2e_test_adapter",
}

# POST /training/start
req = urllib.request.Request(
    f"{BASE}/training/start",
    data=json.dumps(config).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    print("POST /training/start ->", resp.status, resp.read().decode())


async def stream():
    progress_seen = 0
    completed = False
    async with websockets.connect(WS) as ws:
        # First frame is the initial training_state.
        first = await asyncio.wait_for(ws.recv(), timeout=5)
        print("WS initial:", first)
        # Now stream for up to 90s, looking for progress + completion.
        deadline = asyncio.get_event_loop().time() + 90
        while asyncio.get_event_loop().time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                print("WS: 30s with no frame, stopping")
                break
            obj = json.loads(msg)
            t = obj.get("type")
            if t == "training_progress":
                m = obj["data"]["metrics"]
                progress_seen += 1
                print(
                    f"WS progress #{progress_seen}: step={m.get('current_step')} "
                    f"train_loss={m.get('train_loss')} lr={m.get('learning_rate')}"
                )
                if m.get("train_loss") is not None:
                    progress_seen_with_loss = progress_seen
            elif t == "training_completed":
                print("WS: training_completed", obj["data"])
                completed = True
                break
            elif t == "training_error":
                print("WS: training_error", obj["data"])
                break
            else:
                print(f"WS other: {t}", obj.get("data"))
    return progress_seen, completed


async def main():
    seen, completed = await stream()
    print(f"\n=== SUMMARY ===")
    print(f"progress frames received: {seen}")
    print(f"training_completed received: {completed}")
    # status check
    with urllib.request.urlopen(f"{BASE}/training/status") as resp:
        print(f"final /training/status: {resp.read().decode()}")
    if seen >= 2 and completed:
        print("E2E TEST PASSED")
    else:
        print("E2E TEST INCOMPLETE")


asyncio.run(main())
