"""
=============================================================================
SCRIPT NAME: split_train_valid.py
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/New Writing Dataset/data/finetune/snapshot_v1/sft_b_train.jsonl
  Source SFT dataset in {"messages":[{"role","content"},...]} format (878 rows).

OUTPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Droid Fine Tuning/Droid-FineTuning/data/sft_b_train_split/train.jsonl
  90% of the rows (~790), shuffled deterministically.
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Droid Fine Tuning/Droid-FineTuning/data/sft_b_train_split/valid.jsonl
  10% of the rows (~88), the held-out validation split.

VERSION: 1.0
LAST UPDATED: 2026-06-21
AUTHOR: Arjun

DESCRIPTION:
Creates a deterministic 90/10 train/validation split from the sft_b SFT
dataset so Droid-FineTuning can report a real val_loss during training
(the app/MLX-LM-LoRA does NOT auto-split; without a valid.jsonl the trainer
just skips validation entirely).

The shuffle is seeded (seed=42) so the split is reproducible: running this
script twice produces identical train.jsonl / valid.jsonl files. Rows are
kept whole (each is one {"messages":[...]} conversation) and written as
one JSON object per line, which is the format mlx-lm-lora's SFT loader
expects (load_local_dataset reads train.jsonl / valid.jsonl / test.jsonl).

DEPENDENCIES:
- None beyond the Python standard library.

USAGE:
    python scripts/split_train_valid.py
=============================================================================
"""

import json
import random
from pathlib import Path

# --- Paths ------------------------------------------------------------------
SRC = Path(
    "/Users/arjundivecha/Dropbox/AAA Backup/New Writing Dataset/"
    "data/finetune/snapshot_v1/sft_b_train.jsonl"
)
OUT_DIR = Path(
    "/Users/arjundivecha/Dropbox/AAA Backup/A Working/Droid Fine Tuning/"
    "Droid-FineTuning/data/sft_b_train_split"
)
TRAIN_OUT = OUT_DIR / "train.jsonl"
VALID_OUT = OUT_DIR / "valid.jsonl"

SEED = 42
VALID_FRACTION = 0.10


def load_jsonl(path: Path) -> list:
    """Read a JSONL file into a list of parsed records (skips blank lines)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: list, path: Path) -> None:
    """Write records to a JSONL file (one compact JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"Source not found: {SRC}")

    records = load_jsonl(SRC)
    n = len(records)
    if n == 0:
        raise SystemExit(f"Source file is empty: {SRC}")

    # Deterministic shuffle so the split is reproducible.
    rng = random.Random(SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n_valid = max(1, int(round(n * VALID_FRACTION)))
    n_train = n - n_valid

    train_records = shuffled[:n_train]
    valid_records = shuffled[n_train:]

    write_jsonl(train_records, TRAIN_OUT)
    write_jsonl(valid_records, VALID_OUT)

    print(f"Source: {SRC}")
    print(f"  total rows: {n}")
    print(f"Wrote train: {TRAIN_OUT}  ({len(train_records)} rows)")
    print(f"Wrote valid: {VALID_OUT}  ({len(valid_records)} rows)")

    # Sanity: confirm every record still has a messages list.
    bad = 0
    for rec in train_records + valid_records:
        if not isinstance(rec.get("messages"), list) or not rec["messages"]:
            bad += 1
    if bad:
        print(f"WARNING: {bad} records missing a 'messages' list")
    else:
        print("All records have a 'messages' list — ready for mlx-lm-lora SFT.")


if __name__ == "__main__":
    main()
