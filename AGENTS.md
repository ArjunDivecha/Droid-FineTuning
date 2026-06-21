# AGENTS.md — Droid-FineTuning

Durable workspace facts and conventions for AI agents working in this repo.
Recurring corrections/preferences live in the user's global AGENTS.md; this file
covers what is specific to the Droid-FineTuning codebase.

## Droid-FineTuning v2 backend

- The active backend is the v2 rebuild on `mlx-lm-lora`. Source lives at `/Users/arjundivecha/Dropbox/AAA Backup/A Working/Droid Fine Tuning/Droid-FineTuning/backend/app/{main.py, paths.py, training_runner.py, __init__.py}`.
- The legacy backend (`backend/main.py` + `backend/opd/`) is kept on disk as reference only and is NOT used. Do not edit it expecting changes to take effect.
- Training engine is `mlx-lm-lora` v2.1.0. It is run **in-process** via `mlx_lm_lora.train.run(args, training_callback=...)` in a background thread — NOT via subprocess and NOT via stdout parsing. `main()` does not forward the callback, so `run()` is called directly with `CONFIG_DEFAULTS` applied manually to `args`.
- All paths are derived from `PROJECT_ROOT` in `backend/app/paths.py`. Do not introduce hardcoded `/Users/macbook2024/...` (or any user-specific) paths.
- `src/main.ts` spawns `<repo>/.venv/bin/python -m uvicorn app.main:app` with `cwd=backend/`.
- Frontend contract: HTTP at `localhost:8000`, WebSocket at `ws://127.0.0.1:8000/ws`. SFT endpoints are implemented; OPD / nested / fusion / evaluate endpoints return 501.
- `GET /models` scans BOTH `<repo_root>/models` AND `~/.lmstudio/models` recursively, filtering to MLX-format models only (must contain `config.json` + `*.safetensors`). GGUF models are excluded because `mlx-lm-lora` cannot train them.
- `POST /model/test` and `POST /model/test-base` both default model/adapter from the current `runner.config` when the request omits them. They share `_resolve_model_and_adapter()` + `_run_generation()`. `/model/test-base` generates from the base model with no adapter; `/model/test` applies the adapter.

## mlx-lm-lora integration details

- SFT `load_local_dataset` expects a **directory** containing `train.jsonl` / `valid.jsonl` / `test.jsonl`. The runner stages the user's single JSONL file into `<run_dir>/data/train.jsonl` before training.
- `mlx-lm-lora` SFT dataset processor accepts `{"messages": [...]}` records natively (reads `d[self.chat_key]`, default chat key `messages`) and applies the tokenizer chat template.
- TrainingCallback dict shapes:
  - `on_train_loss_report`: `{iteration, train_loss, learning_rate, iterations_per_second, tokens_per_second, trained_tokens, peak_memory}`
  - `on_val_loss_report`: `{iteration, val_loss, val_time}`
- Resume is **weight-only** via `--resume-adapter-file` (uses `model.load_weights(..., strict=False)`). It does NOT restore step count, optimizer state (Adam momentum/variance), LR schedule position, or RNG state. A resumed run restarts at iteration 1 with a fresh optimizer. The runner does NOT expose resume yet.

## Known limitations

- `runner.stop()` sets state to `stopped`, but the MLX trainer loop does NOT check the stop flag — training runs to completion. The UI reports stopped immediately while the background thread keeps running.
- Resume is not wired into the app UI/runner (see above).

## Conventions

- Train/valid split: `scripts/split_train_valid.py` produces a 90/10 split with `seed=42` (reproducible). Source dataset is `/Users/arjundivecha/Dropbox/AAA Backup/New Writing Dataset/data/finetune/snapshot_v1/sft_b_train.jsonl`.
- `package.json` `build:main` uses `--skipLibCheck` to work around a pre-existing `@types/node` ↔ `electron` `noDeprecation` TypeScript conflict. Do not remove this flag without resolving the underlying type conflict.
- Python venv for the backend is `<repo>/.venv`. Backend runs under that interpreter, launched by Electron.
