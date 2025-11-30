# DebugTrainer for MLX GSPO/GRPO

The debug trainer wraps the default `mlx_lm_lora` training loop with pre-flight
validation, structured step logging, and crash forensics. Use it instead of the
standard `python3 -m mlx_lm_lora.train` command when you need to diagnose
instabilities such as the recurring `ValueError: [logsumexp] Received empty array`.

## Key features

- Token-level safeguards prevent zero-length completions from crashing the run
  and log affected prompts for review.
- Per-step JSONL logging (`steps.jsonl`) with gradients, KL, reward statistics,
  throughput, and token-length summaries.
- Event log (`events.jsonl`) covering evaluations, checkpoints, fallbacks, and
  warnings (NaN gradients, empty completions, etc.).
- Automatic pre-flight dataset scan highlighting empty answers/prompts and token
  length percentiles (logged as `preflight_summary` events).
- Crash bundles in `debug_runs/<timestamp>/failures/...` containing the full
  traceback, the last steps JSON, and the offending batch metadata for
  deterministic replay.

## Usage

```bash
python3 -m debug.debug_trainer \
  --data /path/to/grpo_dataset \
  --model /path/to/model \
  --train-mode grpo \
  --adapter-path adapters/debug_run \
  --group-size 2 \
  --batch-size 2 \
  --iters 200
```

All standard `mlx_lm_lora.train` arguments are accepted. Additional debug
options:

- `--debug-log-root DIR`: explicit directory for debug artifacts
  (default: `<adapter_path>/debug_runs`).
- `--debug-preflight-limit N`: limit dataset rows inspected during pre-flight
  validation (default: 2000).
- `--debug-keep-steps N`: number of recent steps retained for crash bundles
  (default: 32).
- `--debug-no-sequence-fallback`: disable the automatic token→sequence
  importance-sampling fallback when empty token slices are detected.
- `--debug-disable-eval`: skip validation passes (useful in bisection).
- `--debug-disable-checkpoints`: avoid writing periodic adapter checkpoints.
- `--debug-min-new-tokens`: enforce a minimum number of sampled tokens per
  completion (passed to the sampler, default: 1).
- `--debug-skip-nan-check`: allow NaN gradients without aborting the run.

Logs and crash bundles are written under `debug_runs/<timestamp>/` inside the
adapter path (or the directory supplied via `--debug-log-root`). Inspect
`steps.jsonl` and `events.jsonl` with `jq`, `rg`, or any log viewer to analyse
metrics around the failure step.

## Integration notes

- The trainer patches `mlx_lm_lora.trainer.grpo_trainer` at runtime; no upstream
  files are modified.
- Restores the original training functions even if the run raises an exception.
- Works with existing backend entry points: simply invoke this script instead of
  the stock CLI when you need detailed diagnostics.
