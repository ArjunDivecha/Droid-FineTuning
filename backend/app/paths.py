"""
=============================================================================
SCRIPT NAME: paths.py
=============================================================================

INPUT FILES:
- None (this module only resolves filesystem locations).

OUTPUT FILES:
- None directly. Other modules use these paths to read/write:
  - <root>/models/      : MLX base models (gitignored)
  - <root>/adapters/    : trained LoRA adapter weights
  - <root>/sessions/    : saved training sessions (JSON)
  - <root>/runs/        : per-run YAML configs + logs (reproducibility archive)
  - <root>/data/        : generated sample datasets
  - <root>/outputs/logs/: backend log files

VERSION: 1.0
LAST UPDATED: 2026-06-20
AUTHOR: Arjun

DESCRIPTION:
Central, machine-independent path resolution for the v2 backend.

This exists specifically to eliminate the hardcoded `/Users/macbook2024/...`
paths that were baked into the original backend and broke it on any machine
other than the one it was built on. Every directory the backend touches is
derived from PROJECT_ROOT (the repo root, computed relative to this file),
so the app is portable.

All directories are created on import so the rest of the backend can assume
they exist.

DEPENDENCIES:
- None (stdlib only)

USAGE:
    from app.paths import PROJECT_ROOT, ADAPTERS_DIR, MODELS_DIR
    path = ADAPTERS_DIR / "my_adapter"
=============================================================================
"""

import os
from pathlib import Path

# This file lives at <repo_root>/backend/app/paths.py.
# repo_root = paths.py -> app -> backend -> repo_root (3 parents).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Core working directories (all relative to repo root, all gitignored already).
MODELS_DIR: Path = PROJECT_ROOT / "models"
ADAPTERS_DIR: Path = PROJECT_ROOT / "adapters"
SESSIONS_DIR: Path = PROJECT_ROOT / "sessions"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
DATA_DIR: Path = PROJECT_ROOT / "data"
LOGS_DIR: Path = PROJECT_ROOT / "outputs" / "logs"

# Optional override for the models directory via environment variable, so a
# user can point at a shared HF cache / external models folder without code
# changes. Defaults to <root>/models.
MODELS_DIR_ENV = os.environ.get("DROID_MODELS_DIR")
if MODELS_DIR_ENV:
    MODELS_DIR = Path(MODELS_DIR_ENV).expanduser()

# LM Studio stores models under ~/.lmstudio/models (or, on some installs,
# ~/Library/Application Support/LM Studio/models) in a publisher/model layout.
# We scan this directory too (in addition to MODELS_DIR) so models you already
# have in LM Studio show up in the Setup page. Only MLX-format models
# (config.json + *.safetensors) are listed — GGUF models can't be fine-tuned
# by mlx-lm-lora.
LM_STUDIO_MODELS_DIR: Path = Path.home() / ".lmstudio" / "models"
if not LM_STUDIO_MODELS_DIR.is_dir():
    _alt = Path.home() / "Library" / "Application Support" / "LM Studio" / "models"
    if _alt.is_dir():
        LM_STUDIO_MODELS_DIR = _alt


def ensure_dirs() -> None:
    """Create all working directories if they do not already exist.

    Safe to call repeatedly; uses exist_ok=True.
    """
    for d in (MODELS_DIR, ADAPTERS_DIR, SESSIONS_DIR, RUNS_DIR, DATA_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Create directories on import so the rest of the backend can rely on them.
ensure_dirs()
