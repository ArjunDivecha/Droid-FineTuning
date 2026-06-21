# Droid FineTuning

A streamlined MLX fine-tuning desktop application for Apple Silicon Macs. Simple, fast, and focused on core fine-tuning functionality.

> ## ⚡ v2 Backend (current) — rebuilt on `mlx-lm-lora`
>
> The backend was rebuilt on top of the [`mlx-lm-lora`](https://pypi.org/project/mlx-lm-lora/) training engine. The original backend (`backend/main.py`, `backend/opd/`, etc.) is kept on disk as reference but is **no longer used**; Electron now launches the v2 backend at `backend/app/main.py`.
>
> ### What changed
> - **Training engine**: replaced the hand-rolled training code with `mlx-lm-lora` v2.1.0 (SFT/DPO/CPO/ORPO/GRPO/GSPO/Dr.GRPO/DAPO/Online-DPO/XPO/RLHF/PPO, LoRA/DoRA/QLoRA/QAT, synthetic data). Phase 1 (this rebuild) wires up **SFT + LoRA only**; the other modes are deferred.
> - **Machine-independent paths**: all hardcoded `/Users/macbook2024/...` paths are gone. Every directory is derived from `PROJECT_ROOT` in `backend/app/paths.py`. The app now boots on any machine.
> - **In-process training with structured callbacks**: instead of spawning a subprocess and parsing stdout, `backend/app/training_runner.py` calls `mlx_lm_lora.train.run(args, training_callback=...)` in a background thread. A custom `TrainingCallback` receives structured per-step dicts (`iteration`, `train_loss`, `learning_rate`, `iterations_per_second`, ...) and forwards them to the WebSocket — no fragile stdout parsing.
> - **Frontend contract preserved**: the React UI is unchanged. The v2 backend implements the exact same HTTP + WebSocket contract (`/training/status`, `/training/start`, `/training/stop`, `/models`, `/adapters`, `/sessions/*`, `/api/training/generate-sample-data`, `/model/test`, `WS /ws`).
> - **OPD / nested-learning / fusion / tier-eval endpoints return HTTP 501** for now (the frontend degrades gracefully). These will be re-wired to `mlx-lm-lora`'s online/teacher flows in a later phase.
>
> ### v2 file layout
> ```
> backend/app/
> ├── __init__.py
> ├── main.py              # FastAPI app: routes + WS + ConnectionManager
> ├── paths.py             # PROJECT_ROOT + all dir resolution (no macbook2024 paths)
> └── training_runner.py   # in-process mlx-lm-lora run() + TrainingCallback -> TrainingMetrics
> ```
> Working directories (all gitignored, auto-created on boot): `models/`, `adapters/`, `sessions/`, `runs/`, `data/`, `outputs/logs/`.
>
> ### Run it
> ```bash
> # one-time: create the project venv and install deps
> python3.11 -m venv .venv
> .venv/bin/pip install -r backend/requirements.txt
>
> # build + launch (Electron spawns the v2 backend from .venv)
> npm install && cd frontend && npm install && cd ..
> npm run dev:frontend
> ```
> The Electron main process (`src/main.ts`) starts the backend as:
> `(.venv/bin/python) -m uvicorn app.main:app --host 0.0.0.0 --port 8000` with `cwd=backend/`, then health-checks `GET /training/status`.
>
> ### Add a model to fine-tune
> Drop an MLX-format model directory under `models/` (or point `DROID_MODELS_DIR` at an existing folder). It will appear in the Setup page's model list. Any HuggingFace repo id also works directly (mlx-lm downloads it on first use).
>
> ### Verified
> - Smoke test: real SFT on `openai-community/gpt2`, loss 4.72 → 1.67 over 6 steps, adapter saved. See `scripts/smoke_sft.py`.
> - End-to-end: `POST /training/start` → live `training_progress` WS frames with per-step loss → `training_completed` frame. See `scripts/e2e_ws_test.py`.
> - Electron launches, spawns the v2 backend, the React UI connects over WS and polls `/training/status`.
>
> The sections below describe the original (v1) design and are kept for historical context.

---

## ✨ Features

- 🖥️ **Modern Desktop GUI** - Clean Electron app with React interface
- 🚀 **MLX Fine-Tuning** - Optimized for Apple Silicon (M1/M2/M3/M4)
- 🎓 **On-Policy Distillation** - Knowledge distillation from larger teacher models
- 📊 **Real-Time Monitoring** - Live training progress with WebSocket updates
- 🆚 **Model Comparison** - Test base vs fine-tuned model responses
- 💾 **Session Management** - Save and load training sessions
- ⚡ **Lean & Fast** - No bloat, just core fine-tuning functionality

## 🏗️ Architecture

```
Droid-FineTuning/
├── backend/           # FastAPI server for training management
│   ├── main.py       # Core training API with WebSocket support
│   ├── opd/          # On-Policy Distillation module
│   └── requirements.txt
├── frontend/          # React GUI
│   ├── src/          # React components and pages
│   └── package.json  # Frontend dependencies
├── src/              # Electron main process
│   ├── main.ts       # Main Electron process
│   └── preload.ts    # Electron preload script
├── OnPolicyDistill/   # OPD data and outputs
│   ├── checkpoints/  # Distilled model checkpoints
│   ├── teacher_cache/# Cached teacher outputs
│   └── metrics/      # Training metrics
└── package.json      # Root dependencies
```

## 🚀 Quick Start

### Prerequisites
- macOS with Apple Silicon
- Node.js 18+ and npm
- Python 3.9+
- MLX environment set up

### Installation & Setup

```bash
# Clone the repository
git clone <repository-url>
cd Droid-FineTuning

# Install dependencies
npm install

# Install frontend dependencies
cd frontend && npm install && cd ..

# Install backend dependencies (in your MLX virtual environment)
source "/path/to/your/mlx/.venv/bin/activate"
cd backend && pip install -r requirements.txt && cd ..
```

### Run the Application

```bash
# Start the application (starts backend + frontend + electron)
npm run dev
```

## 📋 Usage

### Standard Fine-Tuning (SFT)

1. **Setup Page** - Select your base model and upload training data (JSONL format)
2. **Training Page** - Monitor real-time training progress with live metrics
3. **Results Page** - Review training history and performance
4. **Compare Page** - Test and compare base vs fine-tuned model outputs

### On-Policy Distillation (OPD)

After fine-tuning your model, you can optionally use knowledge distillation to compress knowledge from a larger teacher model into your fine-tuned student model.

#### Quick Start with OPD

```bash
python3 backend/opd/run_distillation.py \
  --teacher-path /path/to/larger/model \
  --student-path /path/to/base/model \
  --adapter-path /path/to/fine-tuned/adapter \
  --prompts-path /path/to/validation_prompts.jsonl \
  --output-path ./OnPolicyDistill/checkpoints/my_distilled_model \
  --steps 1000 \
  --batch-size 4 \
  --temperature 2.0
```

#### OPD Parameters

- `--teacher-path`: Path to teacher model (e.g., Qwen 32B)
- `--student-path`: Path to student base model (e.g., Qwen 7B)
- `--adapter-path`: Path to your fine-tuned LoRA adapter
- `--prompts-path`: Validation prompts in JSONL format
- `--steps`: Number of training steps (default: 1000)
- `--batch-size`: Batch size (default: 4)
- `--temperature`: Distillation temperature (default: 2.0)
- `--kl-weight`: Weight for KL divergence loss (default: 0.8)

#### What OPD Does

1. **Teacher Inference**: Runs the larger teacher model to generate outputs
2. **Loss Computation**: Calculates KL divergence between student and teacher
3. **Knowledge Transfer**: Updates student LoRA adapters to match teacher behavior
4. **Caching**: Automatically caches teacher outputs for efficiency
5. **Checkpointing**: Saves best model based on validation loss

#### Benefits

- 📉 **Better Quality**: Student learns from superior teacher model
- ⚡ **Faster Inference**: Deploy compact model with large model's knowledge
- 💾 **Memory Efficient**: Teacher outputs are cached (50%+ time savings)
- 🎯 **Fine Control**: Adjust temperature and loss weights for your use case

## 🔧 Configuration

The app automatically manages:
- Model loading and validation
- Training data preprocessing  
- LoRA adapter generation
- Training progress monitoring
- Model comparison inference

## 🎯 What's Different

This is a **lean, focused** fine-tuning tool that:
- ✅ **GUI only** - No CLI complexity
- ✅ **Core functionality** - Just fine-tuning, no dataset creation tools
- ✅ **Streamlined** - Minimal dependencies and clean codebase
- ✅ **Apple Silicon optimized** - Built specifically for MLX framework

## 📝 Scripts

```bash
# Development
npm run dev                 # Start full development environment

# Building
npm run build              # Build frontend and main process
npm run build:frontend     # Build React frontend only  
npm run build:main         # Build Electron main process only

# Production
npm run start              # Start built application
npm run dist               # Package for distribution
```

## 🔌 Integration

Works with your existing MLX setup:
- Uses your MLX virtual environment
- Compatible with Qwen2.5 models and other MLX-supported models
- Outputs standard LoRA adapters
- Integrates with your existing model directory structure

---

**Simple. Lean. Focused. Just fine-tuning.**
