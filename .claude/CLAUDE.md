# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Droid FineTuning is a streamlined MLX fine-tuning desktop application for Apple Silicon Macs built with Electron, React, and FastAPI. It supports advanced training methods including SFT, GSPO, Dr. GRPO, and GRPO.

**Architecture:** Electron desktop app with React frontend and FastAPI backend
- **Frontend:** React + TypeScript + TailwindCSS + Vite (port 3000 in dev)
- **Backend:** FastAPI + Python 3.11 (port 8000)
- **ML Framework:** MLX (Apple Silicon optimized)

## IMPORTANT: Planned Full-Layer LoRA Upgrade

**Status:** PRD created, awaiting implementation

The Standard Setup Tab (SFT training) will be upgraded to use **full-layer LoRA** (all 7 weight matrices) instead of the current attention-only default. The reference implementation has been validated in the XXX repository.

**DO NOT modify Enhanced Setup Tab** - it should remain unchanged for comparison purposes.

## Development Commands

### Starting the Application

```bash
# Recommended: Use convenience scripts
./startmlxnew          # Start backend and Electron app
./killmlxnew           # Stop all processes

# Manual startup:
# Terminal 1: Backend
cd backend && python3.11 -m uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend + Electron
cd frontend && npm run build && cd .. && npm start
```

**CRITICAL:** Always check and kill ports 8000 and 3000 before starting:
```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

### Building

```bash
# Build everything
npm run build

# Build frontend only (React app)
npm run build:frontend         # In frontend/ directory
cd frontend && npm run build   # Or from root

# Build Electron main process only
npm run build:main

# Package for distribution
npm run dist
```

### Development

```bash
# Frontend development (Vite dev server)
cd frontend && npm run dev

# Backend development (with auto-reload)
npm run dev:backend
```

## Code Architecture

### Backend Structure (`backend/`)

**Dual Training System:**
1. **Standard SFT** (`main.py`) - Uses Apple's `mlx_lm.lora` for supervised fine-tuning
2. **Enhanced RL Methods** (`main_enhancements.py`) - Uses `mlx_lm_lora.train` for GRPO/GSPO/Dr. GRPO

**Key Files:**
- `main.py` (56KB) - Core FastAPI server, TrainingManager, WebSocket streaming, standard SFT
- `main_enhancements.py` (39KB) - EnhancedTrainingManager for GRPO-based methods
- `training_methods.py` - Training method configurations and data validators
- `fusion_api.py` - Adapter fusion and combination endpoints
- `evaluation_api.py` - Model evaluation endpoints
- `model_export.py` - Model export and conversion utilities
- `backend_factory.py` - Factory pattern for training manager selection

**Training Manager Pattern:**
- `TrainingManager` (main.py) - Manages training state, WebSocket clients, session persistence
- `EnhancedTrainingManager` (main_enhancements.py) - Extends base manager with RL methods
- Both use `subprocess.Popen` to run MLX training commands
- Real-time log parsing and WebSocket broadcasting of metrics

**API Endpoints:**
- Standard: `/api/training/*` - SFT training (Setup tab)
- Enhanced: `/api/training/start-enhanced` - GRPO methods (Enhanced Setup tab)
- Models: `/api/models/*` - Model discovery and info
- Fusion: `/api/fusion/*` - Adapter fusion operations
- Evaluation: `/api/evaluation/*` - Model testing

### Frontend Structure (`frontend/src/`)

**Pages:**
- `SetupPage.tsx` - Standard SFT training setup (uses main.py)
- `EnhancedSetupPage.tsx` - GRPO/GSPO/Dr. GRPO setup (uses main_enhancements.py)
- `TrainingPage.tsx` - Real-time training monitoring with WebSocket
- `ResultsPage.tsx` - Training history and session review
- `ComparePage.tsx` - Base vs fine-tuned model comparison
- `FusionPage.tsx` - Adapter fusion interface

**State Management:**
- Redux Toolkit for global state (`store/`)
- WebSocket hook for real-time updates (`hooks/useWebSocket.tsx`)
- Axios for API calls

**Components:**
- Built with React + TypeScript
- TailwindCSS for styling
- Chart.js for training visualizations
- Lucide React for icons

### Data Formats

**SFT (Standard Setup):**
```jsonl
{"text": "User: question\nAssistant: answer"}
```

**GRPO/GSPO/Dr. GRPO (Enhanced Setup):**
```jsonl
{"prompt": "question", "answer": "response", "system": "You are..."}
```

### Training Methods

**SFT (⭐⭐)** - Standard supervised fine-tuning
- Module: `mlx_lm.lora` (Apple's official)
- Resource: Medium (baseline)
- Best for: General instruction following

**GRPO (⭐⭐⭐⭐)** - Group Relative Policy Optimization
- Module: `mlx_lm_lora.train --train-mode grpo`
- Resource: High (generates 4-16 completions per prompt)
- Best for: Reasoning, math, coding tasks
- Key params: `group_size`, `epsilon`, `temperature`, `max_completion_length`

**GSPO (⭐⭐⭐⭐)** - GRPO + importance sampling
- Module: `mlx_lm_lora.train --train-mode grpo --importance-sampling-level token`
- Resource: High (same as GRPO)
- Best for: Sample efficiency improvements over GRPO
- Additional param: `importance_sampling_level` (token/sequence)

**Dr. GRPO (⭐⭐⭐⭐⭐)** - Decoupled rewards GRPO
- Module: `mlx_lm_lora.train --grpo-loss-type dr_grpo`
- Resource: Very High (reward computation overhead)
- Best for: Stable training, large models
- Additional param: `epsilon_high` (for DAPO variant)

### Log Parsing & Metrics

**SFT Training Logs** (main.py:712-850):
- Parses iteration, train_loss, val_loss, learning_rate from MLX output
- Updates `training_metrics` dict
- Broadcasts via WebSocket to frontend

**GRPO Training Logs** (main_enhancements.py:450-600):
- Parses RL-specific metrics: avg_reward, success_rate, KL divergence, entropy
- Handles token_accuracy from debug trainer
- Preserves standard metrics (loss, learning_rate)
- Streams to Training tab without disrupting runs

### Session Management

**Persistence:**
- Sessions saved to `sessions/` directory
- JSON format with config, metrics, and state
- Auto-load latest session on backend startup
- Force idle state if no valid session

**Session File Structure:**
```json
{
  "session_id": "uuid",
  "config": {...},
  "metrics": {...},
  "state": "running|completed|error",
  "timestamp": "ISO8601"
}
```

## Common Development Tasks

### Adding a New Training Method

1. Add enum to `TrainingMethod` in `training_methods.py`
2. Create `TrainingMethodConfig` in `TRAINING_METHODS` dict
3. Update `build_enhanced_training_command()` in `main_enhancements.py`
4. Add UI form in `EnhancedSetupPage.tsx`
5. Update TypeScript types in `frontend/src/types/`

### Modifying Training Parameters

**Backend:** Update `EnhancedTrainingConfig` dataclass in `main_enhancements.py`
**Frontend:** Update `EnhancedTrainingConfig` interface in `EnhancedSetupPage.tsx`
**Validation:** Add validation logic in `training_methods.py`

### Parsing New Log Metrics

1. Add regex pattern to `_parse_training_log()` method
2. Update `training_metrics` dict structure
3. Broadcast via WebSocket: `await self._broadcast_to_clients(metrics)`
4. Update frontend to display new metric in `TrainingPage.tsx`

### WebSocket Protocol

**Connection:** `ws://localhost:8000/ws`
**Message Format:**
```json
{
  "type": "training_update|error|completion",
  "data": {...}
}
```

**Client Registration:**
- Frontend connects via socket.io-client
- Backend stores in `TrainingManager.websocket_clients`
- Automatic cleanup on disconnect

## Important Paths

**Models Directory:** Configurable in `main.py` (default: auto-detect)
- App scans this directory for MLX models
- Populate dropdown in UI

**Output Adapters:** `/path/to/artifacts/lora_adapters/`
- Checkpoints: `{step:07d}_adapters.safetensors`
- Best model: `best_adapters.safetensors`

**Training Data:** User-specified directories
- Must contain `train.jsonl`
- Optionally `valid.jsonl` for validation

**Logs:** `logs/` directory
- `backend.log` - Backend application logs
- `gui_training.log` - Training process logs
- `electron.log` - Electron app logs

## Testing

### Backend Integration Tests

```bash
python3.11 test_backend_integration.py
```

Tests:
- MLX-LM-LORA installation
- Command construction for all methods
- Data validation
- Sample format generation

### Test Data

Available in `test_data/`:
- `train.jsonl` - 10 sample training examples
- `valid.jsonl` - 2 validation examples

Format: `{"prompt": "...", "answer": "...", "system": "..."}`

## Debugging

### Common Issues

**Models not showing in dropdown:**
- Check `models_dir` path in `backend/main.py`
- Ensure models are in MLX format (not PyTorch)
- Restart backend after changing path

**Training not starting:**
- Check data format (JSONL, correct fields)
- Verify model and data paths exist
- Check `logs/backend.log` for errors

**Port conflicts:**
- Always run `./killmlxnew` before starting
- Or manually: `lsof -ti:8000 | xargs kill -9`

**WebSocket not connecting:**
- Ensure backend is running on port 8000
- Check browser console for connection errors
- Verify CORS settings in `main.py`

### Debug Mode for GRPO Methods

GRPO methods use `debug.debug_trainer` wrapper:
- Logs to `debug_runs/` directory
- Adds guards for empty token generation
- Enforces minimum new tokens (default: 2)
- Configure via `--debug-log-root` and `--debug-min-new-tokens`

## Git Workflow

**Current Branch:** `feature/fix-mlx-lm-lora-integration`
**Main Branch:** `main`

**Recent Focus:**
- MLX-LM-LORA integration fixes (GSPO/GRPO/Dr. GRPO)
- Enhanced logging and metrics streaming
- Dataset management and conversion tools

**Excluded from Git:**
- `datasets/` and `*.jsonl` files (see `.gitignore`)
- `grpo_*/` directories
- Local model files
- User training data

## Dependencies

**Backend (Python 3.11):**
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- websockets==12.0
- mlx-lm-lora==0.8.1 (for GRPO methods)
- PyYAML, aiohttp, pydantic

**Frontend (Node.js 18+):**
- React 18.2 + TypeScript 5.3
- Vite 5.0 (build tool)
- TailwindCSS 3.3 (styling)
- Redux Toolkit (state)
- socket.io-client (WebSocket)
- chart.js + react-chartjs-2 (visualizations)

**Electron:**
- electron 28.3.3
- electron-builder 24.8.0

## MLX Integration

**Two MLX Packages:**
1. `mlx_lm.lora` - Apple's official (for SFT)
2. `mlx_lm_lora` - Goekdeniz-Guelmez (for GRPO methods)

**DO NOT confuse these packages!** They have different CLI interfaces and data requirements.

**Command Construction:**
- SFT: `python3.11 -m mlx_lm.lora --model ... --data ...`
- GRPO: `python3.11 -m mlx_lm_lora.train --model ... --train-mode grpo ...`

## Performance Notes

**RL Methods are 3-5x slower than SFT** because:
- Generate multiple completions per prompt (group_size × batch_size)
- Compute policy gradients and rewards
- More complex loss functions

**Memory Usage:**
- SFT: 1.0x baseline
- GRPO/GSPO: 1.3x (multiple generations)
- Dr. GRPO: 1.5x (reward computation overhead)

**Recommended Hardware:**
- Apple Silicon Mac (M1/M2/M3/M4)
- 16GB+ RAM for small models
- 32GB+ RAM for 7B+ models with GRPO

## Documentation

- `README.md` - User-facing quick start
- `ENHANCED_TRAINING_METHODS.md` - Detailed guide to GRPO/GSPO/Dr. GRPO
- `INTEGRATION_COMPLETE.md` - MLX-LM-LORA integration summary
- `INSTALLATION_GUIDE.md` - Detailed setup instructions
- `BACKEND_FIX_SUMMARY.md` - Backend changes and test results

## Support Scripts

- `startmlxnew` - Start app (kills existing processes first)
- `killmlxnew` - Stop app (all processes)
- `convert_to_grpo.py` - Convert SFT data to GRPO format
- `download_gspo_datasets.py` - Download pre-made GRPO datasets
- `test_backend_integration.py` - Backend integration tests
- `check_training_error.sh` - Quick error check in logs

## Key Design Decisions

1. **Dual training managers** - Separate SFT (main.py) from RL methods (main_enhancements.py) for maintainability
2. **WebSocket streaming** - Real-time updates without polling
3. **Session persistence** - Resume interrupted training sessions
4. **Factory pattern** - `backend_factory.py` routes to correct training manager
5. **Debug wrapper** - GRPO methods use debug trainer for better diagnostics
6. **Best model tracking** - Automatically save best checkpoint based on validation loss

---

## Full-Layer LoRA Implementation (Planned Upgrade)

### Overview

The Standard Setup Tab will be upgraded to use full-layer LoRA training based on "LoRA Without Regret" (Schulman et al., 2025). This has been successfully implemented and validated in the **XXX repository** at:
```
/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Fine Tuner XXX
```

### Current vs. Target Implementation

**Current (Droid-FineTuning Standard Setup):**
- Uses `mlx_lm.lora` package (Apple's official)
- Falls back to MLX-LM defaults (Q and V projections only - attention-only)
- Learning rate: `1e-5`
- No explicit `lora_parameters` configuration
- No UI controls for LoRA rank/alpha/layers

**Target (XXX Repository - Validated):**
- Explicit `lora_parameters` with all 7 matrices
- `num_layers: -1` (all transformer layers)
- Learning rate: `1e-4` (10x increase per research)
- Rank: 32, Alpha: 32 (configurable)
- Architecture detection (Qwen2, Mixtral, Qwen2 MoE, etc.)
- Training verified: 3.562% trainable parameters (17.596M/494.033M on Qwen2.5-0.5B)

### Research Findings

**Source:** "LoRA Without Regret" (Schulman et al., 2025)
- URL: https://thinkingmachines.ai/blog/lora/

**Key Findings:**
1. **All-layer training critical**: MLP-only ≈ MLP+attention performance. Attention-only significantly underperforms.
2. **10x learning rate rule**: Optimal LoRA LR ≈ 10x full fine-tuning LR (validated across 14+ models)
3. **Rank selection**: Should exceed dataset information content. Default 32 for medium datasets (100-1000 examples)
4. **Full-layer benefits**: Near-parity with full fine-tuning, better sample efficiency, more stable training

### Backend Implementation (main.py)

**Reference:** XXX repository `mlx-finetune-gui/backend/main.py` (lines 398-488)

#### Key Components:

**1. LoRA Parameter Generation (lines 398-458)**
```python
# Extract from config
lora_rank = max(1, int(getattr(config, "lora_rank", 32) or 32))
lora_alpha = float(getattr(config, "lora_alpha", 32.0) or 32.0)
lora_dropout = float(getattr(config, "lora_dropout", 0.0) or 0.0)
lora_num_layers = getattr(config, "lora_num_layers", -1)

# Architecture detection
model_config_path = os.path.join(config.model_path, "config.json")
model_type = "qwen2"  # Default
if os.path.exists(model_config_path):
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
        model_type = model_config.get("model_type", "qwen2")

# Base keys for all standard transformers
lora_keys = [
    "self_attn.q_proj",    # Query projection
    "self_attn.k_proj",    # Key projection
    "self_attn.v_proj",    # Value projection
    "self_attn.o_proj",    # Output projection
    "mlp.gate_proj",       # MLP gate projection
    "mlp.up_proj",         # MLP up projection
    "mlp.down_proj",       # MLP down projection
]

# Architecture-specific keys
if model_type in ["mixtral", "phimoe"]:
    lora_keys.append("block_sparse_moe.gate")
elif model_type == "qwen2_moe":
    lora_keys.append("mlp.shared_expert_gate")
elif model_type == "qwen3_next":
    lora_keys.extend([
        "mlp.shared_expert_gate",
        "linear_attn.in_proj_qkvz",
        "linear_attn.out_proj",
        "linear_attn.in_proj_ba",
        "linear_attn.dt_bias"
    ])

lora_parameters = {
    "rank": lora_rank,
    "scale": lora_alpha,  # MLX-LM uses "scale" for alpha
    "dropout": lora_dropout,
    "keys": lora_keys,
}
```

**2. Config Data Structure (lines 460-488)**
```python
config_data = {
    # ... other fields ...
    "num_layers": lora_num_layers,          # -1 for all layers
    "lora_parameters": lora_parameters,      # Full dict with keys
    "lora_rank": lora_rank,                  # Also pass individually
    "lora_alpha": lora_alpha,
    "lora_dropout": lora_dropout,
    "learning_rate": config.learning_rate,   # Should be 1e-4
}
```

**3. TrainingConfig Dataclass (lines 44-63)**
```python
@dataclass
class TrainingConfig:
    model_path: str
    train_data_path: str
    val_data_path: str
    learning_rate: float = 1e-5    # CHANGE to 1e-4
    batch_size: int = 1
    max_seq_length: int = 1024
    iterations: int = 7329
    steps_per_report: int = 25
    steps_per_eval: int = 200
    save_every: int = 25
    early_stop: bool = True
    patience: int = 3
    adapter_name: str = "mlx_finetune"
    fine_tune_type: str = "lora"
    lora_rank: int = 32              # ADD
    lora_alpha: float = 32.0         # ADD
    lora_dropout: float = 0.0        # ADD
    lora_num_layers: int = -1        # ADD
```

### Frontend Implementation (SetupPage.tsx)

**Add LoRA Configuration Section:**

```typescript
// Add to Redux store (trainingSlice.ts)
interface TrainingConfig {
  // ... existing fields ...
  lora_rank: number;        // Default: 32
  lora_alpha: number;       // Default: 32
  lora_dropout: number;     // Default: 0.0
  lora_num_layers: number;  // Default: -1
}

// UI Component (SetupPage.tsx)
<div className="bg-gray-800 rounded-lg p-6 space-y-4">
  <h3 className="text-lg font-semibold text-gray-100">
    LoRA Configuration ✨
  </h3>

  {/* Info Banner */}
  <div className="bg-blue-900/30 border border-blue-700/50 rounded p-4">
    <p className="text-sm text-blue-200">
      <strong>Full-Layer LoRA Training</strong> - Applies LoRA to all 7 weight
      matrices (attention + MLP) across all transformer layers. Research shows
      this significantly outperforms attention-only training.
    </p>
  </div>

  {/* LoRA Rank */}
  <div>
    <label className="block text-sm font-medium text-gray-300 mb-2">
      LoRA Rank
    </label>
    <input
      type="number"
      min="1"
      max="512"
      value={config.lora_rank}
      onChange={(e) => dispatch(updateConfig({ lora_rank: parseInt(e.target.value) }))}
      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md"
    />
    <p className="text-xs text-gray-400 mt-1">
      Higher rank = more capacity. Use 16 for small datasets (&lt;100 samples),
      32 for medium (100-1000), 64+ for large (1000+)
    </p>
  </div>

  {/* LoRA Alpha */}
  <div>
    <label className="block text-sm font-medium text-gray-300 mb-2">
      LoRA Alpha
    </label>
    <input
      type="number"
      min="1"
      max="128"
      value={config.lora_alpha}
      onChange={(e) => dispatch(updateConfig({ lora_alpha: parseFloat(e.target.value) }))}
      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md"
    />
    <p className="text-xs text-gray-400 mt-1">
      Scaling factor for LoRA updates. Keep at 32 unless experimenting.
      Higher values = larger updates.
    </p>
  </div>

  {/* LoRA Dropout */}
  <div>
    <label className="block text-sm font-medium text-gray-300 mb-2">
      LoRA Dropout
    </label>
    <input
      type="number"
      min="0.0"
      max="0.5"
      step="0.05"
      value={config.lora_dropout}
      onChange={(e) => dispatch(updateConfig({ lora_dropout: parseFloat(e.target.value) }))}
      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md"
    />
    <p className="text-xs text-gray-400 mt-1">
      Regularization. Use 0.0 for small datasets, 0.05-0.1 for large datasets.
    </p>
  </div>

  {/* Layer Coverage */}
  <div>
    <label className="block text-sm font-medium text-gray-300 mb-2">
      Layer Coverage
    </label>
    <select
      value={config.lora_num_layers}
      onChange={(e) => dispatch(updateConfig({ lora_num_layers: parseInt(e.target.value) }))}
      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md"
    >
      <option value="-1">All Layers (Recommended)</option>
      <option value="16">Last 16 Layers</option>
      <option value="8">Last 8 Layers</option>
    </select>
    <p className="text-xs text-gray-400 mt-1">
      All layers recommended for best performance.
    </p>
  </div>
</div>

{/* Update Learning Rate Section */}
<div>
  <label className="block text-sm font-medium text-gray-300 mb-2">
    Learning Rate
  </label>
  <input
    type="number"
    step="0.00001"
    value={config.learning_rate}
    onChange={(e) => dispatch(updateConfig({ learning_rate: parseFloat(e.target.value) }))}
    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md"
  />
  <p className="text-xs text-gray-400 mt-1">
    LoRA training works best with 10x higher learning rates than full fine-tuning.
    Default: 1e-4 (vs 1e-5 for full fine-tuning)
  </p>
</div>
```

### Inference Improvements (Compare Page)

**Reference:** XXX repository `mlx-finetune-gui/backend/main.py` (lines 1136-1352)

#### Unified Inference Endpoint

**Create new `/models/inference` endpoint** that handles both base and adapted models:

**Key Features:**
1. **Chat Template Support** - Automatically applies model's chat template if available
2. **Adapter Validation** - Validates adapter against base model before inference
3. **Improved Sampling** - Uses `top_p=0.9`, `min_p=0.05`, `top_k=40` for better quality
4. **Best Model Selection** - Automatically uses `best_adapters.safetensors` if available
5. **Diagnostics** - Returns adapter warnings, configuration, and chat template status

**Implementation Pattern:**
```python
@app.post("/models/inference")
async def model_inference(request_data: dict):
    # 1. Extract parameters
    prompt = request_data.get("prompt", "").strip()
    model_name = request_data.get("model_name")
    adapter_name = request_data.get("adapter_name")  # Optional
    max_tokens = request_data.get("max_tokens", 100)
    temperature = request_data.get("temperature", 0.7)

    # 2. Build model path and validate
    model_path = os.path.join(base_model_dir, model_name)

    # 3. Load model config for layer count
    total_layers = None
    base_config_path = os.path.join(model_path, "config.json")
    if os.path.exists(base_config_path):
        with open(base_config_path, "r") as f:
            base_config = json.load(f)
        total_layers = base_config.get("num_hidden_layers")

    # 4. Validate adapter if specified
    adapter_details = {
        "name": adapter_name,
        "type": "none",
        "warnings": [],
        "chat_template_used": False,
    }

    if adapter_name:
        # Validate adapter_config.json
        adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)

            # Check model match
            recorded_model = adapter_config.get("model")
            if recorded_model and os.path.abspath(recorded_model) != os.path.abspath(model_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Adapter trained for '{recorded_model}', not '{model_path}'"
                )

            # Check layer coverage
            recorded_layers = adapter_config.get("num_layers")
            if total_layers and recorded_layers not in (-1, total_layers):
                adapter_details["warnings"].append(
                    f"Adapter covers {recorded_layers} layers; model has {total_layers}"
                )

            # Check LoRA keys
            lora_params = adapter_config.get("lora_parameters", {})
            expected_keys = ["self_attn.q_proj", "self_attn.k_proj", ...]
            missing_keys = [k for k in expected_keys if k not in lora_params.get("keys", [])]
            if missing_keys:
                adapter_details["warnings"].append(f"Missing keys: {', '.join(missing_keys)}")

    # 5. Generate using MLX with chat template and improved sampling
    script = f"""
import json
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

# Load model
model, tokenizer = load("{model_path}", adapter_path={adapter_path_repr})

# Apply chat template if available
template_used = False
if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
    try:
        messages = [{{"role": "user", "content": "{prompt}"}}]
        templated_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        template_used = True
    except:
        templated_prompt = "{prompt}"
else:
    templated_prompt = "{prompt}"

# Tokenize
prompt_tokens = mx.array(tokenizer.encode(templated_prompt))

# Generate with anti-repetition sampling
sampler = make_sampler(temp={temperature}, top_p=0.9, min_p=0.05, top_k=40)
detokenizer = tokenizer.detokenizer

print(f"CHAT_TEMPLATE_USED={{int(template_used)}}")
print("RESPONSE_START")
for token, _ in generate_step(prompt_tokens, model, max_tokens={max_tokens}, sampler=sampler):
    detokenizer.add_token(token.item())
detokenizer.finalize()
print(detokenizer.text)
print("RESPONSE_END")
"""

    # 6. Execute and parse response
    process = subprocess.run([python_path, "-c", script], capture_output=True, text=True)

    # Extract chat_template_used
    for line in process.stdout.splitlines():
        if line.startswith("CHAT_TEMPLATE_USED="):
            adapter_details["chat_template_used"] = bool(int(line.split("=")[1]))

    # Extract response between markers
    start = process.stdout.find("RESPONSE_START") + len("RESPONSE_START")
    end = process.stdout.find("RESPONSE_END")
    response_text = process.stdout[start:end].strip()

    return {
        "success": True,
        "prompt": prompt,
        "response": response_text,
        "model_info": {
            "base_model": model_name,
            "adapter": adapter_name or "none",
            "adapter_type": adapter_type,
            "adapter_details": adapter_details,
        }
    }
```

**Update Compare Page to use this endpoint:**
```typescript
// Call unified endpoint for both base and adapted models
const baseResponse = await axios.post('/models/inference', {
  prompt,
  model_name: modelName,
  adapter_name: null,  // No adapter for base
  max_tokens,
  temperature
});

const adaptedResponse = await axios.post('/models/inference', {
  prompt,
  model_name: modelName,
  adapter_name: adapterName,
  max_tokens,
  temperature
});

// Display adapter diagnostics
{adaptedResponse.data.model_info.adapter_details.warnings.length > 0 && (
  <div className="mt-2 p-2 bg-yellow-900/30 border border-yellow-700/50 rounded">
    <p className="text-xs text-yellow-200">
      ⚠️ Warnings: {adaptedResponse.data.model_info.adapter_details.warnings.join(', ')}
    </p>
  </div>
)}
```

### Implementation Checklist

When implementing full-layer LoRA upgrade:

**Backend Changes:**
- [ ] Update `TrainingConfig` dataclass with LoRA fields (main.py:44-63)
- [ ] Add LoRA parameter generation logic (main.py:398-458)
- [ ] Add architecture detection from `config.json`
- [ ] Pass `lora_parameters` and `num_layers` to training script
- [ ] Change default learning rate from `1e-5` to `1e-4`
- [ ] Update `/training/start` endpoint to accept new fields (main.py:761-793)
- [ ] Create `/models/inference` unified endpoint (main.py:1136-1352)
- [ ] Add adapter validation logic
- [ ] Implement chat template detection and application
- [ ] Add improved sampling parameters (top_p, min_p, top_k)

**Frontend Changes:**
- [ ] Update Redux `trainingSlice.ts` with LoRA fields
- [ ] Add LoRA Configuration section to `SetupPage.tsx`
- [ ] Add rank, alpha, dropout, layer coverage controls
- [ ] Add helper text and info banners
- [ ] Update learning rate default and description
- [ ] Update Compare page to use `/models/inference` endpoint
- [ ] Display adapter diagnostics and warnings
- [ ] Show chat template status
- [ ] Add temperature and max_tokens controls

**Testing:**
- [ ] Verify trainable parameter count (~3.5-4% for rank=32)
- [ ] Test with different architectures (Qwen2, Mixtral, Qwen2 MoE)
- [ ] Compare training curves: full-layer vs attention-only
- [ ] Test inference with chat templates
- [ ] Test adapter validation (mismatched models)
- [ ] Verify Enhanced Setup tab unchanged

**Documentation:**
- [ ] Update this CLAUDE.md with implementation notes
- [ ] Document rank selection guidelines
- [ ] Add troubleshooting section
- [ ] Update README.md

### Expected Results

**Training Log Output:**
```
LoRA Configuration:
  Rank: 32
  Alpha (scale): 32.0
  Dropout: 0.0
  Layer coverage: all transformer layers (-1)
  Target matrices: self_attn.q_proj, self_attn.k_proj, self_attn.v_proj,
                   self_attn.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj

Loading pretrained model: /path/to/Qwen2.5-0.5B-Instruct
Total parameters: 494.033M
Trainable parameters: 3.562% (17.596M/494.033M)  # ✅ Expected range

Starting training...
Iter 1: train_loss 2.453, val_loss 2.198, lr 0.0001
```

**Performance Expectations:**
- Training time: 5-15% increase vs attention-only
- Memory usage: <20% increase
- Final performance: Near-parity with full fine-tuning
- Sample efficiency: 2-3x better than attention-only

### Reference Files

**XXX Repository (Validated Implementation):**
- `/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Fine Tuner XXX/mlx-finetune-gui/backend/main.py`
- `FULL_LAYER_LORA_AUDIT.md` - Implementation audit and verification
- `LORA_ALL_LAYERS_IMPLEMENTATION.md` - Implementation summary

**Research:**
- "LoRA Without Regret" (Schulman et al., 2025): https://thinkingmachines.ai/blog/lora/

**DO NOT MODIFY:**
- Enhanced Setup Tab (`backend/main_enhancements.py`)
- GRPO/GSPO/Dr. GRPO functionality

---
