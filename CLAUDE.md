# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Droid FineTuning** is a desktop MLX fine-tuning application for Apple Silicon Macs with advanced training methods including GSPO (Group Sparse Policy Optimization) and Dr. GRPO (Domain-Refined GRPO).

**Tech Stack:**
- Backend: FastAPI (Python 3.11+) with WebSocket support
- Frontend: React 18 + TypeScript + Vite + TailwindCSS
- Desktop: Electron 28
- ML Framework: MLX (Apple Silicon optimized)

## Build & Development Commands

### Start Application
```bash
# Using convenience scripts (recommended)
./startmlxnew    # Start backend + frontend
./killmlxnew     # Stop all processes

# Or manually
cd backend && python3.11 -m uvicorn main:app --host 0.0.0.0 --port 8000
cd frontend && npm run build && cd .. && npm start
```

### Development
```bash
# Install dependencies
npm install                    # Root dependencies
cd frontend && npm install     # Frontend dependencies
pip3.11 install -r backend/requirements.txt  # Backend dependencies

# Build commands
npm run build                  # Build everything
npm run build:frontend         # Build React app only (in frontend/)
npm run build:main             # Build Electron main process only (TypeScript in src/)

# Development modes
npm run dev                    # Start frontend dev mode
npm run dev:backend            # Start backend with hot reload
npm run dev:frontend           # Build Electron main and start app
```

### Testing & Distribution
```bash
npm run start                  # Start built application
npm run dist                   # Package for distribution
npm run pack                   # Package without creating installer
```

### Port Management
**CRITICAL:** Always check and clear ports before starting:
```bash
lsof -ti:8000 | xargs kill -9  # Backend port
lsof -ti:3000 | xargs kill -9  # Frontend dev port
```

## Architecture

### Three-Layer Structure

1. **Backend** (`backend/`) - FastAPI server managing training
   - `main.py` (50K+ lines) - Core training manager, WebSocket, REST API, session management
   - `training_methods.py` - Training method configurations (SFT, GSPO, Dr. GRPO, GRPO)
   - `main_enhancements.py` - Enhanced training manager for advanced methods
   - `evaluation_api.py` - Adapter evaluation and comparison endpoints

2. **Frontend** (`frontend/src/`) - React GUI
   - `pages/SetupPage.tsx` - Standard SFT training setup
   - `pages/EnhancedSetupPage.tsx` - Advanced training methods (GSPO/Dr. GRPO/GRPO)
   - `pages/TrainingPage.tsx` - Real-time training monitoring
   - `pages/ResultsPage.tsx` - Training history and metrics
   - `pages/ComparePage.tsx` - Base vs fine-tuned model comparison
   - `pages/FusionPage.tsx` - Adapter fusion interface
   - `types/enhancedTraining.ts` - TypeScript definitions for training methods
   - `store/` - Redux state management

3. **Electron** (`src/`) - Desktop wrapper
   - `main.ts` - Electron main process (window management, IPC)
   - `preload.ts` - Electron preload script (security bridge)

### Key Data Flow

1. **Training Session**: Frontend → REST API → TrainingManager → MLX subprocess → WebSocket updates → Frontend
2. **Model Discovery**: Backend scans `models_dir` → Returns available MLX models → Frontend dropdown
3. **Comparison**: Frontend → Inference API → MLX generate → Side-by-side display

### Critical Paths

**Models Directory:** Configured in `backend/main.py` (line ~70):
```python
self.output_dir = "/path/to/lora_adapters"
self.log_file = "/path/to/logs/gui_training.log"
self.sessions_dir = "/path/to/sessions"
```

**Training Data Format:** JSONL with messages:
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Training Methods

### Available Methods (via `backend/training_methods.py`)

1. **SFT** (Supervised Fine-Tuning) - Standard LoRA fine-tuning
2. **GSPO** (Group Sparse Policy Optimization) - 2x faster, resource-efficient reasoning
3. **Dr. GRPO** (Doctor GRPO) - Domain-specialized reasoning (medical/scientific/legal)
4. **GRPO** (Group Relative Policy Optimization) - Multi-step reasoning (DeepSeek-R1 style)

Each method has specific:
- Data format requirements (see `adapter_fusion/GSPO_GRPO_DATASET_GUIDE.md`)
- Resource multipliers (memory/time estimation)
- Parameters (sparse_ratio, expertise_level, reasoning_steps, etc.)

## Adapter Fusion System

**Location:** `adapter_fusion/`

**Purpose:** Merge multiple fine-tuned adapters using different strategies.

**Key Files:**
- `fusion_adapters.py` - Core fusion logic (linear, SLERP, TIES, DARE)
- `evaluate_adapters.py` - Adapter testing and comparison
- `test_fused_models.py` - Fusion testing utilities

**Fusion Strategies:**
- Linear interpolation with weights
- SLERP (Spherical Linear Interpolation)
- TIES (Trim, Elect Sign & Merge)
- DARE (Drop And REscale)

## Important Implementation Details

### Backend Architecture

**TrainingManager Class** (in `main.py`):
- Singleton pattern managing training state
- WebSocket client management for real-time updates
- Session persistence (save/load training configs)
- Best model tracking with automatic saving
- Process lifecycle management (start/stop/monitor)

**Key State Variables:**
- `training_state`: idle, running, paused, completed, error
- `current_process`: Subprocess handle for MLX training
- `websocket_clients`: List of connected WebSocket clients
- `best_val_loss`: Track best validation checkpoint

### Frontend Architecture

**Redux Store** (`frontend/src/store/`):
- `slices/trainingSlice.ts` - Training state and progress
- Centralized state for all training parameters and metrics

**Real-time Updates:**
- WebSocket connection on port 8000
- Live training metrics (loss, tokens/sec, iteration progress)
- Progress bar updates based on iteration count

### Model Discovery

Backend automatically scans for MLX models:
```python
models_dir = "/path/to/models"  # Configurable
# Looks for directories with config.json and safetensors files
```

Frontend dropdown populated via `/api/models` endpoint.

## Testing Integration

**Integration Test:** `test_integration.py` - Validates all training methods and data validation.

**Test Coverage:**
- Training method configuration
- Resource estimation algorithms
- Data format validation
- Sample data generation
- Enhanced training manager functionality

## Development Notes

### When Adding New Training Methods

1. Add enum to `TrainingMethod` in `training_methods.py`
2. Define `TrainingMethodConfig` with parameters
3. Add to `TRAINING_METHODS` dictionary
4. Update TypeScript types in `frontend/src/types/enhancedTraining.ts`
5. Update UI in `EnhancedSetupPage.tsx`

### When Modifying Training Flow

Key areas to update:
1. `TrainingManager.start_training()` - Process spawning
2. `TrainingManager.monitor_training()` - Log parsing and WebSocket broadcasting
3. Frontend WebSocket handlers - Update state based on training events

### WebSocket Event Types

Backend sends these events:
- `training_update` - Progress/metrics updates
- `training_complete` - Training finished
- `training_error` - Error occurred
- `training_paused` - Training paused
- `best_model_saved` - New best checkpoint

### File Paths Are Absolute

All file paths in the backend use absolute paths for compatibility with the existing MLX environment. When modifying paths, update:
- `TrainingManager.__init__()` in `main.py`
- Any file operation code that references adapters, logs, or sessions

## Common Issues

### "Models not showing in dropdown"
- Check `models_dir` path in `backend/main.py` (around line 70)
- Ensure models are in MLX format (have `config.json` and `.safetensors` files)
- Restart backend after changing path

### "Port already in use"
- Run `./killmlxnew` or manually kill processes:
  ```bash
  lsof -ti:8000 | xargs kill -9
  lsof -ti:3000 | xargs kill -9
  ```

### "Training not starting"
- Check training data is JSONL format
- Verify paths are absolute and accessible
- Check logs: `tail -f logs/backend.log`

### "Frontend build fails"
- Ensure you're in `frontend/` directory when running `npm run build`
- Check TypeScript compilation: `cd frontend && tsc`

## Git Workflow

**Current Branch:** `feature/gspo-dr-grpo-integration`

**Main Branch:** Not specified (default to `main`)

**Recent Commits Show:**
- Adapter evaluation system and fusion framework
- Enhanced setup with model dropdown
- GSPO/Dr. GRPO integration
- Electron build fixes

## Logs and Debugging

**Log Locations:**
- Backend: `logs/backend.log`
- Frontend: `logs/frontend.log`
- Training output: Logged by subprocess and parsed by `monitor_training()`

**Debug Mode:**
- Backend has detailed logging with `logger.info/debug/error`
- Frontend uses console.log and Redux DevTools

## Dependencies

**Python (Backend):**
- FastAPI 0.104.1
- uvicorn[standard] 0.24.0
- websockets 12.0
- PyYAML 6.0.1 (for config files)
- MLX and mlx-lm (installed separately)

**Node (Frontend):**
- React 18 + Redux Toolkit
- Vite 5 (build tool)
- TailwindCSS 3 (styling)
- Chart.js + react-chartjs-2 (metrics visualization)
- socket.io-client 4.7.4 (WebSocket)

**Electron:**
- electron 28
- electron-builder 24.8.0

## Session Management

Sessions stored in `sessions_dir` as JSON files:
- Auto-saves training configuration
- Loads most recent session on backend startup
- Session ID tracked in `TrainingManager.current_session_id`
- Includes training config, metrics, and best model info