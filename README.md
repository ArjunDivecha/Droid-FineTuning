# Droid FineTuning

A streamlined MLX fine-tuning desktop application for Apple Silicon Macs. Features a clean Electron/React interface with FastAPI backend for real-time training monitoring and session management.

## ✨ Features

- 🖥️ **Modern Desktop GUI** - Electron app with React/Redux and dark mode
- 🚀 **MLX Fine-Tuning** - LoRA adapters optimized for Apple Silicon (M1/M2/M3/M4)
- 📊 **Real-Time Monitoring** - Live training progress with WebSocket updates
- 🎯 **Best Model Tracking** - Automatically saves best checkpoint based on validation loss
- 🆚 **Model Comparison** - Test and compare base vs fine-tuned model responses
- 💾 **Session Management** - Auto-save and restore training sessions with full state
- ⏹️ **Early Stopping** - Configurable early stopping with patience threshold
- 🔄 **Auto-Resume** - Automatically loads most recent session on startup
- 📁 **Electron File Dialogs** - Native macOS file selection for training data
- ⚡ **Lean & Fast** - Focused on core fine-tuning without unnecessary complexity

## 🏗️ Architecture

```
Droid-FineTuning/
├── backend/                    # FastAPI server (Python)
│   ├── main.py                 # Training API + WebSocket (14 endpoints)
│   └── requirements.txt
├── frontend/                   # React GUI (TypeScript)
│   ├── src/
│   │   ├── pages/              # Main application pages
│   │   │   ├── SetupPage.tsx           # Model selection & config
│   │   │   ├── TrainingPage.tsx        # Real-time monitoring
│   │   │   ├── ResultsPage.tsx         # Training analytics
│   │   │   └── ComparePage.tsx         # Model comparison
│   │   ├── components/         # Reusable UI components
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── LoadSessionModal.tsx
│   │   │   ├── ModelTestModal.tsx
│   │   │   ├── TrainingChart.tsx       # Line charts for metrics
│   │   │   ├── LogViewer.tsx           # Training log display
│   │   │   ├── NotificationCenter.tsx
│   │   │   └── StatusBar.tsx
│   │   ├── store/              # Redux state management
│   │   │   ├── store.ts
│   │   │   └── slices/
│   │   │       ├── trainingSlice.ts    # Training state
│   │   │       ├── modelsSlice.ts      # Model management
│   │   │       └── uiSlice.ts          # UI preferences
│   │   └── hooks/
│   │       └── useWebSocket.ts         # WebSocket connection
│   └── package.json
├── src/                        # Electron main process (TypeScript)
│   ├── main.ts                 # Window management
│   └── preload.ts              # IPC bridge for file dialogs
└── package.json                # Root dependencies & build scripts
```

## 🚀 Quick Start

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Node.js 18+** and npm
- **Python 3.9+**
- **MLX environment** set up with `mlx-lm`
- **External fine-tuning script** at configured path (see Configuration)

### Installation & Setup

```bash
# Clone the repository
git clone <repository-url>
cd Droid-FineTuning

# Install root dependencies
npm install

# Install frontend dependencies
cd frontend && npm install && cd ..

# Install backend dependencies (in your MLX virtual environment)
source "/path/to/your/mlx/.venv/bin/activate"
cd backend && pip install -r requirements.txt && cd ..
```

### Run the Application

**Important:** The GUI requires two components running:

1. **Backend server (FastAPI)** - Must be started separately
2. **Frontend GUI (Electron)** - Starts automatically

```bash
# Terminal 1: Start backend server
cd backend
source "/path/to/your/mlx/.venv/bin/activate"
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Electron GUI
npm run dev
```

The app will:
1. Connect to backend at `http://localhost:8000`
2. Establish WebSocket connection for real-time updates
3. Auto-load the most recent training session (if exists)
4. Display session history and allow resuming previous training

## 📋 How It Works: Deep Dive

### 1. Application Startup

**When you run `npm run dev`:**

1. **Main Process (Electron)** starts:
   - Creates 1400x900 window with dark theme
   - Sets up IPC handlers for native file dialogs
   - Configures window behavior (minimize, close, etc.)

2. **Frontend (React)** initializes:
   - Connects to WebSocket at `ws://localhost:8000/ws`
   - Fetches current training status from `/training/status`
   - Loads available models from `/models`
   - Restores UI state from localStorage (sidebar collapse, theme, etc.)

3. **Backend (FastAPI)** initializes:
   - Creates `TrainingManager` singleton
   - Auto-loads most recent session from `sessions/latest.json`
   - Restores training config, metrics, and state
   - If session was "running", sets state to "idle" (no auto-resume of subprocess)

### 2. Setup Page: Configuring Training

**Model Selection:**
- Fetches available models from hardcoded model directory
- Displays model name, size, and path
- Sets `model_path` in config when selected

**Training Data:**
- Click "Browse" → Opens native macOS file dialog (via Electron IPC)
- Supports `.jsonl` and `.json` files
- Validates file paths before training
- Requires both training and validation datasets

**Hyperparameters (Configurable):**
```python
learning_rate: float = 1e-5           # AdamW learning rate
batch_size: int = 1                   # Training batch size
max_seq_length: int = 32768           # Maximum sequence length
iterations: int = 7329                # Total training steps
steps_per_report: int = 25            # Metrics logging frequency
steps_per_eval: int = 200             # Validation frequency
save_every: int = 25                  # Checkpoint save frequency
early_stop: bool = True               # Enable early stopping
patience: int = 3                     # Early stop patience (evals)
adapter_name: str = "mlx_finetune"    # LoRA adapter folder name
```

**What Happens on "Start Training":**

1. **Frontend** sends `POST /training/start` with config
2. **Backend** creates YAML config file:
   ```yaml
   model: /path/to/model
   train_data: /path/to/train.jsonl
   val_data: /path/to/val.jsonl
   learning_rate: 1e-5
   # ... etc
   ```

3. **Backend** starts training subprocess:
   ```bash
   /path/to/run_finetune.py --config /tmp/config.yaml
   ```

4. **Backend** generates unique session ID (UUID)
5. **Training state** changes to `"running"`
6. **WebSocket** broadcasts `{"type": "training_started"}`
7. **Frontend** navigates to Training Page automatically

### 3. Training Page: Real-Time Monitoring

**What You See:**

1. **Progress Bar:**
   - Current step / Total steps
   - Percentage complete
   - Animated gradient fill

2. **Live Metrics Cards:**
   - **Train Loss** - Current training loss
   - **Val Loss** - Latest validation loss (updated every 200 steps)
   - **Learning Rate** - Current LR (may decay)
   - **Time Remaining** - ETA based on recent step timing

3. **Training Chart:**
   - Line chart showing train/val loss over time
   - Auto-updates every 2 seconds
   - Supports zoom and pan
   - Shows up to 1000 recent data points

4. **Log Viewer** (toggleable):
   - Last 1000 lines of training output
   - Format: `Iter X: Train loss Y, Val loss Z`
   - Auto-scrolls to bottom
   - Syntax highlighting for metrics

**How Real-Time Updates Work:**

1. **Polling (Primary Method):**
   - Frontend polls `/training/status` every 2 seconds
   - Parses metrics from API response
   - Updates Redux store
   - Detects new training runs by comparing start_time

2. **WebSocket (Supplementary):**
   - Receives `training_progress` events
   - Receives `training_completed` / `training_error` events
   - Shows toast notifications for state changes

3. **Log Parsing:**
   - Frontend extracts step number from log lines
   - Only shows new log entries (step tracking)
   - Clears logs when new training starts

**Backend Training Process Management:**

```python
# Backend spawns subprocess
self.current_process = subprocess.Popen(
    ["/path/to/run_finetune.py", "--config", config_path],
    stdout=log_file,
    stderr=subprocess.STDOUT,
    preexec_fn=os.setsid  # Process group for clean shutdown
)

# Monitors process in background
asyncio.create_task(self._monitor_training())
```

**Best Model Tracking:**

- **Monitors** validation loss at each evaluation
- **Compares** to `self.best_val_loss` (starts as `None`)
- **Saves** best checkpoint automatically:
  ```
  lora_adapters/
    mlx_finetune/
      0000025_adapters.safetensors  # Regular checkpoint
      0000050_adapters.safetensors
      0000075_adapters.safetensors
      best_adapters.safetensors     # ← Best model copy
      adapters.safetensors           # Final model
  ```
- **Broadcasts** `best_model_updated` event via WebSocket
- **Displays** best model info in UI (step, val_loss, path)

**Early Stopping:**

- **Tracks** validation loss for `patience` consecutive evaluations
- **Stops** training if no improvement for 3 evals (configurable)
- **Sets** state to `"completed"` (not `"error"`)
- **Saves** session with early_stopped flag
- **Keeps** best model checkpoint

**Stop Button:**

1. Click "Stop Training" → Confirms intent
2. Sends `POST /training/stop`
3. Backend sends `SIGTERM` to process group
4. Waits up to 10 seconds for graceful shutdown
5. Forces `SIGKILL` if timeout
6. Sets state to `"stopped"`
7. Saves session with partial results

### 4. Results Page: Training Analytics

**What You See:**

1. **Final Metrics Summary:**
   - Total steps completed
   - Final train loss
   - Final validation loss
   - Best validation loss achieved
   - Training duration

2. **Loss Chart:**
   - Train and validation loss curves
   - Highlights best model point
   - Shows early stopping point (if applicable)

3. **Training History Table:**
   - Step number
   - Train loss
   - Val loss
   - Timestamp
   - Sortable and filterable

4. **Session Information:**
   - Session ID
   - Model used
   - Adapter name and path
   - Training data paths
   - Hyperparameters used

5. **Action Buttons:**
   - "Load Session" - Restore any previous session
   - "Test Model" - Quick inference test
   - "Go to Compare" - Navigate to comparison page

### 5. Compare Page: Model Testing

**Three-Way Comparison:**

1. **Base Model (No Adapter)**
2. **Fine-Tuned Model (Final Checkpoint)**
3. **Best Model (Best Val Loss)**

**How It Works:**

1. **Enter Prompt** in text area
2. **Select Models** to compare (checkboxes)
3. **Configure Generation:**
   ```python
   max_tokens: int = 1024      # Max response length
   temperature: float = 0.7    # Sampling temperature
   ```

4. **Click "Compare"**

5. **Backend** calls `/models/inference` for each model:
   ```python
   # For base model
   response = mlx_lm.generate(
       model=base_model,
       prompt=prompt,
       max_tokens=max_tokens,
       temp=temperature
   )

   # For fine-tuned model
   response = mlx_lm.generate(
       model=base_model,
       adapter_path="/path/to/adapters.safetensors",
       prompt=prompt,
       max_tokens=max_tokens,
       temp=temperature
   )
   ```

6. **Frontend** displays side-by-side:
   - Prompt (shared)
   - Response 1 (Base)
   - Response 2 (Fine-Tuned)
   - Response 3 (Best)
   - Generation time for each
   - Token count for each

**Model Test Modal:**

- Quick inline testing without full comparison
- Single model at a time
- Faster iteration for checking adapter quality

### 6. Session Management

**Automatic Session Saving:**

- **Triggers:**
  - Every time metrics update
  - When training completes
  - When training stops
  - When training errors
  - On best model update

- **Saved Data:**
  ```json
  {
    "session_id": "uuid-string",
    "timestamp": "2025-01-15T10:30:00",
    "training_state": "completed",
    "config": {
      "model_path": "/path/to/model",
      "train_data_path": "/path/to/train.jsonl",
      // ... full config
    },
    "metrics": {
      "current_step": 1000,
      "total_steps": 1000,
      "train_loss": 0.234,
      "val_loss": 0.456,
      "start_time": "2025-01-15T10:00:00",
      "estimated_time_remaining": 0
    },
    "adapter_path": "/path/to/adapters.safetensors",
    "best_model": {
      "val_loss": 0.445,
      "step": 750,
      "path": "/path/to/best_adapters.safetensors"
    }
  }
  ```

- **Storage:**
  ```
  sessions/
    session_uuid1.json
    session_uuid2.json
    session_uuid3.json
    latest.json        # Points to most recent session
  ```

**Loading Sessions:**

1. **Automatic on Startup:**
   - Reads `latest.json`
   - Loads session file
   - Restores config and metrics
   - Does NOT restart training (subprocess is not restored)
   - Sets state to "idle" if was "running"

2. **Manual Load:**
   - Click "Load Session" button
   - Shows modal with session list
   - Displays: timestamp, model, final metrics
   - Click to load any session
   - Restores full UI state

3. **Session List Endpoint:**
   ```bash
   GET /sessions
   # Returns array of all sessions, sorted by timestamp
   ```

## 🔌 API Endpoints

### Training Endpoints

```bash
# Get current training status
GET /training/status
# Returns: state, config, metrics

# Start training
POST /training/start
# Body: TrainingConfig (JSON)
# Returns: success message

# Stop training
POST /training/stop
# Returns: stopped status

# Get training logs
GET /training/logs
# Returns: last 100 log lines
```

### Model Endpoints

```bash
# List available models
GET /models
# Returns: array of models with name, size, path

# Get all models and adapters
GET /models/available
# Returns: models with their available adapters

# Test base model (no adapter)
POST /model/test-base
# Body: {prompt, max_tokens, temperature}
# Returns: response text

# Test fine-tuned model (with adapter)
POST /model/test
# Body: {prompt, adapter_name, max_tokens, temperature}
# Returns: response text

# Generic inference (specify model + optional adapter)
POST /models/inference
# Body: {prompt, model_name, adapter_name?, max_tokens, temperature}
# Returns: response with metadata
```

### Session Endpoints

```bash
# List all sessions
GET /sessions
# Returns: array of session metadata

# Get specific session
GET /sessions/{session_id}
# Returns: full session data

# Load session
POST /sessions/{session_id}/load
# Restores session as current
# Returns: loaded session data

# Delete session (TODO)
DELETE /sessions/{session_id}
```

### System Endpoints

```bash
# Health check
GET /health
# Returns: {status: "healthy", timestamp}

# WebSocket connection
WS /ws
# Events: training_started, training_progress, training_completed,
#         training_error, best_model_updated
```

## 🔧 Configuration

### Backend Configuration

**Hardcoded Paths** (edit `backend/main.py`):

```python
# Training script path
sys.path.append('/path/to/one_step_finetune')

# Output directory for adapters
self.output_dir = "/path/to/lora_adapters"

# Log file location
self.log_file = "/path/to/logs/gui_training.log"

# Sessions directory
self.sessions_dir = "/path/to/sessions"
```

**Training Script:**

The app calls an external training script:
```bash
/path/to/run_finetune.py --config /tmp/config_uuid.yaml
```

Ensure your training script:
- Accepts `--config` parameter
- Reads YAML config file
- Outputs progress to stdout in expected format
- Creates LoRA adapters in `.safetensors` format
- Saves checkpoints with naming: `{step:07d}_adapters.safetensors`

### Frontend Configuration

**Backend URL** (edit `frontend/src/*/`):

```typescript
const BACKEND_URL = 'http://localhost:8000';
```

### Model Directory

Place MLX models in the configured directory. Expected structure:

```
models/
  Qwen2.5-7B-Instruct/
    config.json
    tokenizer.json
    model.safetensors
    # ... other model files
```

## 📝 Scripts

```bash
# Development
npm run dev                     # Start Electron GUI (frontend only)
npm run dev:backend             # Start FastAPI backend (separate terminal)

# Building
npm run build                   # Build both frontend and main process
npm run build:frontend          # Build React app only
npm run build:main              # Build Electron main/preload only

# Production
npm run start                   # Start built Electron app
npm run pack                    # Package app (unpacked)
npm run dist                    # Create DMG installer for macOS
```

## 🎯 What Makes This Different

### Design Philosophy

This app is **lean and focused**:

- ✅ **GUI-First** - No CLI complexity, everything through UI
- ✅ **Core Functionality** - Just training, no dataset generation tools
- ✅ **Streamlined** - Minimal dependencies, clean codebase
- ✅ **Session-Aware** - Never lose training progress
- ✅ **Best Model Tracking** - Always know your optimal checkpoint
- ✅ **Apple Silicon Optimized** - Built specifically for MLX on M-series chips

### Not Included (By Design)

- ❌ Dataset creation/augmentation tools
- ❌ Multiple training methods (DPO, ORPO, etc.)
- ❌ Quantization options (QLoRA)
- ❌ Distributed training
- ❌ Experiment tracking (W&B, MLflow)
- ❌ Built-in chat interface
- ❌ Model hosting/serving

Keep it simple. Keep it fast. Just fine-tuning.

## 🔬 Technical Details

### Tech Stack

**Frontend:**
- React 18 with TypeScript
- Redux Toolkit for state management
- Tailwind CSS for styling
- Lucide React for icons
- Chart.js for visualizations
- Axios for HTTP requests

**Backend:**
- FastAPI (async Python web framework)
- WebSocket for real-time communication
- Subprocess management for training
- YAML config generation
- JSON session persistence

**Desktop:**
- Electron 28
- Native file dialogs via IPC
- Auto-update ready (electron-builder)

### State Management

```typescript
// Redux Store Structure
{
  training: {
    state: 'idle' | 'running' | 'completed' | 'error' | 'stopped',
    config: TrainingConfig | null,
    metrics: TrainingMetrics | null,
    logs: string[],
    error: string | null,
    isConnected: boolean
  },
  models: {
    models: Model[],
    selectedModel: Model | null,
    isLoading: boolean,
    error: string | null
  },
  ui: {
    theme: 'light' | 'dark' | 'system',
    activeNavItem: 'setup' | 'training' | 'results' | 'compare',
    sidebarCollapsed: boolean,
    showLogs: boolean,
    notifications: Notification[]
  }
}
```

### WebSocket Events

```typescript
// Client → Server (heartbeat only)
// Just keep connection alive

// Server → Client
{type: 'training_started', data: {...}}
{type: 'training_progress', data: {metrics, log_line}}
{type: 'training_completed', data: {final_metrics}}
{type: 'training_error', data: {error}}
{type: 'best_model_updated', data: {step, val_loss, path}}
```

### Error Handling

**Frontend:**
- Try-catch on all API calls
- Toast notifications for errors
- Fallback UI states
- Retry logic for WebSocket reconnection

**Backend:**
- Exception handling on all endpoints
- Proper HTTP status codes (400, 404, 500)
- Logging of all errors
- Graceful subprocess cleanup on failure

## 🐛 Troubleshooting

### "Backend not responding"

**Check:**
1. Backend server is running: `curl http://localhost:8000/health`
2. Port 8000 not blocked by firewall
3. Python virtual environment activated
4. All dependencies installed: `pip install -r backend/requirements.txt`

### "Training won't start"

**Check:**
1. Training script path is correct in `main.py`
2. Config file is being generated in `/tmp/`
3. Training data files exist and are readable
4. Model path is correct and model files present
5. Backend logs for subprocess errors

### "WebSocket disconnected"

**Normal behavior** - WebSocket reconnects automatically every few seconds. If persistent:
1. Check backend logs for WebSocket errors
2. Verify no proxy/VPN blocking WebSocket
3. Try restarting backend

### "Session not loading"

**Check:**
1. `sessions/` directory exists
2. Session JSON files are valid JSON
3. `latest.json` points to existing session
4. Permissions allow reading session files

### "Model comparison fails"

**Check:**
1. Model path is correct
2. Adapter file exists at specified path
3. MLX environment is activated
4. Enough RAM for model inference (7B model ≈ 14GB RAM)

## 📚 Additional Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Electron Docs](https://www.electronjs.org/docs)

## 🤝 Contributing

This is a personal project focused on simplicity. If you want to contribute:

1. Keep it simple
2. No feature bloat
3. Match existing code style
4. Test on Apple Silicon

## 📄 License

MIT License - See LICENSE file for details

---

**Simple. Lean. Focused. Just fine-tuning.**

Built with ❤️ for the MLX community on Apple Silicon.
