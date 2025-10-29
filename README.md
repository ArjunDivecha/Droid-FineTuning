# Droid FineTuning

A streamlined MLX fine-tuning desktop application for Apple Silicon Macs with advanced training methods including GSPO and Dr. GRPO.

## ✨ Features

- 🖥️ **Modern Desktop GUI** - Beautiful dark theme Electron app with React interface
- 🚀 **MLX Fine-Tuning** - Optimized for Apple Silicon (M1/M2/M3/M4)
- 🧠 **Advanced Training Methods** - SFT, GSPO, Dr. GRPO, GRPO, and On-Policy Distillation
- 🎓 **Knowledge Distillation** - Transfer knowledge from large teacher models to smaller students
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
│   │   ├── run_distillation.py    # Main distillation script
│   │   ├── teacher_model.py       # Teacher model wrapper
│   │   ├── student_model.py       # Student model wrapper
│   │   ├── distillation_trainer.py # Training orchestration
│   │   └── distillation_loss.py   # KL divergence loss
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
- macOS with Apple Silicon (M1/M2/M3/M4)
- Node.js 18+ and npm
- Python 3.11+
- MLX framework installed

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/ArjunDivecha/Droid-FineTuning.git
cd Droid-FineTuning

# Install root dependencies
npm install

# Install frontend dependencies
cd frontend && npm install && cd ..

# Install backend dependencies
pip3.11 install uvicorn fastapi websockets pyyaml
```

### 📦 Getting Models

**Important:** Models are NOT included in the repository. You need to download them separately.

#### Option 1: Download from Hugging Face (Recommended)

```bash
# Install MLX if you haven't already
pip3.11 install mlx mlx-lm

# Download a model (example: Qwen2.5-0.5B-Instruct)
python3.11 -m mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B-Instruct --mlx-path ./models/Qwen2.5-0.5B-Instruct

# Or download other models:
# python3.11 -m mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./models/Qwen2.5-7B-Instruct
```

#### Option 2: Use Existing MLX Models

If you already have MLX models, the app will auto-detect them. Update the backend to point to your model directory:

```python
# In backend/main.py, update the models_dir path:
models_dir = "/path/to/your/models"
```

The app will automatically scan this directory and show all available models in the dropdown.

### Run the Application

#### Using the convenience scripts (Recommended):

```bash
# Add to your ~/.zshrc for easy access
source ~/.zshrc

# Start the app
startmlxnew

# Stop the app
killmlxnew
```

#### Or manually:

```bash
# Terminal 1: Start backend
cd backend && python3.11 -m uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2: Build frontend and start Electron
cd frontend && npm run build && cd .. && npm start
```

## 📋 Usage

### Standard Training (Setup Page)
1. **Select Model** - Choose from available models in dropdown
2. **Upload Training Data** - Provide JSONL format training data
3. **Configure Parameters** - Set learning rate, batch size, iterations, etc.
4. **Start Training** - Monitor real-time progress

### Enhanced Training (Enhanced Setup Page) 🆕
1. **Choose Training Method:**
   - **SFT** - Standard supervised fine-tuning
   - **GSPO** - Group Sparse Policy Optimization with importance sampling
   - **Dr. GRPO** - Decoupled Rewards GRPO for stable training
   - **GRPO** - Group Relative Policy Optimization for reasoning tasks

2. **Select Model** - Auto-populated dropdown with all available models
3. **Configure Training** - Method-specific parameters
4. **Start Training** - GPU will run at optimal performance

### On-Policy Distillation (OPD) 🎓

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

## 🐛 Troubleshooting

### Models not showing in dropdown
- Check that `models_dir` in `backend/main.py` points to your models directory
- Ensure models are in MLX format (not PyTorch)
- Restart the backend after changing the path

### Training not starting
- Verify training data is in JSONL format
- Check that model path and data path are correct
- Look at backend logs: `tail -f logs/backend.log`

### GPU not running
- Ensure you're using Python 3.11+ with MLX installed
- Check Activity Monitor for GPU usage
- Verify training actually started (check backend logs)

### Port already in use
```bash
# Kill existing processes
killmlxnew
# Or manually:
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

### Distillation Issues

**Adapter dimension mismatch:**
- Ensure the adapter was trained on the same base model as the student
- Check `adapter_config.json` to verify model compatibility
- Example: Don't use a 0.5B adapter with a 7B student model

**Out of memory:**
- Reduce batch size (`--batch-size 1` or `2`)
- Lower max tokens (`--max-tokens 256`)
- Use smaller models or enable teacher unloading in config

## 📚 Data Format

Training data should be in JSONL format:

```jsonl
{"text": "User: What is MLX?\nAssistant: MLX is Apple's machine learning framework..."}
{"text": "User: How do I fine-tune?\nAssistant: To fine-tune a model..."}
```

---

**Simple. Lean. Focused. Advanced fine-tuning for Apple Silicon.**
