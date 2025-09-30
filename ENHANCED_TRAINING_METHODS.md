# Enhanced Training Methods: GSPO and Dr. GRPO Integration

## Overview

This document describes the integration of advanced RL training methods into the Droid Fine-Tuning system using **mlx-lm-lora v0.8.1**. These methods enable policy optimization and reinforcement learning for improved model reasoning and instruction following.

## 🆕 Training Methods

### 1. SFT (Supervised Fine-Tuning) - ⭐⭐
**Standard approach for instruction following** (Use Standard Setup Tab)

- **Use Case**: General instruction following and task adaptation
- **Data Format**: `instruction/response` or `messages` format
- **Resource Intensity**: Medium (baseline)
- **Best For**: Standard fine-tuning tasks, general domain adaptation
- **Note**: Available in both Standard Setup and Enhanced Setup tabs

### 2. GSPO (Group Sparse Policy Optimization) - ⭐⭐⭐⭐ 🆕
**GRPO with importance sampling for improved efficiency**

- **What it is**: GRPO + token/sequence-level importance sampling
- **Use Case**: Policy optimization with improved sample efficiency
- **Data Format**: `prompt/answer/system` (same as GRPO)
- **Resource Intensity**: Medium-High (generates multiple completions)
- **Best For**: When you want GRPO benefits with better sample efficiency
- **Performance**: Comparable to GRPO with potentially better convergence

**Key Parameters:**
- `group_size` (2-16, default: 4): Number of completions per prompt
- `epsilon` (1e-4 to 1e-2, default: 0.0001): Numerical stability
- `temperature` (0.6-1.2, default: 0.8): Sampling randomness
- `max_completion_length` (128-2048, default: 512): Max tokens per completion
- `importance_sampling_level` ("token" or "sequence"): Sampling granularity

### 3. Dr. GRPO (Decoupled Rewards GRPO) - ⭐⭐⭐⭐⭐ 🆕
**GRPO variant with decoupled reward computation**

- **What it is**: GRPO with separated reward calculation for more stable training
- **Use Case**: When GRPO training is unstable or for complex reward structures
- **Data Format**: `prompt/answer/system` (same as GRPO)
- **Resource Intensity**: High (generates multiple completions + reward computation)
- **Best For**: Complex tasks requiring stable policy optimization
- **Performance**: More stable than standard GRPO, especially with large models

**Key Parameters:**
- `group_size` (2-16, default: 4): Number of completions per prompt
- `epsilon` (1e-4 to 1e-2, default: 0.0001): Lower bound for clipping
- `epsilon_high` (optional): Upper bound for clipping (DAPO variant)
- `temperature` (0.6-1.2, default: 0.8): Sampling randomness
- `max_completion_length` (128-2048, default: 512): Max tokens per completion

### 4. GRPO (Group Relative Policy Optimization) - ⭐⭐⭐⭐
**Multi-completion policy optimization for reasoning tasks**

- **What it is**: Generates multiple completions and learns from relative quality
- **Use Case**: Improving reasoning, instruction following, and response quality
- **Data Format**: `prompt/answer/system`
- **Resource Intensity**: High (generates 4-16 completions per prompt)
- **Best For**: Reasoning tasks, math, coding, complex problem-solving
- **Performance**: Outperforms SFT for reasoning-heavy tasks

**Key Parameters:**
- `group_size` (2-16, default: 4): Number of completions per prompt
- `epsilon` (1e-4 to 1e-2, default: 0.0001): Clipping parameter
- `temperature` (0.6-1.2, default: 0.8): Sampling randomness
- `max_completion_length` (128-2048, default: 512): Max tokens per completion

---

## 📊 Data Format (ALL GRPO METHODS)

**All GRPO-based methods (GSPO, Dr. GRPO, GRPO) use the same simple format:**

```jsonl
{"prompt": "What is the capital of France?", "answer": "The capital of France is Paris, a historic city known for its art, culture, and iconic landmarks like the Eiffel Tower.", "system": "You are a helpful and knowledgeable assistant."}
{"prompt": "Explain machine learning in simple terms.", "answer": "Machine learning is a way for computers to learn from examples and experience, rather than following explicit programmed instructions. It's like teaching a child - you show them many examples and they learn to recognize patterns.", "system": "You are a helpful and knowledgeable assistant."}
```

**Required Fields:**
- `prompt` (string): The user's question or instruction
- `answer` (string): The reference/expected response

**Optional Fields:**
- `system` (string): System message to guide the model's behavior

**Directory Structure:**
Your data should be in a directory with these files:
```
data/
├── train.jsonl  # Training data
└── valid.jsonl  # Validation data (optional but recommended)
```

---

## 🏗️ Architecture

```
backend/
├── training_methods.py      # Training method configurations
├── main_enhancements.py     # Enhanced training manager (mlx-lm-lora integration)
└── main.py                  # Standard SFT training (unchanged)

frontend/src/
├── pages/EnhancedSetupPage.tsx  # Enhanced training UI
└── pages/SetupPage.tsx          # Standard SFT UI (unchanged)
```

**Important:** Standard Setup tab uses `mlx_lm.lora` (Apple's official) for SFT. Enhanced Setup tab uses `mlx_lm_lora.train` (Goekdeniz-Guelmez) for GRPO methods.

---

## 🔄 API Endpoints

### Enhanced Training Endpoints

- `GET /api/training/methods` - Get available training methods
- `POST /api/training/validate-data` - Validate training data format
- `POST /api/training/estimate-resources` - Estimate resource requirements
- `POST /api/training/start-enhanced` - Start enhanced training

### Example API Call

```javascript
// Start GSPO training
const config = {
  training_method: 'gspo',
  model_path: '/path/to/model',
  train_data_path: '/path/to/data/train.jsonl',
  // GRPO parameters
  group_size: 4,
  epsilon: 0.0001,
  temperature: 0.8,
  max_completion_length: 512,
  // GSPO-specific
  importance_sampling_level: 'token',
  // Standard parameters
  learning_rate: 1e-5,
  batch_size: 1,
  iterations: 100,
};

const result = await fetch('/api/training/start-enhanced', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(config)
});
```

---

## 🔧 Configuration Parameters

### Base Configuration (All Methods)
- `model_path`: Path to MLX model directory
- `train_data_path`: Path to training data directory (containing train.jsonl)
- `val_data_path`: Path to validation data (optional)
- `learning_rate`: Learning rate (default: 1e-5)
- `batch_size`: Batch size (default: 1)
- `max_seq_length`: Maximum sequence length (default: 2048)
- `iterations`: Number of training iterations
- `steps_per_report`: Report frequency (default: 10)
- `steps_per_eval`: Evaluation frequency (default: 25)
- `save_every`: Checkpoint save frequency (default: 100)

### GRPO/GSPO/Dr. GRPO Parameters
- `group_size` (2-16, default: 4): How many completions to generate per prompt
  - Lower (2-4): Faster but less learning signal
  - Higher (8-16): Better learning but slower and more memory

- `epsilon` (1e-4 to 1e-2, default: 0.0001): Clipping parameter for stability
  - Lower values: More conservative updates
  - Higher values: More aggressive updates

- `temperature` (0.6-1.2, default: 0.8): Sampling temperature
  - Lower (0.6-0.7): More deterministic, less diverse
  - Higher (0.9-1.2): More creative, more diverse

- `max_completion_length` (128-2048, default: 512): Maximum tokens for generated completions

### GSPO-Specific
- `importance_sampling_level`: "token", "sequence", or None
  - "token": Token-level importance weighting (most fine-grained)
  - "sequence": Sequence-level importance weighting
  - None: No importance sampling (standard GRPO)

### Dr. GRPO-Specific
- `epsilon_high` (optional): Upper epsilon bound for DAPO variant

### Advanced (Optional for All)
- `reward_functions`: Custom reward functions (e.g., "accuracy_reward,format_reward")
- `reward_weights`: Reward weights (e.g., "[0.7, 0.3]")

---

## 🎯 Method Selection Guide

### Choose SFT when:
- ✅ You have instruction-response pairs
- ✅ Standard fine-tuning is sufficient
- ✅ You want faster training
- ✅ You have limited compute resources

### Choose GRPO when:
- ✅ You want to improve reasoning quality
- ✅ You can generate multiple responses per prompt
- ✅ You have reference answers for comparison
- ✅ You want better instruction following

### Choose GSPO when:
- ✅ You want GRPO benefits with better sample efficiency
- ✅ You want faster convergence than standard GRPO
- ✅ You have limited training data

### Choose Dr. GRPO when:
- ✅ Standard GRPO training is unstable
- ✅ You're training large models
- ✅ You need more stable policy optimization
- ✅ You have complex reward structures

---

## 🧪 Resource Requirements

**Approximate multipliers vs. SFT:**

| Method | Memory | Time | Compute |
|--------|--------|------|---------|
| SFT | 1.0x | 1.0x | 1.0x (baseline) |
| GRPO | 1.3x | 3-5x | High (multiple generations) |
| GSPO | 1.3x | 3-5x | High (multiple generations) |
| Dr. GRPO | 1.5x | 3-5x | Very High (reward computation) |

**Note:** RL methods are computationally expensive because they generate multiple completions per training example and compute policy gradients.

---

## 📝 Best Practices

### Data Preparation
1. Start with high-quality instruction-response pairs
2. Use the simple `prompt/answer/system` format
3. Include 10-100+ examples for meaningful training
4. Create validation split to monitor overfitting

### Parameter Tuning
1. Start with defaults: `group_size=4`, `epsilon=0.0001`, `temperature=0.8`
2. Increase `group_size` if you have compute (better learning signal)
3. Adjust `temperature` based on desired creativity
4. Lower `epsilon` if training is unstable

### Resource Management
1. GRPO methods need 3-5x more time than SFT
2. Monitor memory usage (group_size × batch_size)
3. Start with small models for testing
4. Use validation data to prevent overfitting

---

## 🚀 Getting Started

### 1. Prepare Your Data
```bash
# Create data directory
mkdir -p my_training_data

# Format: {"prompt": "...", "answer": "...", "system": "..."}
echo '{"prompt": "What is 2+2?", "answer": "2+2 equals 4.", "system": "You are a helpful math tutor."}' > my_training_data/train.jsonl
```

### 2. Validate Data Format
Use the Enhanced Setup page to validate your data before training.

### 3. Configure Training
- Select training method (GSPO, Dr. GRPO, or GRPO)
- Adjust parameters based on your needs
- Review resource estimation

### 4. Start Training
Monitor real-time progress through the Training page.

---

## 🔍 Troubleshooting

### "Data validation failed"
- Check that you're using `prompt/answer/system` format
- Ensure each line is valid JSON
- Verify no empty strings in required fields

### "Out of memory"
- Reduce `group_size` (try 2 or 3)
- Reduce `batch_size` to 1
- Reduce `max_completion_length`
- Use smaller model

### "Training is slow"
- This is normal for RL methods (3-5x slower than SFT)
- Consider using GSPO for better efficiency
- Reduce `group_size` for faster iteration

### "Loss not improving"
- Check validation loss (might be overfitting)
- Try different `temperature` values
- Increase `group_size` for better learning signal
- Ensure data quality is high

---

## 📦 Requirements

- **mlx-lm-lora==0.8.1** (installed automatically)
- Python 3.11+
- MLX framework
- Apple Silicon Mac (M1/M2/M3/M4)

---

## 🎉 Summary

Enhanced training methods bring reinforcement learning capabilities to Droid Fine-Tuning:

- ✅ **3 RL methods**: GRPO, GSPO, Dr. GRPO
- ✅ **Simple data format**: Just prompt/answer/system
- ✅ **Automatic validation**: Built-in format checking
- ✅ **Resource estimation**: Know before you train
- ✅ **Backward compatible**: Standard SFT still works

Start with GRPO for general use, try GSPO for efficiency, or Dr. GRPO for stability!

---

**Generated:** 2025-09-29
**MLX-LM-LORA Version:** 0.8.1
**Branch:** feature/fix-mlx-lm-lora-integration