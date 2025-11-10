# Nested Learning - Quick Start Guide

## What is Nested Learning?

Nested Learning is a multi-frequency parameter update paradigm that prevents catastrophic forgetting during fine-tuning by updating different parameter groups at different rates:
- **Fast tier** (updates every step): Learns new task-specific information quickly
- **Medium tier** (updates every N steps): Balances old and new knowledge
- **Slow tier** (updates every M steps): Preserves core model capabilities

## How to Use

### Option 1: Via Frontend UI

1. **Navigate to Nested Learning**
   - Click "Nested Learning" in the sidebar
   - Or visit `http://localhost:3000/nested-learning`

2. **Select Your Model**
   - Choose base model from dropdown (e.g., Qwen2.5-7B-Instruct)
   - Select LoRA adapter from dropdown (e.g., 7b)
   - Pick training data file (JSONL format)

3. **Configure Tiers**
   - Set number of tiers (default: 3)
   - Adjust tier frequencies using sliders:
     - Tier 0: 1 (updates every step)
     - Tier 1: 2 (updates every 2 steps)
     - Tier 2: 4 (updates every 4 steps)
   - Choose assignment strategy: layer_depth or parameter_importance

4. **Set Training Parameters**
   - Number of steps (e.g., 1000)
   - Batch size (e.g., 2)
   - Learning rate (e.g., 1e-5)
   - Checkpointing frequency

5. **Start Training**
   - Click "Start Training"
   - Monitor progress in real-time
   - View tier-specific update counts

### Option 2: Via Python Script

```python
from nested_learning import NestedLearningConfig, NestedLoRATrainer

# 1. Create configuration
config = NestedLearningConfig(
    base_model_path="path/to/base/model",
    adapter_path="path/to/adapter",
    train_data_path="path/to/train.jsonl",
    num_tiers=3,
    tier_update_frequencies=[1, 2, 4],
    tier_assignment_strategy="layer_depth",
    learning_rate=1e-5,
    batch_size=2,
    num_steps=1000,
    experiment_name="my_experiment"
)

# 2. Create trainer
trainer = NestedLoRATrainer(config)

# 3. Setup (load model, assign tiers, etc.)
trainer.setup()

# 4. Train
trainer.train()

# 5. Checkpoints saved to: output_path/experiment_name/checkpoints/
```

### Option 3: Via API

```bash
# Start training
curl -X POST http://localhost:8000/nested-learning/start \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_path": "/path/to/model",
    "adapter_path": "/path/to/adapter",
    "train_data_path": "/path/to/data.jsonl",
    "num_tiers": 3,
    "tier_update_frequencies": [1, 2, 4],
    "num_steps": 1000,
    "experiment_name": "my_experiment"
  }'

# Check status
curl http://localhost:8000/nested-learning/status

# Get metrics
curl http://localhost:8000/nested-learning/metrics

# Stop training
curl -X POST http://localhost:8000/nested-learning/stop
```

## Configuration Options

### Essential Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `base_model_path` | Path to base model | Required | `/path/to/Qwen2.5-7B-Instruct` |
| `adapter_path` | Path to LoRA adapter | Required | `/path/to/7b` |
| `train_data_path` | Path to training data (JSONL) | Required | `/path/to/train.jsonl` |
| `num_tiers` | Number of parameter tiers | 3 | 3, 4, 5 |
| `tier_update_frequencies` | Update frequency per tier | [1, 2, 4] | [1, 3, 9] |
| `num_steps` | Total training steps | 1000 | 100, 1000, 10000 |

### Advanced Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tier_assignment_strategy` | How to assign params to tiers | "layer_depth" |
| `learning_rate` | Optimizer learning rate | 1e-5 |
| `batch_size` | Training batch size | 2 |
| `max_seq_length` | Maximum sequence length | 512 |
| `lora_rank` | LoRA rank | 8 |
| `lora_alpha` | LoRA alpha | 16 |
| `warmup_steps` | Learning rate warmup | 5 |
| `checkpoint_every` | Save checkpoint frequency | 10 |
| `eval_every` | Evaluation frequency | 10 |
| `max_grad_norm` | Gradient clipping | 1.0 |

## Tier Assignment Strategies

### 1. Layer Depth Strategy (Default)
Assigns parameters based on layer position:
- **Fast tier (0)**: Shallow layers (e.g., layers 0-8)
- **Medium tier (1)**: Middle layers (e.g., layers 9-19)
- **Slow tier (2)**: Deep layers (e.g., layers 20-28)

**Best for**: General fine-tuning where you want to adapt early representations quickly

### 2. Parameter Importance Strategy
Assigns based on gradient magnitudes (requires initial training):
- **Fast tier (0)**: High-gradient parameters
- **Medium tier (1)**: Medium-gradient parameters
- **Slow tier (2)**: Low-gradient parameters

**Best for**: Domain adaptation where some parameters are more task-relevant

## Data Format

Training data should be in JSONL format (one JSON object per line):

```jsonl
{"text": "Your first training example here"}
{"text": "Your second training example here"}
{"text": "Your third training example here"}
```

## Output Structure

```
output_path/experiment_name/
├── config.json                    # Training configuration
├── checkpoints/
│   ├── best/                      # Best model (lowest val loss)
│   ├── checkpoint_NNNNNNN/        # Periodic checkpoints
│   └── final/                     # Final checkpoint
└── metrics/
    ├── train_metrics.jsonl        # Training metrics
    └── eval_metrics.jsonl         # Evaluation metrics
```

## Monitoring Training

### Real-time Logs
Training logs show:
- Current step / total steps
- Training loss
- Active tiers at current step
- Estimated time remaining

Example:
```
Step 100/1000 | Loss: 1.234 | Active tiers: [0, 1] | ETA: 5.2m
```

### Metrics Files
- `train_metrics.jsonl`: Loss, learning rate, tier stats per step
- `eval_metrics.jsonl`: Validation loss at evaluation intervals

### Tier Statistics
Each checkpoint includes tier-specific statistics:
- Update count per tier
- Parameter count per tier
- Update frequency per tier

## Tips for Best Results

### 1. Choosing Tier Frequencies
- **Conservative**: [1, 2, 4] - Good default, prevents forgetting
- **Moderate**: [1, 3, 9] - More distinction between tiers
- **Aggressive**: [1, 5, 25] - Maximum forgetting prevention

### 2. Number of Tiers
- **3 tiers**: Standard, works well for most cases
- **4 tiers**: More fine-grained control, better for complex domains
- **2 tiers**: Simpler, faster training, less forgetting prevention

### 3. Training Steps
- Start with 100-200 steps to test
- Full training: 1000-5000 steps
- Large datasets: 10000+ steps

### 4. Learning Rate
- Default: 1e-5 (good for most cases)
- Higher (1e-4): Faster learning, more forgetting risk
- Lower (1e-6): Slower, more stable, less forgetting

## Troubleshooting

### Training Fails Immediately
- Check paths exist (model, adapter, data)
- Verify JSONL format is correct
- Ensure sufficient disk space for checkpoints

### Out of Memory
- Reduce batch_size (try 1)
- Reduce max_seq_length (try 256)
- Use smaller model/adapter

### Loss Not Decreasing
- Increase learning_rate
- Check data quality
- Increase num_steps
- Try different tier frequencies

### Too Much Forgetting
- Increase slow tier frequency (e.g., [1, 2, 8])
- Add more tiers
- Lower learning_rate

### Training Too Slow
- Reduce checkpoint_every and eval_every
- Increase batch_size if memory allows
- Reduce number of tiers

## Comparison with Standard Fine-tuning

| Aspect | Standard SFT | Nested Learning |
|--------|-------------|-----------------|
| Update Frequency | All params every step | Multi-frequency |
| Catastrophic Forgetting | High risk | Low risk |
| Task Adaptation | Fast | Fast (tier 0) + stable |
| Training Time | Baseline | ~5-10% slower |
| Checkpoint Size | Same | Same + tier metadata |
| Use Case | Simple tasks | Complex, multi-domain |

## Example Use Cases

### 1. Domain Adaptation
Fine-tune a general model for medical domain while preserving general knowledge:
```python
config = NestedLearningConfig(
    tier_update_frequencies=[1, 3, 9],  # Conservative
    tier_assignment_strategy="layer_depth",
    learning_rate=1e-5
)
```

### 2. Continual Learning
Add new capabilities without losing old ones:
```python
config = NestedLearningConfig(
    tier_update_frequencies=[1, 5, 25],  # Very conservative
    tier_assignment_strategy="parameter_importance",
    learning_rate=5e-6
)
```

### 3. Quick Task Adaptation
Fast adaptation with minimal forgetting:
```python
config = NestedLearningConfig(
    tier_update_frequencies=[1, 2, 4],  # Standard
    tier_assignment_strategy="layer_depth",
    learning_rate=2e-5,
    num_steps=500  # Quick training
)
```

## Next Steps

1. **Run Test**: Start with the provided test script to verify setup
2. **Small Experiment**: Train on 100-200 steps with your data
3. **Tune Parameters**: Adjust tier frequencies based on results
4. **Full Training**: Scale up to production dataset
5. **Evaluate**: Compare with standard SFT on held-out test set

For more details, see:
- `NESTED_LEARNING_EXPLAINED.md` - Detailed theory and comparisons
- `NESTED_LEARNING_GUIDE.md` - Technical implementation details
- `NESTED_LEARNING_TEST_RESULTS.md` - Test validation results
