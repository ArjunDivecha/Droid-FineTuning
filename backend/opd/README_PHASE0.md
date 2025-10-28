# Phase 0: Setup & Testing - COMPLETED

**Status**: Ready for testing
**Date**: January 28, 2025

---

## What Was Created

### 1. Directory Structure ✅
```
OnPolicyDistill/
├── configs/              # Training run configurations
├── teacher_cache/        # Cached teacher outputs
├── student_rollouts/     # Student-generated samples
├── checkpoints/          # Saved LoRA adapters
└── metrics/              # Training metrics logs
```

### 2. Backend Module ✅
```
backend/opd/
├── __init__.py           # Module initialization
├── config.py             # OPDConfig, OPDMetrics, presets
├── test_model_loading.py # Testing script
└── README_PHASE0.md      # This file
```

### 3. Configuration System ✅

**`backend/opd/config.py`** includes:

- **`OPDConfig`**: Complete configuration dataclass
  - Model paths (teacher, student, adapters)
  - Training hyperparameters (batch size, learning rate, etc.)
  - Distillation settings (temperature, KL weight)
  - System settings (memory, caching)
  - Validation and error checking

- **`OPDMetrics`**: Comprehensive metrics tracking
  - Loss components (KL, CE, total)
  - KL statistics (mean, std, max, min)
  - Alignment metrics (token agreement, top-5 agreement)
  - Distribution metrics (entropy, JS divergence)
  - Performance metrics (throughput, latency)
  - Validation metrics
  - Memory usage

- **Preset Configurations**:
  - `get_fast_iteration_config()` - For testing/debugging
  - `get_high_quality_config()` - For production
  - `get_memory_efficient_config()` - For lower RAM systems

### 4. Test Script ✅

**`backend/opd/test_model_loading.py`** tests:

1. ✅ Loading Qwen 32B teacher model
2. ✅ Loading Qwen 7B student model (with optional LoRA)
3. ✅ Loading both models simultaneously
4. ✅ Memory profiling at each stage
5. ✅ `generate_with_logprobs()` implementation (manual loop approach)

---

## Next Steps: Run the Tests

### Step 1: Locate Your Models

You need the paths to:
- **Teacher model**: Qwen 32B (you mentioned you have this)
- **Student model**: Qwen 7B base model
- **Student adapter** (optional): Your fine-tuned LoRA from SFT

Example paths (adjust to your system):
```bash
TEACHER="/Users/macbook2024/Dropbox/mlx/base_model/qwen2.5-32b"
STUDENT="/Users/macbook2024/Dropbox/mlx/base_model/qwen2.5-7b"
ADAPTER="/Users/macbook2024/Dropbox/mlx/lora_adapters/my_sft_adapter"
```

### Step 2: Run the Test Script

**Basic test** (teacher and student loading):
```bash
cd /home/user/Droid-FineTuning
source /path/to/your/mlx/.venv/bin/activate

python backend/opd/test_model_loading.py \
  --teacher-path "$TEACHER" \
  --student-path "$STUDENT"
```

**With adapter** (if you have a fine-tuned adapter):
```bash
python backend/opd/test_model_loading.py \
  --teacher-path "$TEACHER" \
  --student-path "$STUDENT" \
  --adapter-path "$ADAPTER"
```

**With generation test** (tests `generate_with_logprobs`):
```bash
python backend/opd/test_model_loading.py \
  --teacher-path "$TEACHER" \
  --student-path "$STUDENT" \
  --test-generation
```

### Step 3: What the Test Will Show

The test script will:
1. ✅ Load teacher (32B) and measure memory
2. ✅ Load student (7B) and measure memory
3. ✅ Load both simultaneously and measure total memory
4. ✅ Test `generate_with_logprobs()` function
5. ✅ Verify everything works on your 128GB system

**Expected output**:
```
============================================================
TEST 1: Loading Teacher Model (Qwen 32B)
============================================================
Loading teacher from: /path/to/qwen32b
✓ Teacher loaded successfully in 15.2s
  Model type: <class 'mlx_lm.models.qwen2.Model'>
  Tokenizer vocab size: 151643

============================================================
Memory Stats: After Teacher Load
============================================================
  Active Memory:  62.45 GB
  Peak Memory:    64.12 GB
  Cache Memory:   1.23 GB
============================================================

... (similar for student and simultaneous loading)
```

### Step 4: Verify Results

**Success Criteria**:
- ✅ Teacher loads without errors
- ✅ Student loads without errors
- ✅ Both load simultaneously without OOM
- ✅ Peak memory < 110 GB (you have 128 GB)
- ✅ `generate_with_logprobs()` produces text and logprobs

**If tests pass**: Phase 0 complete! Ready for Phase 1.

**If tests fail**: Check error messages and adjust paths.

---

## What Comes Next

After Phase 0 tests pass, we proceed to **Phase 1: Core Components** (12-16 hours):

### Phase 1 Tasks:
1. **Task 1.1**: Implement `TeacherModel` class (4-5 hours)
   - Load teacher model
   - Extract logprobs using manual generation loop
   - Caching system with SHA256 keys

2. **Task 1.2**: Implement `StudentModel` class (3-4 hours)
   - Load student with LoRA
   - Forward pass with gradient tracking
   - Freeze base model, enable LoRA gradients

3. **Task 1.3**: Implement `DistillationLoss` class (3-4 hours)
   - Temperature scaling
   - Reverse KL divergence computation
   - Metrics calculation

4. **Task 1.4**: Implement `DataLoader` (2-3 hours)
   - Load prompts from JSONL
   - Train/val split
   - Batch creation

---

## Configuration Examples

### Test with Fast Iteration Config

```python
from backend.opd.config import get_fast_iteration_config

config = get_fast_iteration_config(
    base_model_path="/path/to/qwen7b",
    teacher_model_path="/path/to/qwen32b",
    student_adapter_path="/path/to/adapter",
    output_adapter_path="./OnPolicyDistill/checkpoints/test_run",
    validation_prompts_path="/path/to/val.jsonl"
)

print(config.num_steps)  # 100 (fast)
print(config.batch_size)  # 2 (small)
```

### Production Config

```python
from backend.opd.config import OPDConfig

config = OPDConfig(
    base_model_path="/path/to/qwen7b",
    teacher_model_path="/path/to/qwen32b",
    student_adapter_path="/path/to/adapter",
    output_adapter_path="./OnPolicyDistill/checkpoints/production",
    validation_prompts_path="/path/to/val.jsonl",

    # Optimized for 128GB RAM
    num_steps=1000,
    batch_size=4,
    gradient_accumulation_steps=2,
    temperature=2.0,
    learning_rate=1e-5,

    # Caching
    use_cache=True,
    cache_size_mb=4096,
    keep_teacher_loaded=True
)

# Save config
config.save("./OnPolicyDistill/configs/my_run.yaml")
```

---

## Troubleshooting

### Issue: "Teacher model not found"
**Solution**: Verify the path exists:
```bash
ls -la /path/to/qwen32b
# Should show config.json, tokenizer files, model weights, etc.
```

### Issue: Out of Memory (OOM)
**Unlikely with 128GB**, but if it happens:
- Reduce batch_size to 2 or 1
- Set `keep_teacher_loaded=False`
- Use `get_memory_efficient_config()`

### Issue: MLX not found
**Solution**: Install MLX in your virtual environment:
```bash
source /path/to/mlx/.venv/bin/activate
pip install mlx mlx-lm
```

### Issue: generate_with_logprobs fails
**Check**:
- Model outputs logits correctly
- Tokenizer has proper encode/decode methods
- EOS token is defined

---

## Files Created in Phase 0

| File | Lines | Purpose |
|------|-------|---------|
| `OnPolicyDistill/` (dirs) | - | Data storage structure |
| `backend/opd/__init__.py` | 5 | Module init |
| `backend/opd/config.py` | 450+ | Configuration classes |
| `backend/opd/test_model_loading.py` | 400+ | Testing script |
| `backend/opd/README_PHASE0.md` | This file | Phase 0 documentation |

**Total**: ~850 lines of code + infrastructure

---

## Phase 0 Checklist

- [x] Create `OnPolicyDistill/` directory structure
- [x] Create `backend/opd/` module
- [x] Implement `OPDConfig` dataclass with validation
- [x] Implement `OPDMetrics` dataclass
- [x] Add configuration presets
- [x] Create comprehensive test script
- [x] Document Phase 0 setup

**Next**: Run test script to verify everything works!

---

**Phase 0 Status**: ✅ COMPLETED (code written, awaiting testing)

**Ready to test?** Run the test script with your model paths!
