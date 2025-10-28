# Phase 2 Testing Guide

This guide will help you test the OPD distillation implementation before proceeding to Phase 3.

---

## What We're Testing

- ✅ Configuration system
- ✅ Teacher model loading and logprob extraction
- ✅ Student model loading with adapter
- ✅ Data loading from JSONL
- ✅ Loss computation
- ✅ Training loop orchestration
- ✅ Checkpoint saving
- ✅ Metrics logging

---

## Test Setup

### Prerequisites

1. **Your models** (already validated in Phase 0):
   - Teacher: Qwen 32B 4-bit
   - Student: Qwen 7B

2. **A fine-tuned adapter** (from previous SFT training)
   - If you don't have one, we can test with the base model as a placeholder
   - For a real test, you need LoRA adapters from your SFT training

3. **Validation prompts** (created for you):
   - Located at: `./OnPolicyDistill/test_prompts.jsonl`
   - 20 sample prompts

---

## Quick Test (10 steps)

Run this command from your Mac:

```bash
cd /path/to/Droid-FineTuning

# Make sure you're in the right directory
pwd

# Run the test
python3 backend/opd/run_distillation.py \
  --teacher-path "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen3-32B-MLX-4bit" \
  --student-path "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct" \
  --adapter-path YOUR_ADAPTER_PATH \
  --prompts-path ./OnPolicyDistill/test_prompts.jsonl \
  --output-path ./OnPolicyDistill/checkpoints/test_run \
  --steps 10 \
  --batch-size 2 \
  --max-prompts 20 \
  --max-tokens 50 \
  --checkpoint-every 5 \
  --eval-every 5
```

**Replace `YOUR_ADAPTER_PATH` with**:
- Path to your fine-tuned LoRA adapter, OR
- Use the base student path as placeholder (for initial testing)

---

## Expected Output

### 1. **Startup (First 10 seconds)**

```
============================================================
On-Policy Distillation Training
============================================================

Configuration:
  Run ID: distill_20250128_143022
  Teacher: /Users/.../Qwen3-32B-MLX-4bit
  Student: /Users/.../Qwen2.5-7B-Instruct
  ...

Validating paths...
✓ base_model_path: /Users/.../Qwen2.5-7B-Instruct
✓ teacher_model_path: /Users/.../Qwen3-32B-MLX-4bit
✓ student_adapter_path: ...
✓ validation_prompts_path: ./OnPolicyDistill/test_prompts.jsonl

Estimating memory requirements...
============================================================
Memory Requirements Estimate
============================================================
  Teacher model:      16.0 GB
  Student model:      14.0 GB
  Gradients:          2.0 GB
  Activations:        2.0 GB
  Cache:              4.0 GB
  ------------------------------
  Total estimated:    38.0 GB

  System total:       128.0 GB
  System available:   100.0 GB

  ✓ Sufficient memory available
============================================================
```

### 2. **Setup (30-60 seconds)**

```
============================================================
Setting up distillation training
============================================================
Loading teacher model...
✓ Teacher loaded successfully in 3.2s

Loading student model...
✓ Student loaded successfully in 4.5s

Initializing loss function...
DistillationLoss: T=2.0, KL=0.8, CE=0.2

Initializing optimizer...
Optimizer: Adam(lr=1e-05)
Trainable parameters: 4,194,304

Loading dataset...
Loaded 20 items from ./OnPolicyDistill/test_prompts.jsonl
Split: 16 train, 4 val
============================================================
Setup complete!
  Teacher: /Users/.../Qwen3-32B-MLX-4bit
  Student: /Users/.../Qwen2.5-7B-Instruct
  Train prompts: 16
  Val prompts: 4
  Total steps: 10
============================================================
```

### 3. **Training Loop (1-2 minutes)**

```
============================================================
Starting distillation training
============================================================

Step 1/10 | Loss: 0.5234 | KL: 0.4187 | Agree: 45.2% | ETA: 2.0m
Step 2/10 | Loss: 0.4987 | KL: 0.3990 | Agree: 48.1% | ETA: 1.8m
Step 3/10 | Loss: 0.4756 | KL: 0.3805 | Agree: 51.3% | ETA: 1.6m
Step 4/10 | Loss: 0.4543 | KL: 0.3635 | Agree: 54.0% | ETA: 1.4m
Step 5/10 | Loss: 0.4347 | KL: 0.3478 | Agree: 56.8% | ETA: 1.2m

  ✓ Checkpoint saved: ./OnPolicyDistill/checkpoints/test_run/step_0000005

Evaluating on validation set (4 prompts)...
  Val KL Loss: 0.4123
  Val Token Agreement: 58.2%

Step 6/10 | Loss: 0.4165 | KL: 0.3332 | Agree: 59.1% | ETA: 1.0m
...
```

### 4. **Completion**

```
Training complete! Saving final checkpoint...
  ✓ Checkpoint saved: ./OnPolicyDistill/checkpoints/test_run/final

============================================================
Training Summary
============================================================
  Total time: 2.45 minutes
  Steps: 10
  Avg step time: 14.7s
  Best val loss: 0.4123
  Final checkpoint: ./OnPolicyDistill/checkpoints/test_run/final
============================================================

Teacher Cache Statistics:
  Cache hits: 18
  Cache misses: 22
  Hit rate: 45.0%
  Cached prompts: 22

✓ Training completed successfully!
```

---

## Known Issues & TODOs

### ⚠️ Expected Issues (We'll Fix These)

1. **Gradient Update Not Implemented**
   - Error: `_update_parameters()` is a placeholder
   - **Impact**: Training loop runs but parameters don't actually update
   - **Fix needed**: Implement proper MLX gradient computation
   - **For testing**: The loop should still run and log metrics

2. **Teacher/Student Logit Alignment**
   - Error: Might see shape mismatches in `_compute_loss()`
   - **Impact**: Loss computation may fail
   - **Fix needed**: Proper tensor alignment
   - **For testing**: May need to skip or mock this

3. **Adapter Loading**
   - Error: If you don't have a LoRA adapter yet
   - **Impact**: Student model may load base model instead
   - **Workaround**: Use base model path as adapter path for initial test

---

## What to Check

### ✅ Success Indicators

1. **Models load without OOM** ✓
   - Both teacher and student load successfully
   - Memory usage ~30-60 GB

2. **Teacher caching works** ✓
   - Cache hit rate increases over time
   - Second run with same prompts should be faster

3. **Metrics are logged** ✓
   - Files created in `./OnPolicyDistill/metrics/`
   - JSONL format with step-by-step data

4. **Checkpoints are saved** ✓
   - Files in `./OnPolicyDistill/checkpoints/test_run/`
   - `best/`, `step_0000005/`, `final/` directories

5. **Training loop completes** ✓
   - Reaches step 10/10
   - No crashes

### ⚠️ Expected Failures (OK for Now)

1. **Loss doesn't decrease** - Parameter updates not implemented yet
2. **Shape mismatch errors** - Tensor alignment needs work
3. **Gradient errors** - MLX gradient API needs proper integration

---

## Troubleshooting

### Issue: "Module 'mlx.core' has no attribute 'X'"
**Solution**: MLX API might differ from what we expect. We'll adjust based on actual API.

### Issue: "Path not found"
**Solution**: Check your model paths are correct. Copy-paste from Phase 0 test results.

### Issue: "Out of memory"
**Solution**: Reduce `--batch-size` to 1 or `--max-tokens` to 20

### Issue: "No adapter found"
**Solution**: For testing, use base model path:
```bash
--adapter-path "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct"
```

---

## After Testing

### If Successful ✅
- Share output logs
- Note which features work
- We'll fix gradient updates and move to Phase 3

### If Errors ❌
- Share the error messages
- Note at which step it failed
- We'll debug together before Phase 3

---

## Testing Checklist

- [ ] Models load successfully
- [ ] Teacher generates text with logprobs
- [ ] Student runs forward pass
- [ ] Data loads from JSONL
- [ ] Training loop runs for 10 steps
- [ ] Checkpoints are saved
- [ ] Metrics are logged
- [ ] Memory stays within bounds (<100 GB)
- [ ] No crashes or OOM errors

---

## Next Steps

After testing:

**If it works well**:
→ Proceed to Phase 3 (FastAPI endpoints)
→ Complete Week 1 backend
→ Move to GUI (Week 2)

**If there are issues**:
→ Fix critical bugs first
→ Re-test
→ Then continue to Phase 3

---

**Ready to test!** Run the command above and share the results! 🧪
