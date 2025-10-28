# OPD Implementation Configuration Decisions

**Date**: January 28, 2025
**Branch**: OPD
**Status**: Ready for Implementation

---

## Configuration Decisions

### 1. MLX Logprob Extraction Method

**Decision**: **Option C - Manual Generation Loop**

**Rationale**:
- Guaranteed to work with current MLX version
- Full control over generation and logprob extraction
- Easy to debug and modify
- Production-ready reliability

**Implementation**:
```python
def generate_with_logprobs(model, tokenizer, prompt, max_tokens):
    tokens = tokenizer.encode(prompt)
    all_logprobs = []

    for step in range(max_tokens):
        logits = model(tokens)
        last_logits = logits[-1]
        logprobs = mx.log_softmax(last_logits, axis=-1)

        # Sample next token
        next_token = mx.argmax(logprobs)
        tokens.append(next_token)
        all_logprobs.append(logprobs)

        if next_token == eos_token:
            break

    return tokens, all_logprobs
```

**Alternative**: Can optimize to use MLX's native `generate()` if it exposes logprobs in future versions.

---

### 2. Validation Data Source

**Decision**: **Separate validation file required**

**Rationale**:
- Clear separation between training and validation
- User has full control over validation prompts
- Prevents data leakage
- Aligns with existing SFT workflow

**User Workflow**:
1. User uploads training data for SFT (existing flow)
2. User uploads separate validation prompts for OPD
3. System uses validation file for distillation training
4. Can be same prompts used in SFT validation, or new ones

**File Format**: JSONL with structure:
```json
{"prompt": "What is the capital of France?"}
{"prompt": "Explain quantum computing."}
{"prompt": "Write a Python function to reverse a string."}
```

**UI**: File picker in OPD Setup page for "Validation Prompts" file

---

### 3. Temperature Default

**Decision**: **T = 2.0**

**Rationale**:
- Standard practice in distillation literature (Hinton et al. 2015)
- Balances between sharp (T=1.0) and very soft (T=4.0)
- Transfers good amount of "dark knowledge"
- Can be tuned per task if needed

**Configuration**:
- Default: 2.0
- Configurable in UI: range [1.0 - 4.0]
- Advanced users can experiment with different values

**Temperature Effects**:
- T=1.0: Sharp distributions (less knowledge transfer)
- T=2.0: Moderate softening (balanced) ⭐
- T=3.0: Softer distributions (more dark knowledge)
- T=4.0: Very soft (may oversmooth)

---

### 4. System Memory

**Available**: **128 GB RAM**

**Impact**: **Excellent for distillation!**

**Memory Breakdown**:
- Qwen 32B (FP16): ~64 GB
- Qwen 7B (FP16): ~14 GB
- Gradients (LoRA only): ~2 GB
- Activations (batch_size=2): ~4 GB
- OS + overhead: ~8 GB
- **Total**: ~92 GB
- **Headroom**: 36 GB (plenty of buffer)

**Optimizations Enabled**:
- ✅ Can keep both models in memory simultaneously
- ✅ Can use larger batch sizes (up to 4)
- ✅ Can cache more teacher outputs in RAM
- ✅ No need for model swapping or offloading
- ✅ Smooth, fast training

**Recommended Settings for 128GB**:
```yaml
batch_size: 4  # Can go higher than 2
gradient_accumulation_steps: 2  # Effective batch = 8
teacher_cache_size_mb: 4096  # 4GB in-memory cache
keep_teacher_loaded: true  # Don't unload between batches
```

---

## Updated Implementation Plan Parameters

### Training Configuration

```yaml
# Optimized for 128GB RAM system
distillation:
  # Models
  teacher_model: qwen2.5-32b
  student_model: qwen2.5-7b

  # Training
  num_steps: 1000
  batch_size: 4  # Increased from 2 (we have plenty of RAM)
  gradient_accumulation_steps: 2
  learning_rate: 0.00001

  # Distillation
  temperature: 2.0  # Default, configurable
  kl_weight: 0.8
  ce_weight: 0.2

  # Generation
  max_generation_tokens: 512
  logprob_method: "manual_loop"  # Option C

  # Data
  validation_source: "separate_file"  # User uploads

  # Performance
  mixed_precision: true
  teacher_cache_size_mb: 4096
  keep_teacher_loaded: true

  # Checkpointing
  checkpoint_every: 100
  eval_every: 100
```

---

## Performance Estimates (Updated for 128GB)

### Training Speed
- **Steps per minute**: ~15-20 (with caching)
- **1000 steps**: ~45-60 minutes
- **Faster than originally estimated** (more memory = less swapping)

### Memory Usage
- **Peak**: ~92 GB (well under 128 GB limit)
- **Average**: ~80 GB (comfortable)
- **No OOM risk**: Excellent headroom

### Batch Size Options
| Batch Size | Memory | Speed | Recommended |
|------------|--------|-------|-------------|
| 1 | ~78 GB | Slow | No |
| 2 | ~84 GB | Good | Yes (safe) |
| 4 | ~92 GB | Best | **Yes (optimal)** ⭐ |
| 8 | ~108 GB | Fastest | Yes (if needed) |

**Recommendation**: Use **batch_size=4** as default (optimal speed/memory balance)

---

## File Structure Updates

### Validation Data Location
```
Droid-FineTuning/
├── data/
│   ├── training/           # SFT training data (existing)
│   ├── validation/         # NEW: OPD validation prompts
│   │   └── opd_val_prompts.jsonl
│   └── ...
└── OnPolicyDistill/
    └── configs/
        └── distill_{run_id}.yaml
```

### UI File Picker
```typescript
// OPD Setup Panel
<FileInput
  label="Validation Prompts (JSONL)"
  accept=".jsonl,.json"
  onChange={handleValidationFileSelect}
  required={true}
/>
```

---

## Implementation Checklist

### Phase 0: Setup (Priority Updates)
- [x] Confirm system has 128 GB RAM
- [x] Decision: Use manual generation loop (Option C)
- [x] Decision: Separate validation file
- [x] Decision: Temperature = 2.0
- [ ] Test loading 32B + 7B simultaneously
- [ ] Measure actual memory usage
- [ ] Verify can run with batch_size=4

### Next Steps
1. Begin Phase 0 implementation
2. Create directory structure
3. Implement `generate_with_logprobs()` function
4. Test with Qwen 32B model
5. Proceed to Phase 1 (core components)

---

## Success Criteria (Updated)

### Memory
✅ Peak memory < 110 GB (plenty of headroom)
✅ No OOM errors during training
✅ Can run batch_size=4 comfortably

### Speed
✅ 1000 steps complete in < 60 minutes
✅ 15-20 steps per minute average
✅ Teacher cache saves 50%+ time on repeated prompts

### Quality
✅ KL loss decreases by 80%+
✅ Token agreement > 85% by end
✅ Distilled model outputs close to teacher

---

## Risk Assessment (Updated)

### Original Risks (Now Mitigated)
- ~~Out of Memory~~ → 128 GB provides ample headroom ✅
- ~~Slow Training~~ → Larger batches + caching = fast ✅

### Remaining Risks (Low)
- **MLX API changes**: Mitigated by using manual generation loop
- **Poor quality**: Mitigated by T=2.0 default + tuning option
- **Cache misses**: Mitigated by 4GB in-memory cache

---

## Summary

All key decisions made:
1. ✅ **Logprob method**: Manual generation loop (reliable)
2. ✅ **Validation data**: Separate file (clean separation)
3. ✅ **Temperature**: 2.0 (standard practice)
4. ✅ **RAM**: 128 GB (excellent, enables batch_size=4)

**Status**: **Ready to begin implementation** 🚀

**Next Action**: Start Phase 0 - Setup & Testing

