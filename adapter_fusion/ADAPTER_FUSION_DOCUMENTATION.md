# Adapter Fusion Module Documentation

## Overview
This module provides tools for fusing (blending) multiple LoRA adapters to create hybrid models that combine characteristics from different fine-tuned adapters.

---

## 📁 File Structure

```
adapter_fusion/
├── fusion_adapters.py              # Core fusion library (MAIN)
├── test_fusion.py                  # Quick fusion experiments
├── test_fused_models.py            # Test fused adapters with prompts
├── investment_question_test.py     # Domain-specific testing
├── fusion_test/                    # 50/50 fusion results
├── fusion_70_30/                   # 70/30 fusion results
└── fusion_slerp/                   # SLERP fusion results
```

---

## 🔧 Core Files

### 1. **fusion_adapters.py** (Main Library)

**Purpose:** Core library for fusing multiple LoRA adapters

**Key Features:**
- Load adapter weights from safetensors files
- Validate adapter compatibility (dimensions, keys)
- Multiple fusion methods:
  - **Weighted Average**: Simple linear combination
  - **SLERP**: Spherical Linear Interpolation (smooth rotation in weight space)
- Save fused adapters in MLX-compatible format
- Generate fusion reports

**Key Classes:**
```python
class AdapterFusion:
    - list_available_adapters()        # List all adapters in directory
    - load_adapter_weights()           # Load .safetensors files
    - validate_adapter_compatibility() # Check if adapters can be fused
    - weighted_average_fusion()        # Linear blend: w1*A1 + w2*A2
    - slerp_fusion()                   # Spherical interpolation
    - save_fused_adapter()             # Save to disk
    - generate_fusion_report()         # Create documentation
```

**Usage:**
```bash
# List available adapters
python fusion_adapters.py --list-adapters

# 50/50 fusion
python fusion_adapters.py \
  --adapters adapter1 adapter2 \
  --weights 0.5 0.5 \
  --output-dir ./fused_output

# 70/30 fusion (favor first adapter)
python fusion_adapters.py \
  --adapters adapter1 adapter2 \
  --weights 0.7 0.3 \
  --output-dir ./fused_70_30

# SLERP fusion
python fusion_adapters.py \
  --adapters adapter1 adapter2 \
  --weights 0.0 0.5 \
  --method slerp \
  --output-dir ./fused_slerp
```

**Input Files:**
- Multiple adapter directories containing `adapters.safetensors` or `best_adapters.safetensors`

**Output Files:**
- `fused_adapter.safetensors` - The blended adapter
- `adapters.safetensors` - MLX-compatible copy
- `fusion_report.txt` - Documentation of fusion process

---

### 2. **test_fusion.py** (Quick Experiments)

**Purpose:** Automated testing script for running multiple fusion experiments

**What It Does:**
1. Lists all available adapters
2. Runs 4 different fusion tests:
   - Test 1: 50/50 weighted average
   - Test 2: 70/30 weighted average
   - Test 3: SLERP interpolation (t=0.5)
   - Test 4: Three-way equal fusion (if 3+ adapters available)

**Usage:**
```bash
python test_fusion.py
```

**Output:**
Creates `./fusion_experiments/` directory with:
- `test1_50_50/` - Equal blend
- `test2_70_30/` - Favoring first adapter
- `test3_slerp_50/` - Spherical interpolation
- `test4_three_way/` - Three adapter blend

**Use Case:** Quick experimentation to see which fusion ratios work best

---

### 3. **test_fused_models.py** (Inference Testing)

**Purpose:** Test fused adapters with actual prompts to compare outputs

**What It Does:**
1. Tests base model (no adapter)
2. Tests each fused adapter with same prompt
3. Compares responses
4. Saves results to JSON

**Usage:**
```bash
python test_fused_models.py
```

**Test Prompt:** "Write a short poem about artificial intelligence:"

**Output Files:**
- `fusion_test_results.json` - All responses in structured format

**Use Case:** Evaluate if fusion actually improves model behavior

---

### 4. **investment_question_test.py** (Domain-Specific Testing)

**Purpose:** Compare investment advice quality across different adapter versions

**What It Does:**
1. Tests 5 configurations:
   - Original Adapter 1
   - Original Adapter 2
   - 50/50 Fusion
   - 70/30 Fusion
   - SLERP Fusion
2. Asks investment-related question
3. Compares response quality
4. Saves detailed comparison

**Usage:**
```bash
python investment_question_test.py
```

**Test Question:** "What are the benefits of investing in emerging markets?"

**Output Files:**
- `investment_advice_comparison.json` - Detailed results
- `investment_advice_summary.txt` - Human-readable comparison

**Use Case:** Domain-specific evaluation (finance, medical, legal, etc.)

---

## 🔬 Fusion Methods Explained

### Weighted Average Fusion
**Formula:** `Fused = w1 × Adapter1 + w2 × Adapter2`

**Characteristics:**
- Simple linear combination
- Weights should sum to 1.0
- Good for: Balancing different training objectives

**Example:**
- 50/50: Equal influence from both adapters
- 70/30: Favor one adapter's characteristics
- 80/20: Mostly one adapter, slight influence from other

### SLERP Fusion (Spherical Linear Interpolation)
**Formula:** Interpolates along the geodesic (shortest path) on a sphere

**Characteristics:**
- Maintains magnitude of weight vectors
- Smoother transitions than linear interpolation
- Better for: Preserving model behavior during blending

**When to Use:**
- When adapters have very different characteristics
- When you want smooth interpolation
- For creative/generative tasks

---

## 📊 Key Differences Between Files

| File | Purpose | Input | Output | Use When |
|------|---------|-------|--------|----------|
| **fusion_adapters.py** | Core library | Adapter names, weights | Fused adapter | Building custom fusion |
| **test_fusion.py** | Batch experiments | None (auto-detects) | Multiple fusions | Exploring fusion ratios |
| **test_fused_models.py** | Inference testing | Fused adapters | Response comparison | Evaluating fusion quality |
| **investment_question_test.py** | Domain testing | Adapters + question | Domain-specific comparison | Testing specific use case |

---

## 🎯 Typical Workflow

### Step 1: Create Fusions
```bash
# Run automated experiments
python test_fusion.py
```

### Step 2: Test General Quality
```bash
# Test with general prompts
python test_fused_models.py
```

### Step 3: Test Domain-Specific
```bash
# Test with your specific use case
python investment_question_test.py
```

### Step 4: Choose Best Fusion
- Review results
- Select fusion that performs best for your use case
- Use that fused adapter in production

---

## 💡 When to Use Adapter Fusion

### Good Use Cases:
1. **Multi-Domain Models**: Combine finance + legal adapters
2. **Balancing Objectives**: Blend accuracy + creativity adapters
3. **Incremental Learning**: Merge old + new training
4. **Style Transfer**: Combine formal + casual writing styles

### Not Recommended:
1. **Incompatible Architectures**: Different model sizes
2. **Conflicting Objectives**: Contradictory training goals
3. **Low-Quality Adapters**: Garbage in = garbage out

---

## 🔍 Technical Details

### Adapter Compatibility Requirements:
- ✅ Same base model
- ✅ Same LoRA rank
- ✅ Same target layers
- ✅ Same tensor shapes
- ✅ Same key names

### File Format:
- Input: `.safetensors` (safer than pickle)
- Output: `.safetensors` + MLX-compatible copy

### Memory Requirements:
- Loads all adapters into RAM
- ~100MB per adapter (typical)
- Fusion is fast (seconds)

---

## 🐛 Common Issues

### "Adapters are not compatible"
**Cause:** Different architectures or ranks
**Fix:** Only fuse adapters from same training setup

### "No adapter file found"
**Cause:** Wrong path or missing files
**Fix:** Check adapter directory has `adapters.safetensors`

### "SLERP fusion only supports exactly 2 adapters"
**Cause:** Tried SLERP with 3+ adapters
**Fix:** Use weighted average for 3+ adapters

---

## 📈 Performance Tips

1. **Start with 50/50**: Baseline for comparison
2. **Try 70/30 and 30/70**: See which direction is better
3. **Use SLERP for creative tasks**: Better interpolation
4. **Test with real prompts**: Don't trust theory alone
5. **Keep original adapters**: Fusion is non-destructive

---

## 🚀 Future Enhancements

Potential additions to this module:
- [ ] Task-specific weighting (different weights per layer)
- [ ] Automatic weight optimization
- [ ] Multi-adapter fusion (3+ adapters with SLERP)
- [ ] Fusion quality metrics
- [ ] GUI for fusion experiments
- [ ] Integration with Droid-FineTuning main app

---

**Last Updated:** 2025-01-29
**Maintainer:** Droid-FineTuning Project
