# Full-Layer LoRA Implementation Guide
## Step-by-Step Instructions for Droid-FineTuning Repository

**Version:** 1.0  
**Date:** October 11, 2025  
**Target:** Standard Setup Tab (SFT Training Only)

---

## Table of Contents
1. [Overview](#overview)
2. [What You'll Modify](#what-youll-modify)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Testing](#testing)
5. [Verification](#verification)

---

## Overview

### Current vs Target State

| Aspect | Current | Target |
|--------|---------|--------|
| **LoRA Layers** | Attention only (Q, V) | All 7 matrices (Q, K, V, O + gate, up, down) |
| **Trainable Params** | ~1.5-2% | ~3.5-4% |
| **Learning Rate** | 1e-5 | 1e-4 (10x increase) |
| **UI Controls** | None | Full LoRA configuration section |
| **Layer Coverage** | Default | Configurable (all or top N layers) |

### Research Foundation
"LoRA Without Regret" (Schulman et al., 2025): https://thinkingmachines.ai/blog/lora/
- MLP layers are critical for performance
- 10x learning rate rule for LoRA
- Full-layer training achieves near full fine-tuning parity

### CRITICAL: What NOT to Change
- **DO NOT modify** `backend/main_enhancements.py` (Enhanced Setup Tab)
- **DO NOT modify** GRPO/GSPO/Dr. GRPO functionality
- **DO NOT modify** training data formats

---

## What You'll Modify

### Backend Files
1. **`backend/main.py`** - 3 sections to modify:
   - TrainingConfig dataclass (add LoRA fields)
   - start_training method (add LoRA parameter generation)
   - /training/start endpoint (accept new parameters)

### Frontend Files
2. **`frontend/src/store/slices/trainingSlice.ts`** - Add LoRA fields to state
3. **`frontend/src/pages/SetupPage.tsx`** - Add LoRA configuration UI

---

## Step-by-Step Implementation

### STEP 1: Backend - Update TrainingConfig

**File:** `backend/main.py`

**1.1** Find the `@dataclass` line for `TrainingConfig` (around line 44-63)

**1.2** Change learning rate default:
```python
learning_rate: float = 1e-4  # Changed from 1e-5
```

**1.3** Add these new fields at the end (after `adapter_name`):
```python
    adapter_name: str = "mlx_finetune"
    # Full-Layer LoRA Configuration
    fine_tune_type: str = "lora"
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_num_layers: int = -1  # -1 = all layers
```

---

### STEP 2: Backend - Add LoRA Parameter Generation

**File:** `backend/main.py`

**2.1** Find the `start_training` method in `TrainingManager` class

**2.2** Add imports at top of file if not present:
```python
import json
import os
```

**2.3** Insert this code BEFORE `config_data = {` dictionary is created:

```python
    # ============================================================
    # LoRA Parameter Generation for Full-Layer Training
    # ============================================================
    
    # Extract and validate LoRA parameters
    lora_rank = max(1, int(getattr(config, "lora_rank", 32) or 32))
    lora_alpha = float(getattr(config, "lora_alpha", 32.0) or 32.0)
    lora_dropout = float(getattr(config, "lora_dropout", 0.0) or 0.0)
    lora_num_layers = getattr(config, "lora_num_layers", -1)
    
    # Normalize num_layers
    try:
        lora_num_layers = int(lora_num_layers)
    except (TypeError, ValueError):
        lora_num_layers = -1
    if lora_num_layers == 0:
        lora_num_layers = -1
    
    # Detect model architecture
    model_config_path = os.path.join(config.model_path, "config.json")
    model_type = "qwen2"
    
    try:
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                model_config = json.load(f)
                model_type = model_config.get("model_type", "qwen2")
                logger.info(f"Detected model architecture: {model_type}")
    except Exception as e:
        logger.warning(f"Could not read model config: {e}")
    
    # Define 7 base matrices for full-layer training
    lora_keys = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    
    # Add architecture-specific keys
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
    
    # Create lora_parameters dict
    lora_parameters = {
        "rank": lora_rank,
        "scale": lora_alpha,
        "dropout": lora_dropout,
        "keys": lora_keys,
    }
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("LoRA Configuration:")
    logger.info(f"  Rank: {lora_rank}")
    logger.info(f"  Alpha: {lora_alpha}")
    logger.info(f"  Dropout: {lora_dropout}")
    logger.info(f"  Layers: {'all' if lora_num_layers == -1 else lora_num_layers}")
    logger.info(f"  Matrices: {', '.join(lora_keys)}")
    logger.info("=" * 60)
```

**2.4** Update the `config_data` dictionary by ADDING these fields:
```python
        # ... existing fields ...
        "no_improve_patience_evals": config.patience,
        # ADD THESE NEW FIELDS:
        "fine_tune_type": getattr(config, "fine_tune_type", "lora") or "lora",
        "num_layers": lora_num_layers,
        "lora_parameters": lora_parameters,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
```

---

### STEP 3: Backend - Update Training Endpoint

**File:** `backend/main.py`

**3.1** Find `@app.post("/training/start")`

**3.2** Update `TrainingConfig(...)` instantiation to include new fields:

Change the learning_rate line:
```python
learning_rate=config_data.get("learning_rate", 1e-4),  # Changed from 1e-5
```

Add these NEW lines after `adapter_name`:
```python
            adapter_name=config_data.get("adapter_name", "mlx_finetune"),
            # NEW: Full-layer LoRA parameters
            fine_tune_type=config_data.get("fine_tune_type", "lora") or "lora",
            lora_rank=int(config_data.get("lora_rank", 32) or 32),
            lora_alpha=float(config_data.get("lora_alpha", 32.0) or 32.0),
            lora_dropout=float(config_data.get("lora_dropout", 0.0) or 0.0),
            lora_num_layers=int(config_data.get("lora_num_layers", -1) or -1)
```

---

### STEP 4: Frontend - Update Redux Store

**File:** `frontend/src/store/slices/trainingSlice.ts`

**4.1** Find `interface TrainingConfig {`

**4.2** Add these fields to the interface:
```typescript
  adapter_name: string;
  // NEW: Full-layer LoRA configuration
  fine_tune_type: string;
  lora_rank: number;
  lora_alpha: number;
  lora_dropout: number;
  lora_num_layers: number;
```

**4.3** Find `initialState` and update `config`:

Change learning_rate:
```typescript
learning_rate: 0.0001,  // Changed from 0.00001
```

Add these NEW fields after `adapter_name`:
```typescript
    adapter_name: 'mlx_finetune',
    // NEW: Full-layer LoRA defaults
    fine_tune_type: 'lora',
    lora_rank: 32,
    lora_alpha: 32.0,
    lora_dropout: 0.0,
    lora_num_layers: -1,
```

---

### STEP 5: Frontend - Add LoRA Configuration UI

**File:** `frontend/src/pages/SetupPage.tsx`

**5.1** Ensure imports include:
```typescript
import { useDispatch, useSelector } from 'react-redux';
import { updateConfig } from '../store/slices/trainingSlice';
```

**5.2** Add this section AFTER your Training Parameters section:

```tsx
{/* Full-Layer LoRA Configuration */}
<div className="bg-gray-800 rounded-lg p-6 space-y-4">
  <h3 className="text-lg font-semibold text-gray-100">
    Full-Layer LoRA Configuration
  </h3>

  {/* Info Banner */}
  <div className="bg-blue-900/30 border border-blue-700/50 rounded p-4">
    <p className="text-sm text-blue-200">
      <strong>Full-Layer LoRA Training</strong> - Trains all 7 matrices (Q, K, V, O + gate, up, down) 
      across all transformer layers for significantly better performance.
    </p>
    <p className="text-xs text-blue-300 mt-2">
      Research: <a href="https://thinkingmachines.ai/blog/lora/" target="_blank" 
        rel="noopener noreferrer" className="underline">"LoRA Without Regret"</a>
    </p>
  </div>

  {/* Parameters Grid */}
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
    
    {/* Rank */}
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">
        LoRA Rank
      </label>
      <input
        type="number"
        min={1}
        max={256}
        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
        value={config.lora_rank}
        onChange={(e) => {
          const val = parseInt(e.target.value, 10);
          dispatch(updateConfig({ lora_rank: isNaN(val) ? 32 : val }));
        }}
      />
      <p className="text-xs text-gray-400 mt-1">Recommended: 32</p>
    </div>

    {/* Alpha */}
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">
        LoRA Alpha
      </label>
      <input
        type="number"
        min={1}
        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
        value={config.lora_alpha}
        onChange={(e) => {
          const val = parseFloat(e.target.value);
          dispatch(updateConfig({ lora_alpha: isNaN(val) ? 32 : val }));
        }}
      />
      <p className="text-xs text-gray-400 mt-1">Typically equals rank</p>
    </div>

    {/* Dropout */}
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">
        LoRA Dropout
      </label>
      <input
        type="number"
        min={0}
        max={1}
        step="0.01"
        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
        value={config.lora_dropout}
        onChange={(e) => {
          const val = parseFloat(e.target.value);
          const clamped = isNaN(val) ? 0 : Math.min(1, Math.max(0, val));
          dispatch(updateConfig({ lora_dropout: clamped }));
        }}
      />
      <p className="text-xs text-gray-400 mt-1">0.0 recommended</p>
    </div>

    {/* Layer Coverage */}
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">
        Layer Coverage
      </label>
      <select
        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
        value={config.lora_num_layers}
        onChange={(e) => {
          const val = parseInt(e.target.value, 10);
          dispatch(updateConfig({ lora_num_layers: isNaN(val) ? -1 : val }));
        }}
      >
        <option value={-1}>All Layers (Recommended)</option>
        <option value={24}>Top 24 Layers</option>
        <option value={16}>Top 16 Layers</option>
        <option value={8}>Top 8 Layers</option>
      </select>
      <p className="text-xs text-gray-400 mt-1">-1 = all transformer blocks</p>
    </div>
  </div>

  {/* Matrix Coverage Visualization */}
  <div className="mt-4 p-4 bg-gray-900/50 border border-gray-700 rounded-lg">
    <h4 className="text-sm font-semibold text-gray-200 mb-3">Matrix Coverage</h4>
    <div className="grid grid-cols-2 gap-6">
      <div>
        <p className="text-xs font-medium text-blue-400 mb-2">Attention (4):</p>
        <ul className="text-xs text-gray-300 space-y-1">
          <li>✓ Query (q_proj)</li>
          <li>✓ Key (k_proj)</li>
          <li>✓ Value (v_proj)</li>
          <li>✓ Output (o_proj)</li>
        </ul>
      </div>
      <div>
        <p className="text-xs font-medium text-green-400 mb-2">MLP (3):</p>
        <ul className="text-xs text-gray-300 space-y-1">
          <li>✓ Gate (gate_proj)</li>
          <li>✓ Up (up_proj)</li>
          <li>✓ Down (down_proj)</li>
        </ul>
      </div>
    </div>
    <div className="mt-3 pt-3 border-t border-gray-700">
      <p className="text-xs text-gray-400">
        <strong>Total:</strong> 7 matrices × {config.lora_num_layers === -1 ? 'all' : config.lora_num_layers} layers
      </p>
      <p className="text-xs text-gray-500 mt-1">
        Expected: ~3.5-4% trainable parameters
      </p>
    </div>
  </div>
</div>
```

---

## Testing

### 1. Backend Validation
Start your backend server and check logs:
```bash
cd backend
python main.py
```

Look for startup without errors.

### 2. Frontend Validation
Start your frontend:
```bash
cd frontend
npm start
```

Navigate to Setup page - you should see the new LoRA Configuration section.

### 3. Training Test
1. Configure a small model (e.g., Qwen2.5-0.5B)
2. Set LoRA parameters:
   - Rank: 32
   - Alpha: 32
   - Dropout: 0.0
   - Layers: All (-1)
3. Start training
4. Check logs for LoRA configuration output

---

## Verification

### Expected Log Output
When training starts, you should see:
```
============================================================
LoRA Configuration:
  Rank: 32
  Alpha: 32.0
  Dropout: 0.0
  Layers: all
  Matrices: self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, 
            self_attn.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj
============================================================
```

### Training Metrics
- **Trainable parameters:** Should be ~3.5-4% of total model parameters
- For Qwen2.5-0.5B: ~17-18M trainable out of ~494M total
- **Training progress:** Should show regular progress updates

### Adapter Output
After training, check `adapters.safetensors` and `adapter_config.json`:
```json
{
  "lora_parameters": {
    "rank": 32,
    "scale": 32.0,
    "dropout": 0.0,
    "keys": [
      "self_attn.q_proj",
      "self_attn.k_proj",
      "self_attn.v_proj",
      "self_attn.o_proj",
      "mlp.gate_proj",
      "mlp.up_proj",
      "mlp.down_proj"
    ]
  },
  "num_layers": -1
}
```

---

## Troubleshooting

### Issue: LoRA fields not showing in UI
- **Check:** Redux store updated in `trainingSlice.ts`?
- **Check:** Component importing `updateConfig` action?
- **Fix:** Clear browser cache, restart dev server

### Issue: Training fails with parameter error
- **Check:** All new fields added to `/training/start` endpoint?
- **Check:** `config_data` dict includes `lora_parameters`?
- **Fix:** Review Step 2 and Step 3 carefully

### Issue: Still seeing attention-only training
- **Check:** LoRA parameter generation code added to `start_training`?
- **Check:** `lora_parameters` dict includes all 7 keys?
- **Check:** Logs show full matrix list?
- **Fix:** Verify Step 2 implementation

### Issue: Model architecture not detected
- **Check:** Model has `config.json` file?
- **Check:** File parsing succeeds (check logs)?
- **Fix:** Defaults to "qwen2" - standard keys will still work

---

## Summary

### Files Modified (5 total)
1. ✅ `backend/main.py` - TrainingConfig class
2. ✅ `backend/main.py` - start_training method
3. ✅ `backend/main.py` - /training/start endpoint
4. ✅ `frontend/src/store/slices/trainingSlice.ts` - State
5. ✅ `frontend/src/pages/SetupPage.tsx` - UI

### Key Changes
- **7 LoRA matrices** instead of 2 (Q, V only)
- **10x learning rate** (1e-4 vs 1e-5)
- **Architecture detection** for model-specific keys
- **Complete UI controls** for all LoRA parameters
- **Configurable layer coverage** (all or top N)

### Expected Results
- ✅ ~3.5-4% trainable parameters
- ✅ Better training efficiency
- ✅ Improved model performance
- ✅ Near full fine-tuning quality

---

## Next Steps

1. **Test on small model** (Qwen2.5-0.5B recommended)
2. **Compare with Enhanced Setup** (attention-only)
3. **Evaluate performance** on your specific tasks
4. **Adjust parameters** based on results

---

## Support & References

### Research Paper
"LoRA Without Regret" - https://thinkingmachines.ai/blog/lora/

### Key Findings
- MLP layers are critical (MLP-only ≈ MLP+attention)
- 10x learning rate rule validated across 14+ models
- Full-layer training essential for best results
- Rank should exceed dataset information content

### Documentation Files
- `CLAUDENEW.md` - Detailed implementation analysis
- `droid.md` - Product requirements document
- This file - Step-by-step implementation guide

---

**End of Implementation Guide**
