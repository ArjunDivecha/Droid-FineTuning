# PRD: Full-Layer LoRA Setup Tab Implementation

## Overview
This Product Requirements Document (PRD) outlines the modifications needed to update the existing Setup tab in the Droid Fine-Tuning GUI to support full-layer LoRA fine-tuning, while preserving the existing Enhanced Setup tab for attention-only training comparison.

## Current State Analysis

### Existing Setup Tab (Current Repository)
- **Training Scope**: Attention-only LoRA training (limited to attention matrices)
- **Configuration**: Basic LoRA parameters (rank, alpha, dropout) without explicit layer coverage control
- **Backend Integration**: Uses original MLX LoRA defaults that target only attention layers
- **UI Elements**: Standard training parameters without comprehensive LoRA configuration

### SFT-Only Repository (Reference Implementation)
- **Training Scope**: Full-layer LoRA training (attention + MLP matrices)
- **Configuration**: Complete `lora_parameters` configuration with explicit matrix targeting
- **Backend Integration**: Explicit `lora_parameters` dict with 7 matrix keys and `num_layers: -1`
- **UI Elements**: Dedicated LoRA Configuration section with layer coverage controls

### Enhanced Setup Tab (Current Repository)
- **Training Scope**: GSPO/GRPO enhanced training methods
- **Configuration**: Comprehensive parameters for reinforcement learning fine-tuning
- **Status**: **DO NOT MODIFY** - must remain unchanged for comparison purposes

## Key Differences Identified

### 1. LoRA Matrix Coverage
- **Current Setup**: Implicit attention-only (MLX default behavior)
- **Target Implementation**: Explicit full-layer coverage:
  ```yaml
  lora_parameters:
    keys:
      - "self_attn.q_proj"     # Query projection
      - "self_attn.k_proj"     # Key projection  
      - "self_attn.v_proj"     # Value projection
      - "self_attn.o_proj"     # Output projection
      - "mlp.gate_proj"        # MLP gate
      - "mlp.up_proj"          # MLP up-projection
      - "mlp.down_proj"        # MLP down-projection
  ```

### 2. Layer Coverage Control
- **Current Setup**: No explicit layer coverage specification
- **Target Implementation**: `num_layers: -1` for all transformer blocks

### 3. Parameter Structure
- **Current Setup**: Individual LoRA parameters (rank, alpha, dropout) as separate fields
- **Target Implementation**: Structured `lora_parameters` object with comprehensive configuration

## Implementation Requirements

### 1. Backend Modifications

#### TrainingConfig Data Class
**File**: `backend/main.py`
```python
@dataclass
class TrainingConfig:
    # ... existing fields ...
    
    # Full-layer LoRA parameters
    fine_tune_type: str = "lora"
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_num_layers: int = -1  # -1 for all layers
```

#### Training Pipeline Integration
**File**: `backend/main.py`
- Update `start_training()` method to generate comprehensive `lora_parameters` dict
- Implement model-type detection for appropriate matrix targeting
- Add validation for layer coverage and parameter consistency

**Required Implementation**:
```python
def generate_lora_parameters(config: TrainingConfig, model_type: str) -> dict:
    """Generate comprehensive lora_parameters for full-layer training"""
    
    # Standard model matrices (Qwen2.5, Llama, etc.)
    standard_keys = [
        "self_attn.q_proj",
        "self_attn.k_proj", 
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj", 
        "mlp.down_proj"
    ]
    
    lora_parameters = {
        "rank": config.lora_rank,
        "scale": config.lora_alpha,
        "dropout": config.lora_dropout,
        "keys": standard_keys
    }
    
    return lora_parameters
```

### 2. Frontend Modifications

#### SetupPage Component
**File**: `frontend/src/pages/SetupPage.tsx`

##### New State Management
```typescript
const [formData, setFormData] = useState<Partial<TrainingConfig>>({
  // ... existing fields ...
  lora_rank: 32,
  lora_alpha: 32,
  lora_dropout: 0,
  lora_num_layers: -1
});
```

##### New UI Components

**LoRA Configuration Section** (insert after Training Parameters):
```tsx
{/* Full-Layer LoRA Configuration */}
<div className="card">
  <div className="card-header">
    <div className="flex items-center space-x-2">
      <Settings className="h-5 w-5 text-primary-600" />
      <h2 className="text-xl font-semibold">Full-Layer LoRA Configuration</h2>
    </div>
    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
      Applies adapters to ALL transformer layers (attention + MLP) for comprehensive fine-tuning.
    </p>
  </div>
  <div className="card-body">
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {/* LoRA Rank */}
      <div>
        <label className="block text-sm font-medium mb-2">LoRA Rank</label>
        <input
          type="number"
          min={1}
          className="input-field"
          value={formData.lora_rank}
          onChange={(e) => {
            const value = parseInt(e.target.value, 10);
            setFormData(prev => ({ 
              ...prev, 
              lora_rank: Number.isNaN(value) || value <= 0 ? 32 : value 
            }));
          }}
        />
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Rank 32 recommended for full-layer training
        </p>
      </div>

      {/* LoRA Alpha (Scale) */}
      <div>
        <label className="block text-sm font-medium mb-2">LoRA Alpha (Scale)</label>
        <input
          type="number"
          min={1}
          className="input-field"
          value={formData.lora_alpha}
          onChange={(e) => {
            const value = parseFloat(e.target.value);
            setFormData(prev => ({ 
              ...prev, 
              lora_alpha: Number.isNaN(value) || value <= 0 ? 32 : value 
            }));
          }}
        />
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Scale factor (alpha), typically equals rank
        </p>
      </div>

      {/* LoRA Dropout */}
      <div>
        <label className="block text-sm font-medium mb-2">LoRA Dropout</label>
        <input
          type="number"
          min={0}
          max={1}
          step="0.01"
          className="input-field"
          value={formData.lora_dropout}
          onChange={(e) => {
            const value = parseFloat(e.target.value);
            const clamped = Number.isNaN(value) ? 0 : Math.min(1, Math.max(0, value));
            setFormData(prev => ({ ...prev, lora_dropout: clamped }));
          }}
        />
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          0.0 recommended for stable training
        </p>
      </div>

      {/* Layer Coverage */}
      <div>
        <label className="block text-sm font-medium mb-2">Layer Coverage</label>
        <select
          className="select-field"
          value={formData.lora_num_layers}
          onChange={(e) => {
            const value = parseInt(e.target.value, 10);
            setFormData(prev => ({ 
              ...prev, 
              lora_num_layers: Number.isNaN(value) ? -1 : value 
            }));
          }}
        >
          <option value={-1}>All Layers (-1)</option>
          <option value={24}>Top 24 Layers</option>
          <option value={16}>Top 16 Layers</option>
          <option value={8}>Top 8 Layers</option>
        </select>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Qwen2.5-0.5B has 24 transformer blocks
        </p>
      </div>
    </div>

    {/* Coverage Visualization */}
    <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
      <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
        Full-Layer Coverage Matrix
      </h4>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="font-medium text-blue-800 dark:text-blue-200">Attention Layers:</p>
          <ul className="text-blue-700 dark:text-blue-300 mt-1">
            <li>• Query (q_proj)</li>
            <li>• Key (k_proj)</li>
            <li>• Value (v_proj)</li>
            <li>• Output (o_proj)</li>
          </ul>
        </div>
        <div>
          <p className="font-medium text-blue-800 dark:text-blue-200">MLP Layers:</p>
          <ul className="text-blue-700 dark:text-blue-300 mt-1">
            <li>• Gate (gate_proj)</li>
            <li>• Up (up_proj)</li>
            <li>• Down (down_proj)</li>
          </ul>
        </div>
      </div>
      <p className="text-xs text-blue-600 dark:text-blue-400 mt-3">
        Total: 7 matrices × {formData.lora_num_layers === -1 ? 'all' : formData.lora_num_layers} layers
      </p>
    </div>
  </div>
</div>
```

#### Training Slice Type Updates
**File**: `frontend/src/store/slices/trainingSlice.ts`

```typescript
export interface TrainingConfig {
  // ... existing fields ...
  
  // Full-layer LoRA parameters
  lora_rank?: number;
  lora_alpha?: number; 
  lora_dropout?: number;
  lora_num_layers?: number;
}
```

### 3. Configuration Generation

#### Backend Config Builder
**File**: `backend/main.py` - modify training config generation:

```python
# In start_training method
config_data = {
    # ... existing config ...
    
    # Full-layer LoRA configuration
    "lora_parameters": generate_lora_parameters(config, model_type),
    "num_layers": config.lora_num_layers,
    "fine_tune_type": "lora"
}
```

### 4. Validation & Error Handling

#### Parameter Validation
- Validate LoRA rank >= 1
- Validate alpha scale >= 1  
- Validate dropout range [0, 1]
- Validate layer coverage (-1 or positive integer)
- Model-type compatibility checks

#### Error Messaging
- Clear validation feedback for invalid LoRA parameters
- Guidance for optimal parameter ranges
- Model-specific recommendations

### 5. User Experience Enhancements

#### Default Values
- LoRA Rank: 32 (recommended for full-layer)
- LoRA Alpha: 32 (typically equals rank)
- LoRA Dropout: 0.0 (recommended for stability)
- Layer Coverage: -1 (all layers)

#### Help Text & Tooltips
- Explanations for each parameter's impact
- Recommendations based on model size
- Comparison guidance (attention-only vs full-layer)

#### Visual Indicators
- Coverage matrix visualization
- Parameter range indicators
- Model-specific recommendations

## Comparison Framework Preservation

### Enhanced Setup Tab
**CRITICAL**: The Enhanced Setup tab must remain **UNCHANGED** to enable:
- Side-by-side comparison of training methods
- A/B testing of attention-only vs full-layer training
- Benchmarking and evaluation workflows

### Comparison Capabilities
The updated Setup tab will enable:
- **Training Method Comparison**: Full-layer vs attention-only SFT
- **Performance Analysis**: Quality improvements with comprehensive fine-tuning
- **Resource Usage Analysis**: Memory/time trade-offs
- **Model Quality Assessment**: Inference comparisons via Compare tab

## Technical Implementation Details

### Backend Configuration Flow
1. **Parameter Reception**: Receive enhanced LoRA parameters from frontend
2. **Model Detection**: Identify model architecture type
3. **Parameter Generation**: Create comprehensive `lora_parameters` dict
4. **Config Assembly**: Build complete MLX training configuration
5. **Training Execution**: Pass full configuration to MLX LoRA trainer

### Frontend Data Flow
1. **Parameter Input**: User configures full-layer LoRA settings
2. **Validation**: Client-side parameter validation
3. **Submission**: Send enhanced configuration to backend
4. **Status Updates**: Real-time training progress with full-layer metrics
5. **Completion**: Model adapter with full-layer coverage

### Integration Points
- **Training Slice**: Extended type definitions for new parameters
- **Backend API**: Enhanced configuration generation
- **MLX Integration**: Seamless parameter passing to training pipeline
- **Compare Tab**: Compatibility with full-layer trained adapters

## Success Criteria

### Functional Requirements
- [ ] Full-layer LoRA training produces adapters with 7 matrix coverage
- [ ] Layer coverage control works correctly (-1, 24, 16, 8 options)
- [ ] Parameter validation prevents invalid configurations
- [ ] Training completion with full-layer adapters
- [ ] Compare tab compatibility with full-layer trained models

### Quality Requirements  
- [ ] Default parameters optimized for common use cases
- [ ] Clear user guidance for parameter selection
- [ ] Robust error handling and validation
- [ ] Consistent UI/UX with existing interface

### Performance Requirements
- [ ] No regression in existing training workflows
- [ ] Enhanced training completes successfully
- [ ] Full-layer adapters work correctly with inference

## Testing Strategy

### Unit Tests
- Parameter validation logic
- Configuration generation correctness
- Model-type detection accuracy

### Integration Tests
- End-to-end full-layer training workflow
- Backend-frontend parameter passing
- MLX configuration generation

### User Acceptance Tests
- Complete training runs with various parameter combinations
- Compare tab functionality with full-layer models
- Error handling and validation scenarios

## Implementation Timeline

### Phase 1: Backend Foundation (Week 1)
- Extend TrainingConfig data class
- Implement lora_parameters generation
- Add parameter validation
- Update configuration assembly

### Phase 2: Frontend UI (Week 2)  
- Add LoRA configuration section to SetupPage
- Implement state management for new parameters
- Add validation and help text
- Create coverage visualization

### Phase 3: Integration & Testing (Week 3)
- End-to-end integration testing
- Compare tab compatibility verification
- User acceptance testing
- Documentation updates

## Risk Mitigation

### Technical Risks
- **MLX Compatibility**: Ensure parameter structure matches MLX expectations
- **Model Detection**: Robust architecture identification for different models
- **Backward Compatibility**: Preserve existing attention-only training capability

### User Experience Risks
- **Parameter Complexity**: Provide clear guidance and sensible defaults
- **Training Time**: Full-layer training may take longer - set expectations
- **Resource Usage**: Monitor memory requirements for comprehensive coverage

## Conclusion

This PRD outlines a comprehensive approach to implementing full-layer LoRA training in the Setup tab while preserving the existing Enhanced Setup for comparison capabilities. The implementation will enable users to train models with comprehensive coverage (attention + MLP matrices) and compare results against attention-only training, providing valuable insights into the effectiveness of different fine-tuning approaches.

The design maintains backward compatibility, provides enhanced user experience with clear parameter guidance, and ensures seamless integration with the existing training and comparison infrastructure.
