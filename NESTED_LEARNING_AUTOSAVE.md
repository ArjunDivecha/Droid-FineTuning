# Nested Learning Auto-Save Feature

## Overview

The Nested Learning page now automatically saves and restores your training configuration, so you don't have to re-enter all parameters every time you restart the application or if a test fails.

## How It Works

### Auto-Save on Training Start
When you click **"Start Nested Learning"**, the system automatically saves:
- All file paths (base model, adapter, training data, validation data)
- All tier configuration (number of tiers, frequencies, assignment strategy)
- All training parameters (learning rate, batch size, steps, etc.)
- All LoRA settings (rank, alpha, dropout)
- All advanced settings (warmup, checkpointing, evaluation frequency)

The configuration is saved to your browser's localStorage, so it persists across sessions.

### Auto-Restore on Page Load
When you navigate to the Nested Learning page:
1. The system checks if a saved configuration exists
2. If found, it automatically loads all your previous settings
3. You'll see a notification: **"Configuration Restored"**
4. All fields will be pre-populated with your last used values

### Manual Reset
If you want to start fresh with default values:
1. Click the **"Reset Form"** button
2. This will:
   - Clear all fields to default values
   - Remove the saved configuration from localStorage
   - Show a notification: **"Configuration Reset"**

## What Gets Saved

```json
{
  "base_model_path": "/path/to/model",
  "adapter_path": "/path/to/adapter",
  "train_data_path": "/path/to/train.jsonl",
  "val_data_path": "/path/to/val.jsonl",
  "num_tiers": 3,
  "tier_update_frequencies": [1, 2, 4],
  "tier_assignment_strategy": "layer_depth",
  "learning_rate": 0.00001,
  "batch_size": 4,
  "num_steps": 1000,
  "max_seq_length": 2048,
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.0,
  "warmup_steps": 100,
  "gradient_accumulation_steps": 2,
  "checkpoint_every": 100,
  "eval_every": 100,
  "output_path": "./nested_learning/checkpoints",
  "experiment_name": "nested_learning_experiment"
}
```

## Benefits

1. **Time Saving**: No need to re-enter 20+ parameters every time
2. **Error Prevention**: Reduces risk of typos when re-entering paths
3. **Quick Iteration**: Test different configurations faster
4. **Recovery**: If training fails, you can immediately retry with the same settings

## Use Cases

### Iterative Testing
```
1. Set up configuration with your paths and parameters
2. Click "Start Nested Learning" (config is saved)
3. Training fails due to some issue
4. Fix the issue (e.g., fix a bug in code)
5. Reload the page
6. Configuration is automatically restored
7. Click "Start Nested Learning" again
```

### Multiple Experiments
```
1. Run experiment with config A (saved)
2. Modify some parameters for config B
3. Click "Start Nested Learning" (config B is now saved)
4. Want to go back to config A?
   - Manually set parameters back
   - Or use "Reset Form" and re-enter config A
```

### Clean Slate
```
1. Have previous config loaded
2. Want to start completely fresh
3. Click "Reset Form"
4. All fields reset to defaults
5. Saved config is cleared
```

## Technical Details

### Storage Location
- **Browser**: localStorage (key: `nested_learning_last_config`)
- **Scope**: Per-browser, per-origin
- **Persistence**: Survives browser restarts
- **Size Limit**: ~5-10MB (more than enough for config)

### Data Format
- Stored as JSON string
- Automatically serialized/deserialized
- Includes all form fields

### Privacy
- Stored locally in your browser
- Never sent to any server
- Only accessible to the Droid-FineTuning app
- Can be cleared by:
  - Clicking "Reset Form"
  - Clearing browser data
  - Opening browser DevTools > Application > localStorage > delete key

## Console Logging

For debugging, check the browser console:
- **On Load**: `"Loaded saved configuration: {...}"`
- **On Save**: `"Configuration saved to localStorage"`
- **On Error**: `"Failed to load/save config: ..."`

## Troubleshooting

### Configuration Not Restoring
**Check**:
1. Open browser DevTools (F12)
2. Go to Application > Local Storage
3. Look for key: `nested_learning_last_config`
4. If missing, configuration wasn't saved

**Solution**:
- Make sure you click "Start Nested Learning" to trigger save
- Check console for error messages

### Want to Clear Manually
**Option 1**: Use Reset Button
- Click "Reset Form" button in the UI

**Option 2**: Browser DevTools
1. F12 to open DevTools
2. Application tab > Local Storage
3. Find `nested_learning_last_config`
4. Right-click > Delete
5. Or click "Clear All"

**Option 3**: Browser Settings
- Clear browsing data > Cookies and site data

### Configuration from Different Version
If you update the app and saved config has old/incompatible fields:
1. Click "Reset Form" to clear old config
2. Re-enter your settings
3. Click "Start Nested Learning" to save new format

## Example Workflow

### Day 1: Initial Setup
```
1. Navigate to Nested Learning page
2. Select base model: Qwen2.5-7B-Instruct
3. Select adapter: 7b
4. Select training data: my_data.jsonl
5. Set num_steps: 500
6. Set batch_size: 2
7. Click "Start Nested Learning"
   → Configuration saved
   → Training starts
```

### Day 2: Continue Testing
```
1. Navigate to Nested Learning page
   → Notification: "Configuration Restored"
   → All fields pre-filled with yesterday's values
2. Change num_steps: 1000 (only change needed)
3. Click "Start Nested Learning"
   → New configuration saved
   → Training starts with updated steps
```

### Day 3: New Experiment
```
1. Navigate to Nested Learning page
   → Previous config loaded
2. Click "Reset Form"
   → Notification: "Configuration Reset"
   → All fields cleared
3. Enter new experiment settings
4. Click "Start Nested Learning"
   → New configuration saved
```

## Future Enhancements

Potential improvements:
- Save multiple named configurations
- Export/import configurations as JSON files
- Configuration history/versioning
- Share configurations between team members
- Pre-set templates for common use cases

## Related Documentation

- `NESTED_LEARNING_QUICKSTART.md` - General usage guide
- `NESTED_LEARNING_TEST_RESULTS.md` - Test validation
- `NESTED_LEARNING_EXPLAINED.md` - Technical deep-dive
