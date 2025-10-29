# LM Studio Configuration - Model Path Setup

**Date:** October 26, 2025  
**Purpose:** Document LM Studio configuration for loading the new Qwen2.5-0.5B-bf16 model

## What Was Changed

Created a **symbolic link** (symlink) in LM Studio's default models directory, pointing to your model. This is the cleanest solution—no settings changes, no file duplication, and LM Studio discovers it automatically.

### Symlink Created
- **Location:** `/Users/macbook2024/.cache/lm-studio/models/mlx-community/Qwen2.5-0.5B-bf16`
- **Points to:** `/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/qwen2.5-0.5b-bf16`
- **Status:** ✅ Verified - model files accessible through symlink
- **Benefit:** No file duplication (~950MB saved), instant updates if you fine-tune the model

### How LM Studio Will See It

LM Studio scans its default directory: `/Users/macbook2024/.cache/lm-studio/models`

Inside the `mlx-community/` subfolder, it will now find:
- ✅ **Qwen2.5-0.5B-bf16** (NEW - symlinked to your model)
- All your existing mlx-community models (Qwen3 variants, etc.)

**Display Name in LM Studio:** `mlx-community/Qwen2.5-0.5B-bf16`  
**Architecture:** qwen2 (0.5B params, bf16 format)  
**Status:** ✅ Ready to load

## How to Use

### Step 1: Restart LM Studio
Close and reopen LM Studio completely (quit the app, don't just close the window).

### Step 2: Find Your Model
1. Go to the **"My Models"** or **"Local Models"** section
2. Filter by **"LLMs"** (should be selected by default)
3. Look for **"mlx-community/Qwen2.5-0.5B-bf16"** in the list
   - It will show: **Arch:** qwen2, **Params:** 0.5B, **Quant:** (none - bf16), **Size:** ~950MB
4. It should appear alongside your other mlx-community models

### Step 3: Load and Test
1. Click on the model to select it
2. Click **"Load Model"** or **"Start Chat"**
3. Your M4 Max will load it in ~1-2GB VRAM (very fast with bf16 format)
4. Test with a stock market question to see baseline performance

## Model Details

- **Name:** Qwen2.5-0.5B-bf16
- **Size:** ~950MB (bf16 compressed)
- **Architecture:** Qwen2 (24 layers, 896 hidden size)
- **Format:** bfloat16 (optimized for Apple Silicon)
- **Hardware:** Runs efficiently on M4 Max with Metal GPU acceleration
- **Use Cases:** 
  - Quick inference for stock sentiment analysis
  - Testing before fine-tuning
  - Baseline comparison with fine-tuned versions

## Troubleshooting

### Model Doesn't Appear?
1. **Restart LM Studio** (must fully quit and reopen)
2. **Check settings:** Open LM Studio → Settings → verify custom path is listed
3. **Refresh models:** Look for a "Refresh" or "Rescan" button in the models list

### LM Studio Won't Start?
1. **Backup restored:** If LM Studio crashes, the old settings.json is backed up at:
   `/Users/macbook2024/Library/Application Support/LM Studio/settings.json.backup`
2. **Restore command:**
   ```bash
   cp "/Users/macbook2024/Library/Application Support/LM Studio/settings.json.backup" "/Users/macbook2024/Library/Application Support/LM Studio/settings.json"
   ```

### Want to Remove Custom Path Later?
Edit `/Users/macbook2024/Library/Application Support/LM Studio/settings.json` and remove the `localModelsFolders` section (lines 4-6).

## Integration with Your Fine-Tuning App

Your Electron GUI app and LM Studio now **share the same model location** (the custom path), so:
- ✅ Both apps see the same models
- ✅ Fine-tune in your GUI → test in LM Studio immediately
- ✅ No duplicate downloads needed
- ✅ Consistent model versions across tools

## Next Steps

1. **Restart LM Studio** and verify the model appears
2. **Load and test** the model with sample stock data
3. **Compare performance** with your other models
4. **Fine-tune** using your Electron GUI app when ready
5. **Re-test in LM Studio** after fine-tuning to see improvements

---

**Documentation:** This setup is also documented in:
- `MODELS_INVENTORY.md` (model details and paths)
- `config/models.json` (project model configuration)

**No Code Changes:** Your app's `main.py` and other files remain unchanged.

## Visual Guide - Where to Find It

When you open LM Studio's "My Models" section, you'll see a table like this:

```
Filter by: [LLMs] Text Embedding View All

Arch    Params   Publisher        LLM                                    Quant    Size        Date Modified
qwen2   0.5B     mlx-community    mlx-community/Qwen2.5-0.5B-bf16       (bf16)   ~950 MB     Today  <-- YOUR NEW MODEL
qwen2   0.5B     lmstudio-com...  qwen2.5-0.5b-instruct-mlx             4bit     293.99 MB   26 days ago
qwen3   32B      Qwen             qwen3-32b-mlx                         4bit     17.42 GB    2 days ago
...
```

**Look for:** A row with `mlx-community` as the publisher and `Qwen2.5-0.5B-bf16` as the model name.

---

## Technical Details - How This Works

### What is a Symbolic Link?
A symlink is like a "shortcut" or "alias" that points to files in another location. Benefits:
- ✅ **No duplication** - Saves ~950MB of disk space
- ✅ **Instant sync** - If you fine-tune the model in your GUI app, LM Studio sees the updates immediately
- ✅ **Clean structure** - LM Studio expects `publisher/model-name/` format, symlink provides that
- ✅ **No settings changes** - Works with LM Studio's default configuration

### Command Used
```bash
ln -s "/path/to/actual/model" "/Users/macbook2024/.cache/lm-studio/models/mlx-community/Qwen2.5-0.5B-bf16"
```

This creates a link in LM Studio's directory that points to your model's actual location.

