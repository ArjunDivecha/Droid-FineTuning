# Nested Learning UI Improvements

**Date**: November 9, 2025
**Changes**: Model/Adapter Dropdowns + Default Directories

---

## ✅ Changes Made

### 1. **Model Selection → Dropdown**

**Before**: Text input + Browse button

**After**:
- **Dropdown populated with all available models** from `/models` endpoint
- Shows model name and type in dropdown (e.g., "Qwen2.5-7B-Instruct (qwen2)")
- Displays full path below dropdown when selected
- "Browse for Custom Model" button for non-listed models

### 2. **Adapter Selection → Dropdown**

**Before**: Text input + Browse button

**After**:
- **Dropdown populated with all available adapters** from `/adapters` endpoint
- Shows adapter name in dropdown (e.g., "7b", "medical_adapter")
- Displays full path below dropdown when selected
- "Browse for Custom Adapter" button for non-listed adapters

### 3. **Default Directories for File Picker**

**Before**: File picker opened at random location

**After**: File picker defaults to:
- **Models**: `/local_qwen/artifacts/base_model/`
- **Adapters**: `/local_qwen/artifacts/lora_adapters/`
- **Training Data**: `/local_qwen/`

---

## 🔧 Backend Changes

### New Endpoint: `/adapters`

```python
@app.get("/adapters")
async def list_adapters():
    """List available LoRA adapters"""
    # Scans: /local_qwen/artifacts/lora_adapters/
    # Returns: List of adapter directories with metadata
```

**Response**:
```json
{
  "adapters": [
    {
      "name": "7b",
      "path": "/path/to/7b",
      "lora_rank": 8,
      "lora_alpha": 16
    },
    {
      "name": "medical_adapter",
      "path": "/path/to/medical_adapter",
      "lora_rank": 16,
      "lora_alpha": 32
    }
  ]
}
```

---

## 📝 Frontend Changes

### NestedLearningPage.tsx

**Lines 278-355**: Replaced text inputs with dropdown + browse pattern

```tsx
{/* Dropdown */}
<select className="select-field" value={formData.base_model_path}>
  <option value="">Select a base model...</option>
  {availableModels.map((model) => (
    <option key={model.path} value={model.path}>
      {model.name} ({model.model_type})
    </option>
  ))}
</select>

{/* Show selected path */}
{formData.base_model_path && (
  <div className="bg-gray-50 p-3 rounded-lg">
    <p className="text-xs font-mono">{formData.base_model_path}</p>
  </div>
)}

{/* Browse for custom */}
<button onClick={() => handleFileSelect('model')}>
  Browse for Custom Model
</button>
```

**Lines 104-142**: Updated `handleFileSelect` with default paths

```tsx
const handleFileSelect = async (type) => {
  let defaultPath = '';

  switch (type) {
    case 'model':
      defaultPath = '/local_qwen/artifacts/base_model';
      break;
    case 'adapter':
      defaultPath = '/local_qwen/artifacts/lora_adapters';
      break;
    // ...
  }

  const result = await window.electronAPI.showOpenDialog({
    defaultPath,  // Opens here by default
    // ...
  });
};
```

---

## 🎨 UI Flow (Updated)

### Model Selection Flow

```
1. User opens Nested Learning tab
   ↓
2. Dropdown shows all models from /models endpoint
   - Qwen2.5-7B-Instruct (qwen2)
   - Qwen3-32B-MLX-4bit (qwen2)
   ↓
3. User selects from dropdown
   ↓
4. Full path displayed below
   ↓
5. (Optional) User clicks "Browse for Custom Model"
   - File picker opens at /local_qwen/artifacts/base_model/
```

### Adapter Selection Flow

```
1. Dropdown shows all adapters from /adapters endpoint
   - 7b
   - medical_adapter
   - bf16-full
   ↓
2. User selects from dropdown
   ↓
3. Full path displayed below
   ↓
4. (Optional) User clicks "Browse for Custom Adapter"
   - File picker opens at /local_qwen/artifacts/lora_adapters/
```

---

## 🧪 Testing

### Test Dropdowns

1. Open app
2. Go to Nested Learning tab
3. Check **Base Model** dropdown → Should show your models
4. Check **LoRA Adapter** dropdown → Should show your adapters

### Test File Picker Defaults

1. Click "Browse for Custom Model"
   - Should open at `/local_qwen/artifacts/base_model/`
2. Click "Browse for Custom Adapter"
   - Should open at `/local_qwen/artifacts/lora_adapters/`

---

## 📁 Files Modified

```
backend/main.py
├── Added: @app.get("/adapters") endpoint (lines 1150-1188)

frontend/src/pages/NestedLearningPage.tsx
├── Modified: Model selection UI (lines 278-315)
├── Modified: Adapter selection UI (lines 317-355)
└── Modified: handleFileSelect() with defaults (lines 104-142)
```

---

## ✨ Benefits

### Before
- ❌ Had to manually type model paths
- ❌ Had to remember adapter locations
- ❌ File picker opened at random location
- ❌ Easy to make typos in paths

### After
- ✅ Select models from dropdown
- ✅ Select adapters from dropdown
- ✅ File picker opens at correct location
- ✅ No typing required for standard models/adapters
- ✅ Full path shown for verification
- ✅ Browse option still available for custom locations

---

## 🚀 How to Use

### Quick Path (Dropdown Selection)

1. Open dropdown
2. Select model/adapter
3. Done! ✅

### Custom Path (Browse)

1. Click "Browse for Custom Model/Adapter"
2. File picker opens at default location
3. Navigate to custom location if needed
4. Select directory
5. Done! ✅

---

## 🔄 Restart to See Changes

```bash
cd /Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning
npm run dev
```

The Nested Learning tab will now have:
- ✅ Model dropdown with all your base models
- ✅ Adapter dropdown with all your LoRA adapters
- ✅ File pickers that default to the correct directories

---

**Enjoy the improved workflow!** 🎉
