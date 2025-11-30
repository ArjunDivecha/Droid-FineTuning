#!/usr/bin/env python3
"""Check implementation status - what's done and what's pending"""

import sys
import os

sys.path.insert(0, '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend')

print("=" * 70)
print("IMPLEMENTATION STATUS CHECK")
print("=" * 70)
print()

# Check 1: TrainingConfig
print("✓ MILESTONE 1: TrainingConfig Dataclass")
print("-" * 70)
try:
    from main import TrainingConfig
    config = TrainingConfig(
        model_path="/test",
        train_data_path="/test",
        val_data_path="/test"
    )
    
    # Check if new fields exist
    has_lora_fields = all(hasattr(config, f) for f in ['fine_tune_type', 'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_num_layers'])
    
    if has_lora_fields:
        print("✅ COMPLETE - All LoRA fields present")
        print(f"   Learning Rate: {config.learning_rate} (should be 0.0001)")
        print(f"   LoRA Rank: {config.lora_rank}")
        print(f"   LoRA Alpha: {config.lora_alpha}")
        print(f"   LoRA Dropout: {config.lora_dropout}")
        print(f"   LoRA Layers: {config.lora_num_layers}")
    else:
        print("❌ INCOMPLETE - LoRA fields missing")
        print("   Action: Accept proposed changes in IDE")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

print()

# Check 2: LoRA Generation Code
print("✓ MILESTONE 2: LoRA Parameter Generation")
print("-" * 70)
try:
    with open('/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/main.py', 'r') as f:
        content = f.read()
    
    # Check for key indicators
    has_lora_keys = 'self_attn.q_proj' in content and 'mlp.gate_proj' in content
    has_lora_params_dict = 'lora_parameters' in content and '"keys":' in content
    has_architecture_detection = 'model_type' in content and 'config.json' in content
    
    if has_lora_keys and has_lora_params_dict:
        print("✅ COMPLETE - LoRA generation code present")
        if has_architecture_detection:
            print("   ✓ Architecture detection included")
        else:
            print("   ⚠️  Architecture detection may be missing")
    else:
        print("❌ INCOMPLETE - LoRA generation code not found")
        print("   Action: Accept proposed changes for start_training method")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

print()

# Check 3: Training Endpoint
print("✓ MILESTONE 3: Training Endpoint")
print("-" * 70)
try:
    with open('/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/main.py', 'r') as f:
        content = f.read()
    
    # Look for endpoint accepting LoRA params
    endpoint_section = content[content.find('@app.post("/training/start")'):content.find('@app.post("/training/start")')+2000] if '@app.post("/training/start")' in content else ""
    
    has_lora_params_in_endpoint = 'lora_rank' in endpoint_section and 'lora_alpha' in endpoint_section
    
    if has_lora_params_in_endpoint:
        print("✅ COMPLETE - Endpoint accepts LoRA parameters")
    else:
        print("❌ INCOMPLETE - Endpoint not updated")
        print("   Action: Accept proposed changes for /training/start endpoint")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

print()

# Check 4: Frontend Redux
print("✓ MILESTONE 4: Redux Store")
print("-" * 70)
try:
    ts_file = '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/frontend/src/store/slices/trainingSlice.ts'
    
    if os.path.exists(ts_file):
        with open(ts_file, 'r') as f:
            ts_content = f.read()
        
        has_lora_fields = all(f in ts_content for f in ['lora_rank', 'lora_alpha', 'lora_dropout', 'lora_num_layers'])
        
        if has_lora_fields:
            print("✅ COMPLETE - Redux store has LoRA fields")
        else:
            print("❌ INCOMPLETE - LoRA fields not in Redux store")
            print("   Action: Accept proposed changes for trainingSlice.ts")
    else:
        print("❌ File not found")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

print()

# Check 5: Frontend UI
print("✓ MILESTONE 5: Frontend UI")
print("-" * 70)
try:
    setup_file = '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/frontend/src/pages/SetupPage.tsx'
    
    if os.path.exists(setup_file):
        with open(setup_file, 'r') as f:
            setup_content = f.read()
        
        has_lora_section = 'Full-Layer LoRA Configuration' in setup_content
        has_lora_inputs = 'lora_rank' in setup_content and 'lora_alpha' in setup_content
        
        if has_lora_section and has_lora_inputs:
            print("✅ COMPLETE - UI has LoRA Configuration section")
        else:
            print("❌ INCOMPLETE - LoRA UI section not added")
            print("   Action: Accept proposed changes for SetupPage.tsx")
    else:
        print("❌ File not found")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("The implementation code has been PROPOSED but may not be APPLIED yet.")
print()
print("If you see ❌ INCOMPLETE above, you need to:")
print("1. Review the proposed changes in your IDE")
print("2. Accept/Apply the changes")
print("3. Run this script again to verify")
print()
print("Once all show ✅ COMPLETE, run:")
print("  python3 quick_test.py")
print()
print("=" * 70)
