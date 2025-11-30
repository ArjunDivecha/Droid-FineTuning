#!/usr/bin/env python3
"""Quick verification test for Full-Layer LoRA implementation"""

import sys
import os

# Add backend to path
sys.path.insert(0, '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend')

print("=" * 70)
print("QUICK VERIFICATION TEST - Full-Layer LoRA Implementation")
print("=" * 70)
print()

# Test 1: Import and syntax check
print("TEST 1: Checking imports and syntax...")
try:
    from main import TrainingConfig, TrainingManager
    print("✅ Imports successful - no syntax errors")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: TrainingConfig has new fields
print("\nTEST 2: Verifying TrainingConfig fields...")
try:
    config = TrainingConfig(
        model_path="/test",
        train_data_path="/test",
        val_data_path="/test"
    )
    
    # Check new fields exist
    required_fields = ['fine_tune_type', 'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_num_layers']
    missing = [f for f in required_fields if not hasattr(config, f)]
    
    if missing:
        print(f"❌ Missing fields: {missing}")
        sys.exit(1)
    
    print("✅ All LoRA fields present")
    
    # Check defaults
    assert config.learning_rate == 1e-4, f"Wrong LR: {config.learning_rate}"
    assert config.lora_rank == 32, f"Wrong rank: {config.lora_rank}"
    assert config.lora_alpha == 32.0, f"Wrong alpha: {config.lora_alpha}"
    assert config.lora_dropout == 0.0, f"Wrong dropout: {config.lora_dropout}"
    assert config.lora_num_layers == -1, f"Wrong layers: {config.lora_num_layers}"
    
    print("✅ All defaults correct")
    print(f"   • Learning Rate: {config.learning_rate}")
    print(f"   • LoRA Rank: {config.lora_rank}")
    print(f"   • LoRA Alpha: {config.lora_alpha}")
    print(f"   • LoRA Dropout: {config.lora_dropout}")
    print(f"   • LoRA Layers: {config.lora_num_layers}")
    
except AssertionError as e:
    print(f"❌ Assertion failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check LoRA keys are defined in code
print("\nTEST 3: Verifying LoRA parameter generation code...")
try:
    # Read the main.py file to verify lora_keys are defined
    with open('/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/main.py', 'r') as f:
        content = f.read()
    
    required_keys = [
        'self_attn.q_proj',
        'self_attn.k_proj',
        'self_attn.v_proj',
        'self_attn.o_proj',
        'mlp.gate_proj',
        'mlp.up_proj',
        'mlp.down_proj'
    ]
    
    missing_keys = [k for k in required_keys if k not in content]
    
    if missing_keys:
        print(f"❌ Missing LoRA keys in code: {missing_keys}")
        sys.exit(1)
    
    print("✅ All 7 LoRA matrices defined in code")
    for key in required_keys:
        print(f"   • {key}")
    
    # Check for lora_parameters dict
    if 'lora_parameters' not in content:
        print("❌ lora_parameters dict not found in code")
        sys.exit(1)
    
    print("✅ lora_parameters dict present")
    
    # Check for architecture detection
    if 'model_type' not in content or 'config.json' not in content:
        print("⚠️  Warning: Architecture detection may not be implemented")
    else:
        print("✅ Architecture detection code present")
    
except Exception as e:
    print(f"❌ Code verification failed: {e}")
    sys.exit(1)

# Test 4: Frontend TypeScript file check
print("\nTEST 4: Verifying frontend files...")
try:
    ts_file = '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/frontend/src/store/slices/trainingSlice.ts'
    
    if not os.path.exists(ts_file):
        print(f"❌ File not found: {ts_file}")
        sys.exit(1)
    
    with open(ts_file, 'r') as f:
        ts_content = f.read()
    
    # Check for new fields
    ts_fields = ['lora_rank', 'lora_alpha', 'lora_dropout', 'lora_num_layers']
    missing_ts = [f for f in ts_fields if f not in ts_content]
    
    if missing_ts:
        print(f"❌ Missing fields in trainingSlice.ts: {missing_ts}")
        sys.exit(1)
    
    print("✅ trainingSlice.ts has LoRA fields")
    
    # Check SetupPage
    setup_file = '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/frontend/src/pages/SetupPage.tsx'
    
    if not os.path.exists(setup_file):
        print(f"❌ File not found: {setup_file}")
        sys.exit(1)
    
    with open(setup_file, 'r') as f:
        setup_content = f.read()
    
    # Check for LoRA Configuration section
    if 'Full-Layer LoRA Configuration' not in setup_content:
        print("❌ LoRA Configuration section not found in SetupPage.tsx")
        sys.exit(1)
    
    print("✅ SetupPage.tsx has LoRA Configuration section")
    
    # Check for input fields
    if 'lora_rank' not in setup_content or 'lora_alpha' not in setup_content:
        print("❌ LoRA input fields not found in SetupPage.tsx")
        sys.exit(1)
    
    print("✅ LoRA input fields present")
    
except Exception as e:
    print(f"❌ Frontend verification failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("🎉 ALL VERIFICATION TESTS PASSED!")
print("=" * 70)
print()
print("✅ Summary:")
print("   • Backend TrainingConfig updated")
print("   • LoRA parameter generation code present")
print("   • All 7 matrices configured")
print("   • Frontend Redux store updated")
print("   • Frontend UI components added")
print()
print("📋 Next Steps:")
print("   1. Start backend server:")
print("      cd backend && python main.py")
print()
print("   2. Start frontend (in new terminal):")
print("      cd frontend && npm start")
print()
print("   3. Open browser to http://localhost:3000")
print("      Navigate to Setup page")
print("      Verify 'Full-Layer LoRA Configuration' section appears")
print()
print("   4. Test with actual training:")
print("      - Use a small model (Qwen2.5-0.5B)")
print("      - Set iterations to 100")
print("      - Check logs for LoRA configuration output")
print("      - Verify ~3.5-4% trainable parameters")
print()
print("=" * 70)
