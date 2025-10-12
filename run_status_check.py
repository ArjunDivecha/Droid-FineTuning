#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend')

print("\n" + "="*60)
print("QUICK STATUS CHECK")
print("="*60 + "\n")

# Test 1: Can we import?
try:
    from main import TrainingConfig
    config = TrainingConfig(model_path="/t", train_data_path="/t", val_data_path="/t")
    
    print("✅ TrainingConfig imported successfully")
    print(f"   Learning Rate: {config.learning_rate}")
    
    if hasattr(config, 'lora_rank'):
        print(f"   LoRA Rank: {config.lora_rank}")
        print(f"   LoRA Alpha: {config.lora_alpha}")
        print("\n✅ MILESTONE 1 COMPLETE: TrainingConfig has LoRA fields\n")
    else:
        print("\n❌ MILESTONE 1 INCOMPLETE: LoRA fields missing\n")
        
except Exception as e:
    print(f"❌ Import failed: {e}\n")

# Test 2: Check for LoRA generation code
try:
    with open('/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/main.py') as f:
        code = f.read()
    
    if 'lora_parameters' in code and 'self_attn.q_proj' in code:
        print("✅ MILESTONE 2 COMPLETE: LoRA generation code found\n")
    else:
        print("❌ MILESTONE 2 INCOMPLETE: LoRA generation code not found")
        print("   You may need to ACCEPT the proposed changes in your IDE\n")
        
except Exception as e:
    print(f"❌ Check failed: {e}\n")

print("="*60)
print("Run 'python3 quick_test.py' for full verification")
print("="*60 + "\n")
