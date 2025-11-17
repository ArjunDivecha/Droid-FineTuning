#!/usr/bin/env python3
"""
Quick test to debug the LoRA parameter mismatch issue
"""

import sys
from pathlib import Path
from mlx_lm import load
from mlx.utils import tree_flatten

# Use paths from the "big" experiment that was failing
BASE_MODEL = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct"
ADAPTER = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints/big/checkpoints/best"

print("=" * 60)
print("Testing LoRA Parameter Loading")
print("=" * 60)

# Load model with adapter
print(f"\n1. Loading model from: {BASE_MODEL}")
print(f"2. With adapter from: {ADAPTER}")

model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER)

print("\n3. Checking parameters BEFORE any freeze/unfreeze:")
print("   Calling model.parameters()...")
all_params = tree_flatten(model.parameters(), destination={})
print(f"   Total parameters: {len(all_params)}")
print(f"   Sample keys: {list(all_params.keys())[:5]}")

print("\n4. Checking trainable parameters with model.train():")
model.train()
trainable_after_train = tree_flatten(model.trainable_parameters(), destination={})
print(f"   Trainable parameters: {len(trainable_after_train)}")
print(f"   Sample keys: {list(trainable_after_train.keys())[:5]}")

# Check if any have 'lora' in the name
lora_params = [k for k in trainable_after_train.keys() if 'lora' in k.lower()]
print(f"   Parameters with 'lora' in name: {len(lora_params)}")
if lora_params:
    print(f"   Sample LoRA params: {lora_params[:5]}")

print("\n5. Now trying freeze/unfreeze approach:")
model.freeze()
print("   Model frozen")

# Check what's trainable now
trainable_after_freeze = tree_flatten(model.trainable_parameters(), destination={})
print(f"   Trainable after freeze: {len(trainable_after_freeze)}")

# Try to unfreeze LoRA params
print("\n6. Attempting to unfreeze LoRA parameters...")
unfrozen_count = 0
for name, module in model.named_modules():
    if 'lora' in name.lower():
        try:
            module.unfreeze()
            unfrozen_count += 1
            if unfrozen_count <= 3:
                print(f"   Unfroze: {name}")
        except Exception as e:
            print(f"   Failed to unfreeze {name}: {e}")

print(f"\n   Total modules unfrozen: {unfrozen_count}")

# Check trainable params after unfreeze
trainable_after_unfreeze = tree_flatten(model.trainable_parameters(), destination={})
print(f"   Trainable after unfreeze: {len(trainable_after_unfreeze)}")
if trainable_after_unfreeze:
    print(f"   Sample keys: {list(trainable_after_unfreeze.keys())[:5]}")
    lora_params_final = [k for k in trainable_after_unfreeze.keys() if 'lora' in k.lower()]
    print(f"   LoRA parameters: {len(lora_params_final)}")
    if lora_params_final:
        print(f"   Sample LoRA keys: {lora_params_final[:5]}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
