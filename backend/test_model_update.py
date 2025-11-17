#!/usr/bin/env python3
"""
Test to verify the model.update() issue with LoRA adapters
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx.utils import tree_flatten, tree_map_with_path

# Test paths
BASE_MODEL = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct"
ADAPTER = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints/big/checkpoints/best"

print("=" * 60)
print("Testing model.update() with LoRA Adapters")
print("=" * 60)

# Load model
print("\n1. Loading model...")
model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER)
model.train()

# Get trainable parameters
print("\n2. Getting trainable parameters...")
trainable = model.trainable_parameters()
flat = tree_flatten(trainable, destination={})
print(f"   Trainable params: {len(flat)}")

# Find a LoRA parameter to test with
lora_keys = [k for k in flat.keys() if 'lora_a' in k.lower()]
test_key = lora_keys[0] if lora_keys else list(flat.keys())[0]
print(f"   Testing with: {test_key}")

# Get original value
original_param = flat[test_key]
print(f"   Original mean: {float(mx.mean(original_param)):.8f}")
print(f"   Original shape: {original_param.shape}")

# Create modified value (add 1.0 to all elements)
modified_param = original_param + 1.0
mx.eval(modified_param)
print(f"   Modified mean: {float(mx.mean(modified_param)):.8f}")
print(f"   Difference: {float(mx.mean(mx.abs(modified_param - original_param))):.8f}")

# Approach 1: Using tree_map_with_path (current approach)
print("\n3. Test Approach 1: tree_map_with_path")
updates = {test_key: modified_param}
updated_params_1 = tree_map_with_path(
    lambda path, value: updates.get('.'.join(path), value),
    trainable
)
mx.eval(updated_params_1)
model.update(updated_params_1)

# Verify
after_1 = tree_flatten(model.trainable_parameters(), destination={})[test_key]
mx.eval(after_1)
print(f"   After model.update() mean: {float(mx.mean(after_1)):.8f}")
print(f"   Update successful: {float(mx.max(mx.abs(after_1 - modified_param))) < 1e-6}")
print(f"   Still original: {float(mx.max(mx.abs(after_1 - original_param))) < 1e-6}")

# Reset to original (for fair comparison of approach 2)
model.update(trainable)
mx.eval(model.parameters())

# Approach 2: Direct nested dict construction
print("\n4. Test Approach 2: Direct nested dict")
# Split key into path components
path_parts = test_key.split('.')
print(f"   Path parts: {path_parts}")

# Build nested structure manually
nested_update = trainable
# Navigate to the right location in the tree and replace
# This requires understanding the exact nested structure

# Get current value at that path to verify structure
current_value = trainable
for part in path_parts:
    if isinstance(current_value, dict):
        current_value = current_value.get(part)
    elif isinstance(current_value, list):
        current_value = current_value[int(part)]
    else:
        # It's a module - get attribute
        current_value = getattr(current_value, part, None)
        if current_value is None:
            break

print(f"   Can navigate to parameter: {current_value is not None}")

# Approach 3: Using model.parameters() instead of trainable_parameters()
print("\n5. Test Approach 3: Using model.parameters()")
all_params = model.parameters()
flat_all = tree_flatten(all_params, destination={})
print(f"   Total params: {len(flat_all)}")
print(f"   Test key exists: {test_key in flat_all}")

# Try updating via model.parameters()
updates_3 = {test_key: modified_param}
updated_all = tree_map_with_path(
    lambda path, value: updates_3.get('.'.join(path), value),
    all_params
)
mx.eval(updated_all)
model.update(updated_all)

# Verify
after_3 = tree_flatten(model.trainable_parameters(), destination={})[test_key]
mx.eval(after_3)
print(f"   After model.update() mean: {float(mx.mean(after_3)):.8f}")
print(f"   Update successful: {float(mx.max(mx.abs(after_3 - modified_param))) < 1e-6}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
