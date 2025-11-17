#!/usr/bin/env python3
"""
Test the fix: use copy.deepcopy + manual assignment instead of tree_map_with_path
"""

import mlx.core as mx
from mlx_lm import load
from mlx.utils import tree_flatten
import copy

BASE_MODEL = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct"
ADAPTER = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints/big/checkpoints/best"

print("=" * 60)
print("Testing Fixed Update Approach")
print("=" * 60)

# Load model
print("\n1. Loading model...")
model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER)
model.train()

# Get trainable parameters
trainable = model.trainable_parameters()
flat = tree_flatten(trainable, destination={})

# Create fake "updates" for all LoRA parameters
print("\n2. Creating fake updates (adding 1.0 to all LoRA params)...")
updates = {}
lora_keys = [k for k in flat.keys() if 'lora_a' in k.lower() or 'lora_b' in k.lower()]
print(f"   Found {len(lora_keys)} LoRA parameters")

for key in lora_keys[:5]:  # Test with first 5 to save time
    updates[key] = flat[key] + 1.0

print(f"   Created {len(updates)} updates")

# OLD BROKEN APPROACH: tree_map_with_path
print("\n3. Testing OLD approach (tree_map_with_path)...")
from mlx.utils import tree_map_with_path
old_approach = tree_map_with_path(
    lambda path, value: updates.get('.'.join(path), value),
    trainable
)
mx.eval(old_approach)
model.update(old_approach)

# Verify
test_key = list(updates.keys())[0]
after_old = tree_flatten(model.trainable_parameters(), destination={})[test_key]
mx.eval(after_old)
original_val = flat[test_key]
expected_val = updates[test_key]
print(f"   Test key: {test_key}")
print(f"   Original mean: {float(mx.mean(original_val)):.8f}")
print(f"   Expected mean: {float(mx.mean(expected_val)):.8f}")
print(f"   Actual mean: {float(mx.mean(after_old)):.8f}")
print(f"   OLD APPROACH WORKED: {float(mx.max(mx.abs(after_old - expected_val))) < 1e-6}")

# Reset model
model.update(trainable)
mx.eval(model.parameters())

# NEW FIXED APPROACH: deepcopy + manual assignment
print("\n4. Testing NEW approach (deepcopy + manual assignment)...")

def apply_flat_updates_to_nested(nested_params, flat_updates):
    """
    Apply flat updates to nested parameter structure.

    Args:
        nested_params: Nested dict/list structure from trainable_parameters()
        flat_updates: Flat dict mapping 'path.to.param' -> updated_value

    Returns:
        Updated nested structure
    """
    # Deep copy to avoid modifying original
    updated = copy.deepcopy(nested_params)

    # Apply each update by navigating the nested structure
    for param_path, new_value in flat_updates.items():
        parts = param_path.split('.')

        # Navigate to the parent container
        current = updated
        for part in parts[:-1]:
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, list):
                current = current[int(part)]
            else:
                raise ValueError(f"Cannot navigate to {param_path}: unexpected type {type(current)}")

        # Set the final value
        final_key = parts[-1]
        if isinstance(current, dict):
            current[final_key] = new_value
        elif isinstance(current, list):
            current[int(final_key)] = new_value
        else:
            raise ValueError(f"Cannot set {param_path}: parent is {type(current)}")

    return updated

# Apply updates using new approach
new_approach = apply_flat_updates_to_nested(trainable, updates)
mx.eval(new_approach)
model.update(new_approach)

# Verify
after_new = tree_flatten(model.trainable_parameters(), destination={})[test_key]
mx.eval(after_new)
print(f"   Test key: {test_key}")
print(f"   Original mean: {float(mx.mean(original_val)):.8f}")
print(f"   Expected mean: {float(mx.mean(expected_val)):.8f}")
print(f"   Actual mean: {float(mx.mean(after_new)):.8f}")
print(f"   NEW APPROACH WORKED: {float(mx.max(mx.abs(after_new - expected_val))) < 1e-6}")

# Verify ALL updates worked
print("\n5. Verifying all updates...")
all_correct = True
for key in updates.keys():
    after_val = tree_flatten(model.trainable_parameters(), destination={})[key]
    mx.eval(after_val)
    expected = updates[key]
    matches = float(mx.max(mx.abs(after_val - expected))) < 1e-6
    if not matches:
        print(f"   FAILED: {key}")
        all_correct = False

if all_correct:
    print(f"   ✓ All {len(updates)} updates verified successfully!")
else:
    print(f"   ✗ Some updates failed")

print("\n" + "=" * 60)
print("FIX VERIFIED!" if all_correct else "FIX FAILED!")
print("=" * 60)
