#!/usr/bin/env python3
"""
End-to-end test of the fixed NestedAdam optimizer
"""

import sys
import json
import mlx.core as mx
from pathlib import Path
from mlx_lm import load
from mlx.utils import tree_flatten

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from nested_learning.nested_optimizer import NestedAdam

# Test paths
BASE_MODEL = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct"
ADAPTER = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints/big/checkpoints/best"

print("=" * 60)
print("Testing Fixed NestedAdam Optimizer")
print("=" * 60)

# 1. Load model
print("\n1. Loading model with adapter...")
model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER)
model.train()

trainable = model.trainable_parameters()
flat = tree_flatten(trainable, destination={})
lora_keys = [k for k in flat.keys() if 'lora_a' in k.lower() or 'lora_b' in k.lower()]
print(f"   Trainable params: {len(flat)}")
print(f"   LoRA params: {len(lora_keys)}")

# 2. Setup optimizer with simple tier configuration
print("\n2. Setting up NestedAdam optimizer...")
# Simple: 2 tiers, first 50% of LoRA params in tier 0 (every step), rest in tier 1 (every 2 steps)
split_point = len(lora_keys) // 2
parameter_tier_map = {}
for i, key in enumerate(lora_keys):
    parameter_tier_map[key] = 0 if i < split_point else 1

tier_update_frequencies = [1, 2]  # Tier 0 every step, tier 1 every 2 steps

optimizer = NestedAdam(
    learning_rate=1e-4,
    tier_update_frequencies=tier_update_frequencies,
    parameter_tier_map=parameter_tier_map
)

print(f"   Tier 0: {split_point} params (every step)")
print(f"   Tier 1: {len(lora_keys) - split_point} params (every 2 steps)")

# 3. Create fake gradients and run a few optimization steps
print("\n3. Running optimization steps...")

# Store original values for verification
original_values = {}
for key in lora_keys[:5]:  # Track first 5 params
    original_values[key] = flat[key]  # MLX arrays are immutable, no need to copy
    mx.eval(original_values[key])

# Run 3 steps
for step in range(1, 4):
    print(f"\n   Step {step}:")

    # Create fake gradients (small random values)
    gradients = {}
    for key in lora_keys:
        param = tree_flatten(model.trainable_parameters(), destination={})[key]
        # Gradient = random noise with mean 0.01
        grad = mx.random.normal(param.shape) * 0.01
        gradients[key] = grad

    mx.eval(gradients)

    # Apply gradients
    optimizer.apply_gradients(gradients, model)

    # Get tier stats
    stats = optimizer.get_tier_stats()
    print(f"      Tier 0 updates: {stats['tier_parameters']['tier_0']['update_count']}")
    print(f"      Tier 1 updates: {stats['tier_parameters']['tier_1']['update_count']}")

    # Verify parameters actually changed
    current_flat = tree_flatten(model.trainable_parameters(), destination={})
    test_key = lora_keys[0]  # Tier 0 param - should update every step
    current_val = current_flat[test_key]
    mx.eval(current_val)
    original_val = original_values[test_key]

    max_diff = float(mx.max(mx.abs(current_val - original_val)))
    print(f"      Test param ({test_key[:50]}...)")
    print(f"      Original mean: {float(mx.mean(original_val)):.8f}")
    print(f"      Current mean: {float(mx.mean(current_val)):.8f}")
    print(f"      Max diff from original: {max_diff:.8f}")
    print(f"      PARAMETER UPDATED: {max_diff > 1e-8}")

# 4. Final verification
print("\n4. Final verification...")
current_flat = tree_flatten(model.trainable_parameters(), destination={})

all_updated = True
for key in lora_keys[:5]:
    current_val = current_flat[key]
    mx.eval(current_val)
    original_val = original_values[key]

    max_diff = float(mx.max(mx.abs(current_val - original_val)))
    updated = max_diff > 1e-8

    if not updated:
        print(f"   FAILED: {key} not updated (diff={max_diff:.2e})")
        all_updated = False

if all_updated:
    print(f"   ✓ All {len(original_values)} test parameters updated successfully!")
else:
    print(f"   ✗ Some parameters were not updated")

# 5. Verify tier scheduling worked
print("\n5. Verifying tier scheduling...")
stats = optimizer.get_tier_stats()
tier_0_updates = stats['tier_parameters']['tier_0']['update_count']
tier_1_updates = stats['tier_parameters']['tier_1']['update_count']

print(f"   Tier 0 (freq=1): {tier_0_updates} updates (expected 3)")
print(f"   Tier 1 (freq=2): {tier_1_updates} updates (expected 1)")

tier_schedule_correct = (tier_0_updates == 3 and tier_1_updates == 1)
print(f"   Tier scheduling correct: {tier_schedule_correct}")

# 6. Final result
print("\n" + "=" * 60)
if all_updated and tier_schedule_correct:
    print("✓ FIX VERIFIED - OPTIMIZER WORKING CORRECTLY!")
    print("=" * 60)
    print("\nSUMMARY:")
    print("- model.update() now correctly updates LoRA parameters")
    print("- Tier scheduling working as expected")
    print("- Parameters are being modified by optimizer")
    sys.exit(0)
else:
    print("✗ FIX INCOMPLETE - ISSUES REMAIN")
    print("=" * 60)
    if not all_updated:
        print("- Parameters not being updated")
    if not tier_schedule_correct:
        print("- Tier scheduling incorrect")
    sys.exit(1)
