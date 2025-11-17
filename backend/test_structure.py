#!/usr/bin/env python3
"""
Deep dive into the nested structure to understand why model.update() fails
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx.utils import tree_flatten

BASE_MODEL = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct"
ADAPTER = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints/big/checkpoints/best"

print("=" * 60)
print("Investigating PyTree Structure")
print("=" * 60)

# Load model
model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER)
model.train()

# Get trainable parameters
trainable = model.trainable_parameters()
print("\n1. Structure of trainable_parameters():")
print(f"   Type: {type(trainable)}")
print(f"   Keys: {list(trainable.keys()) if isinstance(trainable, dict) else 'not a dict'}")

# If it's a dict, check the 'model' key
if isinstance(trainable, dict) and 'model' in trainable:
    model_params = trainable['model']
    print(f"\n2. Structure of trainable['model']:")
    print(f"   Type: {type(model_params)}")
    if isinstance(model_params, dict):
        print(f"   Keys: {list(model_params.keys())[:5]}...")

    # Check layers
    if isinstance(model_params, dict) and 'layers' in model_params:
        layers = model_params['layers']
        print(f"\n3. Structure of trainable['model']['layers']:")
        print(f"   Type: {type(layers)}")
        if isinstance(layers, list):
            print(f"   Length: {len(layers)}")
            print(f"   First layer type: {type(layers[0])}")
            if isinstance(layers[0], dict):
                print(f"   First layer keys: {list(layers[0].keys())}")
                # Check self_attn
                if 'self_attn' in layers[0]:
                    self_attn = layers[0]['self_attn']
                    print(f"\n4. Structure of layer[0]['self_attn']:")
                    print(f"   Type: {type(self_attn)}")
                    if isinstance(self_attn, dict):
                        print(f"   Keys: {list(self_attn.keys())}")
                        # Check q_proj
                        if 'q_proj' in self_attn:
                            q_proj = self_attn['q_proj']
                            print(f"\n5. Structure of q_proj:")
                            print(f"   Type: {type(q_proj)}")
                            if isinstance(q_proj, dict):
                                print(f"   Keys: {list(q_proj.keys())}")
                                # Check lora_a
                                if 'lora_a' in q_proj:
                                    lora_a = q_proj['lora_a']
                                    print(f"\n6. lora_a value:")
                                    print(f"   Type: {type(lora_a)}")
                                    print(f"   Shape: {lora_a.shape if hasattr(lora_a, 'shape') else 'N/A'}")
                                    print(f"   Mean: {float(mx.mean(lora_a)) if isinstance(lora_a, mx.array) else 'N/A'}")

# Now test if we can manually navigate and modify
print("\n" + "=" * 60)
print("Testing Manual Structure Modification")
print("=" * 60)

# Get original
flat = tree_flatten(trainable, destination={})
test_key = 'model.layers.0.self_attn.q_proj.lora_a'
original = flat[test_key]
print(f"\n1. Original value at {test_key}:")
print(f"   Mean: {float(mx.mean(original)):.8f}")

# Create modified
modified = original + 1.0
mx.eval(modified)
print(f"\n2. Modified value:")
print(f"   Mean: {float(mx.mean(modified)):.8f}")

# Manually construct the nested dict
print("\n3. Manually constructing nested update dict...")
import copy

# Deep copy the structure
updated_tree = copy.deepcopy(trainable)

# Navigate and replace
updated_tree['model']['layers'][0]['self_attn']['q_proj']['lora_a'] = modified

# Verify the modified tree has the new value
verify = tree_flatten(updated_tree, destination={})
print(f"   Modified tree has new value: {float(mx.mean(verify[test_key])):.8f}")

# Now try model.update()
print("\n4. Calling model.update() with manually constructed tree...")
mx.eval(updated_tree)
model.update(updated_tree)

# Check if it worked
after = tree_flatten(model.trainable_parameters(), destination={})[test_key]
mx.eval(after)
print(f"   After model.update() mean: {float(mx.mean(after)):.8f}")
print(f"   Update successful: {float(mx.max(mx.abs(after - modified))) < 1e-6}")

print("\n" + "=" * 60)
print("Complete!")
print("=" * 60)
