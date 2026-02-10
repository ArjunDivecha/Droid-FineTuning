
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import time
import psutil
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from nested_learning.nested_optimizer import NestedAdam

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)

def verify_fix():
    print("Verifying fix with NestedAdam...")
    
    model = SimpleModel()
    mx.eval(model.parameters())
    
    # Setup optimizer
    # Tier 0 updates every step, Tier 1 every 2 steps
    optimizer = NestedAdam(
        learning_rate=0.001,
        tier_update_frequencies=[1, 2],
        parameter_tier_map={
            'layer1.weight': 0, 'layer1.bias': 0,
            'layer2.weight': 1, 'layer2.bias': 1
        }
    )
    
    # Fake LoRA params for the filter
    # We need to monkey-patch the filter in nested_optimizer or rename params
    # Let's just rename params in the map and model for this test
    # Actually, simpler to just mock the optimizer's apply_gradients to NOT filter
    # But we want to test the actual code.
    # So let's rename the model layers to look like LoRA
    
    class LoraModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_a = nn.Linear(1000, 1000)
            self.lora_b = nn.Linear(1000, 1000)
            
    model = LoraModel()
    mx.eval(model.parameters())
    
    optimizer = NestedAdam(
        learning_rate=0.001,
        tier_update_frequencies=[1, 1],
        parameter_tier_map={
            'lora_a.weight': 0, 'lora_a.bias': 0,
            'lora_b.weight': 0, 'lora_b.bias': 0
        }
    )
    
    initial_mem = get_memory_usage()
    print(f"Initial memory: {initial_mem:.2f} MB")
    
    for i in range(500):
        # Create dummy gradients
        grads = tree_flatten(model.parameters(), destination={})
        # Make them non-zero
        grads = {k: mx.random.uniform(shape=v.shape) for k, v in grads.items()}
        
        # Apply gradients
        optimizer.apply_gradients(grads, model)
        
        if i % 50 == 0:
            print(f"Step {i}: {get_memory_usage():.2f} MB")
            
    final_mem = get_memory_usage()
    print(f"Final memory: {final_mem:.2f} MB")
    
    if final_mem - initial_mem > 50:
        print("FAIL: Memory leak detected in optimizer loop.")
    else:
        print("SUCCESS: No memory leak detected in optimizer loop.")

if __name__ == "__main__":
    verify_fix()
