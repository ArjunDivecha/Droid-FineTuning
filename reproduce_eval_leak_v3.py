
import mlx.core as mx
import time
import psutil
import os
import numpy as np

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def test_eval_nested_dict():
    print("Testing mx.eval with NESTED dict...")
    
    # Create a large array
    shape = (1000, 2500)
    a = mx.random.uniform(shape=shape)
    mx.eval(a)
    
    # Nested state
    state = {'level1': {'level2': {'a': a}}}
    
    initial_mem = get_memory_usage()
    print(f"Initial memory: {initial_mem:.2f} MB")
    
    for i in range(1000):
        # Update: a = a + 1
        old_a = state['level1']['level2']['a']
        new_a = old_a + 1.0
        
        updates = {'level1': {'level2': {'a': new_a}}}
        
        # Try to eval the nested dict
        mx.eval(updates)
        
        # Update state
        state['level1']['level2']['a'] = updates['level1']['level2']['a']
        
        if i % 100 == 0:
            print(f"Step {i}: {get_memory_usage():.2f} MB")
            
    final_mem = get_memory_usage()
    print(f"Final memory: {final_mem:.2f} MB")
    
    if final_mem - initial_mem > 50:
        print("LEAK DETECTED! mx.eval(nested_dict) does not evaluate deep values.")
    else:
        print("No leak detected with nested dict.")

if __name__ == "__main__":
    test_eval_nested_dict()
