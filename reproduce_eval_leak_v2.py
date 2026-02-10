
import mlx.core as mx
import time
import psutil
import os
import numpy as np

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def test_eval_dict():
    print("Testing mx.eval with dict and LARGE arrays...")
    
    # Create a large array (10MB)
    shape = (1000, 2500)  # 2.5M floats * 4 bytes = 10MB
    a = mx.random.uniform(shape=shape)
    mx.eval(a)
    
    state = {'a': a}
    
    initial_mem = get_memory_usage()
    print(f"Initial memory: {initial_mem:.2f} MB")
    
    # We want to see if memory grows.
    # If we don't eval, the graph grows.
    # But more importantly, if we don't eval, the intermediate arrays might be kept alive if the graph holds them?
    # In a loop: a_new = a_old + 1.
    # a_new depends on a_old.
    # If we don't eval a_new, we have a chain.
    # If we keep a_new (as state['a']), we keep the head of the chain.
    # The chain keeps all previous a's alive?
    # Yes, because a_new needs a_old to be computed.
    
    for i in range(1000):
        # Update: a = a + 1
        # Use a computation that allocates memory if evaluated
        new_a = state['a'] + 1.0
        updates = {'a': new_a}
        
        # Try to eval the dict
        mx.eval(updates)
        
        # Update state
        state['a'] = updates['a']
        
        if i % 100 == 0:
            print(f"Step {i}: {get_memory_usage():.2f} MB")
            
    final_mem = get_memory_usage()
    print(f"Final memory: {final_mem:.2f} MB")
    
    if final_mem - initial_mem > 100:
        print("LEAK DETECTED! Memory grew significantly.")
    else:
        print("No leak detected.")

if __name__ == "__main__":
    test_eval_dict()
