
import mlx.core as mx
import time
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def test_eval_dict():
    print("Testing mx.eval with dict...")
    
    # Create a simple graph loop
    a = mx.array([1.0])
    state = {'a': a}
    
    initial_mem = get_memory_usage()
    print(f"Initial memory: {initial_mem:.2f} MB")
    
    for i in range(10000):
        # Update: a = a + 1
        new_a = state['a'] + 1.0
        updates = {'a': new_a}
        
        # Try to eval the dict
        mx.eval(updates)
        
        # Update state
        state['a'] = updates['a']
        
        if i % 1000 == 0:
            print(f"Step {i}: {get_memory_usage():.2f} MB")
            
    final_mem = get_memory_usage()
    print(f"Final memory: {final_mem:.2f} MB")
    
    if final_mem - initial_mem > 10:
        print("LEAK DETECTED! mx.eval(dict) does not evaluate values.")
    else:
        print("No leak detected. mx.eval(dict) works.")

if __name__ == "__main__":
    test_eval_dict()
