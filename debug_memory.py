
import mlx.core as mx
from mlx_lm import load
import psutil
import os
import sys
import time
from mlx.utils import tree_flatten

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def analyze_model():
    print(f"Initial memory: {get_memory_usage():.2f} MB")
    
    model_path = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-0.5B-Instruct"
    # Try to find a local path from the logs or config if possible, but for now let's try to load a small one or mock it
    # User mentioned "Qwen2.5-0.5B-Instruct"
    
    print(f"Loading model: {model_path}")
    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        print("Trying to find a local model path...")
        # Look for a model in the user's likely directories
        possible_paths = [
            "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/models/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct"
        ]
        for p in possible_paths:
            try:
                print(f"Trying {p}...")
                model, tokenizer = load(p)
                break
            except:
                continue
        else:
            print("Could not load model. Exiting.")
            return

    print(f"Memory after load: {get_memory_usage():.2f} MB")
    
    # Count parameters
    total_params = sum(v.size for v in tree_flatten(model.parameters(), destination={}).values())
    print(f"Total parameters: {total_params:,}")
    
    trainable_params = model.trainable_parameters()
    trainable_count = sum(v.size for v in tree_flatten(trainable_params, destination={}).values())
    print(f"Trainable parameters (default): {trainable_count:,}")
    
    # Check if everything is trainable
    print(f"Is everything trainable? {total_params == trainable_count}")
    
    # Simulate optimizer state size
    # If we use Adam, we have 2 states per trainable param
    optimizer_memory = trainable_count * 4 * 2 / 1024 / 1024 # MB (assuming float32)
    print(f"Estimated Adam state size: {optimizer_memory:.2f} MB")
    
    # Check for large attributes
    print("Checking for large attributes in model...")
    for k, v in model.__dict__.items():
        try:
            if hasattr(v, 'nbytes'):
                print(f"  {k}: {v.nbytes / 1024 / 1024:.2f} MB")
        except:
            pass

if __name__ == "__main__":
    analyze_model()
