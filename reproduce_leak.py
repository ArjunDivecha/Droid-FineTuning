
import os
import sys
import time
import psutil
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import TrainingArgs, train, default_loss, iterate_batches, grad_checkpoint
from mlx_lm.tuner.datasets import CacheDataset

# Monitor memory
process = psutil.Process(os.getpid())

def get_memory():
    # Returns memory in GB
    return process.memory_info().rss / 1024 / 1024 / 1024

print(f"Initial Memory: {get_memory():.2f} GB")

# 1. Load Model
model_path = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-0.5B-Instruct"
print(f"Loading model {model_path}...")
model, tokenizer = load(model_path)
print(f"Model loaded. Memory: {get_memory():.2f} GB")

# 2. Prepare Dummy Data
# Create dummy tokens
vocab_size = tokenizer.vocab_size
seq_len = 2048
num_examples = 100

print("Creating dummy dataset...")
# Simple list of lists
train_data = [np.random.randint(0, vocab_size, size=(seq_len,)).tolist() for _ in range(num_examples)]
valid_data = [np.random.randint(0, vocab_size, size=(seq_len,)).tolist() for _ in range(10)]

# Wrap in a dummy class that looks like TextDataset but pre-tokenized
class DummyDataset:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return (self.data[idx], 0) # (tokens, offset)
    def __len__(self):
        return len(self.data)
    def process(self, d):
        return d # Already processed

train_set = DummyDataset(train_data)
valid_set = DummyDataset(valid_data)

# 3. Setup LoRA
print("Setting up LoRA...")
model.freeze()
linear_to_lora_layers(model, 16, {"rank": 8, "scale": 20.0, "dropout": 0.0})
print(f"LoRA setup done. Memory: {get_memory():.2f} GB")

# 4. Setup Training Args
args = TrainingArgs(
    batch_size=1,
    iters=100,
    val_batches=5,
    steps_per_report=10,
    steps_per_eval=20,
    adapter_file="test_adapters.safetensors",
    grad_checkpoint=True, # ENABLED as per issue
    max_seq_length=seq_len
)

# 5. Optimizer
optimizer = optim.AdamW(learning_rate=1e-5)

# 6. Custom Train Loop (Inline to debug)
# We will use the library's train function first to see if it leaks
print("Starting training loop...")

# Monkey patch print to see memory
original_print = print
def memory_print(*args, **kwargs):
    mem = get_memory()
    original_print(f"[Mem: {mem:.2f} GB]", *args, **kwargs)

import builtins
builtins.print = memory_print

try:
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=args
    )
except Exception as e:
    builtins.print = original_print
    print(f"Error: {e}")
    raise

builtins.print = original_print
print(f"Final Memory: {get_memory():.2f} GB")
