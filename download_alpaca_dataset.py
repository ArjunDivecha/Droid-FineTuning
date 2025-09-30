#!/usr/bin/env python3
"""
Download and convert Alpaca-cleaned dataset to GRPO format

The Alpaca dataset has 52K instruction-following examples with:
- instruction: Task description
- input: Optional context
- output: Expected completion

Format for GRPO: {"prompt": "...", "answer": "...", "system": "..."}
"""

import json
from datasets import load_dataset
from pathlib import Path

def download_and_convert(max_samples=1000):
    print("=" * 70)
    print(f"Downloading Alpaca-cleaned dataset (limiting to {max_samples} examples)")
    print("=" * 70)

    # Download dataset
    print("\n📥 Downloading dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned")

    print(f"✅ Downloaded {len(dataset['train'])} total examples")
    print(f"   Using first {max_samples} for faster testing")

    # Create output directory
    output_dir = Path("grpo_alpaca_data")
    output_dir.mkdir(exist_ok=True)

    # Take first N samples and split into train (80%) and validation (20%)
    samples = list(dataset['train'].select(range(min(max_samples, len(dataset['train'])))))
    train_size = int(len(samples) * 0.8)

    train_examples = []
    val_examples = []

    print("\n🔄 Converting to GRPO format...")
    for idx, example in enumerate(samples):
        # Combine instruction and input (if present) into prompt
        instruction = example['instruction']
        input_text = example.get('input', '')

        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}"
        else:
            prompt = instruction

        # Convert to GRPO format
        grpo_example = {
            "prompt": prompt,
            "answer": example['output'],
            "system": "You are a helpful AI assistant. Follow the user's instructions carefully."
        }

        if idx < train_size:
            train_examples.append(grpo_example)
        else:
            val_examples.append(grpo_example)

    # Write training data
    train_file = output_dir / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Write validation data
    val_file = output_dir / "valid.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"\n✅ Conversion complete!")
    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")
    print(f"\n📁 Files saved to:")
    print(f"   {train_file}")
    print(f"   {val_file}")

    # Show sample
    print(f"\n📝 Sample data:")
    print("-" * 70)
    sample = train_examples[0]
    print(f"System: {sample['system']}")
    print(f"Prompt: {sample['prompt'][:100]}...")
    print(f"Answer: {sample['answer'][:100]}...")
    print("-" * 70)

    print("\n✨ Ready for GRPO training!")
    print(f"   Data path: {output_dir.absolute()}")
    print(f"\n💡 Tip: Start with 50-100 iterations for testing")

if __name__ == "__main__":
    # Download 1000 examples (you can increase this)
    download_and_convert(max_samples=1000)