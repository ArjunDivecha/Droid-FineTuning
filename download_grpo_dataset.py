#!/usr/bin/env python3
"""
Download and convert awesome-chatgpt-prompts dataset to GRPO format

This dataset has 203 diverse prompts for different roles/scenarios.
Format: {"prompt": "...", "answer": "...", "system": "..."}
"""

import json
from datasets import load_dataset
from pathlib import Path

def download_and_convert():
    print("=" * 70)
    print("Downloading awesome-chatgpt-prompts dataset from HuggingFace")
    print("=" * 70)

    # Download dataset
    print("\n📥 Downloading dataset...")
    dataset = load_dataset("fka/awesome-chatgpt-prompts")

    print(f"✅ Downloaded {len(dataset['train'])} examples")

    # Create output directory
    output_dir = Path("grpo_chatgpt_prompts")
    output_dir.mkdir(exist_ok=True)

    # Split into train (80%) and validation (20%)
    train_size = int(len(dataset['train']) * 0.8)

    train_examples = []
    val_examples = []

    print("\n🔄 Converting to GRPO format...")
    for idx, example in enumerate(dataset['train']):
        # Convert to GRPO format
        grpo_example = {
            "prompt": example['prompt'],
            "answer": "",  # No ground truth answer - GRPO will generate and rank
            "system": f"You are a {example['act']}. Respond appropriately to the user's request."
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
    print(f"System: {sample['system'][:80]}...")
    print(f"Prompt: {sample['prompt'][:80]}...")
    print("-" * 70)

    print("\n✨ Ready for GRPO training!")
    print(f"   Use data path: {output_dir.absolute()}")

if __name__ == "__main__":
    download_and_convert()