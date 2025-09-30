#!/usr/bin/env python3
"""
Convert SFT training data (messages format) to GRPO format (prompt/answer/system)

Usage:
    python3 convert_to_grpo.py

This script converts your existing writing_dataset_train.jsonl and writing_dataset_valid.jsonl
from messages format to the GRPO format required by mlx-lm-lora.

Input format: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
Output format: {"prompt": "...", "answer": "...", "system": "..."}
"""

import json
import os
from pathlib import Path

# Source and destination paths
SOURCE_DIR = Path("/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/Writing")
DEST_DIR = Path("/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/grpo_training_data")

# File mappings
FILES_TO_CONVERT = [
    ("writing_dataset_train.jsonl", "train.jsonl"),
    ("writing_dataset_valid.jsonl", "valid.jsonl"),
]

def convert_messages_to_grpo(messages):
    """
    Convert messages format to GRPO format.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Dict with 'prompt', 'answer', and 'system' keys
    """
    system = "You are Arjun Divecha, an expert in emerging markets and investment strategy. Write insightful, analytical articles based on your expertise."
    prompt = ""
    answer = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system = content
        elif role == "user":
            prompt = content
        elif role == "assistant":
            answer = content

    return {
        "prompt": prompt,
        "answer": answer,
        "system": system
    }

def convert_file(input_path, output_path):
    """Convert a single JSONL file from messages to GRPO format."""
    converted_count = 0
    skipped_count = 0

    print(f"\nConverting: {input_path.name}")
    print(f"  → {output_path}")

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                messages = data.get("messages", [])

                if not messages:
                    print(f"  Warning: Line {line_num} has no messages, skipping")
                    skipped_count += 1
                    continue

                grpo_data = convert_messages_to_grpo(messages)

                # Validate that we have required fields
                if not grpo_data["prompt"] or not grpo_data["answer"]:
                    print(f"  Warning: Line {line_num} missing prompt or answer, skipping")
                    skipped_count += 1
                    continue

                f_out.write(json.dumps(grpo_data, ensure_ascii=False) + '\n')
                converted_count += 1

            except json.JSONDecodeError as e:
                print(f"  Error: Line {line_num} is not valid JSON: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"  Error: Line {line_num} failed to convert: {e}")
                skipped_count += 1

    print(f"  ✅ Converted: {converted_count} examples")
    if skipped_count > 0:
        print(f"  ⚠️  Skipped: {skipped_count} examples")

    return converted_count, skipped_count

def main():
    """Main conversion function."""
    print("=" * 70)
    print("Converting SFT Data (messages format) → GRPO Format")
    print("=" * 70)

    # Create destination directory
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {DEST_DIR}")

    total_converted = 0
    total_skipped = 0

    # Convert each file
    for input_filename, output_filename in FILES_TO_CONVERT:
        input_path = SOURCE_DIR / input_filename
        output_path = DEST_DIR / output_filename

        if not input_path.exists():
            print(f"\n⚠️  Warning: {input_path} not found, skipping")
            continue

        converted, skipped = convert_file(input_path, output_path)
        total_converted += converted
        total_skipped += skipped

    # Summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Total converted: {total_converted} examples")
    if total_skipped > 0:
        print(f"Total skipped:   {total_skipped} examples")

    print(f"\n📁 Your GRPO training data is ready at:")
    print(f"   {DEST_DIR}/")
    print(f"   ├── train.jsonl")
    print(f"   └── valid.jsonl")

    print("\n✨ Next steps:")
    print("   1. Review the converted data (spot check a few examples)")
    print("   2. Use Enhanced Setup page to validate data format")
    print("   3. Start GRPO/GSPO/Dr. GRPO training!")

    # Show a sample
    sample_path = DEST_DIR / "train.jsonl"
    if sample_path.exists():
        print("\n📝 Sample converted data (first example):")
        print("-" * 70)
        with open(sample_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            print(f"Prompt: {sample['prompt'][:100]}...")
            print(f"Answer: {sample['answer'][:100]}...")
            print(f"System: {sample['system'][:100]}...")
        print("-" * 70)

if __name__ == "__main__":
    main()