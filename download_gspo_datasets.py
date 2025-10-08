#!/usr/bin/env python3
"""
Download and convert datasets for GSPO/GRPO training testing.

This script downloads recommended datasets and converts them to the 
GSPO format: {"prompt": "...", "answer": "...", "system": "..."}

Datasets included:
1. UltraFeedback prompts - General instruction following
2. NuminaMath-CoT - Mathematical reasoning
3. Investopedia datasets - Financial domain knowledge
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path("/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/gspo_datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_to_gspo_format(dataset, prompt_field="prompt", answer_field="answer", 
                          system_message="You are a helpful AI assistant."):
    """
    Convert dataset to GSPO format.
    
    Args:
        dataset: Hugging Face dataset
        prompt_field: Field name containing the prompt
        answer_field: Field name containing the answer
        system_message: Default system message
    
    Returns:
        List of dictionaries in GSPO format
    """
    gspo_data = []
    
    for example in dataset:
        # Extract prompt and answer based on field names
        if prompt_field in example and answer_field in example:
            prompt = example[prompt_field]
            answer = example[answer_field]
        elif 'instruction' in example and 'output' in example:
            prompt = example['instruction']
            answer = example['output']
        elif 'question' in example and 'response' in example:
            prompt = example['question']
            answer = example['response']
        elif 'problem' in example and 'solution' in example:
            prompt = example['problem']
            answer = example['solution']
        else:
            # Skip examples we can't convert
            continue
        
        # Clean up the data
        if isinstance(prompt, str) and isinstance(answer, str):
            prompt = prompt.strip()
            answer = answer.strip()
            
            if prompt and answer:  # Only include non-empty examples
                gspo_example = {
                    "prompt": prompt,
                    "answer": answer,
                    "system": system_message
                }
                gspo_data.append(gspo_example)
    
    return gspo_data

def save_gspo_dataset(data, filename, description):
    """
    Save dataset in GSPO format.
    
    Args:
        data: List of GSPO format examples
        filename: Output filename
        description: Dataset description
    """
    output_path = OUTPUT_DIR / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Saved {len(data)} examples to {output_path}")
    logger.info(f"   Description: {description}")
    
    # Save a small sample for inspection
    sample_path = OUTPUT_DIR / f"sample_{filename}"
    with open(sample_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(data[:5]):  # First 5 examples
            f.write(json.dumps(example, ensure_ascii=False, indent=2) + '\n')
            if i < 4:  # Add separator between examples
                f.write('\n---\n\n')
    
    logger.info(f"   Sample saved to {sample_path}")

def download_ultrafeedback():
    """Download and convert UltraFeedback prompts dataset."""
    logger.info("📥 Downloading UltraFeedback prompts dataset...")
    
    try:
        # Load the UltraFeedback prompts dataset
        dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")
        
        # Convert to GSPO format
        gspo_data = convert_to_gspo_format(
            dataset,
            prompt_field="prompt",
            answer_field="chosen_response",  # Use the best response
            system_message="You are a helpful AI assistant that provides accurate and informative responses."
        )
        
        # Split into train/validation
        train_size = int(len(gspo_data) * 0.9)
        train_data = gspo_data[:train_size]
        val_data = gspo_data[train_size:]
        
        # Save datasets
        save_gspo_dataset(
            train_data, 
            "ultrafeedback_train.jsonl",
            "UltraFeedback prompts - Training set (90%)"
        )
        save_gspo_dataset(
            val_data, 
            "ultrafeedback_valid.jsonl", 
            "UltraFeedback prompts - Validation set (10%)"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download UltraFeedback: {e}")
        return False

def download_numinamath():
    """Download and convert NuminaMath-CoT dataset."""
    logger.info("📥 Downloading NuminaMath-CoT dataset...")
    
    try:
        # Load the NuminaMath-CoT dataset
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
        
        # Convert to GSPO format
        gspo_data = convert_to_gspo_format(
            dataset,
            prompt_field="problem",
            answer_field="solution",
            system_message="You are an expert mathematician. Solve mathematical problems step-by-step with clear reasoning."
        )
        
        # Create a smaller subset for testing (first 1000 examples)
        test_data = gspo_data[:1000]
        remaining_data = gspo_data[1000:10000]  # Next 9000 for larger testing
        
        # Split test data into train/validation
        train_size = int(len(test_data) * 0.9)
        train_data = test_data[:train_size]
        val_data = test_data[train_size:]
        
        # Save datasets
        save_gspo_dataset(
            train_data,
            "numinamath_small_train.jsonl",
            "NuminaMath-CoT - Small training set (900 examples)"
        )
        save_gspo_dataset(
            val_data,
            "numinamath_small_valid.jsonl",
            "NuminaMath-CoT - Small validation set (100 examples)"
        )
        save_gspo_dataset(
            remaining_data,
            "numinamath_extended_train.jsonl",
            "NuminaMath-CoT - Extended training set (9000 examples)"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download NuminaMath: {e}")
        return False

def download_investopedia():
    """Download and convert Investopedia datasets."""
    logger.info("📥 Downloading Investopedia datasets...")
    
    try:
        # Load Investopedia instruction tuning dataset
        dataset = load_dataset("FinLang/investopedia-instruction-tuning-dataset", split="train")
        
        # Convert to GSPO format
        gspo_data = convert_to_gspo_format(
            dataset,
            prompt_field="instruction",
            answer_field="output",
            system_message="You are an expert financial analyst and educator specializing in investment concepts and market analysis."
        )
        
        # Split into train/validation
        train_size = int(len(gspo_data) * 0.9)
        train_data = gspo_data[:train_size]
        val_data = gspo_data[train_size:]
        
        # Save datasets
        save_gspo_dataset(
            train_data,
            "investopedia_train.jsonl",
            "Investopedia financial concepts - Training set (90%)"
        )
        save_gspo_dataset(
            val_data,
            "investopedia_valid.jsonl",
            "Investopedia financial concepts - Validation set (10%)"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download Investopedia dataset: {e}")
        return False

def create_sample_datasets():
    """Create small sample datasets for quick testing."""
    logger.info("📝 Creating sample datasets for quick testing...")
    
    # Sample mathematical reasoning data
    math_samples = [
        {
            "prompt": "What is the derivative of x^2 + 3x + 2?",
            "answer": "The derivative of x^2 + 3x + 2 is 2x + 3. Using the power rule: d/dx(x^n) = nx^(n-1), so d/dx(x^2) = 2x, d/dx(3x) = 3, and d/dx(2) = 0. Therefore, the derivative is 2x + 3.",
            "system": "You are a mathematics tutor providing step-by-step solutions."
        },
        {
            "prompt": "Solve for x: 2x + 5 = 13",
            "answer": "To solve 2x + 5 = 13: 1) Subtract 5 from both sides: 2x = 8. 2) Divide both sides by 2: x = 4. Therefore, x = 4.",
            "system": "You are a mathematics tutor providing step-by-step solutions."
        },
        {
            "prompt": "What is the integral of sin(x) dx?",
            "answer": "The integral of sin(x) dx is -cos(x) + C, where C is the constant of integration. This follows from the fact that the derivative of cos(x) is -sin(x), so the antiderivative of sin(x) is -cos(x).",
            "system": "You are a mathematics tutor providing step-by-step solutions."
        }
    ]
    
    # Sample financial data
    finance_samples = [
        {
            "prompt": "What is the difference between stocks and bonds?",
            "answer": "Stocks represent ownership in a company with potential for capital gains and dividends, while bonds are debt instruments that pay fixed interest. Stocks offer higher potential returns but with greater risk, while bonds provide more stable income with lower risk.",
            "system": "You are an expert investment analyst and educator."
        },
        {
            "prompt": "Explain the concept of diversification in investing.",
            "answer": "Diversification is the practice of spreading investments across various assets, sectors, or geographic regions to reduce risk. By not putting all eggs in one basket, investors can minimize the impact of poor performance in any single investment on their overall portfolio.",
            "system": "You are an expert investment analyst and educator."
        }
    ]
    
    # Save sample datasets
    save_gspo_dataset(
        math_samples,
        "sample_math.jsonl",
        "Sample mathematical reasoning problems (3 examples)"
    )
    save_gspo_dataset(
        finance_samples,
        "sample_finance.jsonl",
        "Sample financial concepts (2 examples)"
    )

def main():
    """Main function to download and convert datasets."""
    logger.info("🚀 Starting dataset download and conversion for GSPO/GRPO testing")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    
    success_count = 0
    total_attempts = 4
    
    # Download datasets
    if download_ultrafeedback():
        success_count += 1
    
    if download_numinamath():
        success_count += 1
        
    if download_investopedia():
        success_count += 1
    
    # Always create sample datasets
    create_sample_datasets()
    success_count += 1
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("📊 DATASET DOWNLOAD SUMMARY")
    logger.info("="*70)
    logger.info(f"✅ Successfully processed: {success_count}/{total_attempts} datasets")
    
    # List all created files
    logger.info("\n📋 Created files:")
    for file_path in sorted(OUTPUT_DIR.glob("*.jsonl")):
        file_size = file_path.stat().st_size
        logger.info(f"   📄 {file_path.name} ({file_size:,} bytes)")
    
    logger.info("\n🎯 Ready for GSPO/GRPO training!")
    logger.info("   Use these datasets to test your GSPO implementation.")
    logger.info("   Start with sample datasets for quick testing.")

if __name__ == "__main__":
    main()
