"""
Data Loader for On-Policy Distillation

Handles loading and processing validation prompts for distillation training.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries (one per line)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    logger.info(f"Loaded {len(data)} items from {file_path}")
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """
    Save data to JSONL file.

    Args:
        data: List of dictionaries
        file_path: Path where file will be saved
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(data)} items to {file_path}")


def load_prompts(
    prompts_path: str,
    max_prompts: Optional[int] = None,
    split_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Load prompts from JSONL file and split into train/val.

    Expected JSONL format:
        {"prompt": "What is the capital of France?"}
        {"prompt": "Explain quantum computing."}

    Or with additional fields:
        {"prompt": "...", "task": "qa", "difficulty": "easy"}

    Args:
        prompts_path: Path to JSONL file containing prompts
        max_prompts: Maximum number of prompts to load (None = all)
        split_ratio: Train/val split ratio (0.8 = 80% train, 20% val)
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility

    Returns:
        (train_prompts, val_prompts) - Lists of prompt strings
    """
    # Load data
    data = load_jsonl(prompts_path)

    if len(data) == 0:
        raise ValueError(f"No data loaded from {prompts_path}")

    # Extract prompts
    prompts = []
    for item in data:
        if 'prompt' in item:
            prompts.append(item['prompt'])
        elif 'text' in item:
            # Alternative field name
            prompts.append(item['text'])
        else:
            logger.warning(f"Item missing 'prompt' or 'text' field: {item}")

    logger.info(f"Extracted {len(prompts)} prompts from {len(data)} items")

    # Limit to max_prompts if specified
    if max_prompts is not None and max_prompts < len(prompts):
        if shuffle:
            random.seed(seed)
            random.shuffle(prompts)
        prompts = prompts[:max_prompts]
        logger.info(f"Limited to {max_prompts} prompts")

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(prompts)

    # Split into train/val
    split_idx = int(len(prompts) * split_ratio)
    train_prompts = prompts[:split_idx]
    val_prompts = prompts[split_idx:]

    logger.info(f"Split: {len(train_prompts)} train, {len(val_prompts)} val")

    return train_prompts, val_prompts


def create_batches(
    prompts: List[str],
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> List[List[str]]:
    """
    Create batches from list of prompts.

    Args:
        prompts: List of prompt strings
        batch_size: Number of prompts per batch
        shuffle: Whether to shuffle before batching
        seed: Random seed (optional)

    Returns:
        List of batches (each batch is a list of prompts)
    """
    prompts_copy = prompts.copy()

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(prompts_copy)

    # Create batches
    batches = []
    for i in range(0, len(prompts_copy), batch_size):
        batch = prompts_copy[i:i + batch_size]
        batches.append(batch)

    logger.debug(f"Created {len(batches)} batches from {len(prompts)} prompts")

    return batches


class PromptDataset:
    """
    Dataset class for managing prompts during distillation.
    """

    def __init__(
        self,
        prompts_path: str,
        max_prompts: Optional[int] = None,
        split_ratio: float = 0.8,
        batch_size: int = 4,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize prompt dataset.

        Args:
            prompts_path: Path to JSONL file
            max_prompts: Maximum prompts to load
            split_ratio: Train/val split ratio
            batch_size: Batch size for training
            shuffle: Whether to shuffle
            seed: Random seed
        """
        self.prompts_path = prompts_path
        self.max_prompts = max_prompts
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Load and split data
        self.train_prompts, self.val_prompts = load_prompts(
            prompts_path,
            max_prompts=max_prompts,
            split_ratio=split_ratio,
            shuffle=shuffle,
            seed=seed
        )

        # Create batches
        self.train_batches = create_batches(
            self.train_prompts,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed
        )

        self.val_batches = create_batches(
            self.val_prompts,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            seed=seed
        )

        logger.info(f"PromptDataset initialized:")
        logger.info(f"  Train: {len(self.train_prompts)} prompts, {len(self.train_batches)} batches")
        logger.info(f"  Val: {len(self.val_prompts)} prompts, {len(self.val_batches)} batches")

    def get_train_batches(self, shuffle: bool = True) -> List[List[str]]:
        """Get training batches, optionally reshuffled"""
        if shuffle:
            return create_batches(
                self.train_prompts,
                self.batch_size,
                shuffle=True,
                seed=None  # Use different shuffle each epoch
            )
        return self.train_batches

    def get_val_batches(self) -> List[List[str]]:
        """Get validation batches (always in same order)"""
        return self.val_batches

    def get_random_train_batch(self) -> List[str]:
        """Get a random training batch"""
        return random.choice(self.train_batches)

    def get_random_val_batch(self) -> List[str]:
        """Get a random validation batch"""
        return random.choice(self.val_batches)

    def __len__(self) -> int:
        """Total number of prompts"""
        return len(self.train_prompts) + len(self.val_prompts)

    def __repr__(self) -> str:
        return (
            f"PromptDataset(train={len(self.train_prompts)}, "
            f"val={len(self.val_prompts)}, batch_size={self.batch_size})"
        )


# Utility functions for creating sample datasets

def create_sample_prompts_file(output_path: str, num_prompts: int = 100):
    """
    Create a sample prompts JSONL file for testing.

    Args:
        output_path: Where to save the file
        num_prompts: Number of sample prompts to generate
    """
    sample_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to reverse a string.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis.",
        "How does a neural network work?",
        "What is the difference between AI and ML?",
        "Explain the theory of relativity.",
        "Write a haiku about technology.",
        "What is the meaning of life?",
    ]

    # Repeat and shuffle to get desired number
    data = []
    for i in range(num_prompts):
        prompt = sample_prompts[i % len(sample_prompts)]
        # Add variation
        if i >= len(sample_prompts):
            prompt = f"{prompt} (variation {i // len(sample_prompts)})"

        data.append({"prompt": prompt})

    save_jsonl(data, output_path)
    logger.info(f"Created sample prompts file with {num_prompts} prompts at {output_path}")


def validate_prompts_file(file_path: str) -> Dict[str, any]:
    """
    Validate a prompts JSONL file and return statistics.

    Args:
        file_path: Path to JSONL file

    Returns:
        Dictionary with statistics:
            - num_items: Total items in file
            - num_valid_prompts: Items with valid 'prompt' field
            - avg_prompt_length: Average prompt length (characters)
            - min_prompt_length, max_prompt_length
    """
    data = load_jsonl(file_path)

    prompts = []
    for item in data:
        if 'prompt' in item:
            prompts.append(item['prompt'])
        elif 'text' in item:
            prompts.append(item['text'])

    if not prompts:
        raise ValueError("No valid prompts found in file")

    lengths = [len(p) for p in prompts]

    stats = {
        'num_items': len(data),
        'num_valid_prompts': len(prompts),
        'avg_prompt_length': sum(lengths) / len(lengths),
        'min_prompt_length': min(lengths),
        'max_prompt_length': max(lengths)
    }

    logger.info(f"Validation results for {file_path}:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    return stats
