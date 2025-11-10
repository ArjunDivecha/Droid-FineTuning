"""
Nested Learning Configuration

Configuration schema for Nested Learning training.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional
from pathlib import Path


@dataclass
class NestedLearningConfig:
    """Configuration for Nested Learning training."""

    # Model paths
    base_model_path: str

    # Data paths
    train_data_path: str
    val_data_path: Optional[str] = None

    # Adapter path (optional)
    adapter_path: str = ""  # Optional - empty string means train from base model

    # Nested Learning parameters
    num_tiers: int = 3
    tier_update_frequencies: List[int] = field(default_factory=lambda: [1, 2, 4])
    tier_assignment_strategy: Literal['layer_depth', 'parameter_importance', 'manual'] = 'layer_depth'

    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 1  # Reduced to prevent memory explosion
    num_steps: int = 1000
    max_seq_length: int = 128  # CRITICAL: Must be 128 or less for nested learning memory management

    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Advanced training settings
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 2
    checkpoint_every: int = 100
    eval_every: int = 100
    max_grad_norm: float = 1.0

    # Output configuration
    output_path: str = './nested_learning/checkpoints'
    experiment_name: str = 'nested_learning_experiment'
    save_best_only: bool = False
    keep_last_n_checkpoints: int = 5

    # Misc
    seed: int = 42
    mixed_precision: bool = True

    def __post_init__(self):
        """Validate configuration."""
        # Ensure num_tiers matches length of tier_update_frequencies
        if len(self.tier_update_frequencies) != self.num_tiers:
            raise ValueError(
                f"Length of tier_update_frequencies ({len(self.tier_update_frequencies)}) "
                f"must match num_tiers ({self.num_tiers})"
            )

        # Ensure frequencies are in ascending order
        if not all(self.tier_update_frequencies[i] <= self.tier_update_frequencies[i + 1]
                   for i in range(len(self.tier_update_frequencies) - 1)):
            raise ValueError(
                "tier_update_frequencies must be in ascending order (e.g., [1, 2, 4])"
            )

        # Validate paths exist (only if they're provided)
        if self.base_model_path and not Path(self.base_model_path).exists():
            raise FileNotFoundError(f"Base model path not found: {self.base_model_path}")

        if self.adapter_path and not Path(self.adapter_path).exists():
            raise FileNotFoundError(f"Adapter path not found: {self.adapter_path}")

        if self.train_data_path and not Path(self.train_data_path).exists():
            raise FileNotFoundError(f"Training data path not found: {self.train_data_path}")

        if self.val_data_path and not Path(self.val_data_path).exists():
            raise FileNotFoundError(f"Validation data path not found: {self.val_data_path}")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'base_model_path': self.base_model_path,
            'adapter_path': self.adapter_path,
            'train_data_path': self.train_data_path,
            'val_data_path': self.val_data_path,
            'num_tiers': self.num_tiers,
            'tier_update_frequencies': self.tier_update_frequencies,
            'tier_assignment_strategy': self.tier_assignment_strategy,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_steps': self.num_steps,
            'max_seq_length': self.max_seq_length,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'warmup_steps': self.warmup_steps,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'checkpoint_every': self.checkpoint_every,
            'eval_every': self.eval_every,
            'max_grad_norm': self.max_grad_norm,
            'output_path': self.output_path,
            'experiment_name': self.experiment_name,
            'save_best_only': self.save_best_only,
            'keep_last_n_checkpoints': self.keep_last_n_checkpoints,
            'seed': self.seed,
            'mixed_precision': self.mixed_precision,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'NestedLearningConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
