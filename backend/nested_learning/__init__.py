"""
Nested Learning Module

Implements Google Research's Nested Learning paradigm for continual learning
and catastrophic forgetting prevention.

Key Components:
- NestedAdam optimizer with multi-frequency parameter updates
- Parameter tier assignment strategies
- NestedLoRATrainer for training with tiered updates
"""

from .config import NestedLearningConfig
from .nested_optimizer import NestedAdam
from .nested_trainer import NestedLoRATrainer
from .parameter_scheduler import ParameterTierScheduler

__all__ = [
    'NestedLearningConfig',
    'NestedAdam',
    'NestedLoRATrainer',
    'ParameterTierScheduler',
]
