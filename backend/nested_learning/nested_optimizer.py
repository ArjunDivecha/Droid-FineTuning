"""
Nested Adam Optimizer

Custom Adam optimizer with multi-frequency parameter updates for Nested Learning.
Each parameter tier updates at a different frequency based on the tier schedule.
"""

import mlx.core as mx
import mlx.optimizers as optim
from typing import Dict, List, Any, Optional


class NestedAdam(optim.Adam):
    """
    Adam optimizer with nested learning support.

    Parameters are grouped into tiers, and each tier updates at a different
    frequency. This implements the multi-frequency update schedule from
    Google's Nested Learning research.

    Args:
        learning_rate: Base learning rate
        tier_update_frequencies: List of update frequencies for each tier
            e.g., [1, 2, 4] means tier 0 updates every step, tier 1 every 2 steps, etc.
        parameter_tier_map: Dict mapping parameter names to tier indices
        betas: Adam beta parameters (default: [0.9, 0.999])
        eps: Adam epsilon (default: 1e-8)
    """

    def __init__(
        self,
        learning_rate: float,
        tier_update_frequencies: List[int],
        parameter_tier_map: Dict[str, int],
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-8
    ):
        super().__init__(learning_rate=learning_rate, betas=betas, eps=eps)

        self.tier_update_frequencies = tier_update_frequencies
        self.parameter_tier_map = parameter_tier_map
        self.num_tiers = len(tier_update_frequencies)

        # Track update counts for each tier
        self.tier_update_counts = [0] * self.num_tiers

        # Global step counter
        self.global_step = 0

    def apply_gradients(self, gradients: Dict[str, mx.array], model: Any) -> None:
        """
        Apply gradients with tier-based update scheduling.

        Only updates parameters in tiers that are scheduled to update at this step.

        Args:
            gradients: Dictionary of parameter gradients
            model: Model to update
        """
        self.global_step += 1

        # Determine which tiers should update at this step
        active_tiers = self._get_active_tiers()

        # Filter gradients to:
        # 1. Only LoRA parameters (.lora_a, .lora_b)
        # 2. Only parameters from active tiers
        filtered_gradients = {}
        discarded_gradients = {}

        for param_name, grad in gradients.items():
            # First check: Is this a LoRA parameter?
            is_lora = '.lora_a' in param_name or '.lora_b' in param_name

            if not is_lora:
                # Not LoRA - always discard (don't update base model weights)
                discarded_gradients[param_name] = grad
                continue

            # Second check: Is this LoRA param in an active tier?
            tier_idx = self.parameter_tier_map.get(param_name, 0)  # Default to tier 0
            if tier_idx in active_tiers:
                filtered_gradients[param_name] = grad
            else:
                discarded_gradients[param_name] = grad

        # CRITICAL: Force evaluation of discarded gradients to free computation graphs
        # This is the main memory leak - unevaluated gradients keep their entire
        # computation graph in memory, including activations and intermediate tensors
        if discarded_gradients:
            mx.eval(discarded_gradients)
            del discarded_gradients

        # Force evaluation of filtered gradients before update
        mx.eval(filtered_gradients)

        # Update tier counters for active tiers
        for tier_idx in active_tiers:
            self.tier_update_counts[tier_idx] += 1

        # Apply gradients only for active parameters
        if filtered_gradients:
            super().apply_gradients(filtered_gradients, model)

        # Clear MLX cache after gradient application
        mx.metal.clear_cache()

    def _get_active_tiers(self) -> List[int]:
        """
        Determine which tiers should update at the current step.

        Returns:
            List of tier indices that should update
        """
        active_tiers = []
        for tier_idx, frequency in enumerate(self.tier_update_frequencies):
            # Tier updates if global_step is divisible by its frequency
            if self.global_step % frequency == 0:
                active_tiers.append(tier_idx)
        return active_tiers

    def get_tier_stats(self) -> Dict[str, Any]:
        """
        Get statistics about tier updates.

        Returns:
            Dictionary with tier update statistics
        """
        stats = {
            'global_step': self.global_step,
            'tier_update_counts': self.tier_update_counts,
            'tier_update_frequencies': self.tier_update_frequencies,
            'tier_parameters': {}
        }

        # Count parameters per tier
        tier_param_counts = [0] * self.num_tiers
        for param_name, tier_idx in self.parameter_tier_map.items():
            tier_param_counts[tier_idx] += 1

        for tier_idx in range(self.num_tiers):
            stats['tier_parameters'][f'tier_{tier_idx}'] = {
                'frequency': self.tier_update_frequencies[tier_idx],
                'update_count': self.tier_update_counts[tier_idx],
                'parameter_count': tier_param_counts[tier_idx]
            }

        return stats

    def reset_tier_counters(self):
        """Reset tier update counters (useful for tracking within an epoch)."""
        self.tier_update_counts = [0] * self.num_tiers


class NestedAdamW(NestedAdam):
    """
    AdamW optimizer with nested learning support.

    Adds weight decay to NestedAdam.

    Args:
        learning_rate: Base learning rate
        tier_update_frequencies: List of update frequencies for each tier
        parameter_tier_map: Dict mapping parameter names to tier indices
        weight_decay: Weight decay coefficient (default: 0.01)
        betas: Adam beta parameters (default: [0.9, 0.999])
        eps: Adam epsilon (default: 1e-8)
    """

    def __init__(
        self,
        learning_rate: float,
        tier_update_frequencies: List[int],
        parameter_tier_map: Dict[str, int],
        weight_decay: float = 0.01,
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-8
    ):
        super().__init__(
            learning_rate=learning_rate,
            tier_update_frequencies=tier_update_frequencies,
            parameter_tier_map=parameter_tier_map,
            betas=betas,
            eps=eps
        )
        self.weight_decay = weight_decay

    def apply_gradients(self, gradients: Dict[str, mx.array], model: Any) -> None:
        """
        Apply gradients with weight decay and tier-based scheduling.

        Args:
            gradients: Dictionary of parameter gradients
            model: Model to update
        """
        # Add weight decay to gradients
        if self.weight_decay > 0:
            for param_name in gradients:
                param = getattr(model, param_name, None)
                if param is not None:
                    gradients[param_name] = gradients[param_name] + self.weight_decay * param

        # Call parent's apply_gradients with tier scheduling
        super().apply_gradients(gradients, model)
