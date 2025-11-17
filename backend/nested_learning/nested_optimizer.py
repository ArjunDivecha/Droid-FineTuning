"""
Nested Adam Optimizer

Custom Adam optimizer with multi-frequency parameter updates for Nested Learning.
Each parameter tier updates at a different frequency based on the tier schedule.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from typing import Dict, List, Any, Optional
import logging
import copy

logger = logging.getLogger(__name__)


class NestedAdam:
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
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps

        self.tier_update_frequencies = tier_update_frequencies
        self.parameter_tier_map = parameter_tier_map
        self.num_tiers = len(tier_update_frequencies)

        # Track update counts for each tier
        self.tier_update_counts = [0] * self.num_tiers

        # Global step counter
        self.global_step = 0

        # Adam state: stores m, v, and step for each parameter
        self.state = {}

    def apply_gradients(self, gradients: Dict[str, mx.array], model: Any) -> None:
        """
        Apply gradients with tier-based update scheduling.

        Only updates parameters in tiers that are scheduled to update at this step.

        Args:
            gradients: Dictionary of parameter gradients (may be nested PyTree)
            model: Model to update
        """
        self.global_step += 1

        # Determine which tiers should update at this step
        active_tiers = self._get_active_tiers()

        # Update tier counters for active tiers
        for tier_idx in active_tiers:
            self.tier_update_counts[tier_idx] += 1

        # Get trainable parameters as flat dict
        trainable_params = tree_flatten(model.trainable_parameters(), destination={})

        # Flatten gradients to flat dict
        flat_gradients = tree_flatten(gradients, destination={})

        # Build updates dict with Adam-updated parameters
        updates = {}

        for param_name in trainable_params.keys():
            # Skip if no gradient for this parameter
            if param_name not in flat_gradients:
                continue

            param = trainable_params[param_name]
            grad = flat_gradients[param_name]

            # Filter 1: Only update LoRA parameters
            param_name_lower = param_name.lower()
            is_lora = ('lora_a' in param_name_lower or
                      'lora_b' in param_name_lower or
                      'adapter' in param_name_lower)

            if not is_lora:
                # Not LoRA - skip update (keep original param)
                continue

            # Filter 2: Only update parameters in active tiers
            if param_name not in self.parameter_tier_map:
                tier_idx = self.num_tiers - 1  # Slowest tier
                logger.warning(f"Parameter {param_name} not in tier map, assigning to slowest tier {tier_idx}")
            else:
                tier_idx = self.parameter_tier_map[param_name]

            if tier_idx not in active_tiers:
                # Not in active tier - skip update
                continue

            # Initialize Adam state if needed
            if param_name not in self.state:
                self.state[param_name] = {
                    'm': mx.zeros_like(param),
                    'v': mx.zeros_like(param),
                    'step': 0
                }

            state = self.state[param_name]
            state['step'] += 1

            # Adam update
            beta1, beta2 = self.betas
            m = beta1 * state['m'] + (1 - beta1) * grad
            v = beta2 * state['v'] + (1 - beta2) * (grad * grad)

            # Bias correction
            m_hat = m / (1 - beta1 ** state['step'])
            v_hat = v / (1 - beta2 ** state['step'])

            # Compute parameter update
            param_update = self.learning_rate * m_hat / (mx.sqrt(v_hat) + self.eps)
            new_param = param - param_update

            # Update state
            state['m'] = m
            state['v'] = v

            # Store updated parameter
            updates[param_name] = new_param

        # Evaluate all updates
        if updates:
            mx.eval(updates)

            logger.info(f"Applying {len(updates)} parameter updates via model.update()")
            if len(updates) > 0:
                sample_key = list(updates.keys())[0]
                sample_before = tree_flatten(model.trainable_parameters(), destination={})[sample_key]
                sample_update = updates[sample_key]
                logger.info(f"Sample param '{sample_key}': before_mean={float(mx.mean(sample_before)):.8f}, update_mean={float(mx.mean(sample_update)):.8f}, max_diff={float(mx.max(mx.abs(sample_before - sample_update))):.8f}")

            # CRITICAL FIX: Rebuild nested PyTree structure using deepcopy + manual assignment
            # tree_map_with_path does NOT properly reconstruct the nested structure for model.update()
            updated_params = self._apply_flat_updates_to_nested(
                model.trainable_parameters(),
                updates
            )

            # Evaluate all parameters
            mx.eval(updated_params)

            # Update model with new parameters
            model.update(updated_params)

            # Verify update worked
            sample_after = tree_flatten(model.trainable_parameters(), destination={})[sample_key]
            logger.info(f"After model.update(): mean={float(mx.mean(sample_after)):.8f}, matches_update={float(mx.max(mx.abs(sample_after - sample_update))) < 1e-10}")

        # Clear MLX cache after gradient application
        mx.clear_cache()

    def _apply_flat_updates_to_nested(
        self,
        nested_params: Any,
        flat_updates: Dict[str, mx.array]
    ) -> Any:
        """
        Apply flat updates dict to nested parameter structure.

        CRITICAL: This is the correct way to update MLX models. tree_map_with_path
        does NOT properly reconstruct the nested structure needed by model.update().

        Args:
            nested_params: Nested dict/list structure from trainable_parameters()
            flat_updates: Flat dict mapping 'path.to.param' -> updated_value

        Returns:
            Updated nested structure suitable for model.update()
        """
        # Deep copy to avoid modifying original structure
        updated = copy.deepcopy(nested_params)

        # Apply each update by navigating the nested structure
        for param_path, new_value in flat_updates.items():
            parts = param_path.split('.')

            # Navigate to the parent container
            current = updated
            for part in parts[:-1]:
                if isinstance(current, dict):
                    current = current[part]
                elif isinstance(current, list):
                    current = current[int(part)]
                else:
                    raise ValueError(
                        f"Cannot navigate to {param_path}: "
                        f"unexpected type {type(current)} at part '{part}'"
                    )

            # Set the final value
            final_key = parts[-1]
            if isinstance(current, dict):
                current[final_key] = new_value
            elif isinstance(current, list):
                current[int(final_key)] = new_value
            else:
                raise ValueError(
                    f"Cannot set {param_path}: "
                    f"parent is {type(current)}, expected dict or list"
                )

        return updated

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
            gradients: Dictionary of parameter gradients (may be nested PyTree)
            model: Model to update
        """
        # Apply weight decay to gradients
        if self.weight_decay > 0:
            flat_gradients = tree_flatten(gradients, destination={})
            flat_params = tree_flatten(model.trainable_parameters(), destination={})

            for param_name in flat_gradients:
                if param_name in flat_params:
                    flat_gradients[param_name] = (
                        flat_gradients[param_name] + self.weight_decay * flat_params[param_name]
                    )

            # Update gradients dict with weight decay applied
            gradients = flat_gradients

        # Call parent's apply_gradients with tier scheduling
        super().apply_gradients(gradients, model)
