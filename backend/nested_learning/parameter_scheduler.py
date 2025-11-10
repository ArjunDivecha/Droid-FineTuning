"""
Parameter Tier Scheduler

Assigns model parameters to different update frequency tiers based on
various strategies (layer depth, parameter importance, etc.).
"""

import mlx.core as mx
from typing import Dict, List, Literal, Optional, Any
import numpy as np


class ParameterTierScheduler:
    """
    Assigns parameters to update frequency tiers.

    Supports multiple strategies for tier assignment:
    - layer_depth: Shallow layers → fast updates, deep layers → slow updates
    - parameter_importance: Based on gradient magnitude
    - manual: User-specified tier assignments
    """

    def __init__(
        self,
        num_tiers: int,
        strategy: Literal['layer_depth', 'parameter_importance', 'manual'] = 'layer_depth'
    ):
        """
        Initialize parameter tier scheduler.

        Args:
            num_tiers: Number of tiers to create
            strategy: Strategy for assigning parameters to tiers
        """
        self.num_tiers = num_tiers
        self.strategy = strategy
        self.tier_map: Dict[str, int] = {}

    def assign_tiers(
        self,
        model: Any,
        gradient_history: Optional[Dict[str, List[mx.array]]] = None,
        manual_assignments: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """
        Assign parameters to tiers based on the selected strategy.

        Args:
            model: MLX model
            gradient_history: Historical gradients for importance-based assignment
            manual_assignments: Manual tier assignments (for 'manual' strategy)

        Returns:
            Dictionary mapping parameter names to tier indices
        """
        if self.strategy == 'layer_depth':
            self.tier_map = self._assign_by_layer_depth(model)
        elif self.strategy == 'parameter_importance':
            if gradient_history is None:
                raise ValueError("gradient_history required for 'parameter_importance' strategy")
            self.tier_map = self._assign_by_importance(model, gradient_history)
        elif self.strategy == 'manual':
            if manual_assignments is None:
                raise ValueError("manual_assignments required for 'manual' strategy")
            self.tier_map = manual_assignments
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self.tier_map

    def _assign_by_layer_depth(self, model: Any) -> Dict[str, int]:
        """
        Assign tiers based on layer depth.

        Shallow layers (early in the network) get faster update frequencies,
        deep layers (late in the network) get slower update frequencies.

        This follows the intuition that early layers learn general features
        (should adapt quickly) while late layers learn task-specific features
        (should be more stable).

        Args:
            model: MLX model with LoRA adapters

        Returns:
            Mapping of parameter names to tier indices
        """
        tier_map = {}

        # Get all trainable parameters (LoRA adapters)
        trainable_params = self._get_trainable_param_names(model)

        if not trainable_params:
            return {}

        # Extract layer numbers from parameter names
        # Example: "model.layers.0.self_attn.lora_a" → layer 0
        param_layer_nums = []
        for param_name in trainable_params:
            layer_num = self._extract_layer_number(param_name)
            param_layer_nums.append((param_name, layer_num))

        # Sort by layer number
        param_layer_nums.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))

        # Divide parameters into tiers based on layer depth
        num_params = len(param_layer_nums)
        params_per_tier = num_params // self.num_tiers

        for idx, (param_name, layer_num) in enumerate(param_layer_nums):
            # Assign tier: early layers → tier 0 (fastest), late layers → tier N-1 (slowest)
            tier_idx = min(idx // max(params_per_tier, 1), self.num_tiers - 1)
            tier_map[param_name] = tier_idx

        return tier_map

    def _assign_by_importance(
        self,
        model: Any,
        gradient_history: Dict[str, List[mx.array]]
    ) -> Dict[str, int]:
        """
        Assign tiers based on parameter importance (gradient magnitude).

        Parameters with larger gradients are considered more important and
        assigned to faster-updating tiers.

        Args:
            model: MLX model
            gradient_history: Historical gradients for each parameter

        Returns:
            Mapping of parameter names to tier indices
        """
        tier_map = {}

        # Compute average gradient magnitude for each parameter
        param_importance = {}
        for param_name, grad_list in gradient_history.items():
            if len(grad_list) == 0:
                param_importance[param_name] = 0.0
            else:
                # Average L2 norm of gradients
                avg_magnitude = sum(mx.sum(g ** 2).item() for g in grad_list) / len(grad_list)
                param_importance[param_name] = avg_magnitude

        # Sort parameters by importance (descending)
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)

        # Divide into tiers: high importance → fast updates (tier 0)
        num_params = len(sorted_params)
        params_per_tier = num_params // self.num_tiers

        for idx, (param_name, importance) in enumerate(sorted_params):
            tier_idx = min(idx // max(params_per_tier, 1), self.num_tiers - 1)
            tier_map[param_name] = tier_idx

        return tier_map

    def _get_trainable_param_names(self, model: Any) -> List[str]:
        """
        Get names of all trainable parameters in the model.

        For LoRA models, this typically includes lora_a and lora_b parameters.

        Args:
            model: MLX model

        Returns:
            List of trainable parameter names
        """
        trainable = []

        def _recurse(module, prefix=''):
            if hasattr(module, 'parameters'):
                params = module.parameters()
                for name, param in params.items():
                    full_name = f"{prefix}.{name}" if prefix else name
                    # Check if parameter is trainable (LoRA adapters)
                    if 'lora' in name.lower():
                        trainable.append(full_name)

            if hasattr(module, 'children'):
                for child_name, child in module.children().items():
                    child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                    _recurse(child, child_prefix)

        _recurse(model)
        return trainable

    def _extract_layer_number(self, param_name: str) -> Optional[int]:
        """
        Extract layer number from parameter name.

        Examples:
            "model.layers.0.self_attn.lora_a" → 0
            "model.layers.15.mlp.lora_b" → 15
            "model.embed_tokens.weight" → None

        Args:
            param_name: Parameter name

        Returns:
            Layer number or None if not found
        """
        import re

        # Pattern to match layer numbers: "layers.{number}."
        match = re.search(r'layers\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))

        return None

    def get_tier_summary(self) -> Dict[str, Any]:
        """
        Get summary of tier assignments.

        Returns:
            Dictionary with tier assignment summary
        """
        if not self.tier_map:
            return {'error': 'No tier assignments yet'}

        tier_counts = [0] * self.num_tiers
        tier_params = {i: [] for i in range(self.num_tiers)}

        for param_name, tier_idx in self.tier_map.items():
            tier_counts[tier_idx] += 1
            tier_params[tier_idx].append(param_name)

        summary = {
            'num_tiers': self.num_tiers,
            'strategy': self.strategy,
            'total_parameters': len(self.tier_map),
            'tiers': {}
        }

        for tier_idx in range(self.num_tiers):
            summary['tiers'][f'tier_{tier_idx}'] = {
                'parameter_count': tier_counts[tier_idx],
                'parameters': tier_params[tier_idx][:5],  # Show first 5
                'parameter_count_total': len(tier_params[tier_idx])
            }

        return summary
