#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: fusion_adapters.py
=============================================================================

INPUT FILES:
- Multiple adapter directories containing .safetensors files
- Each adapter should have same rank and target layers

OUTPUT FILES:
- fused_adapter.safetensors: Blended adapter weights
- fusion_report.txt: Details about the fusion process

VERSION: 1.0
LAST UPDATED: 2025-08-31
AUTHOR: Claude Code

DESCRIPTION:
Fuses multiple LoRA adapters by weighted averaging their parameters.
Supports different fusion strategies including simple averaging,
weighted fusion, and SLERP (Spherical Linear Interpolation).

DEPENDENCIES:
- safetensors
- numpy
- torch

USAGE:
python fusion_adapters.py --adapters adapter1 adapter2 --weights 0.5 0.5 --output fused_adapter

NOTES:
- All adapters must have compatible dimensions
- Weights should sum to 1.0 for best results
- Default fusion method is weighted averaging
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError:
    print("Error: safetensors not installed. Install with: pip install safetensors")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Warning: PyTorch not available. Some fusion methods may not work.")
    torch = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdapterFusion:
    """Handles fusion of multiple LoRA adapters."""
    
    def __init__(self, base_adapter_dir: str = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters"):
        self.base_adapter_dir = base_adapter_dir
        
    def list_available_adapters(self) -> List[str]:
        """List all available adapter names."""
        adapters = []
        if os.path.exists(self.base_adapter_dir):
            for item in os.listdir(self.base_adapter_dir):
                adapter_path = os.path.join(self.base_adapter_dir, item)
                if os.path.isdir(adapter_path):
                    # Check if it has adapter files
                    best_file = os.path.join(adapter_path, "best_adapters.safetensors")
                    latest_file = os.path.join(adapter_path, "adapters.safetensors")
                    if os.path.exists(best_file) or os.path.exists(latest_file):
                        adapters.append(item)
        return sorted(adapters)
    
    def load_adapter_weights(self, adapter_name: str, use_best: bool = True) -> Dict[str, np.ndarray]:
        """Load adapter weights from safetensors file."""
        adapter_dir = os.path.join(self.base_adapter_dir, adapter_name)
        
        # Choose which file to load
        if use_best:
            adapter_file = os.path.join(adapter_dir, "best_adapters.safetensors")
            if not os.path.exists(adapter_file):
                adapter_file = os.path.join(adapter_dir, "adapters.safetensors")
                logger.warning(f"No best adapter found for {adapter_name}, using latest")
        else:
            adapter_file = os.path.join(adapter_dir, "adapters.safetensors")
        
        if not os.path.exists(adapter_file):
            raise FileNotFoundError(f"No adapter file found at {adapter_file}")
        
        logger.info(f"Loading adapter weights from: {adapter_file}")
        
        weights = {}
        with safe_open(adapter_file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        
        return weights
    
    def validate_adapter_compatibility(self, adapters: List[Dict[str, np.ndarray]]) -> bool:
        """Validate that adapters have compatible dimensions.

        Allows adapters with different sparsity patterns (e.g., partial layer LoRA)
        as long as overlapping tensors share the same shapes.
        """
        if len(adapters) < 2:
            return True

        shape_map: Dict[str, Tuple[Tuple[int, ...], str]] = {}

        for idx, adapter in enumerate(adapters):
            for key, value in adapter.items():
                current_shape = tuple(value.shape)
                dtype_name = str(value.dtype)

                if key not in shape_map:
                    shape_map[key] = (current_shape, dtype_name)
                    continue

                reference_shape, reference_dtype = shape_map[key]
                if reference_shape != current_shape:
                    logger.error(
                        "Shape mismatch for key '%s' between adapters. "
                        "Expected %s but adapter %d has %s",
                        key,
                        reference_shape,
                        idx,
                        current_shape,
                    )
                    return False

                if reference_dtype != dtype_name:
                    logger.warning(
                        "Dtype mismatch for key '%s' (expected %s, adapter %d has %s). "
                        "Casting to reference dtype during fusion.",
                        key,
                        reference_dtype,
                        idx,
                        dtype_name,
                    )
        
        logger.info("All adapters are compatible for fusion")
        return True
    
    def weighted_average_fusion(self, adapters: List[Dict[str, np.ndarray]], weights: List[float]) -> Dict[str, np.ndarray]:
        """Fuse adapters using weighted averaging."""
        if len(adapters) != len(weights):
            raise ValueError("Number of adapters must match number of weights")
        
        # Normalize weights
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            weights = [w / total_weight for w in weights]
        
        logger.info(f"Fusing {len(adapters)} adapters with weights: {weights}")
        
        # Initialize result with zeros
        # Collect union of keys across adapters and remember canonical shapes/dtypes
        key_specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        for adapter in adapters:
            for key, tensor in adapter.items():
                shape = tensor.shape
                dtype = tensor.dtype
                if key in key_specs:
                    ref_shape, ref_dtype = key_specs[key]
                    if ref_shape != shape:
                        raise ValueError(
                            f"Incompatible shapes for key '{key}': {ref_shape} vs {shape}"
                        )
                    if ref_dtype != dtype:
                        logger.warning(
                            "Casting key '%s' from %s to %s to keep dtype consistent",
                            key,
                            dtype,
                            ref_dtype,
                        )
                        # Cast to reference dtype so downstream math stays stable
                        adapter[key] = adapter[key].astype(ref_dtype, copy=False)
                else:
                    key_specs[key] = (shape, dtype)

        fused: Dict[str, np.ndarray] = {}
        for key, (shape, dtype) in key_specs.items():
            fused[key] = np.zeros(shape, dtype=dtype)

        # Weighted sum, treating missing tensors as zeros
        for adapter, weight in zip(adapters, weights):
            for key, (shape, dtype) in key_specs.items():
                if key not in adapter:
                    continue
                tensor = adapter[key]
                if tensor.dtype != dtype:
                    tensor = tensor.astype(dtype, copy=False)
                fused[key] += weight * tensor
        
        return fused
    
    def slerp_fusion(self, adapter1: Dict[str, np.ndarray], adapter2: Dict[str, np.ndarray], t: float = 0.5) -> Dict[str, np.ndarray]:
        """Spherical Linear Interpolation fusion for two adapters."""
        if torch is None:
            logger.warning("PyTorch not available, falling back to linear interpolation")
            return self.weighted_average_fusion([adapter1, adapter2], [1-t, t])
        
        logger.info(f"SLERP fusion with interpolation factor t={t}")
        
        fused: Dict[str, np.ndarray] = {}
        all_keys = set(adapter1.keys()) | set(adapter2.keys())

        for key in all_keys:
            if key in adapter1:
                w1_np = adapter1[key]
            elif key in adapter2:
                w1_np = np.zeros_like(adapter2[key])
            else:
                # Should not happen because key is in union
                continue

            if key in adapter2:
                w2_np = adapter2[key]
            else:
                w2_np = np.zeros_like(w1_np)

            # Ensure dtypes align before converting to torch
            target_dtype = w1_np.dtype
            if w2_np.dtype != target_dtype:
                w2_np = w2_np.astype(target_dtype, copy=False)

            w1 = torch.from_numpy(w1_np).float()
            w2 = torch.from_numpy(w2_np).float()
            
            # Flatten for SLERP computation
            w1_flat = w1.flatten()
            w2_flat = w2.flatten()
            
            # Compute dot product (cosine of angle)
            dot = torch.dot(w1_flat, w2_flat)
            
            # If vectors are nearly parallel, use linear interpolation
            if abs(dot) > 0.9995:
                result = (1 - t) * w1 + t * w2
            else:
                # Compute angle and SLERP
                theta = torch.acos(torch.clamp(dot / (torch.norm(w1_flat) * torch.norm(w2_flat)), -1, 1))
                sin_theta = torch.sin(theta)
                
                if sin_theta.abs() < 1e-6:
                    # Handle degenerate case
                    result = (1 - t) * w1 + t * w2
                else:
                    # SLERP formula
                    a = torch.sin((1 - t) * theta) / sin_theta
                    b = torch.sin(t * theta) / sin_theta
                    result = a * w1 + b * w2
            
            fused[key] = result.numpy().astype(w1_np.dtype, copy=False)
        
        return fused
    
    def save_fused_adapter(self, fused_weights: Dict[str, np.ndarray], output_dir: str, adapter_name: str = "fused_adapter", source_adapter_names: List[str] = None):
        """Save fused adapter weights and create adapter_config.json."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as safetensors
        output_file = os.path.join(output_dir, f"{adapter_name}.safetensors")
        save_file(fused_weights, output_file)
        
        # Also save as 'adapters.safetensors' for MLX compatibility
        mlx_file = os.path.join(output_dir, "adapters.safetensors")
        save_file(fused_weights, mlx_file)
        
        # Create adapter_config.json by copying from first source adapter
        if source_adapter_names and len(source_adapter_names) > 0:
            source_adapter_dir = os.path.join(self.base_adapter_dir, source_adapter_names[0])
            source_config_path = os.path.join(source_adapter_dir, "adapter_config.json")
            
            if os.path.exists(source_config_path):
                try:
                    with open(source_config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Save to fused adapter directory
                    dest_config_path = os.path.join(output_dir, "adapter_config.json")
                    with open(dest_config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    logger.info(f"Copied adapter_config.json from {source_adapter_names[0]}")
                except Exception as e:
                    logger.warning(f"Failed to copy adapter_config.json: {e}")
        
        logger.info(f"Saved fused adapter to: {output_file}")
        logger.info(f"MLX-compatible file saved to: {mlx_file}")
        
        return output_file
    
    def generate_fusion_report(self, adapter_names: List[str], weights: List[float], 
                             fusion_method: str, output_dir: str) -> str:
        """Generate a report about the fusion process."""
        report_file = os.path.join(output_dir, "fusion_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("=== Adapter Fusion Report ===\n\n")
            f.write(f"Fusion Method: {fusion_method}\n")
            f.write(f"Number of adapters fused: {len(adapter_names)}\n\n")
            
            f.write("Source Adapters:\n")
            for i, (name, weight) in enumerate(zip(adapter_names, weights)):
                f.write(f"  {i+1}. {name} (weight: {weight:.4f})\n")
            
            f.write(f"\nOutput directory: {output_dir}\n")
            from datetime import datetime
            f.write(f"Generated on: {datetime.now()}\n")
        
        logger.info(f"Fusion report saved to: {report_file}")
        return report_file

def main():
    parser = argparse.ArgumentParser(description="Fuse multiple LoRA adapters")
    parser.add_argument("--adapters", nargs="+", help="Names of adapters to fuse")
    parser.add_argument("--weights", nargs="+", type=float, help="Fusion weights (default: equal weights)")
    parser.add_argument("--method", choices=["weighted", "slerp"], default="weighted", 
                       help="Fusion method (default: weighted)")
    parser.add_argument("--output-dir", default="./fused_adapters", help="Output directory")
    parser.add_argument("--output-name", default="fused_adapter", help="Output adapter name")
    parser.add_argument("--use-best", action="store_true", default=True, 
                       help="Use best_adapters.safetensors if available")
    parser.add_argument("--list-adapters", action="store_true", 
                       help="List available adapters and exit")
    
    args = parser.parse_args()
    
    fusion = AdapterFusion()
    
    # List available adapters if requested
    if args.list_adapters:
        adapters = fusion.list_available_adapters()
        print("Available adapters:")
        for adapter in adapters:
            print(f"  - {adapter}")
        return
    
    # Validate inputs (skip if just listing adapters)
    if not args.list_adapters:
        if not args.adapters or len(args.adapters) < 2:
            logger.error("At least 2 adapters are required for fusion")
            return
    
    # Set default weights if not provided
    if args.weights is None:
        args.weights = [1.0 / len(args.adapters)] * len(args.adapters)
    
    if len(args.weights) != len(args.adapters):
        logger.error("Number of weights must match number of adapters")
        return
    
    # Load adapters
    logger.info("Loading adapter weights...")
    loaded_adapters = []
    for adapter_name in args.adapters:
        try:
            weights = fusion.load_adapter_weights(adapter_name, use_best=args.use_best)
            loaded_adapters.append(weights)
            logger.info(f"Loaded adapter: {adapter_name} ({len(weights)} tensors)")
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_name}: {e}")
            return
    
    # Validate compatibility
    if not fusion.validate_adapter_compatibility(loaded_adapters):
        logger.error("Adapters are not compatible for fusion")
        return
    
    # Perform fusion
    logger.info(f"Starting fusion using method: {args.method}")
    if args.method == "weighted":
        fused_weights = fusion.weighted_average_fusion(loaded_adapters, args.weights)
    elif args.method == "slerp" and len(loaded_adapters) == 2:
        t = args.weights[1]  # Use second weight as interpolation factor
        fused_weights = fusion.slerp_fusion(loaded_adapters[0], loaded_adapters[1], t)
    else:
        logger.error("SLERP fusion only supports exactly 2 adapters")
        return
    
    # Save fused adapter
    output_file = fusion.save_fused_adapter(fused_weights, args.output_dir, args.output_name, source_adapter_names=args.adapters)
    
    # Generate report
    fusion.generate_fusion_report(args.adapters, args.weights, args.method, args.output_dir)
    
    logger.info("Fusion completed successfully!")
    logger.info(f"Fused adapter saved to: {output_file}")

if __name__ == "__main__":
    main()
