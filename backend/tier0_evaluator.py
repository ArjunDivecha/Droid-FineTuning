#!/usr/bin/env python3
"""
Tier 0 Evaluator - Mathematical Analysis (INSTANT - <1 second)

Analyzes LoRA adapter weights directly without running inference.
Provides instant quality assessment based on:
- Spectral analysis (singular values, effective rank)
- Training dynamics (loss curves, overfitting)
- Weight statistics (norms, sparsity)
- Quality scoring and issue detection

Usage:
    python tier0_evaluator.py --adapter 4B
    python tier0_evaluator.py --adapter 4b-nested --compare-to 4B
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    logger.warning("MLX not available - will use numpy fallback")
    HAS_MLX = False

class Tier0Evaluator:
    """
    Instant mathematical evaluation of adapters - NO inference needed.

    Analyzes:
    1. Spectral properties (singular values, effective rank, concentration)
    2. Training dynamics (loss curves, overfitting detection)
    3. Weight statistics (norms, sparsity, magnitude)
    4. Quality scoring and issue warnings
    """

    def __init__(self,
                 adapter_base_dir: str = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters"):
        self.adapter_base_dir = adapter_base_dir
        self.nested_base_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints"

    def load_adapter_weights(self, adapter_name: str) -> Dict[str, np.ndarray]:
        """Load adapter weights from safetensors file."""
        # Check if nested learning adapter
        nested_path = Path(self.nested_base_dir) / adapter_name / "checkpoints" / "best"
        regular_path = Path(self.adapter_base_dir) / adapter_name

        if nested_path.exists():
            adapter_path = nested_path
            logger.info(f"Loading nested learning adapter from: {adapter_path}")
        elif regular_path.exists():
            adapter_path = regular_path
            logger.info(f"Loading regular adapter from: {adapter_path}")
        else:
            raise FileNotFoundError(f"Adapter not found: {adapter_name}")

        # Look for adapter file
        adapter_file = None
        for filename in ["adapters.safetensors", "best_adapters.safetensors"]:
            candidate = adapter_path / filename
            if candidate.exists():
                adapter_file = candidate
                break

        if not adapter_file:
            raise FileNotFoundError(f"No adapter safetensors file found in {adapter_path}")

        logger.info(f"Loading weights from: {adapter_file}")

        # Load using safetensors
        try:
            from safetensors import safe_open
            import numpy as np
            
            # Try PyTorch first (better bfloat16 support)
            try:
                import torch
                weights = {}
                with safe_open(adapter_file, framework="pt") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        # Convert bfloat16 to float32
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.float()
                        weights[key] = tensor.numpy().astype(np.float32)
                logger.info(f"Loaded {len(weights)} weight tensors using PyTorch")
                return weights
            except (ImportError, Exception) as pt_error:
                # Fallback to numpy
                logger.debug(f"PyTorch loading failed, using numpy: {pt_error}")
            weights = {}
            with safe_open(adapter_file, framework="numpy") as f:
                for key in f.keys():
                        tensor = f.get_tensor(key)
                        # Convert float16 to float32 if needed
                        if tensor.dtype == np.float16:
                            tensor = tensor.astype(np.float32)
                        weights[key] = tensor
                logger.info(f"Loaded {len(weights)} weight tensors using numpy")
            return weights
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            raise

    def compute_svd_metrics(self, weight_matrix: np.ndarray) -> Dict:
        """Compute SVD-based metrics for a weight matrix."""
        # Handle different shapes
        if len(weight_matrix.shape) > 2:
            # Reshape to 2D
            original_shape = weight_matrix.shape
            weight_matrix = weight_matrix.reshape(original_shape[0], -1)

        # Compute SVD
        try:
            U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            logger.warning("SVD failed for matrix, using fallback")
            return {
                'spectral_norm': 0.0,
                'effective_rank': 0,
                'concentration_ratio': 0.0,
                'spectral_decay': 0.0
            }

        # Spectral norm (largest singular value)
        spectral_norm = float(S[0]) if len(S) > 0 else 0.0

        # Effective rank (number of significant singular values)
        threshold = 1e-3 * spectral_norm
        effective_rank = int(np.sum(S > threshold))

        # Concentration ratio (top 5 singular values / total)
        top_k = min(5, len(S))
        concentration_ratio = float(np.sum(S[:top_k]) / (np.sum(S) + 1e-10))

        # Spectral decay (how fast singular values decay)
        if len(S) > 1:
            spectral_decay = float(S[1] / (S[0] + 1e-10))
        else:
            spectral_decay = 0.0

        return {
            'spectral_norm': spectral_norm,
            'effective_rank': effective_rank,
            'concentration_ratio': concentration_ratio,
            'spectral_decay': spectral_decay,
            'singular_values': S[:10].tolist()  # Top 10 for analysis
        }

    def compute_weight_statistics(self, weights: Dict[str, np.ndarray]) -> Dict:
        """Compute basic statistics about adapter weights."""
        all_weights = np.concatenate([w.flatten() for w in weights.values()])

        return {
            'mean_abs': float(np.abs(all_weights).mean()),
            'std': float(all_weights.std()),
            'l2_norm': float(np.linalg.norm(all_weights)),
            'l1_norm': float(np.abs(all_weights).sum()),
            'sparsity': float((np.abs(all_weights) < 1e-4).mean()),
            'max_abs': float(np.abs(all_weights).max()),
            'num_parameters': int(len(all_weights))
        }

    def load_training_metrics(self, adapter_name: str) -> Dict:
        """Load training metrics from logs."""
        # Check nested learning first
        nested_metrics_path = Path(self.nested_base_dir) / adapter_name / "metrics"
        regular_config_path = Path(self.adapter_base_dir) / adapter_name / "adapter_config.json"

        metrics = {
            'train_loss': [],
            'val_loss': [],
            'has_metrics': False
        }

        # Try nested learning metrics
        if nested_metrics_path.exists():
            train_file = nested_metrics_path / "train_metrics.jsonl"
            eval_file = nested_metrics_path / "eval_metrics.jsonl"

            if train_file.exists():
                with open(train_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if 'loss' in data:
                                metrics['train_loss'].append(data['loss'])
                metrics['has_metrics'] = True

            if eval_file.exists():
                with open(eval_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if 'loss' in data:
                                metrics['val_loss'].append(data['loss'])

        # Try regular adapter config (may have final loss)
        elif regular_config_path.exists():
            with open(regular_config_path) as f:
                config = json.load(f)
                # Some configs store final metrics
                if 'final_train_loss' in config:
                    metrics['train_loss'] = [config['final_train_loss']]
                if 'final_val_loss' in config:
                    metrics['val_loss'] = [config['final_val_loss']]
                    metrics['has_metrics'] = True

        return metrics

    def analyze_training_dynamics(self, metrics: Dict) -> Dict:
        """Analyze training loss curves for quality indicators."""
        if not metrics['has_metrics'] or not metrics['train_loss']:
            return {
                'has_data': False,
                'quality_score': 50  # Neutral score if no data
            }

        train_loss = np.array(metrics['train_loss'])
        val_loss = np.array(metrics['val_loss']) if metrics['val_loss'] else train_loss

        analysis = {'has_data': True}

        # Final losses
        analysis['final_train_loss'] = float(train_loss[-1])
        analysis['final_val_loss'] = float(val_loss[-1]) if len(val_loss) > 0 else None

        # Loss improvement
        if len(train_loss) > 1:
            initial_loss = train_loss[0]
            final_loss = train_loss[-1]
            analysis['loss_improvement'] = float((initial_loss - final_loss) / initial_loss)
        else:
            analysis['loss_improvement'] = 0.0

        # Overfitting detection
        if len(val_loss) > 0:
            analysis['overfitting_gap'] = float(val_loss[-1] - train_loss[-1])
            analysis['min_val_loss'] = float(val_loss.min())

            # Check if validation loss stopped improving
            if len(val_loss) > 5:
                recent_val = val_loss[-5:]
                analysis['val_trend'] = 'improving' if recent_val[-1] < recent_val[0] else 'plateaued'
        else:
            analysis['overfitting_gap'] = 0.0
            analysis['val_trend'] = 'unknown'

        # Learning curve quality (smooth vs erratic)
        if len(train_loss) > 3:
            loss_changes = np.diff(train_loss)
            loss_volatility = float(loss_changes.std())
            analysis['loss_volatility'] = loss_volatility
            analysis['learning_stable'] = loss_volatility < 0.1

        return analysis

    def compute_quality_score(self, spectral_metrics: Dict, weight_stats: Dict,
                             training_analysis: Dict) -> Tuple[float, List[str]]:
        """
        Compute overall quality score (0-100) and detect issues.

        Scoring rubric:
        - Spectral properties: 40 points
        - Training dynamics: 40 points
        - Weight statistics: 20 points
        """
        score = 0.0
        warnings = []

        # Spectral score (40 points)
        spectral_score = 0

        # Good: High concentration (focused learning)
        if spectral_metrics['concentration_ratio'] > 0.8:
            spectral_score += 15
        elif spectral_metrics['concentration_ratio'] > 0.6:
            spectral_score += 10
        else:
            warnings.append("Low singular value concentration - may be learning noise")

        # Good: Moderate effective rank (efficient but not too low)
        if 2 <= spectral_metrics['effective_rank'] <= 32:
            spectral_score += 15
        elif spectral_metrics['effective_rank'] < 2:
            warnings.append("Very low effective rank - may underfit")
            spectral_score += 5
        else:
            spectral_score += 10

        # Good: Reasonable spectral norm (not too high or low)
        if 0.1 < spectral_metrics['spectral_norm'] < 10.0:
            spectral_score += 10
        elif spectral_metrics['spectral_norm'] < 0.01:
            warnings.append("Very low spectral norm - adapter may have minimal effect")
        elif spectral_metrics['spectral_norm'] > 50:
            warnings.append("Very high spectral norm - risk of catastrophic forgetting")

        score += spectral_score

        # Training dynamics score (40 points)
        training_score = 0

        if training_analysis['has_data']:
            # Good: Significant loss improvement
            if training_analysis['loss_improvement'] > 0.3:
                training_score += 15
            elif training_analysis['loss_improvement'] > 0.1:
                training_score += 10
            else:
                warnings.append("Low loss improvement - training may not be effective")
                training_score += 5

            # Good: Small overfitting gap
            if training_analysis.get('overfitting_gap') is not None:
                gap = training_analysis['overfitting_gap']
                if abs(gap) < 0.1:
                    training_score += 15
                elif abs(gap) < 0.3:
                    training_score += 10
                else:
                    if gap > 0:
                        warnings.append(f"Overfitting detected - train/val gap: {gap:.3f}")
                    training_score += 5
            else:
                training_score += 10  # No val data, give benefit of doubt

            # Good: Final loss is low
            if training_analysis['final_train_loss'] < 1.0:
                training_score += 10
            elif training_analysis['final_train_loss'] < 2.0:
                training_score += 5
        
        # Note: If no training data, training_score remains 0
        # and we re-normalize in the final calculation step.

        # Weight statistics score (20 points)
        weight_score = 0

        # Good: Moderate sparsity (focused learning)
        if 0.2 < weight_stats['sparsity'] < 0.8:
            weight_score += 10
        elif weight_stats['sparsity'] > 0.95:
            warnings.append("Very high sparsity - adapter may be undertrained")

        # Good: Reasonable L2 norm
        if 1.0 < weight_stats['l2_norm'] < 1000.0:
            weight_score += 10
        elif weight_stats['l2_norm'] < 0.1:
            warnings.append("Very low L2 norm - adapter may have minimal effect")

        score += weight_score

        # Final score calculation
        if training_analysis['has_data']:
            score += training_score
            # Max score is 100 (40 + 40 + 20)
        else:
            # Re-normalize if no training data
            # Available points: 40 (Spectral) + 20 (Weight) = 60
            # We scale this up to 100
            raw_score = score
            score = (raw_score / 60.0) * 100.0
            warnings.append(f"No training metrics found - score re-normalized (Raw: {raw_score:.1f}/60)")

        return min(100.0, score), warnings

    def evaluate_adapter(self, adapter_name: str) -> Dict:
        """
        Evaluate adapter using mathematical analysis - NO inference.

        Returns comprehensive quality report in <1 second.
        """
        import time
        start_time = time.time()

        logger.info(f"Tier 0 Evaluation: {adapter_name}")

        # Load weights
        weights = self.load_adapter_weights(adapter_name)

        # Compute spectral metrics for each layer
        logger.info("Computing spectral analysis...")
        layer_metrics = {}
        for layer_name, weight_matrix in weights.items():
            layer_metrics[layer_name] = self.compute_svd_metrics(weight_matrix)

        # Aggregate spectral metrics
        spectral_metrics = {
            'spectral_norm': float(np.mean([m['spectral_norm'] for m in layer_metrics.values()])),
            'effective_rank': float(np.mean([m['effective_rank'] for m in layer_metrics.values()])),
            'concentration_ratio': float(np.mean([m['concentration_ratio'] for m in layer_metrics.values()])),
            'spectral_decay': float(np.mean([m['spectral_decay'] for m in layer_metrics.values()])),
            'avg_spectral_norm': float(np.mean([m['spectral_norm'] for m in layer_metrics.values()])),
            'avg_effective_rank': float(np.mean([m['effective_rank'] for m in layer_metrics.values()])),
            'avg_concentration_ratio': float(np.mean([m['concentration_ratio'] for m in layer_metrics.values()])),
            'avg_spectral_decay': float(np.mean([m['spectral_decay'] for m in layer_metrics.values()])),
            'layer_metrics': {k: {kk: vv for kk, vv in v.items() if kk != 'singular_values'}
                             for k, v in layer_metrics.items()}
        }

        # Compute weight statistics
        logger.info("Computing weight statistics...")
        weight_stats = self.compute_weight_statistics(weights)

        # Load and analyze training metrics
        logger.info("Analyzing training dynamics...")
        training_metrics = self.load_training_metrics(adapter_name)
        training_analysis = self.analyze_training_dynamics(training_metrics)

        # Compute quality score
        logger.info("Computing quality score...")
        quality_score, warnings = self.compute_quality_score(
            spectral_metrics, weight_stats, training_analysis
        )

        elapsed_time = time.time() - start_time

        report = {
            'adapter_name': adapter_name,
            'tier': 0,
            'evaluation_method': 'mathematical_analysis',
            'time_taken_seconds': elapsed_time,
            'quality_score': round(quality_score, 1),
            'grade': self._score_to_grade(quality_score),
            'spectral_analysis': spectral_metrics,
            'weight_statistics': weight_stats,
            'training_dynamics': training_analysis,
            'warnings': warnings,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Evaluation complete in {elapsed_time:.3f}s - Score: {quality_score:.1f}/100")

        return report

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    def compare_adapters(self, adapter1: str, adapter2: str) -> Dict:
        """Compare two adapters."""
        logger.info(f"Comparing {adapter1} vs {adapter2}")

        report1 = self.evaluate_adapter(adapter1)
        report2 = self.evaluate_adapter(adapter2)

        comparison = {
            'adapter1': adapter1,
            'adapter2': adapter2,
            'scores': {
                adapter1: report1['quality_score'],
                adapter2: report2['quality_score']
            },
            'winner': adapter1 if report1['quality_score'] > report2['quality_score'] else adapter2,
            'score_difference': abs(report1['quality_score'] - report2['quality_score']),
            'detailed_reports': {
                adapter1: report1,
                adapter2: report2
            }
        }

        return comparison


def main():
    parser = argparse.ArgumentParser(description='Tier 0 Adapter Evaluator - Instant Mathematical Analysis')
    parser.add_argument('--adapter', required=True, help='Adapter name to evaluate')
    parser.add_argument('--compare-to', help='Compare to another adapter')
    parser.add_argument('--output', help='Output JSON file path')
    args = parser.parse_args()

    evaluator = Tier0Evaluator()

    if args.compare_to:
        # Comparison mode
        result = evaluator.compare_adapters(args.adapter, args.compare_to)

        print("\n" + "="*70)
        print("TIER 0 ADAPTER COMPARISON")
        print("="*70)
        print(f"\n{args.adapter}: {result['scores'][args.adapter]:.1f}/100")
        print(f"{args.compare_to}: {result['scores'][args.compare_to]:.1f}/100")
        print(f"\nWinner: {result['winner']} (+{result['score_difference']:.1f} points)")
        print("="*70)

    else:
        # Single adapter evaluation
        result = evaluator.evaluate_adapter(args.adapter)

        print("\n" + "="*70)
        print(f"TIER 0 EVALUATION: {args.adapter}")
        print("="*70)
        print(f"\n✨ Quality Score: {result['quality_score']:.1f}/100 (Grade: {result['grade']})")
        print(f"⏱️  Evaluation Time: {result['time_taken_seconds']:.3f} seconds")

        print(f"\n📊 Spectral Analysis:")
        print(f"   - Spectral Norm: {result['spectral_analysis']['avg_spectral_norm']:.3f}")
        print(f"   - Effective Rank: {result['spectral_analysis']['avg_effective_rank']:.1f}")
        print(f"   - Concentration: {result['spectral_analysis']['avg_concentration_ratio']:.3f}")

        print(f"\n📈 Training Dynamics:")
        if result['training_dynamics']['has_data']:
            print(f"   - Final Loss: {result['training_dynamics']['final_train_loss']:.3f}")
            print(f"   - Improvement: {result['training_dynamics']['loss_improvement']*100:.1f}%")
            if result['training_dynamics'].get('overfitting_gap') is not None:
                print(f"   - Overfitting Gap: {result['training_dynamics']['overfitting_gap']:.3f}")
        else:
            print("   - No training data available")

        print(f"\n⚖️  Weight Statistics:")
        print(f"   - L2 Norm: {result['weight_statistics']['l2_norm']:.3f}")
        print(f"   - Sparsity: {result['weight_statistics']['sparsity']:.3f}")
        print(f"   - Parameters: {result['weight_statistics']['num_parameters']:,}")

        if result['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in result['warnings']:
                print(f"   - {warning}")
        else:
            print(f"\n✅ No issues detected")

        print("="*70)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Report saved to: {args.output}")


if __name__ == "__main__":
    main()
