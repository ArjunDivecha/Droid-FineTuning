#!/usr/bin/env python3
"""
Combined Tier 0+1 Evaluator - Best of Both Worlds

Combines:
- Tier 0: Instant mathematical analysis (<5s)
- Tier 1: Fast perplexity measurement (10-20s)

Usage:
    python combined_evaluator.py --adapter 4B
    python combined_evaluator.py --adapter 4B --compare-to 4b-nested
    python combined_evaluator.py --adapter 4B --include-base
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime

# Import tier evaluators
from tier0_evaluator import Tier0Evaluator
from tier1_evaluator import Tier1Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CombinedEvaluator:
    """
    Combines Tier 0 (mathematical) and Tier 1 (perplexity) evaluation.

    Tier 0: Instant (<5s) - spectral analysis, training dynamics
    Tier 1: Fast (10-20s) - perplexity on validation data

    Together provides comprehensive quality assessment.
    """

    def __init__(self):
        self.tier0 = Tier0Evaluator()
        self.tier1 = Tier1Evaluator()
        self.base_model_path = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/qwen3-4b-mlx"

    def evaluate_adapter(self, adapter_name: str, include_base: bool = False,
                        max_samples: int = 20) -> Dict:
        """
        Evaluate adapter using both Tier 0 and Tier 1.

        Args:
            adapter_name: Name of adapter to evaluate
            include_base: Also evaluate base model for comparison
            max_samples: Number of validation samples for Tier 1

        Returns:
            Combined evaluation report
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"COMBINED EVALUATION: {adapter_name}")
        logger.info(f"{'='*70}\n")

        # Tier 0: Mathematical analysis (instant)
        logger.info("Running Tier 0 (Mathematical Analysis)...")
        tier0_report = self.tier0.evaluate_adapter(adapter_name)

        # Tier 1: Perplexity (fast)
        logger.info("\nRunning Tier 1 (Perplexity Analysis)...")

        # Get adapter path
        adapter_path = self._get_adapter_path(adapter_name)

        if include_base:
            # Compare to base model
            tier1_comparison = self.tier1.compare_to_base(
                self.base_model_path,
                adapter_path,
                max_samples=max_samples
            )
            tier1_report = tier1_comparison['adapter']
            base_report = tier1_comparison['base_model']
            improvement = tier1_comparison['improvement']
        else:
            # Just evaluate adapter
            tier1_report = self.tier1.evaluate(
                self.base_model_path,
                adapter_path=adapter_path,
                max_samples=max_samples
            )
            base_report = None
            improvement = None

        # Compute combined score
        # Tier 0: 40% weight (mathematical properties)
        # Tier 1: 60% weight (actual performance)
        combined_score = (tier0_report['quality_score'] * 0.4 +
                         tier1_report['quality_score'] * 0.6)

        report = {
            'adapter_name': adapter_name,
            'evaluation_method': 'combined_tier0_tier1',
            'timestamp': datetime.now().isoformat(),
            'combined_score': round(combined_score, 1),
            'grade': self._score_to_grade(combined_score),
            'tier0': {
                'quality_score': tier0_report['quality_score'],
                'spectral_norm': tier0_report['spectral_analysis']['spectral_norm'],
                'effective_rank': tier0_report['spectral_analysis']['effective_rank'],
                'concentration': tier0_report['spectral_analysis']['concentration_ratio'],
                'l2_norm': tier0_report['weight_statistics']['l2_norm'],
                'sparsity': tier0_report['weight_statistics']['sparsity'],
                'warnings': tier0_report['warnings'],
                'time_seconds': tier0_report['time_taken_seconds']
            },
            'tier1': {
                'quality_score': tier1_report['quality_score'],
                'perplexity': tier1_report['perplexity'],
                'avg_loss': tier1_report['avg_loss'],
                'time_seconds': tier1_report['time_taken_seconds']
            },
            'total_time_seconds': round(
                tier0_report['time_taken_seconds'] + tier1_report['time_taken_seconds'], 2
            )
        }

        if base_report:
            report['base_model_comparison'] = {
                'base_perplexity': base_report['perplexity'],
                'adapter_perplexity': tier1_report['perplexity'],
                'perplexity_reduction_pct': improvement['perplexity_reduction_pct'],
                'quality_improvement': improvement['quality_score_increase']
            }

        return report

    def compare_adapters(self, adapter1: str, adapter2: str,
                        include_base: bool = False, max_samples: int = 20) -> Dict:
        """Compare two adapters using combined evaluation."""
        logger.info(f"\n{'='*70}")
        logger.info(f"COMPARING: {adapter1} vs {adapter2}")
        logger.info(f"{'='*70}\n")

        report1 = self.evaluate_adapter(adapter1, include_base, max_samples)
        report2 = self.evaluate_adapter(adapter2, include_base, max_samples)

        winner = adapter1 if report1['combined_score'] > report2['combined_score'] else adapter2
        score_diff = abs(report1['combined_score'] - report2['combined_score'])

        comparison = {
            'adapter1': adapter1,
            'adapter2': adapter2,
            'winner': winner,
            'score_difference': round(score_diff, 1),
            'reports': {
                adapter1: report1,
                adapter2: report2
            }
        }

        return comparison

    def _get_adapter_path(self, adapter_name: str) -> str:
        """Get full path to adapter."""
        # Check nested learning
        nested_path = f"/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints/{adapter_name}"
        if os.path.exists(nested_path):
            return nested_path

        # Check regular adapters
        regular_path = f"/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters/{adapter_name}"
        if os.path.exists(regular_path):
            return regular_path

        raise FileNotFoundError(f"Adapter not found: {adapter_name}")

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

    def cleanup(self):
        """Free resources."""
        self.tier1.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Combined Tier 0+1 Evaluator')
    parser.add_argument('--adapter', required=True, help='Adapter name')
    parser.add_argument('--compare-to', help='Compare to another adapter')
    parser.add_argument('--include-base', action='store_true',
                       help='Include base model comparison')
    parser.add_argument('--max-samples', type=int, default=20,
                       help='Number of validation samples for Tier 1')
    parser.add_argument('--output', help='Output JSON file')
    args = parser.parse_args()

    evaluator = CombinedEvaluator()

    try:
        if args.compare_to:
            # Comparison mode
            result = evaluator.compare_adapters(
                args.adapter,
                args.compare_to,
                include_base=args.include_base,
                max_samples=args.max_samples
            )

            print("\n" + "="*70)
            print("COMBINED EVALUATION COMPARISON")
            print("="*70)
            print(f"\n{args.adapter}:")
            print(f"  Combined Score: {result['reports'][args.adapter]['combined_score']}/100 "
                  f"(Grade: {result['reports'][args.adapter]['grade']})")
            print(f"  Tier 0: {result['reports'][args.adapter]['tier0']['quality_score']}/100")
            print(f"  Tier 1: {result['reports'][args.adapter]['tier1']['quality_score']}/100 "
                  f"(perplexity: {result['reports'][args.adapter]['tier1']['perplexity']:.2f})")

            print(f"\n{args.compare_to}:")
            print(f"  Combined Score: {result['reports'][args.compare_to]['combined_score']}/100 "
                  f"(Grade: {result['reports'][args.compare_to]['grade']})")
            print(f"  Tier 0: {result['reports'][args.compare_to]['tier0']['quality_score']}/100")
            print(f"  Tier 1: {result['reports'][args.compare_to]['tier1']['quality_score']}/100 "
                  f"(perplexity: {result['reports'][args.compare_to]['tier1']['perplexity']:.2f})")

            print(f"\n✨ Winner: {result['winner']} (+{result['score_difference']:.1f} points)")
            print("="*70)

        else:
            # Single evaluation
            result = evaluator.evaluate_adapter(
                args.adapter,
                include_base=args.include_base,
                max_samples=args.max_samples
            )

            print("\n" + "="*70)
            print(f"COMBINED EVALUATION: {args.adapter}")
            print("="*70)
            print(f"\n🏆 Combined Score: {result['combined_score']}/100 (Grade: {result['grade']})")
            print(f"⏱️  Total Time: {result['total_time_seconds']:.1f}s")

            print(f"\n📊 Tier 0 (Mathematical Analysis):")
            print(f"   Score: {result['tier0']['quality_score']}/100")
            print(f"   Spectral Norm: {result['tier0']['spectral_norm']:.3f}")
            print(f"   Effective Rank: {result['tier0']['effective_rank']:.1f}")
            print(f"   Concentration: {result['tier0']['concentration']:.3f}")
            print(f"   Time: {result['tier0']['time_seconds']:.1f}s")

            print(f"\n📊 Tier 1 (Perplexity Analysis):")
            print(f"   Score: {result['tier1']['quality_score']}/100")
            print(f"   Perplexity: {result['tier1']['perplexity']:.2f}")
            print(f"   Avg Loss: {result['tier1']['avg_loss']:.4f}")
            print(f"   Time: {result['tier1']['time_seconds']:.1f}s")

            if 'base_model_comparison' in result:
                comp = result['base_model_comparison']
                print(f"\n✨ vs Base Model:")
                print(f"   Base Perplexity: {comp['base_perplexity']:.2f}")
                print(f"   Adapter Perplexity: {comp['adapter_perplexity']:.2f}")
                print(f"   Improvement: {comp['perplexity_reduction_pct']:.1f}% better")

            if result['tier0']['warnings']:
                print(f"\n⚠️  Warnings:")
                for warning in result['tier0']['warnings']:
                    print(f"   - {warning}")

            print("="*70)

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Report saved to: {args.output}")

    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()
