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

    @staticmethod
    def print_score_card(report: Dict, show_details: bool = True):
        """
        Print enhanced score card with separate Tier 0 and Tier 1 scores.
        
        This is the standard scoring display format going forward.
        Shows separate scores for Tier 0 (adapters only) and Tier 1 (both base and adapters).
        """
        adapter_name = report.get('adapter_name', report.get('model_name', 'Unknown'))
        is_base_model = report.get('is_base_model', False)
        
        tier0_data = report.get('tier0')
        tier0_score = tier0_data.get('quality_score') if tier0_data else None
        tier1_score = report['tier1']['quality_score']
        tier1_grade = report['tier1'].get('grade') or CombinedEvaluator._score_to_grade(tier1_score)
        
        print("\n" + "="*80)
        if is_base_model:
            print(f"📊 EVALUATION SCORE CARD: BASE MODEL")
        else:
            print(f"📊 EVALUATION SCORE CARD: {adapter_name}")
        print("="*80)
        
        # Tier 1 score (always available)
        print(f"\n🎯 TIER 1 SCORE (Perplexity Analysis): {tier1_score:.1f}/100")
        print(f"   Grade: {tier1_grade}")
        print(f"   ⏱️  Time: {report['tier1']['time_seconds']:.1f}s")
        
        # Tier 0 score (adapters only)
        if tier0_score is not None and tier0_data:
            tier0_grade = tier0_data.get('grade') or CombinedEvaluator._score_to_grade(tier0_score)
            print(f"\n🔬 TIER 0 SCORE (Mathematical Analysis): {tier0_score:.1f}/100")
            print(f"   Grade: {tier0_grade}")
            print(f"   ⏱️  Time: {tier0_data['time_seconds']:.1f}s")
        else:
            print(f"\n🔬 TIER 0 SCORE: N/A (Base models cannot be evaluated with Tier 0)")
        
        # Visual bar chart
        print(f"\n📊 VISUAL SCORE BREAKDOWN:")
        if tier0_score is not None:
            CombinedEvaluator._print_score_bar("Tier 0 (Mathematical)", tier0_score, tier0_score)
        CombinedEvaluator._print_score_bar("Tier 1 (Perplexity)", tier1_score, tier1_score)
        
        print(f"\n⏱️  Total Time: {report['total_time_seconds']:.1f}s")
        
        if show_details:
            # Tier 0 details (only if available)
            if tier0_score is not None and tier0_data:
                print(f"\n🔬 TIER 0 DETAILS (Mathematical Analysis):")
                print(f"   Spectral Norm:      {tier0_data['spectral_norm']:.4f}")
                print(f"   Effective Rank:     {tier0_data['effective_rank']:.1f}")
                print(f"   Concentration:      {tier0_data['concentration']:.4f}")
                print(f"   L2 Norm:            {tier0_data['l2_norm']:.4f}")
                print(f"   Sparsity:          {tier0_data['sparsity']:.4f}")
                print(f"   ⏱️  Time:            {tier0_data['time_seconds']:.2f}s")
            
            # Tier 1 details
            print(f"\n🎯 TIER 1 DETAILS (Perplexity Analysis):")
            print(f"   Perplexity:         {report['tier1']['perplexity']:.4f}")
            print(f"   Avg Loss:           {report['tier1']['avg_loss']:.4f}")
            print(f"   ⏱️  Time:            {report['tier1']['time_seconds']:.2f}s")
            
            # Base model comparison if available
            if 'base_model_comparison' in report:
                comp = report['base_model_comparison']
                print(f"\n📊 BASE MODEL COMPARISON:")
                print(f"   Base Perplexity:   {comp['base_perplexity']:.4f}")
                print(f"   Adapter Perplexity: {comp['adapter_perplexity']:.4f}")
                print(f"   ✨ Improvement:     {comp['perplexity_reduction_pct']:.1f}% better")
                print(f"   Quality Increase:  +{comp['quality_improvement']:.1f} points")
            
            # Warnings (only for adapters with Tier 0)
            if tier0_score is not None and tier0_data and tier0_data.get('warnings'):
                print(f"\n⚠️  WARNINGS:")
                for warning in tier0_data['warnings']:
                    print(f"   • {warning}")
        
        print("="*80)
    
    @staticmethod
    def _print_score_bar(label: str, score: float, weighted_score: float, is_combined: bool = False):
        """Print a visual bar chart for a score."""
        bar_length = 50
        filled = int((score / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"   {label:25s} │{bar}│ {score:5.1f}/100")

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

        # Separate Tier 0 and Tier 1 scores (no combined score)
        report = {
            'adapter_name': adapter_name,
            'is_base_model': False,
            'evaluation_method': 'tier0_tier1_separate',
            'timestamp': datetime.now().isoformat(),
            'tier0': {
                'quality_score': tier0_report['quality_score'],
                'grade': tier0_report['grade'],
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
                'grade': tier1_report['grade'],
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

    def evaluate_base_model(self, max_samples: int = 20) -> Dict:
        """
        Evaluate base model using Tier 1 only (Tier 0 doesn't work for base models).
        
        Args:
            max_samples: Number of validation samples for Tier 1
            
        Returns:
            Evaluation report with Tier 1 score only
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUATING BASE MODEL (Tier 1 only)")
        logger.info(f"{'='*70}\n")
        
        tier1_report = self.tier1.evaluate(
            self.base_model_path,
            adapter_path=None,
            max_samples=max_samples
        )
        
        report = {
            'model_name': 'base_model',
            'adapter_name': 'base_model',
            'is_base_model': True,
            'evaluation_method': 'tier1_only',
            'timestamp': datetime.now().isoformat(),
            'tier0': None,  # No Tier 0 for base model
            'tier1': {
                'quality_score': tier1_report['quality_score'],
                'grade': tier1_report['grade'],
                'perplexity': tier1_report['perplexity'],
                'avg_loss': tier1_report['avg_loss'],
                'time_seconds': tier1_report['time_taken_seconds']
            },
            'total_time_seconds': round(tier1_report['time_taken_seconds'], 2)
            }

        return report

    def compare_adapters(self, adapter1: str, adapter2: str,
                        include_base: bool = False, max_samples: int = 20) -> Dict:
        """Compare two adapters using Tier 1 scores (since that's comparable)."""
        logger.info(f"\n{'='*70}")
        logger.info(f"COMPARING: {adapter1} vs {adapter2}")
        logger.info(f"{'='*70}\n")

        report1 = self.evaluate_adapter(adapter1, include_base, max_samples)
        report2 = self.evaluate_adapter(adapter2, include_base, max_samples)

        # Compare by Tier 1 score (since that's what base model has)
        tier1_1 = report1['tier1']['quality_score']
        tier1_2 = report2['tier1']['quality_score']
        
        winner = adapter1 if tier1_1 > tier1_2 else adapter2
        score_diff = abs(tier1_1 - tier1_2)

        comparison = {
            'adapter1': adapter1,
            'adapter2': adapter2,
            'winner': winner,
            'tier1_score_difference': round(score_diff, 1),
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

    @staticmethod
    def _score_to_grade(score: float) -> str:
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

            print("\n" + "="*80)
            print("📊 COMBINED EVALUATION COMPARISON")
            print("="*80)
            
            # Print score cards for both adapters
            print(f"\n{'='*80}")
            print(f"ADAPTER 1: {args.adapter}")
            print("="*80)
            CombinedEvaluator.print_score_card(result['reports'][args.adapter], show_details=True)
            
            print(f"\n{'='*80}")
            print(f"ADAPTER 2: {args.compare_to}")
            print("="*80)
            CombinedEvaluator.print_score_card(result['reports'][args.compare_to], show_details=True)
            
            # Winner announcement
            print("\n" + "="*80)
            print(f"🏆 WINNER: {result['winner']} (by Tier 1 score)")
            print(f"   Tier 1 Score Difference: +{result['tier1_score_difference']:.1f} points")
            print("="*80)

        else:
            # Single evaluation
            result = evaluator.evaluate_adapter(
                args.adapter,
                include_base=args.include_base,
                max_samples=args.max_samples
            )

            # Use enhanced score card display
            CombinedEvaluator.print_score_card(result, show_details=True)

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Report saved to: {args.output}")

    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()
