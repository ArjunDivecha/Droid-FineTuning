#!/usr/bin/env python3
"""
INPUT FILES:
- None (uses adapters from nested_learning/checkpoints)

OUTPUT FILES:
- test_combined_evaluation_results.json: Full evaluation results for all adapters
- test_combined_evaluation_summary.txt: Human-readable summary report

Test script for Combined Tier 0+1 Evaluation System

This script tests the combined evaluation system by:
1. Running Tier 0 (mathematical analysis) on multiple adapters
2. Running Tier 1 (perplexity analysis) on the same adapters
3. Combining scores with 40/60 weighting
4. Displaying comprehensive scoring breakdown
5. Comparing adapters side-by-side

Version: 1.0
Last Updated: 2025-11-10
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from combined_evaluator import CombinedEvaluator

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ScoringVisualizer:
    """
    Additional visualization utilities for testing.
    Uses CombinedEvaluator.print_score_card() for individual score cards.
    Provides comparison table and scoring system explanation.
    """
    
    @staticmethod
    def print_score_card(report: Dict, show_details: bool = True):
        """Print a formatted score card for an adapter."""
        adapter_name = report['adapter_name']
        combined_score = report['combined_score']
        grade = report['grade']
        tier0_score = report['tier0']['quality_score']
        tier1_score = report['tier1']['quality_score']
        
        # Score breakdown
        tier0_weighted = tier0_score * 0.4
        tier1_weighted = tier1_score * 0.6
        
        print("\n" + "="*80)
        print(f"📊 EVALUATION SCORE CARD: {adapter_name}")
        print("="*80)
        
        # Overall score (large and prominent)
        print(f"\n🏆 OVERALL SCORE: {combined_score:.1f}/100")
        print(f"   Grade: {grade}")
        print(f"   ⏱️  Total Time: {report['total_time_seconds']:.1f}s")
        
        # Score breakdown visualization
        print(f"\n📈 SCORE BREAKDOWN:")
        print(f"   ┌─────────────────────────────────────────────────────────┐")
        print(f"   │ Tier 0 (Mathematical): {tier0_score:5.1f}/100 × 40% = {tier0_weighted:5.1f} points │")
        print(f"   │ Tier 1 (Perplexity):   {tier1_score:5.1f}/100 × 60% = {tier1_weighted:5.1f} points │")
        print(f"   ├─────────────────────────────────────────────────────────┤")
        print(f"   │ Combined Score:                    {combined_score:5.1f}/100      │")
        print(f"   └─────────────────────────────────────────────────────────┘")
        
        # Visual bar chart
        print(f"\n📊 VISUAL SCORE BREAKDOWN:")
        ScoringVisualizer._print_score_bar("Tier 0 (40%)", tier0_score, tier0_weighted)
        ScoringVisualizer._print_score_bar("Tier 1 (60%)", tier1_score, tier1_weighted)
        ScoringVisualizer._print_score_bar("COMBINED", combined_score, combined_score, is_combined=True)
        
        if show_details:
            # Tier 0 details
            print(f"\n🔬 TIER 0 DETAILS (Mathematical Analysis):")
            print(f"   Spectral Norm:      {report['tier0']['spectral_norm']:.4f}")
            print(f"   Effective Rank:     {report['tier0']['effective_rank']:.1f}")
            print(f"   Concentration:      {report['tier0']['concentration']:.4f}")
            print(f"   L2 Norm:            {report['tier0']['l2_norm']:.4f}")
            print(f"   Sparsity:          {report['tier0']['sparsity']:.4f}")
            print(f"   ⏱️  Time:            {report['tier0']['time_seconds']:.2f}s")
            
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
            
            # Warnings
            if report['tier0']['warnings']:
                print(f"\n⚠️  WARNINGS:")
                for warning in report['tier0']['warnings']:
                    print(f"   • {warning}")
        
        print("="*80)
    
    @staticmethod
    def _print_score_bar(label: str, score: float, weighted_score: float, is_combined: bool = False):
        """Print a visual bar chart for a score."""
        bar_length = 50
        filled = int((score / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        if is_combined:
            print(f"   {label:20s} │{bar}│ {score:5.1f}/100")
        else:
            print(f"   {label:20s} │{bar}│ {score:5.1f}/100 (weighted: {weighted_score:5.1f})")
    
    @staticmethod
    def print_comparison_table(reports: List[Dict]):
        """Print a comparison table for multiple adapters."""
        print("\n" + "="*100)
        print("📊 COMPARISON TABLE: ALL ADAPTERS")
        print("="*100)
        
        # Sort by Tier 1 score (descending) since that's comparable across all
        sorted_reports = sorted(reports, key=lambda x: x['tier1']['quality_score'], reverse=True)
        
        # Header
        print(f"\n{'Adapter Name':<20} {'Tier 1 Score':<15} {'Tier 1 Grade':<12} {'Tier 0 Score':<15} {'Tier 0 Grade':<12} {'Time':<10}")
        print("-" * 100)
        
        # Rows
        for report in sorted_reports:
            adapter_name = report.get('adapter_name', report.get('model_name', 'Unknown'))[:18]
            tier1_score = report['tier1']['quality_score']
            tier1_grade = report['tier1'].get('grade', 'N/A')
            tier0_score = report.get('tier0', {}).get('quality_score')
            tier0_grade = report.get('tier0', {}).get('grade', 'N/A') if tier0_score else 'N/A'
            total_time = report['total_time_seconds']
            
            tier0_str = f"{tier0_score:.1f}/100" if tier0_score else "N/A"
            
            print(f"{adapter_name:<20} {tier1_score:>6.1f}/100      {tier1_grade:<12} {tier0_str:>15}  {tier0_grade:<12} {total_time:>6.1f}s")
        
        print("="*100)
        
        # Winner
        winner = sorted_reports[0]
        winner_name = winner.get('adapter_name', winner.get('model_name', 'Unknown'))
        print(f"\n🏆 WINNER (by Tier 1 Score): {winner_name} with {winner['tier1']['quality_score']:.1f}/100 (Grade: {winner['tier1'].get('grade', 'N/A')})")
    
    @staticmethod
    def print_scoring_system_explanation():
        """Print explanation of the scoring system."""
        print("\n" + "="*80)
        print("📚 SCORING SYSTEM EXPLANATION")
        print("="*80)
        print("""
The Evaluation System uses a two-tier approach with SEPARATE scores:

TIER 0 (Mathematical Analysis) - Adapters Only:
  • Analyzes adapter weights directly (no inference needed)
  • Measures spectral properties (singular values, effective rank)
  • Evaluates training dynamics (loss curves, overfitting)
  • Time: <5 seconds
  • Score Range: 0-100
  • NOT available for base models

TIER 1 (Perplexity Analysis) - Base Models AND Adapters:
  • Computes perplexity on validation data
  • Measures actual language modeling performance
  • Works for both base models and adapters
  • Time: 10-20 seconds
  • Score Range: 0-100
  • This is the COMPARABLE score between base model and adapters

SCORING:
  Each adapter gets TWO separate scores:
  - Tier 0 Score: Mathematical quality (adapters only)
  - Tier 1 Score: Perplexity-based performance (all models)
  
  Base model gets ONE score:
  - Tier 1 Score: Perplexity-based performance
  
  Compare adapters to base model using Tier 1 scores!

GRADE SCALE:
  A: 90-100  (Excellent)
  B: 80-89   (Good)
  C: 70-79   (Fair)
  D: 60-69   (Poor)
  F: 0-59    (Failing)
        """)
        print("="*80)


def test_combined_evaluation(adapter_names: List[str], include_base: bool = True, 
                             max_samples: int = 20) -> Dict:
    """
    Test evaluation on multiple adapters and optionally base model.
    
    Args:
        adapter_names: List of adapter names to test
        include_base: Whether to evaluate base model for comparison
        max_samples: Number of validation samples for Tier 1
    
    Returns:
        Dictionary with all evaluation results
    """
    print("\n" + "="*80)
    print("🧪 EVALUATION SYSTEM TEST")
    print("="*80)
    print(f"Testing {len(adapter_names)} adapters...")
    print(f"Include base model: {include_base}")
    print(f"Max validation samples: {max_samples}")
    
    evaluator = CombinedEvaluator()
    all_reports = []
    
    start_time = time.time()
    
    try:
        # Evaluate base model first if requested
        if include_base:
            print(f"\n[Base Model] Evaluating base model (Tier 1 only)...")
            try:
                base_report = evaluator.evaluate_base_model(max_samples=max_samples)
                all_reports.append(base_report)
                CombinedEvaluator.print_score_card(base_report, show_details=True)
            except Exception as e:
                print(f"❌ Error evaluating base model: {e}")
        
        # Evaluate adapters
        for i, adapter_name in enumerate(adapter_names, 1):
            print(f"\n[{i}/{len(adapter_names)}] Evaluating: {adapter_name}")
            
            try:
                report = evaluator.evaluate_adapter(
                    adapter_name,
                    include_base=False,  # We already evaluated base separately
                    max_samples=max_samples
                )
                all_reports.append(report)
                
                # Print score card using CombinedEvaluator's standard method
                CombinedEvaluator.print_score_card(report, show_details=True)
                
            except Exception as e:
                print(f"❌ Error evaluating {adapter_name}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Print comparison table
        if len(all_reports) > 1:
            ScoringVisualizer.print_comparison_table(all_reports)
        
        # Print scoring system explanation
        ScoringVisualizer.print_scoring_system_explanation()
        
        # Summary
        print(f"\n✅ Test completed in {total_time:.1f} seconds")
        print(f"   Evaluated {len(all_reports)} models successfully")
        
        return {
            'test_timestamp': datetime.now().isoformat(),
            'total_time_seconds': round(total_time, 2),
            'num_adapters_tested': len(adapter_names),
            'num_successful': len(all_reports),
            'include_base': include_base,
            'max_samples': max_samples,
            'reports': all_reports
        }
        
    finally:
        evaluator.cleanup()


def main():
    """Main test function."""
    # Default adapters to test (from nested learning checkpoints)
    default_adapters = [
        "4b-nested",
        "b1",
        "b2",
        "b3"
    ]
    
    # Check which adapters actually exist
    nested_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints"
    available_adapters = []
    
    if os.path.exists(nested_dir):
        for adapter_name in default_adapters:
            adapter_path = os.path.join(nested_dir, adapter_name)
            if os.path.exists(adapter_path):
                available_adapters.append(adapter_name)
    
    if not available_adapters:
        print("❌ No adapters found to test!")
        print(f"   Checked directory: {nested_dir}")
        return
    
    print(f"📋 Found {len(available_adapters)} adapters to test:")
    for adapter in available_adapters:
        print(f"   • {adapter}")
    
    # Run test
    results = test_combined_evaluation(
        adapter_names=available_adapters,
        include_base=True,
        max_samples=20
    )
    
    # Save results
    output_dir = Path(__file__).parent
    json_output = output_dir / "test_combined_evaluation_results.json"
    txt_output = output_dir / "test_combined_evaluation_summary.txt"
    
    # Save JSON
    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Full results saved to: {json_output}")
    
    # Save human-readable summary
    with open(txt_output, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMBINED EVALUATION TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Date: {results['test_timestamp']}\n")
        f.write(f"Total Time: {results['total_time_seconds']:.1f} seconds\n")
        f.write(f"Adapters Tested: {results['num_adapters_tested']}\n")
        f.write(f"Successful Evaluations: {results['num_successful']}\n\n")
        
        # Write score cards
        for report in results['reports']:
            f.write("\n" + "="*80 + "\n")
            f.write(f"ADAPTER: {report['adapter_name']}\n")
            f.write("="*80 + "\n")
            f.write(f"Combined Score: {report['combined_score']:.1f}/100 (Grade: {report['grade']})\n")
            f.write(f"Tier 0 Score: {report['tier0']['quality_score']:.1f}/100\n")
            f.write(f"Tier 1 Score: {report['tier1']['quality_score']:.1f}/100\n")
            f.write(f"Tier 0 Weighted: {report['tier0']['quality_score'] * 0.4:.1f}\n")
            f.write(f"Tier 1 Weighted: {report['tier1']['quality_score'] * 0.6:.1f}\n")
            f.write(f"Total Time: {report['total_time_seconds']:.1f}s\n")
            
            if 'base_model_comparison' in report:
                comp = report['base_model_comparison']
                f.write(f"\nBase Model Comparison:\n")
                f.write(f"  Perplexity Reduction: {comp['perplexity_reduction_pct']:.1f}%\n")
                f.write(f"  Quality Increase: +{comp['quality_improvement']:.1f} points\n")
    
    print(f"💾 Summary saved to: {txt_output}")


if __name__ == "__main__":
    main()

