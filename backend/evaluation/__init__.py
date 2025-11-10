"""
Revolutionary Multi-Tier Adapter Evaluation System

This module provides fast, deterministic, scientifically-sound evaluation
of LoRA adapters without relying on LLM judges.

Tier 1: Intrinsic Analysis (< 1s) - SVD/spectral analysis of adapter weights
Tier 2: Fast Metrics (5-10s) - Perplexity and loss on validation set
Tier 3: Thorough Metrics (1-2min) - BLEU/ROUGE/semantic similarity
Tier 4: Comparison Mode (< 1s) - Cached multi-adapter comparison
"""

from .revolutionary_evaluator import RevolutionaryAdapterEvaluator
from .tier1_intrinsic import IntrinsicAdapterAnalyzer
from .tier2_fast import FastDeterministicEvaluator
from .tier3_thorough import ThoroughStatisticalEvaluator
from .tier4_comparison import AdapterComparisonEngine

__all__ = [
    'RevolutionaryAdapterEvaluator',
    'IntrinsicAdapterAnalyzer',
    'FastDeterministicEvaluator',
    'ThoroughStatisticalEvaluator',
    'AdapterComparisonEngine'
]

__version__ = '1.0.0'
