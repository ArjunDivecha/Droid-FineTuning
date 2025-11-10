#!/usr/bin/env python3
"""
Tier 1 Evaluator - Perplexity Analysis (FAST - 10-30 seconds)

Computes perplexity on validation data to measure model quality.
Works for BOTH base models and adapters using the same metric.

Perplexity = exp(average_loss)
- Lower perplexity = better language modeling
- Deterministic (no sampling)
- Fast (single forward pass, no generation)

Usage:
    python tier1_evaluator.py --model qwen3-4b-mlx  # Base model
    python tier1_evaluator.py --model qwen3-4b-mlx --adapter 4B  # With adapter
    python tier1_evaluator.py --model qwen3-4b-mlx --adapter 4B --compare-base
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    HAS_MLX = True
except ImportError:
    logger.error("MLX not available - Tier 1 requires MLX")
    HAS_MLX = False
    sys.exit(1)


class Tier1Evaluator:
    """
    Fast perplexity-based evaluation.

    Computes perplexity on validation set to measure language modeling quality.
    Works for both base models and adapters.
    """

    def __init__(self):
        self.model_cache = {}  # Cache loaded models

    def load_validation_data(self, data_path: str, max_samples: int = 50) -> List[str]:
        """Load validation data from JSONL file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Validation data not found: {data_path}")

        texts = []
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                if line.strip():
                    data = json.loads(line)

                    # Extract text from different formats
                    if 'text' in data:
                        texts.append(data['text'])
                    elif 'messages' in data:
                        # Convert messages to text
                        messages = data['messages']
                        text_parts = []
                        for msg in messages:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                        texts.append('\n'.join(text_parts))

        logger.info(f"Loaded {len(texts)} validation samples")
        return texts

    def get_model(self, model_path: str, adapter_path: Optional[str] = None) -> Tuple:
        """Get or load model (cached)."""
        cache_key = (model_path, adapter_path)

        if cache_key not in self.model_cache:
            logger.info(f"Loading model: {model_path}")
            if adapter_path:
                logger.info(f"  with adapter: {adapter_path}")

                # Handle nested learning adapters
                if "nested_learning/checkpoints" in adapter_path:
                    best_checkpoint = os.path.join(adapter_path, "checkpoints", "best")
                    if os.path.exists(best_checkpoint):
                        adapter_path = best_checkpoint
                        logger.info(f"  using nested best: {best_checkpoint}")

            start = time.time()
            model, tokenizer = load(model_path, adapter_path=adapter_path)
            load_time = time.time() - start
            logger.info(f"Model loaded in {load_time:.1f}s")

            self.model_cache[cache_key] = (model, tokenizer)
        else:
            logger.info("Using cached model")

        return self.model_cache[cache_key]

    def compute_perplexity(self, model, tokenizer, texts: List[str]) -> Dict:
        """
        Compute perplexity on text samples.

        Perplexity = exp(average_cross_entropy_loss)
        """
        total_loss = 0.0
        total_tokens = 0

        logger.info(f"Computing perplexity on {len(texts)} samples...")

        for i, text in enumerate(texts):
            # Tokenize
            tokens = tokenizer.encode(text)

            if len(tokens) < 2:
                continue

            # Truncate if too long
            max_length = 2048
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            # Convert to MLX array
            input_ids = mx.array([tokens[:-1]])  # All but last token
            target_ids = mx.array([tokens[1:]])   # All but first token

            # Forward pass (no generation, just compute loss)
            try:
                logits = model(input_ids)

                # Compute cross-entropy loss
                # logits: (batch, seq_len, vocab_size)
                # target_ids: (batch, seq_len)

                # Reshape for loss computation
                logits_flat = logits.reshape(-1, logits.shape[-1])
                targets_flat = target_ids.reshape(-1)

                # Cross entropy: -log(p(correct_token))
                # Use nn.losses.cross_entropy
                loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')

                total_loss += float(loss) * len(tokens)
                total_tokens += len(tokens)

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(texts)} samples")

        if total_tokens == 0:
            raise ValueError("No valid tokens processed")

        # Compute average loss and perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'num_samples': len(texts)
        }

    def evaluate(self, model_path: str, adapter_path: Optional[str] = None,
                validation_data: str = None, max_samples: int = 50) -> Dict:
        """
        Evaluate model or adapter using perplexity.

        Args:
            model_path: Path to base model
            adapter_path: Optional path to adapter
            validation_data: Path to validation JSONL
            max_samples: Number of validation samples to use

        Returns:
            Evaluation report with perplexity score
        """
        start_time = time.time()

        # Determine validation data path
        if validation_data is None:
            # Try to find it automatically
            if adapter_path:
                # Check adapter config
                nested_config = f"/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints/{os.path.basename(adapter_path)}/config.json"
                regular_config = f"{adapter_path}/adapter_config.json"

                if os.path.exists(nested_config):
                    with open(nested_config) as f:
                        config = json.load(f)
                        validation_data = config.get('train_data_path')
                elif os.path.exists(regular_config):
                    with open(regular_config) as f:
                        config = json.load(f)
                        validation_data = config.get('data')
                        if validation_data and os.path.isdir(validation_data):
                            validation_data = os.path.join(validation_data, "train.jsonl")

            if not validation_data:
                # Use default
                validation_data = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/one_step_finetune/data/train.jsonl"

        logger.info(f"Tier 1 Evaluation")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Adapter: {adapter_path if adapter_path else 'None (base model)'}")
        logger.info(f"  Validation data: {validation_data}")

        # Load validation data
        texts = self.load_validation_data(validation_data, max_samples)

        # Load model
        model, tokenizer = self.get_model(model_path, adapter_path)

        # Compute perplexity
        perplexity_results = self.compute_perplexity(model, tokenizer, texts)

        elapsed_time = time.time() - start_time

        # Compute quality score (perplexity-based)
        # Lower perplexity = higher quality
        # Scale: perplexity 1 = 100, perplexity 100 = 0
        quality_score = max(0, min(100, 100 - (perplexity_results['perplexity'] - 1)))

        report = {
            'tier': 1,
            'evaluation_method': 'perplexity',
            'model_path': model_path,
            'adapter_path': adapter_path,
            'adapter_name': os.path.basename(adapter_path) if adapter_path else 'base',
            'is_base_model': adapter_path is None,
            'perplexity': round(perplexity_results['perplexity'], 2),
            'avg_loss': round(perplexity_results['avg_loss'], 4),
            'quality_score': round(quality_score, 1),
            'grade': self._score_to_grade(quality_score),
            'total_tokens': perplexity_results['total_tokens'],
            'num_samples': perplexity_results['num_samples'],
            'time_taken_seconds': round(elapsed_time, 2),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Evaluation complete in {elapsed_time:.1f}s")
        logger.info(f"  Perplexity: {report['perplexity']:.2f}")
        logger.info(f"  Quality Score: {report['quality_score']:.1f}/100")

        return report

    def compare_to_base(self, model_path: str, adapter_path: str,
                       validation_data: str = None, max_samples: int = 50) -> Dict:
        """
        Compare adapter to base model.

        Returns comparison with improvement percentage.
        """
        logger.info("Comparing adapter to base model...")

        # Evaluate base model
        base_report = self.evaluate(model_path, adapter_path=None,
                                   validation_data=validation_data,
                                   max_samples=max_samples)

        # Evaluate adapter
        adapter_report = self.evaluate(model_path, adapter_path=adapter_path,
                                      validation_data=validation_data,
                                      max_samples=max_samples)

        # Compute improvement
        perplexity_improvement = (
            (base_report['perplexity'] - adapter_report['perplexity']) /
            base_report['perplexity'] * 100
        )

        comparison = {
            'base_model': base_report,
            'adapter': adapter_report,
            'improvement': {
                'perplexity_reduction_pct': round(perplexity_improvement, 1),
                'quality_score_increase': round(
                    adapter_report['quality_score'] - base_report['quality_score'], 1
                ),
                'better': adapter_report['perplexity'] < base_report['perplexity']
            }
        }

        return comparison

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
        """Free model memory."""
        self.model_cache.clear()
        import gc
        gc.collect()
        logger.info("Model cache cleared")


def main():
    parser = argparse.ArgumentParser(description='Tier 1 Adapter Evaluator - Perplexity Analysis')
    parser.add_argument('--model', required=True, help='Base model path')
    parser.add_argument('--adapter', help='Adapter path (omit for base model evaluation)')
    parser.add_argument('--validation-data', help='Validation JSONL file')
    parser.add_argument('--max-samples', type=int, default=50, help='Max validation samples')
    parser.add_argument('--compare-base', action='store_true', help='Compare adapter to base model')
    parser.add_argument('--output', help='Output JSON file path')
    args = parser.parse_args()

    evaluator = Tier1Evaluator()

    try:
        if args.compare_base and args.adapter:
            # Comparison mode
            result = evaluator.compare_to_base(
                args.model,
                args.adapter,
                validation_data=args.validation_data,
                max_samples=args.max_samples
            )

            print("\n" + "="*70)
            print("TIER 1 COMPARISON: ADAPTER VS BASE MODEL")
            print("="*70)
            print(f"\n📊 Base Model:")
            print(f"   Perplexity: {result['base_model']['perplexity']:.2f}")
            print(f"   Quality Score: {result['base_model']['quality_score']:.1f}/100")
            print(f"\n📊 Adapter ({os.path.basename(args.adapter)}):")
            print(f"   Perplexity: {result['adapter']['perplexity']:.2f}")
            print(f"   Quality Score: {result['adapter']['quality_score']:.1f}/100")
            print(f"\n✨ Improvement:")
            print(f"   Perplexity reduced by: {result['improvement']['perplexity_reduction_pct']:.1f}%")
            print(f"   Quality score increased by: {result['improvement']['quality_score_increase']:.1f} points")

            if result['improvement']['better']:
                print(f"   ✅ Adapter is BETTER than base model")
            else:
                print(f"   ❌ Adapter is WORSE than base model")

            print("="*70)

        else:
            # Single evaluation
            result = evaluator.evaluate(
                args.model,
                adapter_path=args.adapter,
                validation_data=args.validation_data,
                max_samples=args.max_samples
            )

            print("\n" + "="*70)
            if result['is_base_model']:
                print("TIER 1 EVALUATION: BASE MODEL")
            else:
                print(f"TIER 1 EVALUATION: {result['adapter_name']}")
            print("="*70)
            print(f"\n📊 Perplexity: {result['perplexity']:.2f}")
            print(f"✨ Quality Score: {result['quality_score']:.1f}/100 (Grade: {result['grade']})")
            print(f"📈 Average Loss: {result['avg_loss']:.4f}")
            print(f"⏱️  Evaluation Time: {result['time_taken_seconds']:.1f}s")
            print(f"📝 Samples: {result['num_samples']} ({result['total_tokens']:,} tokens)")
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
