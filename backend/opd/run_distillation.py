#!/usr/bin/env python3
"""
CLI script to run On-Policy Distillation training

Usage:
    python backend/opd/run_distillation.py \\
        --teacher-path /path/to/qwen32b \\
        --student-path /path/to/qwen7b \\
        --adapter-path /path/to/adapter \\
        --prompts-path /path/to/prompts.jsonl \\
        --output-path ./OnPolicyDistill/checkpoints/my_run

Example:
    python backend/opd/run_distillation.py \\
        --teacher-path "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen3-32B-MLX-4bit" \\
        --student-path "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct" \\
        --adapter-path ./my_adapter \\
        --prompts-path ./validation_prompts.jsonl \\
        --output-path ./distilled_adapter \\
        --steps 100 \\
        --batch-size 4
"""

import argparse
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.opd.config import OPDConfig
from backend.opd.distillation_trainer import DistillationTrainer
from backend.opd.utils import (
    setup_logging,
    validate_paths,
    estimate_memory_requirements,
    print_memory_report,
    save_config,
    create_run_manifest
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run On-Policy Distillation training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--teacher-path',
        type=str,
        required=True,
        help='Path to teacher model (e.g., Qwen 32B)'
    )

    parser.add_argument(
        '--student-path',
        type=str,
        required=True,
        help='Path to student base model (e.g., Qwen 7B)'
    )

    parser.add_argument(
        '--adapter-path',
        type=str,
        required=True,
        help='Path to student LoRA adapter (from SFT training)'
    )

    parser.add_argument(
        '--prompts-path',
        type=str,
        required=True,
        help='Path to validation prompts (JSONL file)'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path where distilled adapter will be saved'
    )

    # Training parameters
    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='Number of training steps (default: 1000)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Learning rate (default: 1e-5)'
    )

    # Distillation parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=2.0,
        help='Temperature for distillation (default: 2.0)'
    )

    parser.add_argument(
        '--kl-weight',
        type=float,
        default=0.8,
        help='Weight for KL loss (default: 0.8)'
    )

    parser.add_argument(
        '--max-prompts',
        type=int,
        default=1000,
        help='Maximum prompts to use (default: 1000)'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate (default: 512)'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=100,
        help='Save checkpoint every N steps (default: 100)'
    )

    parser.add_argument(
        '--eval-every',
        type=int,
        default=100,
        help='Evaluate every N steps (default: 100)'
    )

    # Caching
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable teacher output caching'
    )

    parser.add_argument(
        '--cache-size-mb',
        type=int,
        default=4096,
        help='Cache size in MB (default: 4096)'
    )

    # Other
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run ID (default: auto-generated from timestamp)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(
        level=log_level,
        log_file=f"{args.run_id or 'distillation'}.log" if args.run_id else None
    )

    logger.info("="*70)
    logger.info("On-Policy Distillation Training")
    logger.info("="*70)

    # Create configuration
    config = OPDConfig(
        # Models
        base_model_path=args.student_path,
        teacher_model_path=args.teacher_path,
        student_adapter_path=args.adapter_path,
        output_adapter_path=args.output_path,

        # Data
        validation_prompts_path=args.prompts_path,
        max_prompts=args.max_prompts,

        # Training
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,

        # Distillation
        temperature=args.temperature,
        kl_weight=args.kl_weight,
        ce_weight=1.0 - args.kl_weight,

        # Generation
        max_generation_tokens=args.max_tokens,

        # Checkpointing
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,

        # Caching
        use_cache=not args.no_cache,
        cache_size_mb=args.cache_size_mb,

        # Other
        seed=args.seed,
        run_id=args.run_id
    )

    logger.info(f"\nConfiguration:")
    logger.info(f"  Run ID: {config.run_id}")
    logger.info(f"  Teacher: {config.teacher_model_path}")
    logger.info(f"  Student: {config.base_model_path}")
    logger.info(f"  Adapter: {config.student_adapter_path}")
    logger.info(f"  Output: {config.output_adapter_path}")
    logger.info(f"  Prompts: {config.validation_prompts_path}")
    logger.info(f"  Steps: {config.num_steps}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Temperature: {config.temperature}")

    # Validate paths
    logger.info("\nValidating paths...")
    if not validate_paths(config):
        logger.error("Path validation failed. Please check your paths.")
        return 1

    # Estimate memory requirements
    logger.info("\nEstimating memory requirements...")
    memory_est = estimate_memory_requirements(config)
    print_memory_report(memory_est)

    if not memory_est['sufficient']:
        logger.warning("⚠ Warning: May not have sufficient memory!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            return 1

    # Save configuration
    config_path = f"./OnPolicyDistill/configs/{config.run_id}.yaml"
    save_config(config, config_path)

    # Create run manifest
    manifest_path = f"./OnPolicyDistill/configs/{config.run_id}_manifest.json"
    create_run_manifest(config, manifest_path)

    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = DistillationTrainer(config)

    # Setup (load models, data, etc.)
    logger.info("\nSetting up training...")
    trainer.setup()

    # Run training
    logger.info("\nStarting training...")
    try:
        trainer.train()
        logger.info("\n✓ Training completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("\n⚠ Training interrupted by user")
        logger.info("Saving current checkpoint...")
        trainer.save_checkpoint(trainer.current_step, is_final=False)
        return 1

    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
