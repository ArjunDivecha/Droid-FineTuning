#!/usr/bin/env python3
"""
Nested Learning CLI

Command-line interface for running nested learning experiments.
All parameters from NestedLearningConfig are exposed as command-line arguments.

Example usage:
    python run_nested_learning_cli.py \\
        --base-model-path /path/to/model \\
        --train-data-path /path/to/train.jsonl \\
        --adapter-path /path/to/adapter \\
        --num-tiers 3 \\
        --tier-update-frequencies 1 2 4 \\
        --max-seq-length 128 \\
        --batch-size 1 \\
        --num-steps 1000

For testing loops:
    for lr in 1e-5 1e-4; do
        for bs in 1 2 4; do
            python run_nested_learning_cli.py \\
                --base-model-path /path/to/model \\
                --train-data-path /path/to/train.jsonl \\
                --learning-rate $lr \\
                --batch-size $bs \\
                --experiment-name "test_lr${lr}_bs${bs}"
        done
    done
"""

import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Nested Learning CLI - Run nested learning experiments from command line',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python run_nested_learning_cli.py \\
      --base-model-path /path/to/model \\
      --train-data-path /path/to/train.jsonl

  # Full configuration
  python run_nested_learning_cli.py \\
      --base-model-path /path/to/model \\
      --adapter-path /path/to/adapter \\
      --train-data-path /path/to/train.jsonl \\
      --val-data-path /path/to/val.jsonl \\
      --num-tiers 3 \\
      --tier-update-frequencies 1 2 4 \\
      --tier-assignment-strategy layer_depth \\
      --learning-rate 1e-5 \\
      --batch-size 1 \\
      --num-steps 1000 \\
      --max-seq-length 128 \\
      --lora-rank 8 \\
      --lora-alpha 16 \\
      --experiment-name my_experiment

  # Memory-efficient small model training
  python run_nested_learning_cli.py \\
      --base-model-path /path/to/small_model \\
      --train-data-path /path/to/train.jsonl \\
      --max-seq-length 128 \\
      --batch-size 1 \\
      --num-steps 500
        """
    )

    # Required arguments
    required = parser.add_argument_group('Required Arguments')
    required.add_argument(
        '--base-model-path',
        type=str,
        required=True,
        help='Path to base model directory'
    )
    required.add_argument(
        '--train-data-path',
        type=str,
        required=True,
        help='Path to training data (JSONL format)'
    )

    # Model & Data paths
    paths = parser.add_argument_group('Model & Data Paths')
    paths.add_argument(
        '--adapter-path',
        type=str,
        default='',
        help='Path to LoRA adapter (optional, empty means train from base model)'
    )
    paths.add_argument(
        '--val-data-path',
        type=str,
        default=None,
        help='Path to validation data (optional, JSONL format)'
    )

    # Nested Learning parameters
    nested = parser.add_argument_group('Nested Learning Configuration')
    nested.add_argument(
        '--num-tiers',
        type=int,
        default=3,
        help='Number of parameter tiers (default: 3)'
    )
    nested.add_argument(
        '--tier-update-frequencies',
        type=int,
        nargs='+',
        default=[1, 2, 4],
        help='Update frequency for each tier (e.g., 1 2 4 means tier0 updates every step, tier1 every 2 steps, tier2 every 4 steps). Must be in ascending order and match num-tiers. (default: 1 2 4)'
    )
    nested.add_argument(
        '--tier-assignment-strategy',
        type=str,
        choices=['layer_depth', 'parameter_importance', 'manual'],
        default='layer_depth',
        help='Strategy for assigning parameters to tiers (default: layer_depth)'
    )

    # Training parameters
    training = parser.add_argument_group('Training Parameters')
    training.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Learning rate (default: 1e-5)'
    )
    training.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size - CRITICAL: Keep at 1 to prevent memory explosion (default: 1)'
    )
    training.add_argument(
        '--num-steps',
        type=int,
        default=1000,
        help='Total number of training steps (default: 1000)'
    )
    training.add_argument(
        '--max-seq-length',
        type=int,
        default=128,
        help='Maximum sequence length - CRITICAL: Must be 128 or less for nested learning memory management (default: 128)'
    )

    # LoRA configuration
    lora = parser.add_argument_group('LoRA Configuration')
    lora.add_argument(
        '--lora-rank',
        type=int,
        default=8,
        help='LoRA rank (default: 8)'
    )
    lora.add_argument(
        '--lora-alpha',
        type=int,
        default=16,
        help='LoRA alpha scaling parameter (default: 16)'
    )
    lora.add_argument(
        '--lora-dropout',
        type=float,
        default=0.0,
        help='LoRA dropout rate (default: 0.0)'
    )

    # Advanced training settings
    advanced = parser.add_argument_group('Advanced Training Settings')
    advanced.add_argument(
        '--warmup-steps',
        type=int,
        default=100,
        help='Number of warmup steps for learning rate schedule (default: 100)'
    )
    advanced.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=2,
        help='Number of gradient accumulation steps (default: 2)'
    )
    advanced.add_argument(
        '--checkpoint-every',
        type=int,
        default=100,
        help='Save checkpoint every N steps (default: 100)'
    )
    advanced.add_argument(
        '--eval-every',
        type=int,
        default=100,
        help='Run evaluation every N steps (default: 100)'
    )
    advanced.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0,
        help='Maximum gradient norm for clipping (default: 1.0)'
    )

    # Early stopping
    early_stop = parser.add_argument_group('Early Stopping')
    early_stop.add_argument(
        '--early-stop',
        action='store_true',
        default=True,
        help='Enable early stopping (default: enabled)'
    )
    early_stop.add_argument(
        '--no-early-stop',
        action='store_false',
        dest='early_stop',
        help='Disable early stopping'
    )
    early_stop.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Number of eval cycles without improvement before stopping (default: 5)'
    )
    early_stop.add_argument(
        '--min-delta',
        type=float,
        default=0.0001,
        help='Minimum change in loss to qualify as improvement (default: 0.0001)'
    )

    # Output configuration
    output = parser.add_argument_group('Output Configuration')
    output.add_argument(
        '--output-path',
        type=str,
        default='./nested_learning/checkpoints',
        help='Base output directory for checkpoints (default: ./nested_learning/checkpoints)'
    )
    output.add_argument(
        '--experiment-name',
        type=str,
        default='nested_learning_experiment',
        help='Name for this experiment (creates subdirectory in output-path) (default: nested_learning_experiment)'
    )
    output.add_argument(
        '--save-best-only',
        action='store_true',
        default=False,
        help='Only save the best checkpoint (default: disabled)'
    )
    output.add_argument(
        '--keep-last-n-checkpoints',
        type=int,
        default=5,
        help='Keep only the last N checkpoints (default: 5)'
    )

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    misc.add_argument(
        '--mixed-precision',
        action='store_true',
        default=True,
        help='Use mixed precision training (default: enabled)'
    )
    misc.add_argument(
        '--no-mixed-precision',
        action='store_false',
        dest='mixed_precision',
        help='Disable mixed precision training'
    )

    # Validation mode
    misc.add_argument(
        '--validate-only',
        action='store_true',
        default=False,
        help='Only validate configuration without training (useful for testing parameter combinations)'
    )

    return parser.parse_args()


def validate_config(args):
    """Validate configuration before training."""
    logger.info("Validating configuration...")

    # Validate tier configuration
    if len(args.tier_update_frequencies) != args.num_tiers:
        logger.error(
            f"ERROR: Length of tier-update-frequencies ({len(args.tier_update_frequencies)}) "
            f"must match num-tiers ({args.num_tiers})"
        )
        return False

    # Validate frequencies are in ascending order
    if not all(args.tier_update_frequencies[i] <= args.tier_update_frequencies[i + 1]
               for i in range(len(args.tier_update_frequencies) - 1)):
        logger.error("ERROR: tier-update-frequencies must be in ascending order (e.g., 1 2 4)")
        return False

    # Validate paths exist
    if not Path(args.base_model_path).exists():
        logger.error(f"ERROR: Base model path not found: {args.base_model_path}")
        return False

    if args.adapter_path and not Path(args.adapter_path).exists():
        logger.error(f"ERROR: Adapter path not found: {args.adapter_path}")
        return False

    if not Path(args.train_data_path).exists():
        logger.error(f"ERROR: Training data path not found: {args.train_data_path}")
        return False

    if args.val_data_path and not Path(args.val_data_path).exists():
        logger.error(f"ERROR: Validation data path not found: {args.val_data_path}")
        return False

    # Memory warnings
    if args.max_seq_length > 128:
        logger.warning(
            f"WARNING: max-seq-length is {args.max_seq_length} > 128. "
            "This may cause significant memory usage in nested learning!"
        )

    if args.batch_size > 1:
        logger.warning(
            f"WARNING: batch-size is {args.batch_size} > 1. "
            "This may cause memory issues. Recommended: batch-size=1"
        )

    logger.info("✓ Configuration validated successfully")
    return True


def print_config_summary(args):
    """Print configuration summary."""
    logger.info("="*70)
    logger.info("NESTED LEARNING CONFIGURATION")
    logger.info("="*70)

    logger.info("\n📁 Paths:")
    logger.info(f"  Base Model: {args.base_model_path}")
    if args.adapter_path:
        logger.info(f"  Adapter: {args.adapter_path}")
    logger.info(f"  Training Data: {args.train_data_path}")
    if args.val_data_path:
        logger.info(f"  Validation Data: {args.val_data_path}")
    logger.info(f"  Output: {args.output_path}/{args.experiment_name}")

    logger.info("\n🔧 Nested Learning:")
    logger.info(f"  Tiers: {args.num_tiers}")
    logger.info(f"  Update Frequencies: {args.tier_update_frequencies}")
    logger.info(f"  Assignment Strategy: {args.tier_assignment_strategy}")

    logger.info("\n🎯 Training:")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Steps: {args.num_steps}")
    logger.info(f"  Max Sequence Length: {args.max_seq_length}")
    logger.info(f"  Gradient Accumulation: {args.gradient_accumulation_steps}")

    logger.info("\n🎛️ LoRA:")
    logger.info(f"  Rank: {args.lora_rank}")
    logger.info(f"  Alpha: {args.lora_alpha}")
    logger.info(f"  Dropout: {args.lora_dropout}")

    logger.info("\n📊 Checkpointing & Evaluation:")
    logger.info(f"  Checkpoint Every: {args.checkpoint_every} steps")
    logger.info(f"  Eval Every: {args.eval_every} steps")
    logger.info(f"  Keep Last N Checkpoints: {args.keep_last_n_checkpoints}")
    logger.info(f"  Save Best Only: {args.save_best_only}")

    logger.info("\n⏸️ Early Stopping:")
    logger.info(f"  Enabled: {args.early_stop}")
    if args.early_stop:
        logger.info(f"  Patience: {args.patience}")
        logger.info(f"  Min Delta: {args.min_delta}")

    logger.info("\n⚙️ Other:")
    logger.info(f"  Warmup Steps: {args.warmup_steps}")
    logger.info(f"  Max Grad Norm: {args.max_grad_norm}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Mixed Precision: {args.mixed_precision}")

    logger.info("\n" + "="*70)


def run_training(args):
    """Run nested learning training."""
    from nested_learning.config import NestedLearningConfig
    from nested_learning.nested_trainer import NestedLoRATrainer

    # Create config from args
    config = NestedLearningConfig(
        # Paths
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,

        # Nested Learning
        num_tiers=args.num_tiers,
        tier_update_frequencies=args.tier_update_frequencies,
        tier_assignment_strategy=args.tier_assignment_strategy,

        # Training
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        max_seq_length=args.max_seq_length,

        # LoRA
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,

        # Advanced
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        max_grad_norm=args.max_grad_norm,

        # Early stopping
        early_stop=args.early_stop,
        patience=args.patience,
        min_delta=args.min_delta,

        # Output
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        save_best_only=args.save_best_only,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,

        # Misc
        seed=args.seed,
        mixed_precision=args.mixed_precision,
    )

    # Create trainer
    logger.info("\nCreating trainer...")
    trainer = NestedLoRATrainer(config)
    logger.info("✓ Trainer created")

    # Setup
    logger.info("\nRunning setup...")
    try:
        trainer.setup()
        logger.info("✓ Setup completed")
    except Exception as e:
        logger.error(f"✗ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Train
    logger.info("\n" + "="*70)
    logger.info(f"STARTING TRAINING ({args.num_steps} steps)")
    logger.info("="*70)
    try:
        trainer.train()
        logger.info("\n" + "="*70)
        logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*70)
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Print output location
    output_dir = Path(config.output_path) / config.experiment_name
    logger.info(f"\n📂 Output saved to: {output_dir}")
    logger.info("\nNested Learning completed successfully! 🎉")

    return True


def main():
    """Main entry point."""
    args = parse_args()

    # Print configuration
    print_config_summary(args)

    # Validate configuration
    if not validate_config(args):
        logger.error("\n❌ Configuration validation failed!")
        sys.exit(1)

    # If validate-only mode, exit here
    if args.validate_only:
        logger.info("\n✓ Validation complete (--validate-only mode, not training)")
        sys.exit(0)

    # Run training
    success = run_training(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
