"""
Full End-to-End Test for Nested Learning

This script performs a complete test of the nested learning system with real parameters.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_test():
    """Run complete nested learning test."""

    logger.info("="*70)
    logger.info("NESTED LEARNING END-TO-END TEST")
    logger.info("="*70)

    # Import modules
    from nested_learning.config import NestedLearningConfig
    from nested_learning.nested_trainer import NestedLoRATrainer

    # Test parameters
    base_model_path = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct"
    adapter_path = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters/7b"
    train_data_path = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/test_data_nested_learning.jsonl"

    # Verify paths exist
    logger.info("\nVerifying paths...")
    if not Path(base_model_path).exists():
        logger.error(f"✗ Base model not found: {base_model_path}")
        return False
    logger.info(f"✓ Base model found: {base_model_path}")

    if not Path(adapter_path).exists():
        logger.error(f"✗ Adapter not found: {adapter_path}")
        return False
    logger.info(f"✓ Adapter found: {adapter_path}")

    if not Path(train_data_path).exists():
        logger.error(f"✗ Training data not found: {train_data_path}")
        return False
    logger.info(f"✓ Training data found: {train_data_path}")

    # Create configuration
    logger.info("\nCreating configuration...")
    config = NestedLearningConfig(
        # Model & data
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        train_data_path=train_data_path,
        val_data_path=None,  # Will use train/val split

        # Nested Learning (3 tiers with exponential frequencies)
        num_tiers=3,
        tier_update_frequencies=[1, 2, 4],
        tier_assignment_strategy='layer_depth',

        # Training (SHORT TEST RUN)
        learning_rate=1e-5,
        batch_size=2,
        num_steps=20,  # Just 20 steps for quick test
        max_seq_length=512,  # Short sequences

        # LoRA
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,

        # Advanced
        warmup_steps=5,
        gradient_accumulation_steps=1,
        checkpoint_every=10,
        eval_every=10,

        # Output
        output_path="./test_nested_learning_output",
        experiment_name="test_run_complete"
    )

    logger.info("✓ Configuration created")
    logger.info(f"  Model: Qwen2.5-7B-Instruct")
    logger.info(f"  Adapter: 7b")
    logger.info(f"  Tiers: {config.num_tiers}")
    logger.info(f"  Frequencies: {config.tier_update_frequencies}")
    logger.info(f"  Steps: {config.num_steps}")
    logger.info(f"  Batch size: {config.batch_size}")

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
    logger.info("STARTING TRAINING (20 steps)")
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

    # Verify outputs
    logger.info("\nVerifying outputs...")
    output_dir = Path(config.output_path) / config.experiment_name

    if not output_dir.exists():
        logger.error(f"✗ Output directory not created: {output_dir}")
        return False
    logger.info(f"✓ Output directory: {output_dir}")

    # Check for checkpoints
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        logger.error(f"✗ Checkpoint directory not created")
        return False

    final_checkpoint = checkpoint_dir / "final"
    if not final_checkpoint.exists():
        logger.error(f"✗ Final checkpoint not created")
        return False
    logger.info(f"✓ Final checkpoint: {final_checkpoint}")

    # Check for metrics
    metrics_dir = output_dir / "metrics"
    train_metrics = metrics_dir / "train_metrics.jsonl"
    if not train_metrics.exists():
        logger.error(f"✗ Training metrics not created")
        return False
    logger.info(f"✓ Training metrics: {train_metrics}")

    # Read and display final metrics
    import json
    with open(train_metrics, 'r') as f:
        lines = f.readlines()
        if len(lines) > 0:
            last_metrics = json.loads(lines[-1])
            logger.info("\n" + "="*70)
            logger.info("FINAL METRICS")
            logger.info("="*70)
            logger.info(f"  Step: {last_metrics['step']}")
            logger.info(f"  Loss: {last_metrics['loss']:.4f}")
            logger.info(f"  Step time: {last_metrics['step_time']:.2f}s")

            if 'tier_stats' in last_metrics:
                logger.info("\n  Tier Statistics:")
                tier_stats = last_metrics['tier_stats']
                for tier_name, tier_info in tier_stats.get('tier_parameters', {}).items():
                    logger.info(f"    {tier_name}:")
                    logger.info(f"      Updates: {tier_info['update_count']}")
                    logger.info(f"      Frequency: every {tier_info['frequency']} steps")
                    logger.info(f"      Parameters: {tier_info['parameter_count']}")

    logger.info("\n" + "="*70)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("="*70)
    logger.info(f"\nOutput location: {output_dir}")
    logger.info("\nNested Learning is working correctly! 🎉")

    return True

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
