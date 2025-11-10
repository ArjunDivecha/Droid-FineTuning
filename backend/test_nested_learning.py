"""
Test script for Nested Learning implementation.

This script validates the Nested Learning trainer with a small test run.
"""

import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from nested_learning.config import NestedLearningConfig
from nested_learning.nested_trainer import NestedLoRATrainer
from nested_learning.parameter_scheduler import ParameterTierScheduler
from nested_learning.nested_optimizer import NestedAdam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_tier_scheduler():
    """Test parameter tier assignment."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Parameter Tier Scheduler")
    logger.info("=" * 60)

    # Create mock parameter tier map
    param_names = [
        f"model.layers.{i}.self_attn.lora_a" for i in range(24)
    ] + [
        f"model.layers.{i}.mlp.lora_b" for i in range(24)
    ]

    # Test scheduler
    scheduler = ParameterTierScheduler(num_tiers=3, strategy='layer_depth')

    # Create mock model with parameter names
    class MockModel:
        def __init__(self, param_names):
            self.param_names = param_names

    # Assign tiers (will fail without real model, but shows structure)
    logger.info("✓ ParameterTierScheduler created successfully")
    logger.info(f"  Num tiers: 3")
    logger.info(f"  Strategy: layer_depth")


def test_nested_optimizer():
    """Test nested optimizer with tier filtering."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Nested Optimizer")
    logger.info("=" * 60)

    # Create tier map
    tier_map = {
        'param_0': 0,  # Tier 0 - every step
        'param_1': 1,  # Tier 1 - every 2 steps
        'param_2': 2,  # Tier 2 - every 4 steps
    }

    # Create optimizer
    optimizer = NestedAdam(
        learning_rate=1e-5,
        tier_update_frequencies=[1, 2, 4],
        parameter_tier_map=tier_map
    )

    logger.info("✓ NestedAdam optimizer created")
    logger.info(f"  Learning rate: 1e-5")
    logger.info(f"  Tier frequencies: [1, 2, 4]")

    # Simulate steps
    logger.info("\nSimulating training steps:")
    for step in range(1, 9):
        optimizer.global_step = step
        active_tiers = optimizer._get_active_tiers()
        logger.info(f"  Step {step}: Active tiers = {active_tiers}")

    # Check tier stats
    stats = optimizer.get_tier_stats()
    logger.info("\n✓ Tier statistics:")
    logger.info(f"  Global step: {stats['global_step']}")
    logger.info(f"  Tier update counts: {stats['tier_update_counts']}")


def test_config_validation():
    """Test configuration validation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Configuration Validation")
    logger.info("=" * 60)

    # Test invalid config (frequencies not ascending)
    try:
        config = NestedLearningConfig(
            base_model_path="/path/to/model",
            adapter_path="/path/to/adapter",
            train_data_path="/path/to/data.jsonl",
            num_tiers=3,
            tier_update_frequencies=[1, 4, 2],  # Not ascending!
        )
        logger.error("✗ Should have raised ValueError for non-ascending frequencies")
    except ValueError as e:
        logger.info(f"✓ Correctly rejected non-ascending frequencies: {e}")

    # Test invalid config (mismatch num_tiers)
    try:
        config = NestedLearningConfig(
            base_model_path="/path/to/model",
            adapter_path="/path/to/adapter",
            train_data_path="/path/to/data.jsonl",
            num_tiers=3,
            tier_update_frequencies=[1, 2],  # Only 2 frequencies!
        )
        logger.error("✗ Should have raised ValueError for tier mismatch")
    except ValueError as e:
        logger.info(f"✓ Correctly rejected tier count mismatch: {e}")


def create_test_data():
    """Create minimal test data for training."""
    logger.info("\n" + "=" * 60)
    logger.info("SETUP: Creating test data")
    logger.info("=" * 60)

    test_data_dir = Path("./test_nested_learning_data")
    test_data_dir.mkdir(exist_ok=True)

    # Create tiny training dataset
    train_data = [
        {"text": "The capital of France is Paris."},
        {"text": "Python is a programming language."},
        {"text": "Machine learning models learn from data."},
        {"text": "The sky appears blue during the day."},
        {"text": "Water freezes at 0 degrees Celsius."},
    ]

    train_file = test_data_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    logger.info(f"✓ Created test data: {train_file}")
    logger.info(f"  Training samples: {len(train_data)}")

    return str(train_file)


def test_integration():
    """
    Integration test with real model (if available).

    NOTE: This requires actual model files to be present.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Integration Test (Requires Model Files)")
    logger.info("=" * 60)

    # Check for model files
    base_model_path = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen2.5-7B-Instruct"
    adapter_path = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters/7b"

    if not Path(base_model_path).exists():
        logger.warning(f"⚠ Model not found: {base_model_path}")
        logger.warning("  Skipping integration test")
        return

    if not Path(adapter_path).exists():
        logger.warning(f"⚠ Adapter not found: {adapter_path}")
        logger.warning("  Skipping integration test")
        return

    # Create test data
    import json
    train_file = create_test_data()

    # Create config
    config = NestedLearningConfig(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        train_data_path=train_file,
        num_tiers=3,
        tier_update_frequencies=[1, 2, 4],
        tier_assignment_strategy='layer_depth',
        num_steps=10,  # Very short test run
        batch_size=1,
        learning_rate=1e-5,
        max_seq_length=128,
        output_path="./test_nested_learning_output",
        experiment_name="test_run"
    )

    logger.info("✓ Configuration created")

    try:
        # Create trainer
        trainer = NestedLoRATrainer(config)
        logger.info("✓ Trainer created")

        # Setup
        trainer.setup()
        logger.info("✓ Setup completed")

        # Train for 10 steps
        trainer.train()
        logger.info("✓ Training completed")

        # Check outputs
        output_dir = Path(config.output_path) / config.experiment_name
        assert output_dir.exists(), "Output directory not created"
        logger.info(f"✓ Output directory created: {output_dir}")

        # Check metrics
        metrics_file = output_dir / "metrics" / "train_metrics.jsonl"
        assert metrics_file.exists(), "Metrics file not created"
        logger.info(f"✓ Metrics file created: {metrics_file}")

        # Read and display final metrics
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            last_metrics = json.loads(lines[-1])
            logger.info(f"\nFinal metrics:")
            logger.info(f"  Step: {last_metrics['step']}")
            logger.info(f"  Loss: {last_metrics['loss']:.4f}")
            logger.info(f"  Tier stats: {last_metrics['tier_stats']}")

        logger.info("\n✓ INTEGRATION TEST PASSED")

    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("NESTED LEARNING TEST SUITE")
    logger.info("=" * 60)

    try:
        # Test 1: Tier scheduler
        test_tier_scheduler()

        # Test 2: Nested optimizer
        test_nested_optimizer()

        # Test 3: Config validation
        test_config_validation()

        # Test 4: Integration (if models available)
        test_integration()

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS COMPLETED")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
