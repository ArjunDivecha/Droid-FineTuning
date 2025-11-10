"""
Quick test to verify nested learning imports work correctly.
Run this from the backend directory to test.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all nested learning imports work."""
    logger.info("Testing nested learning imports...")

    try:
        from nested_learning.config import NestedLearningConfig
        logger.info("✓ Config import successful")
    except Exception as e:
        logger.error(f"✗ Config import failed: {e}")
        return False

    try:
        from nested_learning.nested_optimizer import NestedAdam
        logger.info("✓ Optimizer import successful")
    except Exception as e:
        logger.error(f"✗ Optimizer import failed: {e}")
        return False

    try:
        from nested_learning.parameter_scheduler import ParameterTierScheduler
        logger.info("✓ Scheduler import successful")
    except Exception as e:
        logger.error(f"✗ Scheduler import failed: {e}")
        return False

    try:
        from nested_learning.nested_trainer import NestedLoRATrainer
        logger.info("✓ Trainer import successful")
    except Exception as e:
        logger.error(f"✗ Trainer import failed: {e}")
        return False

    logger.info("\n✓ All imports successful!")
    return True

def test_config_creation():
    """Test creating a config object."""
    logger.info("\nTesting config creation...")

    try:
        from nested_learning.config import NestedLearningConfig

        # Create config with mock paths (won't validate yet)
        config = NestedLearningConfig(
            base_model_path="",  # Empty paths to skip validation
            adapter_path="",
            train_data_path="",
            num_tiers=3,
            tier_update_frequencies=[1, 2, 4],
            tier_assignment_strategy='layer_depth',
            num_steps=10
        )

        logger.info("✓ Config object created successfully")
        logger.info(f"  Tiers: {config.num_tiers}")
        logger.info(f"  Frequencies: {config.tier_update_frequencies}")
        return True

    except Exception as e:
        logger.error(f"✗ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimizer_creation():
    """Test creating optimizer."""
    logger.info("\nTesting optimizer creation...")

    try:
        from nested_learning.nested_optimizer import NestedAdam

        tier_map = {
            'param_0': 0,
            'param_1': 1,
            'param_2': 2
        }

        optimizer = NestedAdam(
            learning_rate=1e-5,
            tier_update_frequencies=[1, 2, 4],
            parameter_tier_map=tier_map
        )

        logger.info("✓ Optimizer created successfully")
        logger.info(f"  LR: {optimizer.learning_rate}")
        logger.info(f"  Tiers: {optimizer.num_tiers}")
        return True

    except Exception as e:
        logger.error(f"✗ Optimizer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("NESTED LEARNING IMPORT TEST")
    logger.info("="*60)

    success = True

    # Test imports
    if not test_imports():
        success = False

    # Test config creation
    if not test_config_creation():
        success = False

    # Test optimizer creation
    if not test_optimizer_creation():
        success = False

    logger.info("\n" + "="*60)
    if success:
        logger.info("✓ ALL TESTS PASSED")
        logger.info("Nested learning modules are ready to use!")
    else:
        logger.info("✗ SOME TESTS FAILED")
        logger.info("Check errors above")
    logger.info("="*60)

    sys.exit(0 if success else 1)
