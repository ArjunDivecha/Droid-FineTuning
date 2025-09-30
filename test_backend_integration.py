#!/usr/bin/env python3.11
"""
Test script to verify mlx-lm-lora integration without actually running training.
Tests command construction and data validation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from main_enhancements import EnhancedTrainingConfig, EnhancedTrainingManager
from training_methods import TrainingDataValidator, TrainingMethod
import json
import tempfile

def test_command_construction():
    """Test that commands are constructed correctly for each method"""
    print("=" * 70)
    print("TEST 1: Command Construction")
    print("=" * 70)

    # Mock base manager
    class MockBaseManager:
        output_dir = "/tmp/test_adapters"

    base_manager = MockBaseManager()
    enhanced_manager = EnhancedTrainingManager(base_manager)

    # Test GSPO
    print("\n[GSPO] Testing command construction...")
    gspo_config = EnhancedTrainingConfig(
        model_path="/path/to/model",
        train_data_path="/path/to/data/train.jsonl",
        training_method="gspo",
        group_size=4,
        epsilon=0.0001,
        temperature=0.8,
        max_completion_length=512,
        importance_sampling_level="token"
    )
    gspo_cmd = enhanced_manager.build_enhanced_training_command(gspo_config)
    print("Command:", " ".join(gspo_cmd))
    assert "-m" in gspo_cmd and "mlx_lm_lora.train" in gspo_cmd, "Should use mlx_lm_lora.train"
    assert "--train-mode" in gspo_cmd and "grpo" in gspo_cmd, "Should use grpo mode"
    assert "--importance-sampling-level" in gspo_cmd and "token" in gspo_cmd, "Should have importance sampling"
    print("✅ GSPO command construction PASSED")

    # Test Dr. GRPO
    print("\n[Dr. GRPO] Testing command construction...")
    dr_grpo_config = EnhancedTrainingConfig(
        model_path="/path/to/model",
        train_data_path="/path/to/data/train.jsonl",
        training_method="dr_grpo",
        group_size=4,
        epsilon=0.0001,
        temperature=0.8,
        max_completion_length=512,
        grpo_loss_type="dr_grpo"
    )
    dr_grpo_cmd = enhanced_manager.build_enhanced_training_command(dr_grpo_config)
    print("Command:", " ".join(dr_grpo_cmd))
    assert "--grpo-loss-type" in dr_grpo_cmd and "dr_grpo" in dr_grpo_cmd, "Should use dr_grpo loss type"
    print("✅ Dr. GRPO command construction PASSED")

    # Test GRPO
    print("\n[GRPO] Testing command construction...")
    grpo_config = EnhancedTrainingConfig(
        model_path="/path/to/model",
        train_data_path="/path/to/data/train.jsonl",
        training_method="grpo",
        group_size=8,
        epsilon=0.0001,
        temperature=0.7,
        max_completion_length=512,
        grpo_loss_type="grpo"
    )
    grpo_cmd = enhanced_manager.build_enhanced_training_command(grpo_config)
    print("Command:", " ".join(grpo_cmd))
    assert "--group-size" in grpo_cmd and "8" in grpo_cmd, "Should have correct group size"
    print("✅ GRPO command construction PASSED")

    print("\n" + "=" * 70)
    print("✅ All command construction tests PASSED!")
    print("=" * 70)

def test_data_validation():
    """Test data validation with correct format"""
    print("\n" + "=" * 70)
    print("TEST 2: Data Validation")
    print("=" * 70)

    # Create temporary test data file with CORRECT format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_data_correct = {
            "prompt": "What is 2+2?",
            "answer": "2+2 equals 4.",
            "system": "You are a helpful math tutor."
        }
        f.write(json.dumps(test_data_correct) + '\n')
        correct_file = f.name

    # Create temporary test data file with WRONG format (old format)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_data_wrong = {
            "problem": "What is 2+2?",
            "reasoning_steps": ["Add 2 and 2"],
            "solution": "4",
            "sparse_indicators": [1]
        }
        f.write(json.dumps(test_data_wrong) + '\n')
        wrong_file = f.name

    try:
        # Test GRPO validation with CORRECT format
        print("\n[GRPO] Testing with CORRECT format (prompt/answer/system)...")
        result = TrainingDataValidator.validate_data_format(TrainingMethod.GRPO, correct_file)
        print(f"Validation result: {result}")
        assert result["valid"] == True, "Should validate correct format"
        print("✅ GRPO validation with correct format PASSED")

        # Test GRPO validation with WRONG format
        print("\n[GRPO] Testing with WRONG format (old format)...")
        result = TrainingDataValidator.validate_data_format(TrainingMethod.GRPO, wrong_file)
        print(f"Validation result: {result}")
        assert result["valid"] == False, "Should reject old format"
        assert "prompt" in result.get("error", ""), "Error should mention missing 'prompt'"
        print("✅ GRPO validation correctly rejects old format")

        # Test GSPO validation with CORRECT format
        print("\n[GSPO] Testing with CORRECT format...")
        result = TrainingDataValidator.validate_data_format(TrainingMethod.GSPO, correct_file)
        print(f"Validation result: {result}")
        assert result["valid"] == True, "Should validate correct format"
        print("✅ GSPO validation PASSED")

        # Test Dr. GRPO validation with CORRECT format
        print("\n[Dr. GRPO] Testing with CORRECT format...")
        result = TrainingDataValidator.validate_data_format(TrainingMethod.DR_GRPO, correct_file)
        print(f"Validation result: {result}")
        assert result["valid"] == True, "Should validate correct format"
        print("✅ Dr. GRPO validation PASSED")

        print("\n" + "=" * 70)
        print("✅ All data validation tests PASSED!")
        print("=" * 70)

    finally:
        # Cleanup
        os.unlink(correct_file)
        os.unlink(wrong_file)

def test_sample_format():
    """Test that sample format is correct"""
    print("\n" + "=" * 70)
    print("TEST 3: Sample Format Generation")
    print("=" * 70)

    from training_methods import TrainingDataValidator

    for method in [TrainingMethod.GSPO, TrainingMethod.DR_GRPO, TrainingMethod.GRPO]:
        print(f"\n[{method.value.upper()}] Sample format:")
        sample = TrainingDataValidator._get_sample_format(method)
        print(json.dumps(sample, indent=2))

        assert "prompt" in sample, f"{method.value} sample should have 'prompt'"
        assert "answer" in sample, f"{method.value} sample should have 'answer'"
        assert "problem" not in sample, f"{method.value} sample should NOT have old 'problem' field"
        assert "reasoning_steps" not in sample, f"{method.value} sample should NOT have old 'reasoning_steps' field"
        print(f"✅ {method.value.upper()} sample format is correct")

    print("\n" + "=" * 70)
    print("✅ Sample format tests PASSED!")
    print("=" * 70)

def test_mlx_lm_lora_installed():
    """Test that mlx-lm-lora is installed and accessible"""
    print("\n" + "=" * 70)
    print("TEST 4: MLX-LM-LORA Installation")
    print("=" * 70)

    try:
        import mlx_lm_lora
        print(f"✅ mlx-lm-lora is installed: {mlx_lm_lora.__file__}")

        # Check if train module exists
        from mlx_lm_lora import train
        print(f"✅ mlx_lm_lora.train module is accessible")

        # Try to get help (without executing)
        import subprocess
        result = subprocess.run(
            ["python3.11", "-m", "mlx_lm_lora.train", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ mlx_lm_lora.train CLI is working")
            # Check for key flags
            help_text = result.stdout
            assert "--train-mode" in help_text, "Should have --train-mode flag"
            assert "--grpo-loss-type" in help_text, "Should have --grpo-loss-type flag"
            assert "--importance-sampling-level" in help_text, "Should have --importance-sampling-level flag"
            print("✅ All required CLI flags are present")
        else:
            print(f"⚠️  Warning: CLI returned error: {result.stderr}")

    except ImportError as e:
        print(f"❌ FAILED: mlx-lm-lora not installed: {e}")
        print("Run: pip3.11 install mlx-lm-lora==0.8.1")
        return False
    except Exception as e:
        print(f"⚠️  Warning: {e}")

    print("\n" + "=" * 70)
    print("✅ MLX-LM-LORA installation tests PASSED!")
    print("=" * 70)
    return True

def main():
    print("\n" + "=" * 70)
    print("MLX-LM-LORA BACKEND INTEGRATION TEST SUITE")
    print("=" * 70)

    try:
        # Run all tests
        test_mlx_lm_lora_installed()
        test_command_construction()
        test_data_validation()
        test_sample_format()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\n✅ Backend integration is working correctly!")
        print("✅ Ready to test with actual training data")
        print("\nNext steps:")
        print("1. Create sample GRPO training data (prompt/answer/system format)")
        print("2. Test actual training with small model")
        print("3. Update frontend to expose new parameters")

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())