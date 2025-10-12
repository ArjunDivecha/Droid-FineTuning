#!/usr/bin/env python3
"""Test Milestone 1: TrainingConfig updates"""

from main import TrainingConfig

def test_training_config():
    """Test TrainingConfig has new LoRA fields with correct defaults"""
    
    config = TrainingConfig(
        model_path="/test/path",
        train_data_path="/train/path",
        val_data_path="/val/path"
    )
    
    # Test new fields exist
    assert hasattr(config, 'fine_tune_type'), "❌ Missing fine_tune_type"
    assert hasattr(config, 'lora_rank'), "❌ Missing lora_rank"
    assert hasattr(config, 'lora_alpha'), "❌ Missing lora_alpha"
    assert hasattr(config, 'lora_dropout'), "❌ Missing lora_dropout"
    assert hasattr(config, 'lora_num_layers'), "❌ Missing lora_num_layers"
    
    # Test default values
    assert config.learning_rate == 1e-4, f"❌ Wrong learning_rate: {config.learning_rate}, expected 1e-4"
    assert config.fine_tune_type == "lora", f"❌ Wrong fine_tune_type: {config.fine_tune_type}"
    assert config.lora_rank == 32, f"❌ Wrong lora_rank: {config.lora_rank}"
    assert config.lora_alpha == 32.0, f"❌ Wrong lora_alpha: {config.lora_alpha}"
    assert config.lora_dropout == 0.0, f"❌ Wrong lora_dropout: {config.lora_dropout}"
    assert config.lora_num_layers == -1, f"❌ Wrong lora_num_layers: {config.lora_num_layers}"
    
    print("✅ All TrainingConfig tests passed!")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   LoRA Rank: {config.lora_rank}")
    print(f"   LoRA Alpha: {config.lora_alpha}")
    print(f"   LoRA Dropout: {config.lora_dropout}")
    print(f"   LoRA Layers: {config.lora_num_layers}")
    print(f"   Fine Tune Type: {config.fine_tune_type}")
    
    return True

if __name__ == "__main__":
    try:
        test_training_config()
        print("\n🎉 MILESTONE 1 COMPLETE: TrainingConfig updated successfully")
    except Exception as e:
        print(f"\n❌ MILESTONE 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
