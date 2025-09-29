#!/usr/bin/env python3
"""
Test script for GSPO and Dr. GRPO integration
"""

import sys
import os
import asyncio
import logging
import json

# Add the backend to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from training_methods import (
    TrainingMethod, 
    TRAINING_METHODS, 
    TrainingDataValidator, 
    ResourceEstimator
)

# Import only what we need for testing without external dependencies
try:
    from main_enhancements import EnhancedTrainingManager
    ENHANCED_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: EnhancedTrainingManager not available: {e}")
    ENHANCED_MANAGER_AVAILABLE = False

# Mock TrainingManager for testing
class MockTrainingManager:
    def __init__(self):
        self.output_dir = "/tmp/test_adapters"
        self.current_config = None
        self.training_state = "idle"
        self.training_metrics = {}
        self.current_process = None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

def test_training_methods():
    """Test that all training methods are properly configured"""
    print("🧪 Testing training methods configuration...")
    
    # Test that all methods are available
    assert TrainingMethod.SFT in TRAINING_METHODS
    assert TrainingMethod.GSPO in TRAINING_METHODS
    assert TrainingMethod.DR_GRPO in TRAINING_METHODS
    assert TrainingMethod.GRPO in TRAINING_METHODS
    
    # Test GSPO configuration
    gspo_config = TRAINING_METHODS[TrainingMethod.GSPO]
    assert gspo_config.display_name == "Group Sparse Policy Optimization"
    assert gspo_config.requires_reasoning_chains == True
    assert "sparse_ratio" in gspo_config.additional_params
    assert gspo_config.estimated_speedup == "2x faster than GRPO"
    
    # Test Dr. GRPO configuration
    dr_grpo_config = TRAINING_METHODS[TrainingMethod.DR_GRPO]
    assert dr_grpo_config.display_name == "Doctor GRPO"
    assert dr_grpo_config.requires_reasoning_chains == True
    assert "domain" in dr_grpo_config.additional_params
    assert dr_grpo_config.resource_intensity == "high"
    
    print("✅ All training methods configured correctly")

def test_resource_estimation():
    """Test resource estimation functionality"""
    print("🧪 Testing resource estimation...")
    
    # Test GSPO estimation
    gspo_estimation = ResourceEstimator.estimate_requirements(
        TrainingMethod.GSPO, "7B", 1000
    )
    assert "method" in gspo_estimation
    assert gspo_estimation["method"] == "gspo"
    assert gspo_estimation["estimated_memory_gb"] > 0
    assert gspo_estimation["estimated_time_hours"] > 0
    assert len(gspo_estimation["recommendations"]) > 0
    
    # Test Dr. GRPO estimation
    dr_grpo_estimation = ResourceEstimator.estimate_requirements(
        TrainingMethod.DR_GRPO, "7B", 1000
    )
    assert dr_grpo_estimation["estimated_memory_gb"] > gspo_estimation["estimated_memory_gb"]
    assert "domain-specific" in " ".join(dr_grpo_estimation["recommendations"]).lower()
    
    print("✅ Resource estimation working correctly")

def test_enhanced_training_manager():
    """Test enhanced training manager functionality"""
    print("🧪 Testing enhanced training manager...")
    
    if not ENHANCED_MANAGER_AVAILABLE:
        print("⚠️ Skipping enhanced training manager test (dependencies not available)")
        return
    
    # Create mock base manager
    base_manager = MockTrainingManager()
    enhanced_manager = EnhancedTrainingManager(base_manager)
    
    # Test get available methods
    methods = enhanced_manager.get_available_methods()
    assert "gspo" in methods
    assert "dr_grpo" in methods
    assert "grpo" in methods
    assert "sft" in methods
    
    # Test GSPO method details
    gspo_method = methods["gspo"]
    assert gspo_method["badge"] == "🆕 Most Efficient"
    assert gspo_method["estimated_speedup"] == "2x faster than GRPO"
    
    print("✅ Enhanced training manager working correctly")

def test_sample_data_generation():
    """Test sample data generation for different methods"""
    print("🧪 Testing sample data generation...")
    
    if not ENHANCED_MANAGER_AVAILABLE:
        print("⚠️ Skipping sample data generation test (dependencies not available)")
        return
    
    base_manager = MockTrainingManager()
    enhanced_manager = EnhancedTrainingManager(base_manager)
    
    # Test GSPO sample data
    gspo_result = enhanced_manager.generate_sample_data(
        "gspo", "/tmp/test_gspo_data.jsonl", 5
    )
    assert gspo_result["success"] == True
    assert gspo_result["method"] == "gspo"
    assert gspo_result["sample_count"] == 5
    
    # Verify GSPO data format
    if os.path.exists(gspo_result["output_path"]):
        with open(gspo_result["output_path"], 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                sample = json.loads(first_line)
                assert "sparse_indicators" in sample
                assert "efficiency_markers" in sample
                assert "reasoning_steps" in sample
        os.remove(gspo_result["output_path"])
    
    # Test Dr. GRPO sample data
    dr_grpo_result = enhanced_manager.generate_sample_data(
        "dr_grpo", "/tmp/test_dr_grpo_data.jsonl", 3
    )
    assert dr_grpo_result["success"] == True
    assert dr_grpo_result["method"] == "dr_grpo"
    
    # Verify Dr. GRPO data format
    if os.path.exists(dr_grpo_result["output_path"]):
        with open(dr_grpo_result["output_path"], 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                sample = json.loads(first_line)
                assert "domain" in sample
                assert "expertise_level" in sample
                assert "domain_context" in sample
        os.remove(dr_grpo_result["output_path"])
    
    print("✅ Sample data generation working correctly")

def test_data_validation():
    """Test data format validation"""
    print("🧪 Testing data validation...")
    
    # Create test data files
    gspo_test_data = {
        "problem": "Test optimization problem",
        "reasoning_steps": ["Step 1", "Step 2", "Step 3"],
        "solution": "Test solution",
        "sparse_indicators": [1, 1, 0],
        "efficiency_markers": {"optimization_applied": True}
    }
    
    test_file = "/tmp/test_gspo_validation.jsonl"
    with open(test_file, 'w') as f:
        f.write(json.dumps(gspo_test_data) + "\n")
    
    # Test GSPO validation
    gspo_validation = TrainingDataValidator.validate_data_format(
        TrainingMethod.GSPO, test_file
    )
    assert gspo_validation["valid"] == True
    assert gspo_validation["format"] == "reasoning_chains"
    
    # Clean up
    os.remove(test_file)
    
    print("✅ Data validation working correctly")

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting GSPO and Dr. GRPO integration tests...\n")
    
    try:
        test_training_methods()
        test_resource_estimation()
        test_enhanced_training_manager()
        test_sample_data_generation()
        test_data_validation()
        
        print("\n🎉 All integration tests passed!")
        print("✨ GSPO and Dr. GRPO integration is working correctly")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)