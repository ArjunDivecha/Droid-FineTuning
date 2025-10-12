# backend/training_methods.py
# Enhanced training methods configuration for MLX-LM-LORA v0.8.1
# Integrates with existing Droid-FineTuning architecture

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
import os
import logging

logger = logging.getLogger(__name__)

class TrainingMethod(str, Enum):
    """Available training methods"""
    DPO = "dpo"  # Direct Preference Optimization
    ORPO = "orpo"  # Odds Ratio Preference Optimization

@dataclass
class TrainingMethodConfig:
    """Configuration for a training method"""
    name: str
    display_name: str
    description: str
    complexity: str
    use_case: str
    data_format: str
    requires_preferences: bool = False
    requires_reasoning_chains: bool = False
    supports_batch: bool = True
    resource_intensity: str = "medium"  # low, medium, high, very_high
    estimated_speedup: Optional[str] = None
    badge: Optional[str] = None
    module_name: str = "mlx_lm.lora"  # Default MLX module
    additional_params: List[str] = None

    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = []

# Official MLX-LM Alignment Methods
TRAINING_METHODS = {
    TrainingMethod.DPO: TrainingMethodConfig(
        name="dpo",
        display_name="Direct Preference Optimization (DPO)",
        description="A stable and efficient method for aligning models with human or AI preferences.",
        complexity="⭐⭐⭐",
        use_case="Improving model behavior, style, and safety based on preference pairs.",
        data_format="preference",
        requires_preferences=True,
        supports_batch=True,
        resource_intensity="high",
        badge="Industry Standard",
        module_name="mlx_lm.dpo",
        additional_params=["beta"]
    ),
    
    TrainingMethod.ORPO: TrainingMethodConfig(
        name="orpo",
        display_name="Odds Ratio Preference Optimization (ORPO)",
        description="A newer, simpler, and often more effective alignment method than DPO.",
        complexity="⭐⭐⭐",
        use_case="Simultaneously fine-tuning and aligning a model, improving both instruction-following and preference alignment.",
        data_format="preference",
        requires_preferences=True,
        supports_batch=True,
        resource_intensity="high",
        badge="🆕 State-of-the-Art",
        module_name="mlx_lm.orpo",
        additional_params=["beta", "alpha"]
    )
}

class TrainingDataValidator:
    """Validates training data formats for different methods"""
    
    @staticmethod
    def validate_data_format(method: TrainingMethod, data_path: str) -> Dict[str, Any]:
        """Validate data format for specific training method"""
        config = TRAINING_METHODS.get(method)
        if not config:
            return {"valid": False, "error": f"Unknown training method: {method}"}
        
        try:
            if not os.path.exists(data_path):
                return {"valid": False, "error": f"Data file not found: {data_path}"}
            
            # Read all lines to check format and count samples
            valid_samples = 0
            total_lines = 0
            errors = []
            
            detected_format = None

            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    total_lines += 1
                    
                    try:
                        data_sample = json.loads(line)
                        
                        # Validate this sample
                        if config.requires_preferences:
                            validation = TrainingDataValidator._validate_preference_data(data_sample, method)
                        else:
                            # Fallback for any non-preference formats if added in the future
                            validation = TrainingDataValidator._validate_instruction_data(data_sample, method)
                        
                        if validation.get("valid"):
                            valid_samples += 1
                            if not detected_format and validation.get("format"):
                                detected_format = validation["format"]
                        else:
                            errors.append(f"Line {line_num}: {validation.get('error')}")
                            
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
            
            if total_lines == 0:
                return {"valid": False, "error": "Empty data file"}
            
            if valid_samples == 0:
                error_summary = ". ".join(errors[:3])  # Show first 3 errors
                return {"valid": False, "error": f"No valid samples found. {error_summary}"}
            
            # Return success with sample count
            result = {
                "valid": True,
                "num_samples": valid_samples,
                "total_lines": total_lines,
                "format": detected_format or ("preference" if config.requires_preferences else "instruction_following")
            }
            
            if errors:
                result["warnings"] = f"{len(errors)} samples had errors"
            
            return result
                
        except Exception as e:
            logger.error(f"Data validation failed for {method}: {str(e)}")
            return {"valid": False, "error": f"Data validation failed: {str(e)}"}
    

    
    @staticmethod
    def _validate_preference_data(data_sample: Dict, method: TrainingMethod) -> Dict[str, Any]:
        """Validate preference data format (for future methods)"""
        required_fields = ["prompt", "chosen", "rejected"]
        missing_fields = [field for field in required_fields if field not in data_sample]
        
        if missing_fields:
            return {
                "valid": False,
                "error": f"Missing required fields for preferences: {missing_fields}",
                "required_format": required_fields
            }
        
        return {"valid": True, "format": "preference_pairs"}
    
    @staticmethod
    def _validate_instruction_data(data_sample: Dict, method: TrainingMethod) -> Dict[str, Any]:
        """Validate instruction-response data format"""
        # Support both instruction/response and messages format
        if "messages" in data_sample:
            # Chat format
            if not isinstance(data_sample["messages"], list):
                return {
                    "valid": False,
                    "error": "messages must be a list",
                    "required_format": ["messages"]
                }
            return {"valid": True, "format": "chat_messages"}
        else:
            # Instruction format
            required_fields = ["instruction", "response"]
            missing_fields = [field for field in required_fields if field not in data_sample]
            
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required fields: {missing_fields}",
                    "required_format": required_fields,
                    "note": "Supports both instruction/response format or messages format"
                }
        
        return {"valid": True, "format": "instruction_response"}
    
    @staticmethod
    def _get_sample_format(method: TrainingMethod) -> Dict[str, Any]:
        """Get sample data format for a method"""
        if method in [TrainingMethod.DPO, TrainingMethod.ORPO]:
            return {
                "prompt": "What are the main benefits of using MLX for machine learning on Apple silicon?",
                "chosen": "MLX offers several key advantages: 1) Unified Memory, which avoids data copies between CPU and GPU, 2) Lazy Computation, which only materializes arrays when needed, optimizing performance, and 3) a familiar NumPy-like API, making it easy to adopt.",
                "rejected": "MLX is just another framework, it doesn't do much."
            }
        else: # Fallback for future methods
            return {
                "instruction": "Your instruction here",
                "response": "Expected response here"
            }

class ResourceEstimator:
    """Estimates resource requirements for different training methods"""
    
    @staticmethod
    def estimate_requirements(method: TrainingMethod, model_size: str, dataset_size: int) -> Dict[str, Any]:
        """Estimate memory, time, and compute requirements"""
        config = TRAINING_METHODS.get(method)
        if not config:
            return {"error": f"Unknown method: {method}"}
        
        # Base requirements
        base_memory = ResourceEstimator._get_base_memory(model_size)
        
        # Method-specific multipliers
        method_multipliers = {
            TrainingMethod.SFT: {"memory": 1.0, "time": 1.0},
            TrainingMethod.GSPO: {"memory": 1.2, "time": 0.5},  # More memory, much faster
            TrainingMethod.DR_GRPO: {"memory": 1.5, "time": 1.3},  # More memory and time
            TrainingMethod.GRPO: {"memory": 1.3, "time": 1.0}
        }
        
        multiplier = method_multipliers.get(method, {"memory": 1.0, "time": 1.0})
        
        estimated_memory = base_memory * multiplier["memory"]
        estimated_time = (dataset_size / 1000) * multiplier["time"]  # hours per 1k samples
        
        return {
            "method": method.value,
            "estimated_memory_gb": round(estimated_memory, 1),
            "estimated_time_hours": round(estimated_time, 1),
            "resource_intensity": config.resource_intensity,
            "recommendations": ResourceEstimator._get_recommendations(method, estimated_memory)
        }
    
    @staticmethod
    def _get_base_memory(model_size: str) -> float:
        """Get base memory requirements by model size"""
        size_map = {
            "7B": 8.0, "13B": 16.0, "30B": 32.0,
            "70B": 64.0, "80B": 72.0, "3B": 4.0,
            "1B": 2.0, "500M": 1.0
        }
        
        # Extract size from model name/path
        for size, memory in size_map.items():
            if size in model_size.upper():
                return memory
        
        return 8.0  # Default fallback
    
    @staticmethod
    def _get_recommendations(method: TrainingMethod, memory_gb: float) -> List[str]:
        """Get optimization recommendations"""
        recommendations = []
        
        if memory_gb > 32:
            recommendations.append("Consider using a quantized model (4-bit) to reduce memory usage")
        
        if method in [TrainingMethod.DPO, TrainingMethod.ORPO]:
            recommendations.append("Ensure your dataset contains high-quality preference pairs.")
            recommendations.append("Start with the default beta value (0.1) and tune if needed.")
        
        if memory_gb > 16:
            recommendations.append("Close other applications to free up memory")
        
        return recommendations

# Export main components
__all__ = [
    "TrainingMethod",
    "TrainingMethodConfig", 
    "TRAINING_METHODS",
    "TrainingDataValidator",
    "ResourceEstimator"
]