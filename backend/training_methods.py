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
    SFT = "sft"
    GSPO = "gspo"  # Group Sparse Policy Optimization
    DR_GRPO = "dr_grpo"  # Doctor GRPO
    GRPO = "grpo"  # Group Relative Policy Optimization

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

# Enhanced training methods with GSPO and Dr. GRPO
TRAINING_METHODS = {
    TrainingMethod.SFT: TrainingMethodConfig(
        name="sft",
        display_name="Supervised Fine-Tuning",
        description="Standard instruction following fine-tuning with LoRA adapters",
        complexity="⭐⭐",
        use_case="General instruction following and task adaptation",
        data_format="instruction_response",
        requires_preferences=False,
        requires_reasoning_chains=False,
        supports_batch=True,
        resource_intensity="medium",
        module_name="mlx_lm.lora"
    ),
    
    TrainingMethod.GSPO: TrainingMethodConfig(
        name="gspo",
        display_name="Group Sparse Policy Optimization",
        description="Latest breakthrough in efficient reasoning model training with sparse optimization",
        complexity="⭐⭐⭐⭐",
        use_case="Efficient reasoning tasks with resource constraints",
        data_format="reasoning_chains",
        requires_preferences=False,
        requires_reasoning_chains=True,
        supports_batch=True,
        resource_intensity="medium",  # More efficient than GRPO
        estimated_speedup="2x faster than GRPO",
        badge="🆕 Most Efficient",
        module_name="mlx_lm_lora.gspo",
        additional_params=["sparse_ratio", "efficiency_threshold", "sparse_optimization"]
    ),
    
    TrainingMethod.DR_GRPO: TrainingMethodConfig(
        name="dr_grpo",
        display_name="Doctor GRPO",
        description="Domain-specialized reasoning for expert knowledge applications",
        complexity="⭐⭐⭐⭐⭐",
        use_case="Medical, scientific, and specialized domain reasoning",
        data_format="domain_reasoning_chains",
        requires_preferences=False,
        requires_reasoning_chains=True,
        supports_batch=True,
        resource_intensity="high",
        badge="🆕 Domain Expert",
        module_name="mlx_lm_lora.dr_grpo",
        additional_params=["domain", "expertise_level", "domain_adaptation_strength"]
    ),
    
    TrainingMethod.GRPO: TrainingMethodConfig(
        name="grpo",
        display_name="Group Relative Policy Optimization",
        description="DeepSeek-R1 style multi-step reasoning capabilities",
        complexity="⭐⭐⭐⭐",
        use_case="Complex multi-step reasoning and problem solving",
        data_format="reasoning_chains",
        requires_preferences=False,
        requires_reasoning_chains=True,
        supports_batch=True,
        resource_intensity="high",
        module_name="mlx_lm_lora.grpo",
        additional_params=["reasoning_steps", "multi_step_training"]
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
            
            # Read first line to check format
            with open(data_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    return {"valid": False, "error": "Empty data file"}
                
                try:
                    data_sample = json.loads(first_line)
                except json.JSONDecodeError as e:
                    return {"valid": False, "error": f"Invalid JSON format: {str(e)}"}
            
            # Method-specific validation
            if config.requires_reasoning_chains:
                return TrainingDataValidator._validate_reasoning_data(data_sample, method)
            elif config.requires_preferences:
                return TrainingDataValidator._validate_preference_data(data_sample, method)
            else:
                return TrainingDataValidator._validate_instruction_data(data_sample, method)
                
        except Exception as e:
            logger.error(f"Data validation failed for {method}: {str(e)}")
            return {"valid": False, "error": f"Data validation failed: {str(e)}"}
    
    @staticmethod
    def _validate_reasoning_data(data_sample: Dict, method: TrainingMethod) -> Dict[str, Any]:
        """Validate GRPO/GSPO/Dr.GRPO data format (prompt/answer/system)"""
        # All GRPO-based methods use the same format:
        # Required: "prompt" and "answer"
        # Optional: "system"

        required_fields = ["prompt", "answer"]
        missing_fields = [field for field in required_fields if field not in data_sample]

        if missing_fields:
            return {
                "valid": False,
                "error": f"Missing required fields for {method.value}: {missing_fields}. Format: {{\"prompt\": \"...\", \"answer\": \"...\", \"system\": \"...\" (optional)}}",
                "required_format": required_fields,
                "sample_format": TrainingDataValidator._get_sample_format(method)
            }

        # Validate types
        if not isinstance(data_sample.get("prompt"), str):
            return {
                "valid": False,
                "error": "prompt must be a string",
                "required_format": required_fields
            }

        if not isinstance(data_sample.get("answer"), str):
            return {
                "valid": False,
                "error": "answer must be a string",
                "required_format": required_fields
            }

        # System message is optional but should be string if present
        if "system" in data_sample and not isinstance(data_sample["system"], str):
            return {
                "valid": False,
                "error": "system message must be a string if provided",
                "required_format": required_fields
            }

        return {"valid": True, "format": "grpo_prompt_answer"}
    
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
        if method in [TrainingMethod.GSPO, TrainingMethod.DR_GRPO, TrainingMethod.GRPO]:
            # All GRPO-based methods use the same format
            return {
                "prompt": "What is the capital of France?",
                "answer": "The capital of France is Paris, a historic city known for its art, culture, and iconic landmarks like the Eiffel Tower.",
                "system": "You are a helpful and knowledgeable assistant."  # Optional field
            }
        else:  # SFT
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
        
        if method in [TrainingMethod.GSPO, TrainingMethod.DR_GRPO]:
            recommendations.append("Enable batch processing for optimal efficiency")
        
        if method == TrainingMethod.DR_GRPO:
            recommendations.append("Ensure high-quality domain-specific training data")
            recommendations.append("Consider domain-specific model initialization")
        
        if method == TrainingMethod.GSPO:
            recommendations.append("Enable sparse optimization for best performance")
            recommendations.append("Monitor efficiency metrics during training")
        
        if memory_gb > 16:
            recommendations.append("Close other applications to free up memory")
        
        return recommendations

# Utility functions for data format conversion
def convert_to_reasoning_format(instruction_data: Dict[str, Any], method: TrainingMethod) -> Dict[str, Any]:
    """Convert instruction/response data to reasoning format"""
    if method in [TrainingMethod.GSPO, TrainingMethod.DR_GRPO, TrainingMethod.GRPO]:
        return {
            "problem": instruction_data.get("instruction", ""),
            "reasoning_steps": [
                "Analyze the problem",
                "Develop solution strategy", 
                "Execute solution"
            ],
            "solution": instruction_data.get("response", ""),
            **({"sparse_indicators": [1, 1, 0], "efficiency_markers": {"optimization_applied": True}} 
               if method == TrainingMethod.GSPO else {}),
            **({"domain": "general", "expertise_level": "intermediate", "domain_context": {"specialty": "general"}} 
               if method == TrainingMethod.DR_GRPO else {})
        }
    return instruction_data

# Export main components
__all__ = [
    "TrainingMethod",
    "TrainingMethodConfig", 
    "TRAINING_METHODS",
    "TrainingDataValidator",
    "ResourceEstimator",
    "convert_to_reasoning_format"
]