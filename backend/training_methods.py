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
    # Alignment methods
    DPO = "dpo"   # Direct Preference Optimization
    ORPO = "orpo"  # Odds Ratio Preference Optimization
    # Supervised/Reasoning methods supported by the app UI
    SFT = "sft"
    GSPO = "gspo"
    DR_GRPO = "dr_grpo"
    GRPO = "grpo"

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
    ,
    # Minimal configs to unblock validation/estimation for existing UI flows
    TrainingMethod.SFT: TrainingMethodConfig(
        name="sft",
        display_name="Supervised Fine-Tuning (SFT)",
        description="Standard instruction-following fine-tuning.",
        complexity="⭐⭐",
        use_case="General instruction following.",
        data_format="instruction",
        requires_preferences=False,
        supports_batch=True,
        resource_intensity="medium",
        module_name="mlx_lm.lora",
        additional_params=[]
    ),
    TrainingMethod.GSPO: TrainingMethodConfig(
        name="gspo",
        display_name="GSPO (Guided Stepwise Preference Optimization)",
        description="Reasoning-oriented supervised data with stepwise signals.",
        complexity="⭐⭐⭐",
        use_case="Reasoning with step traces.",
        data_format="reasoning_supervised",
        requires_preferences=False,
        supports_batch=True,
        resource_intensity="high",
        module_name="mlx_lm.lora",
        additional_params=[]
    ),
    TrainingMethod.DR_GRPO: TrainingMethodConfig(
        name="dr_grpo",
        display_name="Dr. GRPO",
        description="Domain‑regularized GRPO style training.",
        complexity="⭐⭐⭐⭐",
        use_case="Preference/reward‑guided optimization with domain regularization.",
        data_format="reasoning_supervised",
        requires_preferences=False,
        supports_batch=True,
        resource_intensity="high",
        module_name="mlx_lm.lora",
        additional_params=[]
    ),
    TrainingMethod.GRPO: TrainingMethodConfig(
        name="grpo",
        display_name="GRPO",
        description="Generalized preference optimization variant.",
        complexity="⭐⭐⭐",
        use_case="Reasoning and preference signals.",
        data_format="reasoning_supervised",
        requires_preferences=False,
        supports_batch=True,
        resource_intensity="high",
        module_name="mlx_lm.lora",
        additional_params=[]
    ),
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
            
            # Support both JSONL (one JSON per line) and JSON/JSON-array files
            valid_samples = 0
            total_lines = 0
            errors = []
            detected_format = None

            def validate_sample(sample: Dict, index_info: str):
                nonlocal valid_samples, detected_format
                if config.requires_preferences:
                    validation = TrainingDataValidator._validate_preference_data(sample, method)
                else:
                    validation = TrainingDataValidator._validate_instruction_data(sample, method)
                if validation.get("valid"):
                    valid_samples += 1
                    if not detected_format and validation.get("format"):
                        detected_format = validation["format"]
                else:
                    errors.append(f"{index_info}: {validation.get('error')}")

            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                return {"valid": False, "error": "Empty data file"}

            # If file starts with '[' or '{', try full JSON parsing (array or object)
            if content[0] in ['[', '{']:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        total_lines = len(parsed)
                        for idx, item in enumerate(parsed, 1):
                            if isinstance(item, dict):
                                validate_sample(item, f"Item {idx}")
                            else:
                                errors.append(f"Item {idx}: Expected object, got {type(item).__name__}")
                    elif isinstance(parsed, dict):
                        # Look for common array containers: data/samples/examples
                        container = None
                        for key in ["data", "samples", "examples"]:
                            if isinstance(parsed.get(key), list):
                                container = parsed[key]
                                break
                        if container is None:
                            return {"valid": False, "error": "JSON object must contain an array under one of: data, samples, examples"}
                        total_lines = len(container)
                        for idx, item in enumerate(container, 1):
                            if isinstance(item, dict):
                                validate_sample(item, f"Item {idx}")
                            else:
                                errors.append(f"Item {idx}: Expected object, got {type(item).__name__}")
                    else:
                        return {"valid": False, "error": "Unsupported JSON root type; expected array or object"}
                except json.JSONDecodeError as e:
                    return {"valid": False, "error": f"Invalid JSON: {str(e)}"}
            else:
                # JSONL: process line by line
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    total_lines += 1
                    try:
                        data_sample = json.loads(line)
                        if isinstance(data_sample, dict):
                            validate_sample(data_sample, f"Line {line_num}")
                        else:
                            errors.append(f"Line {line_num}: Expected object, got {type(data_sample).__name__}")
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")

            if total_lines == 0:
                return {"valid": False, "error": "No samples found"}

            if valid_samples == 0:
                error_summary = ". ".join(errors[:3]) if errors else "No valid samples detected"
                return {"valid": False, "error": f"No valid samples found. {error_summary}"}

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
            # Broaden acceptance: support common schemas used in this repo
            # 1) instruction/response
            if "instruction" in data_sample and "response" in data_sample:
                return {"valid": True, "format": "instruction_response"}
            # 2) prompt/response or prompt/output variants
            if "prompt" in data_sample and ("response" in data_sample or "output" in data_sample):
                return {"valid": True, "format": "prompt_response"}
            # 3) reasoning datasets like GSPO/GRPO: problem + solution (optionally reasoning_steps)
            if "problem" in data_sample and "solution" in data_sample:
                return {"valid": True, "format": "reasoning_supervised"}

            # If none matched, report the most standard requirement
            required_fields = ["instruction", "response"]
            missing_fields = [field for field in required_fields if field not in data_sample]
            return {
                "valid": False,
                "error": f"Missing required fields: {missing_fields}",
                "required_format": required_fields,
                "note": "Also supports messages[], prompt/response, or problem/solution for reasoning data"
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
        else: # Supervised/reasoning fallback
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
            TrainingMethod.GSPO: {"memory": 1.2, "time": 0.5},
            TrainingMethod.DR_GRPO: {"memory": 1.5, "time": 1.3},
            TrainingMethod.GRPO: {"memory": 1.3, "time": 1.0},
            TrainingMethod.DPO: {"memory": 1.6, "time": 1.4},
            TrainingMethod.ORPO: {"memory": 1.6, "time": 1.4},
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