# backend/main_enhancements.py
# Enhanced functionality to integrate with existing main.py for GSPO and Dr. GRPO support

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import subprocess
import json
import os
import yaml
import logging
import asyncio
import tempfile
import uuid
import traceback
from datetime import datetime

from training_methods import (
    TrainingMethod, 
    TRAINING_METHODS, 
    TrainingDataValidator, 
    ResourceEstimator
)

logger = logging.getLogger(__name__)

@dataclass
class EnhancedTrainingConfig:
    """Enhanced training configuration for modern mlx-lm training"""
    # Core parameters (required fields first)
    model_path: str
    train_data_path: str
    training_method: str  # dpo, orpo, lora
    
    # Optional core parameters
    val_data_path: Optional[str] = None
    adapter_name: str = "adapter"

    # LoRA parameters (for all methods)
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_num_layers: int = -1

    # Quantization parameters
    quantize: bool = False
    quantize_bits: int = 4

    # DPO/ORPO parameters
    beta: float = 0.1
    alpha: Optional[float] = None  # ORPO-specific

    # General training parameters
    learning_rate: float = 1e-5
    batch_size: int = 1
    max_seq_length: int = 2048
    iterations: int = 100
    steps_per_report: int = 10
    steps_per_eval: int = 25
    save_every: int = 100
    # New in mlx-lm v0.28.3: effective batch via accumulation
    grad_accumulation_steps: int = 1

class EnhancedTrainingManager:
    """Enhanced training manager that extends the existing TrainingManager functionality"""
    
    def __init__(self, base_training_manager):
        """Initialize with reference to existing TrainingManager"""
        self.base_manager = base_training_manager
        self.logger = logging.getLogger(__name__)
        self.wrapper_python = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/python"
        self.python_executable = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/python"

    def get_available_methods(self) -> Dict[str, Any]:
        """Get available training methods with their configurations"""
        methods = {}
        for method, config in TRAINING_METHODS.items():
            methods[method.value] = {
                "display_name": config.display_name,
                "description": config.description,
                "complexity": config.complexity,
                "use_case": config.use_case,
                "badge": config.badge,
                "resource_intensity": config.resource_intensity,
                "estimated_speedup": config.estimated_speedup,
                "data_format": config.data_format,
                "requires_reasoning_chains": config.requires_reasoning_chains,
                "additional_params": config.additional_params
            }
        return methods
    
    def validate_training_data(self, method: str, data_path: str) -> Dict[str, Any]:
        """Validate training data format for specific method"""
        try:
            training_method = TrainingMethod(method)
            return TrainingDataValidator.validate_data_format(training_method, data_path)
        except ValueError:
            return {"valid": False, "error": f"Unknown training method: {method}"}
    
    def estimate_resources(self, method: str, model_path: str, dataset_size: int) -> Dict[str, Any]:
        """Estimate resource requirements for training"""
        try:
            training_method = TrainingMethod(method)
            return ResourceEstimator.estimate_requirements(training_method, model_path, dataset_size)
        except ValueError:
            return {"error": f"Unknown training method: {method}"}
    

    
    def build_enhanced_training_command(self, config: EnhancedTrainingConfig) -> List[str]:
        """Build a training command for the native mlx-lm DPO and ORPO modules."""
        method = TrainingMethod(config.training_method)
        module_name = TRAINING_METHODS[method].module_name

        adapter_path = os.path.join(self.base_manager.output_dir, config.adapter_name)
        os.makedirs(adapter_path, exist_ok=True)

        # Use subcommand style for mlx-lm modules to avoid deprecation warnings
        if module_name.startswith("mlx_lm."):
            subcmd = module_name.split(".")[-1]
            cmd = [
                self.python_executable,
                "-m", "mlx_lm", subcmd,
                "--model", config.model_path,
            ]
        else:
            # Fallback for non-mlx_lm modules
            subcmd = module_name
            cmd = [
                self.python_executable,
                "-m", module_name,
                "--model", config.model_path,
            ]

        # Common arguments
        cmd += [
            "--train",
            "--data", config.train_data_path,
            "--adapter-path", adapter_path,
            "--lora-layers", str(config.lora_num_layers),
            "--batch-size", str(config.batch_size),
            "--iters", str(config.iterations),
            "--learning-rate", str(config.learning_rate),
            "--steps-per-report", str(config.steps_per_report),
            "--steps-per-eval", str(config.steps_per_eval),
            "--save-every", str(config.save_every),
            "--max-seq-length", str(config.max_seq_length),
        ]

        # Pass gradient accumulation only for lora subcommand
        if subcmd == "lora" and int(getattr(config, "grad_accumulation_steps", 1) or 1) > 0:
            cmd += ["--grad-accumulation-steps", str(int(config.grad_accumulation_steps))]

        # DPO/ORPO-only parameters
        if subcmd in ("dpo", "orpo"):
            cmd += ["--beta", str(config.beta)]

        if config.val_data_path:
            cmd.extend(["--val", config.val_data_path])

        # ORPO has an extra 'alpha' parameter
        if method == TrainingMethod.ORPO and config.alpha is not None and subcmd == "orpo":
            cmd.extend(["--alpha", str(config.alpha)])

        # Add quantization parameters if enabled
        if config.quantize:
            cmd.append("--quantize")
            cmd.extend(["--bits", str(config.quantize_bits)])

        self.logger.info(f"Built {method.value.upper()} command: {' '.join(cmd)}")
        return cmd

    async def start_enhanced_training(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start training with enhanced method support"""
        try:
            # Normalize and validate required fields early to avoid NoneType errors
            if not isinstance(config_data, dict):
                return {"success": False, "error": "Invalid payload: expected JSON object"}

            if not config_data.get("adapter_name"):
                # Generate a simple default
                config_data["adapter_name"] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if not config_data.get("training_method"):
                return {"success": False, "error": "Missing training_method (gspo/grpo/dr_grpo/sft)"}
            if not config_data.get("model_path"):
                return {"success": False, "error": "Missing model_path"}
            if not config_data.get("train_data_path"):
                return {"success": False, "error": "Missing train_data_path"}

            # Convert to enhanced config
            enhanced_config = EnhancedTrainingConfig(**config_data)
            method_enum = TrainingMethod(enhanced_config.training_method)
            
            # Validate data format if specified
            if enhanced_config.train_data_path:
                validation = self.validate_training_data(
                    enhanced_config.training_method, 
                    enhanced_config.train_data_path
                )
                if not validation.get("valid", False):
                    return {
                        "success": False,
                        "error": f"Data validation failed: {validation.get('error', 'Unknown error')}"
                    }

            if method_enum == TrainingMethod.SFT:
                return await self._start_sft_training(enhanced_config)

            # Update base manager's configuration
            self.base_manager.current_config = enhanced_config
            self.base_manager.training_state = "running"
            
            # Reset metrics
            self.base_manager.training_metrics = {
                "current_step": 0,
                "total_steps": enhanced_config.iterations,
                "train_loss": None,
                "val_loss": None,
                "learning_rate": enhanced_config.learning_rate,
                "start_time": datetime.now().isoformat(),
                "estimated_time_remaining": None,
                "method": enhanced_config.training_method,
                "best_val_loss": None,  # Use None instead of float('inf') for JSON compatibility
                "best_model_step": 0
            }
            
            # Build training command (no config file needed for mlx-lm-lora)
            cmd = self.build_enhanced_training_command(enhanced_config)
            
            # Create log file FIRST for debugging
            log_dir = self.base_manager.output_dir
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "training_debug.log")

            # Log command details before execution
            self.logger.info(f"Starting {enhanced_config.training_method.upper()} training...")
            self.logger.info(f"Command: {' '.join(cmd)}")

            with open(log_file, 'w') as f:
                f.write(f"Training started at {datetime.now().isoformat()}\n")
                f.write(f"Method: {enhanced_config.training_method}\n")
                f.write(f"Model: {enhanced_config.model_path}\n")
                f.write(f"Train data: {enhanced_config.train_data_path}\n")
                f.write(f"Data directory: {os.path.dirname(enhanced_config.train_data_path)}\n")
                f.write(f"Adapter path: {os.path.join(self.base_manager.output_dir, enhanced_config.adapter_name)}\n")
                f.write(f"\nFull command:\n")
                for i, arg in enumerate(cmd):
                    f.write(f"  [{i}] {arg}\n")
                f.write(f"\nCommand as string: {' '.join(cmd)}\n\n")

            # Mirror a succinct parameter summary into the primary GUI log so users can see settings immediately
            try:
                with open(self.base_manager.log_file, 'a') as lf:
                    lf.write("\n=== Enhanced Training Start ===\n")
                    lf.write(f"Time: {datetime.now().isoformat()}\n")
                    lf.write(f"Method: {enhanced_config.training_method}\n")
                    lf.write("Parameters:\n")
                    lf.write(f"  group_size: {enhanced_config.group_size}\n")
                    lf.write(f"  epsilon: {enhanced_config.epsilon}\n")
                    lf.write(f"  temperature: {enhanced_config.temperature}\n")
                    lf.write(f"  max_completion_length: {enhanced_config.max_completion_length}\n")
                    lf.write(f"  importance_sampling_level: {enhanced_config.importance_sampling_level}\n")
                    lf.write(f"  grpo_loss_type: {enhanced_config.grpo_loss_type}\n")
                    lf.write(f"  learning_rate: {enhanced_config.learning_rate}\n")
                    lf.write(f"  batch_size: {enhanced_config.batch_size}\n")
                    lf.write(f"  max_seq_length: {enhanced_config.max_seq_length}\n")
                    lf.write(f"  iterations: {enhanced_config.iterations}\n")
                    lf.write("===============================\n\n")
            except Exception:
                pass

            # Validate paths exist
            if not os.path.exists(enhanced_config.model_path):
                error_msg = f"Model path does not exist: {enhanced_config.model_path}"
                self.logger.error(error_msg)
                with open(log_file, 'a') as f:
                    f.write(f"ERROR: {error_msg}\n")
                self.base_manager.training_state = "error"
                self.base_manager.last_error = error_msg
                return {"success": False, "error": error_msg, "log_file": log_file}

            if not os.path.exists(enhanced_config.train_data_path):
                error_msg = f"Training data file does not exist: {enhanced_config.train_data_path}"
                self.logger.error(error_msg)
                with open(log_file, 'a') as f:
                    f.write(f"ERROR: {error_msg}\n")
                self.base_manager.training_state = "error"
                self.base_manager.last_error = error_msg
                return {"success": False, "error": error_msg, "log_file": log_file}

            # Start training process with better error handling
            try:
                # Force unbuffered output for real-time log streaming and ensure repo root is importable
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                env['PYTHONFAULTHANDLER'] = '1'

                # Ensure the project root is on PYTHONPATH so `-m debug.debug_trainer` can import
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')

                # Run from the repo root so local modules are discoverable
                self.base_manager.current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True,
                    bufsize=1,  # Line buffered
                    universal_newlines=True,
                    env=env,
                    cwd=repo_root,
                )

                with open(log_file, 'a') as f:
                    f.write(f"Process started with PID: {self.base_manager.current_process.pid}\n")
                    f.write(f"Working directory: {os.path.dirname(enhanced_config.train_data_path)}\n\n")

            except Exception as e:
                error_msg = f"Failed to start training process: {str(e)}"
                self.logger.error(error_msg)
                with open(log_file, 'a') as f:
                    f.write(f"\nERROR starting process:\n{error_msg}\n")
                    f.write(f"\nTraceback:\n{traceback.format_exc()}\n")
                self.base_manager.training_state = "error"
                return {"success": False, "error": error_msg, "log_file": log_file}

            # Capture immediate errors
            await asyncio.sleep(1.0)  # Give process time to start and potentially fail

            if self.base_manager.current_process.poll() is not None:
                # Process already exited - capture error
                try:
                    stdout, stderr = self.base_manager.current_process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    stdout = stderr = "Timeout reading output"

                error_msg = f"Process exited immediately with code {self.base_manager.current_process.returncode}"
                self.logger.error(error_msg)
                self.logger.error(f"STDERR: {stderr}")
                self.logger.error(f"STDOUT: {stdout}")

                with open(log_file, 'a') as f:
                    f.write(f"\n{'='*70}\n")
                    f.write(f"ERROR: Process exited immediately\n")
                    f.write(f"Exit code: {self.base_manager.current_process.returncode}\n")
                    f.write(f"{'='*70}\n\n")
                    f.write(f"STDERR:\n{stderr}\n\n")
                    f.write(f"STDOUT:\n{stdout}\n")

                self.base_manager.training_state = "error"
                return {
                    "success": False,
                    "error": f"{error_msg}. Check {log_file} for details.",
                    "stderr": stderr[:500],  # First 500 chars
                    "log_file": log_file
                }
            
            # Return success - monitoring will be started by the async endpoint wrapper
            return {
                "success": True,
                "message": f"Enhanced training started with {enhanced_config.training_method.upper()}",
                "method": enhanced_config.training_method,
                "pid": self.base_manager.current_process.pid,
                "needs_monitoring": True  # Signal that monitoring should be started
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced training start failed: {str(e)}")
            self.base_manager.training_state = "error"
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_sample_data(self, method: str, output_path: str, num_samples: int = 10) -> Dict[str, Any]:
        """Generate sample training data for testing"""
        try:
            training_method = TrainingMethod(method)
            method_config = TRAINING_METHODS[training_method]
            
            samples = []
            
            if method == "gspo":
                for i in range(num_samples):
                    sample = {
                        "problem": f"Solve this optimization problem: Find the minimum value of f(x) = x² + {i+1}x + {i}",
                        "reasoning_steps": [
                            f"Identify the quadratic function f(x) = x² + {i+1}x + {i}",
                            "Find the vertex using x = -b/(2a)",
                            f"Calculate x = -{i+1}/2 = {-(i+1)/2}",
                            f"Substitute back to find minimum value"
                        ],
                        "solution": f"The minimum value is {i - (i+1)**2/4}",
                        "sparse_indicators": [1, 1, 0, 1],  # Critical steps
                        "efficiency_markers": {
                            "computation_cost": "low",
                            "reasoning_depth": 4,
                            "optimization_applied": True
                        }
                    }
                    samples.append(sample)
            
            elif method == "dr_grpo":
                medical_cases = [
                    "chest pain and shortness of breath",
                    "fever and rash in pediatric patient", 
                    "confusion and memory loss in elderly",
                    "severe headache with vision changes"
                ]
                
                for i in range(num_samples):
                    case = medical_cases[i % len(medical_cases)]
                    sample = {
                        "problem": f"Patient presents with {case}. Provide differential diagnosis.",
                        "reasoning_steps": [
                            "Obtain comprehensive history",
                            "Perform focused physical examination",
                            "Consider most likely diagnoses",
                            "Order appropriate diagnostic tests",
                            "Formulate evidence-based treatment plan"
                        ],
                        "solution": "Systematic diagnostic approach with evidence-based recommendations",
                        "domain": "medical",
                        "expertise_level": "advanced",
                        "domain_context": {
                            "medical_specialty": "internal_medicine",
                            "complexity": "high",
                            "evidence_level": "grade_A"
                        }
                    }
                    samples.append(sample)
            
            elif method == "grpo":
                for i in range(num_samples):
                    sample = {
                        "problem": f"Complex reasoning: If A implies B, and B implies C, what can we conclude about A and C when we know C is false?",
                        "reasoning_steps": [
                            "Identify the logical structure: A → B → C",
                            "Apply modus tollens: ¬C → ¬B",
                            "Apply modus tollens again: ¬B → ¬A",
                            "Conclude: ¬C → ¬A (if C is false, A must be false)"
                        ],
                        "solution": "By contraposition and logical chaining, if C is false, then A must also be false."
                    }
                    samples.append(sample)
            
            else:  # SFT
                for i in range(num_samples):
                    sample = {
                        "instruction": f"Explain the concept of {['machine learning', 'neural networks', 'deep learning', 'artificial intelligence'][i % 4]}",
                        "response": f"A comprehensive explanation of {['machine learning', 'neural networks', 'deep learning', 'artificial intelligence'][i % 4]} with examples and applications."
                    }
                    samples.append(sample)
            
            # Write samples to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            return {
                "success": True,
                "message": f"Generated {num_samples} samples for {method}",
                "output_path": output_path,
                "method": method,
                "sample_count": num_samples
            }
            
        except Exception as e:
            self.logger.error(f"Sample data generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Enhanced API endpoints to add to your existing FastAPI app
def create_enhanced_endpoints(app, training_manager, enhanced_manager):
    """Create enhanced API endpoints"""
    
    @app.get("/api/training/methods")
    async def get_training_methods():
        """Get available training methods"""
        try:
            methods = enhanced_manager.get_available_methods()
            return {"success": True, "methods": methods}
        except Exception as e:
            logger.error(f"Failed to get training methods: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @app.post("/api/training/validate-data")
    async def validate_training_data(request_data: dict):
        """Validate training data format and syntax"""
        try:
            method = request_data.get("method")
            data_path = request_data.get("data_path")
            
            if not method or not data_path:
                return {"valid": False, "error": "Missing method or data_path"}
            
            validation = enhanced_manager.validate_training_data(method, data_path)
            
            # Return in format expected by frontend
            if validation.get("valid"):
                return {
                    "valid": True,
                    "num_samples": validation.get("num_samples", 0),
                    "format": validation.get("format", "unknown"),
                    "message": validation.get("message", "Data validation passed")
                }
            else:
                return {
                    "valid": False,
                    "error": validation.get("error", "Unknown validation error")
                }
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    @app.post("/api/training/estimate-resources")
    async def estimate_resources(request_data: dict):
        """Estimate resource requirements"""
        try:
            method = request_data.get("method")
            model_path = request_data.get("model_path", "7B")
            dataset_size = request_data.get("dataset_size", 1000)
            
            if not method:
                return {"success": False, "error": "Missing method"}
            
            estimation = enhanced_manager.estimate_resources(method, model_path, dataset_size)
            return {"success": True, "estimation": estimation}
        except Exception as e:
            logger.error(f"Resource estimation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @app.post("/api/training/start-enhanced")
    async def start_enhanced_training(config: dict):
        """Start training with enhanced method support"""
        try:
            # Stop any existing training first
            if training_manager.current_process and training_manager.current_process.poll() is None:
                await training_manager.stop_training()
            
            result = await enhanced_manager.start_enhanced_training(config)
            
            # Start monitoring task if training started successfully
            if result.get("success") and result.get("needs_monitoring"):
                asyncio.create_task(training_manager._monitor_training())
                logger.info("Started monitoring task for enhanced training")
            
            # Broadcast status update
            if result.get("success"):
                await training_manager.broadcast({
                    "type": "training_started", 
                    "method": result.get("method", "unknown"),
                    "message": result.get("message", "Training started")
                })
            
            return result
        except Exception as e:
            tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(f"Enhanced training start failed: {str(e)}")
            # Also append to GUI log if possible
            try:
                with open(training_manager.log_file, 'a') as f:
                    f.write(f"\n[ENHANCED START ERROR] {e}\n{tb}\n")
            except Exception:
                pass
            return {"success": False, "error": str(e)}
    
    @app.post("/api/training/generate-sample-data")
    async def generate_sample_data(request_data: dict):
        """Generate sample training data for testing"""
        try:
            method = request_data.get("method")
            output_path = request_data.get("output_path")
            num_samples = request_data.get("num_samples", 10)
            
            if not method or not output_path:
                return {"success": False, "error": "Missing method or output_path"}
            
            result = enhanced_manager.generate_sample_data(method, output_path, num_samples)
            return result
        except Exception as e:
            logger.error(f"Sample data generation failed: {str(e)}")
            return {"success": False, "error": str(e)}

# Integration function for existing main.py
def integrate_enhanced_training(app, training_manager):
    """
    Integration function to add enhanced training to existing main.py
    
    Add this to your existing main.py:
    
    from main_enhancements import integrate_enhanced_training
    
    # After creating your TrainingManager instance:
    integrate_enhanced_training(app, training_manager)
    """
    
    # Create enhanced manager
    enhanced_manager = EnhancedTrainingManager(training_manager)
    
    # Add enhanced endpoints
    create_enhanced_endpoints(app, training_manager, enhanced_manager)
    
    # Add enhanced manager to app state for access in other endpoints
    app.state.enhanced_manager = enhanced_manager
    
    logger.info("Enhanced training methods integrated successfully")
    return enhanced_manager
