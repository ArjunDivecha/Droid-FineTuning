"""
Nested Learning API Router

FastAPI router for Nested Learning endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
import logging
import json
from datetime import datetime
import asyncio
import uuid
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nested-learning", tags=["nested-learning"])


# Pydantic models for request/response
class NestedLearningRequest(BaseModel):
    """Request model for starting nested learning training."""
    base_model_path: str
    adapter_path: str
    train_data_path: str
    val_data_path: Optional[str] = None

    # Nested Learning parameters
    num_tiers: int = 3
    tier_update_frequencies: List[int] = [1, 2, 4]
    tier_assignment_strategy: Literal['layer_depth', 'parameter_importance', 'manual'] = 'layer_depth'

    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_steps: int = 1000
    max_seq_length: int = 2048

    # LoRA config
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Advanced
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 2
    checkpoint_every: int = 100
    eval_every: int = 100

    output_path: str = './nested_learning/checkpoints'
    experiment_name: str = 'nested_learning_experiment'


class NestedLearningStatus(BaseModel):
    """Status response model."""
    status: Literal['idle', 'running', 'completed', 'error']
    current_step: int = 0
    total_steps: int = 0
    experiment_name: Optional[str] = None
    tier_stats: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class NestedLearningManager:
    """Manages nested learning training sessions."""

    def __init__(self):
        self.current_status = "idle"
        self.current_step = 0
        self.total_steps = 0
        self.experiment_name = None
        self.tier_stats = None
        self.error_message = None
        self.training_task = None
        self.metrics_history = []

        # Session management
        self.current_session_id = None
        self.current_config = None
        self.sessions_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/sessions"
        os.makedirs(self.sessions_dir, exist_ok=True)

    def save_session(self, training_config, final_metrics):
        """Save nested learning session to match regular training format."""
        try:
            # Generate session ID if not already set
            if not self.current_session_id:
                self.current_session_id = str(uuid.uuid4())

            # Get the best checkpoint path
            best_checkpoint_dir = Path(training_config.output_path) / training_config.experiment_name / "checkpoints" / "best"
            adapter_path = str(best_checkpoint_dir / "adapters.safetensors")

            session_data = {
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat(),
                "training_state": self.current_status,
                "config": {
                    "model_path": training_config.base_model_path,
                    "adapter_name": training_config.experiment_name,
                    "train_data": training_config.train_data_path,
                    "val_data": training_config.val_data_path,
                    "batch_size": training_config.batch_size,
                    "num_steps": training_config.num_steps,
                    "learning_rate": training_config.learning_rate,
                    "lora_rank": training_config.lora_rank,
                    "lora_alpha": training_config.lora_alpha,
                    "training_type": "nested_learning"
                },
                "metrics": {
                    "train_loss": final_metrics.get("final_loss"),
                    "val_loss": final_metrics.get("final_val_loss"),
                    "current_step": self.current_step,
                    "total_steps": self.total_steps
                },
                "adapter_path": adapter_path,
                "nested_learning": {
                    "num_tiers": training_config.num_tiers,
                    "tier_frequencies": training_config.tier_update_frequencies,
                    "tier_stats": self.tier_stats
                },
                "best_model": {
                    "val_loss": final_metrics.get("best_val_loss"),
                    "step": final_metrics.get("best_step"),
                    "path": str(best_checkpoint_dir)
                }
            }

            session_file = os.path.join(self.sessions_dir, f"session_{self.current_session_id}.json")
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)

            # Update latest session pointer
            latest_file = os.path.join(self.sessions_dir, "latest.json")
            with open(latest_file, 'w') as f:
                json.dump({"latest_session_id": self.current_session_id}, f)

            logger.info(f"Saved nested learning session: {self.current_session_id}")

        except Exception as e:
            logger.error(f"Failed to save nested learning session: {e}")

    def get_status(self) -> NestedLearningStatus:
        """Get current training status."""
        return NestedLearningStatus(
            status=self.current_status,
            current_step=self.current_step,
            total_steps=self.total_steps,
            experiment_name=self.experiment_name,
            tier_stats=self.tier_stats,
            message=self.error_message
        )

    def _training_step_callback(self, step: int, total_steps: int, metrics: Dict[str, Any]):
        """Callback function called during training to update status."""
        self.current_step = step
        self.total_steps = total_steps
        if metrics.get('tier_stats'):
            self.tier_stats = metrics['tier_stats']
        if metrics:
            self.metrics_history.append(metrics)

    def start_training(self, config: NestedLearningRequest) -> Dict[str, Any]:
        """
        Start nested learning training.

        Args:
            config: Training configuration

        Returns:
            Dictionary with training start info
        """
        if self.current_status == "running":
            logger.warning("Training already in progress")
            return {
                "success": False,
                "message": "Training already in progress"
            }

        # Validate paths
        if not Path(config.base_model_path).exists():
            self.current_status = "error"
            self.error_message = f"Base model not found: {config.base_model_path}"
            logger.error(self.error_message)
            return {"success": False, "message": self.error_message}

        if not Path(config.adapter_path).exists():
            self.current_status = "error"
            self.error_message = f"Adapter not found: {config.adapter_path}"
            logger.error(self.error_message)
            return {"success": False, "message": self.error_message}

        if not Path(config.train_data_path).exists():
            self.current_status = "error"
            self.error_message = f"Training data not found: {config.train_data_path}"
            logger.error(self.error_message)
            return {"success": False, "message": self.error_message}

        # Set up training state
        self.current_status = "running"
        self.current_step = 0
        self.total_steps = config.num_steps
        self.experiment_name = config.experiment_name
        self.error_message = None
        self.metrics_history = []

        # Create output directory
        output_path = Path(config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Nested Learning training: {config.experiment_name}")
        logger.info(f"  Tiers: {config.num_tiers}")
        logger.info(f"  Frequencies: {config.tier_update_frequencies}")
        logger.info(f"  Strategy: {config.tier_assignment_strategy}")

        # Start training in background
        try:
            # Import here to avoid circular imports
            from nested_learning.config import NestedLearningConfig
            from nested_learning.nested_trainer import NestedLoRATrainer

            # Convert request to config
            training_config = NestedLearningConfig(**config.dict())

            # Create trainer with callback
            trainer = NestedLoRATrainer(training_config)

            # Set callback for status updates
            trainer.step_callback = self._training_step_callback

            # Store config and session ID
            self.current_config = training_config
            self.current_session_id = str(uuid.uuid4())

            # Setup and train (this will run in background)
            logger.info("Running trainer setup...")
            trainer.setup()
            logger.info("Starting training loop...")
            trainer.train()

            self.current_status = "completed"
            logger.info("Training completed successfully")

            # Get final metrics from trainer
            final_metrics = {
                "final_loss": getattr(trainer, 'current_train_loss', None),
                "final_val_loss": getattr(trainer, 'current_val_loss', None),
                "best_val_loss": getattr(trainer, 'best_val_loss', None),
                "best_step": getattr(trainer, 'best_model_step', None)
            }

            # Save session for loading in Compare page
            self.save_session(training_config, final_metrics)

            return {
                "success": True,
                "experiment_name": config.experiment_name,
                "output_path": str(training_config.output_path),
                "session_id": self.current_session_id,
                "message": "Nested Learning training completed successfully"
            }

        except Exception as e:
            self.current_status = "error"
            self.error_message = str(e)
            logger.error(f"Training failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def stop_training(self) -> Dict[str, Any]:
        """Stop current training."""
        if self.current_status != "running":
            raise HTTPException(
                status_code=400,
                detail="No training in progress"
            )

        self.current_status = "idle"
        logger.info("Nested Learning training stopped")

        return {
            "success": True,
            "message": "Training stopped"
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return {
            "experiment_name": self.experiment_name,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "tier_stats": self.tier_stats,
            "metrics_history": self.metrics_history
        }


# Global manager instance
nested_learning_manager = NestedLearningManager()


@router.post("/start")
async def start_nested_learning(
    request: NestedLearningRequest,
    background_tasks: BackgroundTasks
):
    """
    Start nested learning training.

    This endpoint initiates training with multi-frequency parameter updates.
    """
    try:
        # Run training in background task
        background_tasks.add_task(nested_learning_manager.start_training, request)

        return {
            "success": True,
            "message": "Nested learning training started in background",
            "experiment_name": request.experiment_name
        }
    except Exception as e:
        logger.error(f"Failed to start nested learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_nested_learning_status():
    """Get current nested learning training status."""
    return nested_learning_manager.get_status()


@router.post("/stop")
async def stop_nested_learning():
    """Stop current nested learning training."""
    try:
        result = nested_learning_manager.stop_training()
        return result
    except Exception as e:
        logger.error(f"Failed to stop nested learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_nested_learning_metrics():
    """Get nested learning training metrics."""
    return nested_learning_manager.get_metrics()


@router.get("/experiments")
async def list_nested_learning_experiments():
    """List all nested learning experiments."""
    try:
        experiments_dir = Path("./nested_learning/checkpoints")
        if not experiments_dir.exists():
            return {"experiments": []}

        experiments = []
        for config_file in experiments_dir.glob("*_config.json"):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    experiments.append({
                        "experiment_name": config.get("experiment_name"),
                        "config_file": str(config_file),
                        "created": datetime.fromtimestamp(config_file.stat().st_ctime).isoformat()
                    })
            except Exception as e:
                logger.warning(f"Failed to load config {config_file}: {e}")

        return {"experiments": experiments}
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tier-info")
async def get_tier_info():
    """
    Get information about nested learning tiers.

    Returns educational information about how tiers work.
    """
    return {
        "description": "Nested Learning organizes parameters into tiers with different update frequencies",
        "tiers": {
            "tier_0": {
                "name": "Fast Tier",
                "description": "Updates every step - captures rapidly changing patterns",
                "typical_frequency": 1,
                "use_case": "Task-specific adaptation, fine-grained learning"
            },
            "tier_1": {
                "name": "Medium Tier",
                "description": "Updates every 2-4 steps - balances stability and adaptation",
                "typical_frequency": 2,
                "use_case": "Intermediate features, moderate learning rate"
            },
            "tier_2": {
                "name": "Slow Tier",
                "description": "Updates every 4-8 steps - maintains stable representations",
                "typical_frequency": 4,
                "use_case": "Core knowledge retention, preventing catastrophic forgetting"
            }
        },
        "benefits": [
            "Prevents catastrophic forgetting in continual learning",
            "Better generalization across tasks",
            "Improved stability during fine-tuning",
            "More efficient parameter updates"
        ],
        "strategies": {
            "layer_depth": "Shallow layers update faster, deep layers update slower",
            "parameter_importance": "Important parameters (high gradient) update faster",
            "manual": "User-defined tier assignments"
        }
    }
