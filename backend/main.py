"""
MLX Fine-Tuning GUI Backend API Server

This FastAPI server provides REST endpoints and WebSocket connections
for the MLX fine-tuning GUI application.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import os
import sys
import subprocess
import signal
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import uuid
import random

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed, environment variables from .env won't be loaded")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to path to import existing modules
sys.path.append('/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/one_step_finetune')

app = FastAPI(title="MLX Fine-Tuning GUI API", version="1.0.0")

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class TrainingConfig:
    """Training configuration data class"""
    model_path: str
    train_data_path: str
    val_data_path: str
    learning_rate: float = 1e-5
    batch_size: int = 1
    max_seq_length: int = 1024
    iterations: int = 7329
    steps_per_report: int = 25
    steps_per_eval: int = 200
    save_every: int = 100
    early_stop: bool = True
    patience: int = 3
    adapter_name: str = "mlx_finetune"

class TrainingManager:
    """Manages training processes and state"""
    
    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.training_state = "idle"  # idle, running, paused, completed, error
        self.current_config: Optional[TrainingConfig] = None
        self.training_metrics: Dict[str, Any] = {}
        self.websocket_clients: List[WebSocket] = []
        self.output_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters"
        self.log_file = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/logs/gui_training.log"
        self.sessions_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/sessions"
        self.current_session_id: Optional[str] = None
        self.current_adapter_path: Optional[str] = None  # Full path to adapter directory
        
        # Best model tracking
        self.best_val_loss: Optional[float] = None
        self.best_model_step: Optional[int] = None
        self.best_model_path: Optional[str] = None
        
        # Ensure sessions directory exists
        os.makedirs(self.sessions_dir, exist_ok=True)
        
        # Load the most recent session on startup
        self.load_latest_session()
        
        # Force idle state if no valid session loaded
        if not self.current_config:
            self.training_state = "idle"
            logger.info("Forcing idle state - no valid session loaded")
    
    async def _save_best_model(self, step: int):
        """Save the current checkpoint as the best model"""
        try:
            if not self.current_config:
                return
                
            adapter_dir = os.path.join(self.output_dir, self.current_config.adapter_name)
            step_file = os.path.join(adapter_dir, f"{step:07d}_adapters.safetensors")
            best_file = os.path.join(adapter_dir, "best_adapters.safetensors")
            
            # Copy the current step checkpoint as the best model
            if os.path.exists(step_file):
                import shutil
                shutil.copy2(step_file, best_file)
                self.best_model_path = best_file
                logger.info(f"Saved best model from step {step} with val_loss {self.best_val_loss:.4f}")
                
                # Broadcast best model update
                await self.broadcast({
                    "type": "best_model_updated",
                    "data": {
                        "step": step,
                        "val_loss": self.best_val_loss,
                        "path": best_file
                    }
                })
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
    
    def save_session(self):
        """Save current training session to persistent storage"""
        if not self.current_config or not self.training_metrics:
            return
            
        try:
            # Generate session ID if not already set
            if not self.current_session_id:
                self.current_session_id = str(uuid.uuid4())
            
            session_data = {
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat(),
                "training_state": self.training_state,
                "config": asdict(self.current_config),
                "metrics": self.training_metrics,
                "adapter_path": os.path.join(self.output_dir, self.current_config.adapter_name, "adapters.safetensors"),
                "best_model": {
                    "val_loss": self.best_val_loss,
                    "step": self.best_model_step,
                    "path": self.best_model_path
                } if self.best_val_loss is not None else None
            }
            
            session_file = os.path.join(self.sessions_dir, f"session_{self.current_session_id}.json")
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            # Update latest session pointer
            latest_file = os.path.join(self.sessions_dir, "latest.json")
            with open(latest_file, 'w') as f:
                json.dump({"latest_session_id": self.current_session_id}, f)
                
            logger.info(f"Saved training session: {self.current_session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def load_session(self, session_id: str) -> bool:
        """Load a specific training session"""
        try:
            session_file = os.path.join(self.sessions_dir, f"session_{session_id}.json")
            if not os.path.exists(session_file):
                logger.warning(f"Session file not found: {session_file}")
                return False

            with open(session_file, 'r') as f:
                session_data = json.load(f)

            # Check if this is a nested learning session
            config_data = session_data["config"]
            if config_data.get("training_type") == "nested_learning":
                # For nested learning, create a compatible TrainingConfig with defaults
                self.current_config = TrainingConfig(
                    model_path=config_data["model_path"],
                    train_data_path=config_data["train_data"],
                    val_data_path=config_data.get("val_data", ""),
                    learning_rate=config_data.get("learning_rate", 1e-5),
                    batch_size=config_data.get("batch_size", 1),
                    max_seq_length=2048,
                    iterations=config_data.get("num_steps", 1000),
                    steps_per_report=10,
                    steps_per_eval=100,
                    save_every=100,
                    early_stop=False,
                    patience=3,
                    adapter_name=config_data["adapter_name"]
                )
            else:
                # Regular training session
                self.current_config = TrainingConfig(**config_data)

            # Restore metrics and state
            self.training_metrics = session_data["metrics"]
            self.training_state = session_data["training_state"]
            self.current_session_id = session_data["session_id"]

            # Get adapter path from session data
            if "adapter_path" in session_data:
                adapter_path = session_data["adapter_path"]
                # Convert relative path to absolute path
                if not os.path.isabs(adapter_path):
                    # Remove the filename if present (e.g., adapters.safetensors)
                    if adapter_path.endswith(".safetensors"):
                        adapter_path = os.path.dirname(adapter_path)
                    adapter_path = os.path.abspath(adapter_path)
                self.current_adapter_path = adapter_path
            else:
                # Fallback to old behavior for regular training sessions
                self.current_adapter_path = os.path.join(self.output_dir, self.current_config.adapter_name)

            logger.info(f"Loaded training session: {session_id}")
            logger.info(f"Adapter path: {self.current_adapter_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False
    
    def load_latest_session(self):
        """Load the most recent training session on startup"""
        try:
            latest_file = os.path.join(self.sessions_dir, "latest.json")
            if not os.path.exists(latest_file):
                logger.info("No previous training sessions found")
                return
            
            with open(latest_file, 'r') as f:
                latest_data = json.load(f)
            
            session_id = latest_data.get("latest_session_id")
            if session_id:
                success = self.load_session(session_id)
                if success:
                    logger.info(f"Restored previous training session: {session_id}")
                    # Only restore if training was completed
                    if self.training_state not in ["completed", "error", "stopped"]:
                        self.training_state = "idle"
                        logger.info("Previous session was not completed, reset to idle state")
                else:
                    logger.warning("Failed to restore previous session")
            
        except Exception as e:
            logger.error(f"Failed to load latest session: {e}")
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get list of all saved training sessions"""
        sessions = []
        try:
            if not os.path.exists(self.sessions_dir):
                return sessions
                
            for filename in os.listdir(self.sessions_dir):
                if filename.startswith("session_") and filename.endswith(".json"):
                    session_file = os.path.join(self.sessions_dir, filename)
                    try:
                        with open(session_file, 'r') as f:
                            session_data = json.load(f)
                        
                        session_summary = {
                            "session_id": session_data["session_id"],
                            "timestamp": session_data["timestamp"],
                            "training_state": session_data["training_state"],
                            "model_name": session_data["config"]["model_path"].split('/')[-1],
                            "adapter_name": session_data["config"]["adapter_name"],
                            "final_train_loss": session_data["metrics"].get("train_loss"),
                            "final_val_loss": session_data["metrics"].get("val_loss"),
                            "steps_completed": session_data["metrics"].get("current_step", 0),
                            "total_steps": session_data["metrics"].get("total_steps", 0)
                        }
                        sessions.append(session_summary)
                        
                    except Exception as e:
                        logger.error(f"Error reading session file {filename}: {e}")
                        continue
            
            # Sort by timestamp, newest first
            sessions.sort(key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get sessions list: {e}")
        
        return sessions
        
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific training session"""
        try:
            session_file = os.path.join(self.sessions_dir, f"session_{session_id}.json")
            if not os.path.exists(session_file):
                logger.warning(f"Session file not found: {session_file}")
                return False
            
            # Remove the session file
            os.remove(session_file)
            
            # Update latest.json if this was the latest session
            latest_file = os.path.join(self.sessions_dir, "latest.json")
            if os.path.exists(latest_file):
                try:
                    with open(latest_file, 'r') as f:
                        latest_data = json.load(f)
                    
                    if latest_data.get("latest_session_id") == session_id:
                        # Find the next most recent session
                        remaining_sessions = self.get_all_sessions()
                        if remaining_sessions:
                            # Update to the most recent remaining session
                            with open(latest_file, 'w') as f:
                                json.dump({"latest_session_id": remaining_sessions[0]["session_id"]}, f)
                        else:
                            # No sessions left, remove latest.json
                            os.remove(latest_file)
                except Exception as e:
                    logger.error(f"Error updating latest.json after deletion: {e}")
            
            logger.info(f"Deleted training session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
        
    async def add_websocket(self, websocket: WebSocket):
        """Add a WebSocket client"""
        self.websocket_clients.append(websocket)
        # Send current state
        await websocket.send_json({
            "type": "training_state",
            "data": {
                "state": self.training_state,
                "metrics": self.training_metrics
            }
        })
        
    def remove_websocket(self, websocket: WebSocket):
        """Remove a WebSocket client"""
        if websocket in self.websocket_clients:
            self.websocket_clients.remove(websocket)
    
    async def _prepare_training_data(self, model_path: str, train_data_path: str, val_data_path: str):
        """Automatically prepare training data using the correct tokenizer for the selected model"""
        try:
            import subprocess
            import os
            
            # Use user-selected data paths
            chat_train_path = train_data_path
            chat_val_path = val_data_path
            data_prep_script = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/scripts/prepare_chat_template_dataset.py"
            output_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/one_step_finetune/data"
            venv_python = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/.venv/bin/python"
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Run data preparation with the selected model's tokenizer
            cmd = [
                venv_python,
                data_prep_script,
                "--train-path", chat_train_path,
                "--val-path", chat_val_path,
                "--model-tokenizer-dir", model_path,
                "--out-dir", output_dir
            ]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Data preparation completed successfully for model: {model_path}")
            logger.info(f"Data prep output: {process.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Data preparation failed: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"Failed to prepare training data: {e.stderr}")
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket clients"""
        if self.websocket_clients:
            disconnected = []
            for client in self.websocket_clients:
                try:
                    await client.send_json(message)
                except:
                    disconnected.append(client)
            
            # Remove disconnected clients
            for client in disconnected:
                self.remove_websocket(client)
    
    async def start_training(self, config: TrainingConfig) -> bool:
        """Start training with the given configuration"""
        if self.current_process and self.current_process.poll() is None:
            raise HTTPException(status_code=400, detail="Training is already running")
        
        self.current_config = config
        self.training_state = "running"
        self.training_metrics = {
            "current_step": 0,
            "total_steps": config.iterations,
            "train_loss": None,  # Don't initialize with 0, wait for first real value
            "val_loss": None,    # Don't initialize with 0, wait for first real value
            "learning_rate": config.learning_rate,
            "start_time": datetime.now().isoformat(),
            "estimated_time_remaining": None
        }
        
        # Generate new session ID for this training run
        self.current_session_id = str(uuid.uuid4())
        
        # Automatically prepare training data with the correct tokenizer for the selected model
        await self._prepare_training_data(config.model_path, config.train_data_path, config.val_data_path)
        
        # Create config file for the training script
        config_data = {
            "venv_python": "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/.venv/bin/python",
            "base_model_dir": config.model_path,
            "prepared_data_dir": "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/one_step_finetune/data",
            "prepare_from_chat": False,  # Disable chat preparation since script is missing
            "adapter_output_dir": self.output_dir,
            "adapter_name": config.adapter_name,
            "optimizer": "adamw",
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "iters": config.iterations,
            "steps_per_report": config.steps_per_report,
            "steps_per_eval": config.steps_per_eval,
            "val_batches": 25,
            "max_seq_length": config.max_seq_length,
            "grad_checkpoint": True,
            "mask_prompt": False,
            "save_every": config.save_every,
            "resume_adapter_file": None,
            "train_log": self.log_file,
            "enable_early_stop": config.early_stop,
            "no_improve_patience_evals": config.patience
        }
        
        # Write config file
        config_path = "/tmp/gui_training_config.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(config_data, f, default_flow_style=False)
        
        # Start training process
        try:
            cmd = [
                "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Fine Tuner XXX/one_step_finetune/run_finetune.py",
                "--config", config_path
            ]
            
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid
            )
            
            # Start monitoring the process
            asyncio.create_task(self._monitor_training())
            
            await self.broadcast({
                "type": "training_started",
                "data": {"config": config_data}
            })
            
            return True
            
        except Exception as e:
            self.training_state = "error"
            logger.error(f"Failed to start training: {e}")
            await self.broadcast({
                "type": "training_error",
                "data": {"error": str(e)}
            })
            return False
    
    async def stop_training(self):
        """Stop the current training process"""
        if self.current_process and self.current_process.poll() is None:
            try:
                # Send SIGTERM to the process group
                os.killpg(self.current_process.pid, signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    self.current_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    os.killpg(self.current_process.pid, signal.SIGKILL)
                
                self.training_state = "stopped"
                # Save stopped session
                self.save_session()
                await self.broadcast({
                    "type": "training_stopped",
                    "data": {}
                })
                
            except Exception as e:
                logger.error(f"Error stopping training: {e}")
        else:
            # If no process is running but state is error, reset to idle
            if self.training_state == "error":
                self.training_state = "idle"
                await self.broadcast({
                    "type": "training_reset",
                    "data": {"message": "Training state reset from error to idle"}
                })
        
    async def _monitor_training(self):
        """Monitor the training process and broadcast updates"""
        if not self.current_process:
            return
            
        try:
            import re
            
            # Patterns to extract metrics from training output
            step_pattern = re.compile(r'Iter (\d+):')
            loss_pattern = re.compile(r'Train loss ([0-9.]+)')
            val_pattern = re.compile(r'Val loss ([0-9.]+)')
            lr_pattern = re.compile(r'Learning Rate ([0-9.e-]+)')
            early_stop_detected = False
            
            while self.current_process and self.current_process.poll() is None:
                try:
                    output = self.current_process.stdout.readline()
                    if not output:
                        # Empty readline means no data available yet, sleep and continue
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Check for early stopping message
                    if "Early stop:" in output:
                        early_stop_detected = True
                        logger.info(f"Early stopping detected: {output.strip()}")
                    
                    # Parse metrics from output
                    step_match = step_pattern.search(output)
                    loss_match = loss_pattern.search(output)
                    val_match = val_pattern.search(output)
                    lr_match = lr_pattern.search(output)
                    
                    # Extract metrics
                    if step_match:
                        current_step = int(step_match.group(1))
                        self.training_metrics["current_step"] = current_step
                        
                    if loss_match:
                        train_loss = float(loss_match.group(1))
                        self.training_metrics["train_loss"] = train_loss
                        
                    if val_match:
                        val_loss = float(val_match.group(1))
                        self.training_metrics["val_loss"] = val_loss
                        
                        # Track best model based on validation loss
                        if self.best_val_loss is None or val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.best_model_step = current_step
                            # Copy current checkpoint as best model
                            await self._save_best_model(current_step)
                        
                    if lr_match:
                        learning_rate = float(lr_match.group(1))
                        self.training_metrics["learning_rate"] = learning_rate
                    
                    # Calculate progress and ETA
                    if "current_step" in self.training_metrics and "total_steps" in self.training_metrics:
                        progress = self.training_metrics["current_step"] / self.training_metrics["total_steps"]
                        if progress > 0 and "start_time" in self.training_metrics:
                            start_time = datetime.fromisoformat(self.training_metrics["start_time"])
                            elapsed = (datetime.now() - start_time).total_seconds()
                            estimated_total = elapsed / progress
                            remaining = estimated_total - elapsed
                            self.training_metrics["estimated_time_remaining"] = remaining
                    
                    # Broadcast update
                    await self.broadcast({
                        "type": "training_progress",
                        "data": {
                            "metrics": self.training_metrics,
                            "log_line": output.strip()
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"Error monitoring training: {e}")
                    break
            
            # Process completed - but read any remaining output first
            return_code = self.current_process.wait()
            
            # Read any remaining output after process completion
            try:
                while True:
                    remaining_output = self.current_process.stdout.readline()
                    if not remaining_output:
                        break
                    
                    # Check for early stopping message
                    if "Early stop:" in remaining_output:
                        early_stop_detected = True
                    
                    # Parse final metrics from remaining output
                    step_match = step_pattern.search(remaining_output)
                    if step_match:
                        # Use iteration number directly as step (Iter 0 = Step 0, Iter 1 = Step 1, etc.)
                        self.training_metrics["current_step"] = int(step_match.group(1))
                    
                    loss_match = loss_pattern.search(remaining_output)
                    if loss_match:
                        self.training_metrics["train_loss"] = float(loss_match.group(1))
                    
                    val_match = val_pattern.search(remaining_output)
                    if val_match:
                        self.training_metrics["val_loss"] = float(val_match.group(1))
            except:
                pass  # Ignore errors when reading final output
            
            if early_stop_detected:
                # Early stopping is a successful completion, not an error
                self.training_state = "completed"
                # Save completed session
                self.save_session()
                await self.broadcast({
                    "type": "training_completed",
                    "data": {
                        "final_metrics": self.training_metrics,
                        "early_stopped": True,
                        "message": "Training completed via early stopping"
                    }
                })
            elif return_code == 0:
                self.training_state = "completed"
                # Save completed session
                self.save_session()
                await self.broadcast({
                    "type": "training_completed",
                    "data": {"final_metrics": self.training_metrics}
                })
            else:
                self.training_state = "error"
                # Save error session
                self.save_session()
                await self.broadcast({
                    "type": "training_error",
                    "data": {"error": f"Training process exited with code {return_code}"}
                })
                
        except Exception as e:
            self.training_state = "error"
            logger.error(f"Training monitoring error: {e}")
            await self.broadcast({
                "type": "training_error",
                "data": {"error": str(e)}
            })

class OPDManager:
    """Manages On-Policy Distillation training processes and state"""

    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.opd_state = "idle"  # idle, running, completed, error
        self.current_run_id: Optional[str] = None
        self.current_config: Optional[Dict[str, Any]] = None
        self.opd_metrics: Dict[str, Any] = {}
        self.runs_dir = "./OnPolicyDistill/runs"
        self.checkpoint_base_dir = "./OnPolicyDistill/checkpoints"

        # Ensure directories exist
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.checkpoint_base_dir, exist_ok=True)

    async def start_distillation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start distillation training with the given configuration"""
        if self.current_process and self.current_process.poll() is None:
            raise HTTPException(status_code=400, detail="Distillation is already running")

        # Generate run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"distill_{timestamp}"

        self.current_run_id = run_id
        self.current_config = config
        self.opd_state = "running"
        self.opd_metrics = {
            "step": 0,
            "total_steps": config.get("num_steps", 1000),
            "progress_pct": 0.0,
            "kl_loss": None,
            "token_agreement_pct": None,
            "started_at": datetime.now().isoformat()
        }

        # Save run metadata
        self._save_run_metadata(run_id, config)

        # Build command
        cmd = [
            "python3",
            "backend/opd/run_distillation.py",
            "--teacher-path", config["teacher_model_path"],
            "--student-path", config["base_model_path"],
            "--adapter-path", config["student_adapter_path"],
            "--prompts-path", config["validation_prompts_path"],
            "--output-path", os.path.join(self.checkpoint_base_dir, run_id),
            "--steps", str(config.get("num_steps", 1000)),
            "--batch-size", str(config.get("batch_size", 4)),
            "--temperature", str(config.get("temperature", 2.0)),
            "--kl-weight", str(config.get("kl_weight", 0.8)),
            "--learning-rate", str(config.get("learning_rate", 1e-5)),
            "--run-id", run_id
        ]

        # Start subprocess
        try:
            log_file = os.path.join(self.runs_dir, f"{run_id}.log")
            log_f = open(log_file, 'w')

            self.current_process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                cwd=os.getcwd()
            )

            # Start monitoring in background
            asyncio.create_task(self._monitor_distillation())

            # Estimate duration (rough estimate: ~30 seconds per step for batch_size=4)
            estimated_minutes = (config.get("num_steps", 1000) * 30) / 60

            # Estimate memory (teacher 4-bit + student + cache)
            memory_gb = 48  # Conservative estimate

            return {
                "status": "success",
                "run_id": run_id,
                "message": "Distillation training started",
                "estimated_duration_minutes": int(estimated_minutes),
                "memory_required_gb": memory_gb
            }

        except Exception as e:
            self.opd_state = "error"
            logger.error(f"Failed to start distillation: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start distillation: {str(e)}")

    async def stop_distillation(self) -> Dict[str, Any]:
        """Stop the current distillation process"""
        if self.current_process and self.current_process.poll() is None:
            try:
                # Send SIGTERM to the process group
                os.killpg(self.current_process.pid, signal.SIGTERM)
                self.current_process.wait(timeout=10)
                self.opd_state = "stopped"

                checkpoint_path = self._get_latest_checkpoint()

                return {
                    "status": "stopped",
                    "final_step": self.opd_metrics.get("step", 0),
                    "checkpoint_path": checkpoint_path,
                    "message": "Distillation stopped by user"
                }
            except Exception as e:
                logger.error(f"Error stopping distillation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            return {
                "status": "not_running",
                "message": "No distillation process is running"
            }

    def get_status(self) -> Dict[str, Any]:
        """Get current distillation status"""
        return {
            "state": self.opd_state,
            "run_id": self.current_run_id,
            "metrics": self.opd_metrics,
            "config": self.current_config
        }

    def get_metrics(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for a specific run"""
        target_run_id = run_id or self.current_run_id

        if not target_run_id:
            return {"error": "No run_id specified and no active run"}

        # Read metrics from file (stored in OnPolicyDistill/metrics/)
        metrics_file = os.path.join("./OnPolicyDistill/metrics", f"{target_run_id}_train.jsonl")

        if not os.path.exists(metrics_file):
            return {
                "run_id": target_run_id,
                "metrics_history": [],
                "error": "Metrics file not found"
            }

        try:
            metrics_history = []
            with open(metrics_file, 'r') as f:
                for line in f:
                    if line.strip():
                        metrics_history.append(json.loads(line))

            return {
                "run_id": target_run_id,
                "total_steps": self.current_config.get("num_steps", 1000) if self.current_config else 1000,
                "metrics_history": metrics_history
            }
        except Exception as e:
            logger.error(f"Error reading metrics: {e}")
            return {
                "run_id": target_run_id,
                "metrics_history": [],
                "error": str(e)
            }

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Get list of all distillation runs"""
        runs = []

        try:
            for filename in os.listdir(self.runs_dir):
                if filename.endswith("_metadata.json"):
                    run_id = filename.replace("_metadata.json", "")
                    metadata_path = os.path.join(self.runs_dir, filename)

                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    runs.append(metadata)

            # Sort by started_at descending
            runs.sort(key=lambda x: x.get("started_at", ""), reverse=True)

            return runs
        except Exception as e:
            logger.error(f"Error getting runs list: {e}")
            return []

    async def _monitor_distillation(self):
        """Monitor the distillation process and update metrics"""
        if not self.current_process:
            return

        try:
            log_file = os.path.join(self.runs_dir, f"{self.current_run_id}.log")

            while self.current_process.poll() is None:
                await asyncio.sleep(2)

                # Parse log file for metrics
                if os.path.exists(log_file):
                    self._parse_metrics_from_log(log_file)

                # Broadcast progress
                await training_manager.broadcast({
                    "type": "opd_progress",
                    "data": self.opd_metrics
                })

            # Process completed
            return_code = self.current_process.returncode

            if return_code == 0:
                self.opd_state = "completed"
                self.opd_metrics["completed_at"] = datetime.now().isoformat()

                # Update metadata
                self._update_run_metadata(self.current_run_id, {
                    "status": "completed",
                    "completed_at": self.opd_metrics["completed_at"],
                    "final_metrics": self.opd_metrics
                })

                await training_manager.broadcast({
                    "type": "opd_completed",
                    "data": {
                        "run_id": self.current_run_id,
                        "final_metrics": self.opd_metrics,
                        "message": "Distillation completed successfully"
                    }
                })
            else:
                self.opd_state = "error"
                self._update_run_metadata(self.current_run_id, {
                    "status": "error",
                    "error": f"Process exited with code {return_code}"
                })

                await training_manager.broadcast({
                    "type": "opd_error",
                    "data": {"error": f"Distillation process exited with code {return_code}"}
                })

        except Exception as e:
            self.opd_state = "error"
            logger.error(f"Distillation monitoring error: {e}")
            await training_manager.broadcast({
                "type": "opd_error",
                "data": {"error": str(e)}
            })

    def _parse_metrics_from_log(self, log_file: str):
        """Parse metrics from the log file"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Look for the most recent metrics line
            for line in reversed(lines[-50:]):  # Check last 50 lines
                if "Step" in line and "KL Loss" in line:
                    # Parse: "Step 123/1000 | KL Loss: 0.234 | Token Agr: 78.5% | ETA: 12m"
                    try:
                        parts = line.split("|")

                        # Parse step
                        step_part = parts[0].strip()
                        if "/" in step_part:
                            step_str = step_part.split()[1].split("/")[0]
                            self.opd_metrics["step"] = int(step_str)

                        # Parse KL loss
                        for part in parts:
                            if "KL Loss" in part:
                                kl_str = part.split(":")[1].strip()
                                self.opd_metrics["kl_loss"] = float(kl_str)
                            elif "Token Agr" in part:
                                agr_str = part.split(":")[1].strip().replace("%", "")
                                self.opd_metrics["token_agreement_pct"] = float(agr_str)

                        # Update progress
                        total_steps = self.opd_metrics.get("total_steps", 1000)
                        current_step = self.opd_metrics.get("step", 0)
                        self.opd_metrics["progress_pct"] = (current_step / total_steps) * 100

                        break
                    except Exception as e:
                        logger.debug(f"Error parsing metrics line: {e}")
                        continue
        except Exception as e:
            logger.debug(f"Error parsing log file: {e}")

    def _save_run_metadata(self, run_id: str, config: Dict[str, Any]):
        """Save run metadata to file"""
        metadata = {
            "run_id": run_id,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "config": config,
            "teacher_model": os.path.basename(config.get("teacher_model_path", "")),
            "student_model": os.path.basename(config.get("base_model_path", "")),
            "student_adapter": os.path.basename(config.get("student_adapter_path", ""))
        }

        metadata_file = os.path.join(self.runs_dir, f"{run_id}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _update_run_metadata(self, run_id: str, updates: Dict[str, Any]):
        """Update run metadata"""
        metadata_file = os.path.join(self.runs_dir, f"{run_id}_metadata.json")

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            metadata.update(updates)

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint path"""
        if not self.current_run_id:
            return None

        checkpoint_dir = os.path.join(self.checkpoint_base_dir, self.current_run_id)

        if not os.path.exists(checkpoint_dir):
            return None

        try:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors")]
            if checkpoints:
                checkpoints.sort()
                return os.path.join(checkpoint_dir, checkpoints[-1])
        except Exception as e:
            logger.error(f"Error finding checkpoint: {e}")

        return None

# Global training manager instance
training_manager = TrainingManager()

# Global OPD manager instance
opd_manager = OPDManager()

# Evaluation Manager
class EvaluationManager:
    """Manages model evaluation processes"""
    
    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.evaluation_state = "idle"  # idle, running, completed, error
        self.progress = 0
        self.total_questions = 0
        self.current_question = 0
        self.adapter_name: Optional[str] = None
        self.is_base_model = False
        self.result: Optional[Dict] = None
        self.error: Optional[str] = None
    
    async def start_evaluation(self, adapter_name: str, training_data_path: Optional[str], 
                               num_questions: int = 20, evaluate_base_model: bool = False) -> bool:
        """Start evaluation process"""
        if self.evaluation_state == "running":
            raise HTTPException(status_code=400, detail="Evaluation already running")
        
        try:
            self.evaluation_state = "running"
            self.progress = 0
            self.total_questions = num_questions
            self.current_question = 0
            self.adapter_name = adapter_name
            self.is_base_model = evaluate_base_model
            self.result = None
            self.error = None
            
            # Build command to run evaluation script
            python_path = '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/.venv/bin/python'
            script_path = '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/evaluate_adapters.py'
            
            cmd = [
                python_path,
                script_path,
                '--adapter', adapter_name,
                '--num-questions', str(num_questions)
            ]
            
            if evaluate_base_model:
                cmd.append('--base-model')
            
            if training_data_path:
                cmd.extend(['--training-data', training_data_path])
            
            # Run in background
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Start monitoring in background
            asyncio.create_task(self._monitor_evaluation())
            
            return True
            
        except Exception as e:
            self.evaluation_state = "error"
            self.error = str(e)
            logger.error(f"Failed to start evaluation: {e}")
            return False
    
    async def _monitor_evaluation(self):
        """Monitor evaluation process"""
        try:
            if not self.current_process:
                return
            
            # Wait for process to complete
            stdout, stderr = self.current_process.communicate(timeout=600)
            
            if self.current_process.returncode == 0:
                # Parse result from stdout
                try:
                    # Look for JSON result in output
                    import re
                    # Try to find JSON between markers first
                    json_match = re.search(r'JSON_RESULT_START\s*\n(.*?)\n\s*JSON_RESULT_END', stdout, re.DOTALL)
                    if json_match:
                        self.result = json.loads(json_match.group(1))
                    else:
                        # Fallback to searching for scores object
                        json_match = re.search(r'\{.*"scores".*\}', stdout, re.DOTALL)
                        if json_match:
                            self.result = json.loads(json_match.group())
                        else:
                            raise ValueError("No valid JSON result found in output")
                    
                    self.evaluation_state = "completed"
                    self.progress = 100
                except Exception as e:
                    self.evaluation_state = "error"
                    self.error = f"Failed to parse result: {str(e)}"
            else:
                self.evaluation_state = "error"
                self.error = stderr or "Evaluation failed"
                
        except subprocess.TimeoutExpired:
            self.evaluation_state = "error"
            self.error = "Evaluation timed out"
        except Exception as e:
            self.evaluation_state = "error"
            self.error = str(e)
    
    def get_status(self) -> Dict:
        """Get current evaluation status"""
        return {
            "running": self.evaluation_state == "running",
            "progress": self.progress,
            "current_question": self.current_question,
            "total_questions": self.total_questions,
            "adapter_name": self.adapter_name,
            "error": self.error
        }
    
    def get_result(self) -> Optional[Dict]:
        """Get evaluation result"""
        return self.result

# Global evaluation manager instance
evaluation_manager = EvaluationManager()

# REST API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/models")
async def list_models():
    """List available models"""
    models_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model"
    models = []
    
    try:
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    # Check if it's a valid model directory (contains config.json)
                    config_path = os.path.join(item_path, "config.json")
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                models.append({
                                    "name": item,
                                    "path": item_path,
                                    "model_type": config.get("model_type", "unknown"),
                                    "vocab_size": config.get("vocab_size", 0)
                                })
                        except Exception as e:
                            logger.warning(f"Could not read config for {item}: {e}")
                            models.append({
                                "name": item,
                                "path": item_path,
                                "model_type": "unknown",
                                "vocab_size": 0
                            })
    except Exception as e:
        logger.error(f"Error listing models: {e}")
    
    return {"models": models}

@app.get("/adapters")
async def list_adapters():
    """List available LoRA adapters (both regular and nested learning)"""
    adapters_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters"
    nested_learning_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints"
    adapters = []

    try:
        # Scan regular LoRA adapters
        if os.path.exists(adapters_dir):
            for item in os.listdir(adapters_dir):
                item_path = os.path.join(adapters_dir, item)
                if os.path.isdir(item_path):
                    # Check if directory contains adapter files
                    has_adapters = False
                    for file in os.listdir(item_path):
                        if file.endswith(('.safetensors', '.npz', '.bin')):
                            has_adapters = True
                            break

                    if has_adapters:
                        # Try to read adapter config
                        metadata = {}
                        adapter_config = os.path.join(item_path, "adapter_config.json")
                        if os.path.exists(adapter_config):
                            try:
                                with open(adapter_config, 'r') as f:
                                    metadata = json.load(f)
                            except Exception as e:
                                logger.warning(f"Could not read adapter config for {item}: {e}")

                        adapters.append({
                            "name": item,
                            "path": item_path,
                            "type": "standard",
                            "lora_rank": metadata.get("r", "unknown"),
                            "lora_alpha": metadata.get("lora_alpha", "unknown")
                        })

        # Scan nested learning adapters
        if os.path.exists(nested_learning_dir):
            for item in os.listdir(nested_learning_dir):
                if item.startswith('.'):  # Skip hidden files
                    continue
                item_path = os.path.join(nested_learning_dir, item)
                if os.path.isdir(item_path):
                    # Check for best checkpoint
                    best_checkpoint = os.path.join(item_path, "checkpoints", "best")
                    if os.path.exists(best_checkpoint):
                        adapter_file = os.path.join(best_checkpoint, "adapters.safetensors")
                        if os.path.exists(adapter_file):
                            # Try to read adapter config
                            metadata = {}
                            adapter_config = os.path.join(best_checkpoint, "adapter_config.json")
                            if os.path.exists(adapter_config):
                                try:
                                    with open(adapter_config, 'r') as f:
                                        metadata = json.load(f)
                                except Exception as e:
                                    logger.warning(f"Could not read adapter config for {item}: {e}")

                            adapters.append({
                                "name": f"{item} (nested)",
                                "path": best_checkpoint,
                                "type": "nested_learning",
                                "lora_rank": metadata.get("lora_rank", "unknown"),
                                "lora_alpha": metadata.get("lora_alpha", "unknown")
                            })
    except Exception as e:
        logger.error(f"Error listing adapters: {e}")

    return {"adapters": adapters}

@app.get("/training/status")
async def get_training_status():
    """Get current training status"""
    return {
        "state": training_manager.training_state,
        "metrics": training_manager.training_metrics,
        "config": asdict(training_manager.current_config) if training_manager.current_config else None
    }

@app.post("/training/start")
async def start_training(config_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Start training with given configuration"""
    try:
        # Build config dict - only include values that were provided by GUI
        config_dict = {
            "model_path": config_data["model_path"],
            "train_data_path": config_data["train_data_path"],
            "val_data_path": config_data.get("val_data_path", ""),
        }

        # Add optional parameters only if provided (to use GUI values, not backend defaults)
        if "learning_rate" in config_data:
            config_dict["learning_rate"] = config_data["learning_rate"]
        if "batch_size" in config_data:
            config_dict["batch_size"] = config_data["batch_size"]
        if "max_seq_length" in config_data:
            config_dict["max_seq_length"] = config_data["max_seq_length"]
        if "iterations" in config_data:
            config_dict["iterations"] = config_data["iterations"]
        if "steps_per_report" in config_data:
            config_dict["steps_per_report"] = config_data["steps_per_report"]
        if "steps_per_eval" in config_data:
            config_dict["steps_per_eval"] = config_data["steps_per_eval"]
        if "save_every" in config_data:
            config_dict["save_every"] = config_data["save_every"]
        if "early_stop" in config_data:
            config_dict["early_stop"] = config_data["early_stop"]
        if "patience" in config_data:
            config_dict["patience"] = config_data["patience"]
        if "adapter_name" in config_data:
            config_dict["adapter_name"] = config_data["adapter_name"]

        config = TrainingConfig(**config_dict)
        
        success = await training_manager.start_training(config)
        if success:
            return {"status": "started", "message": "Training started successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start training")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/training/stop")
async def stop_training():
    """Stop current training"""
    await training_manager.stop_training()
    return {"status": "stopped", "message": "Training stop requested"}

@app.get("/training/logs")
async def get_training_logs():
    """Get training logs"""
    try:
        if os.path.exists(training_manager.log_file):
            with open(training_manager.log_file, 'r') as f:
                logs = f.readlines()[-100:]  # Last 100 lines
            return {"logs": logs}
        else:
            return {"logs": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_sessions():
    """Get list of all training sessions"""
    sessions = training_manager.get_all_sessions()
    return {"sessions": sessions}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get details of a specific training session"""
    try:
        session_file = os.path.join(training_manager.sessions_dir, f"session_{session_id}.json")
        if not os.path.exists(session_file):
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        return session_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/load")
async def load_session(session_id: str):
    """Load a specific training session"""
    try:
        success = training_manager.load_session(session_id)
        if success:
            # Broadcast the loaded session state to any connected clients
            await training_manager.broadcast({
                "type": "session_loaded",
                "data": {
                    "session_id": session_id,
                    "state": training_manager.training_state,
                    "metrics": training_manager.training_metrics,
                    "config": asdict(training_manager.current_config) if training_manager.current_config else None
                }
            })
            return {"status": "success", "message": f"Session {session_id} loaded successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found or could not be loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific training session"""
    try:
        success = training_manager.delete_session(session_id)
        if success:
            return {"status": "success", "message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/test-base")
async def test_base_model(request_data: dict):
    """Test the base model (without adapter) with a prompt"""
    try:
        prompt = request_data.get("prompt", "")
        max_tokens = request_data.get("max_tokens", 1024)
        temperature = request_data.get("temperature", 0.7)
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Get the model path from the latest training config
        if not training_manager.current_config:
            raise HTTPException(status_code=400, detail="No training session found. Please complete a training session first.")
        
        config = training_manager.current_config
        model_path = config.model_path
        
        # Use MLX to generate text with the base model only (no adapter)
        python_path = '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/.venv/bin/python'
        
        # Create a simple inference command using mlx-lm for base model only
        cmd = [
            python_path, '-c', f'''
import mlx.core as mx
from mlx_lm import load, generate

# Load the base model WITHOUT adapter
model, tokenizer = load("{model_path}")

# Generate text
prompt = """{prompt}"""

# Try different parameter combinations based on MLX version
response = None
try:
    # Try with temp parameter
    response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens})
except TypeError:
    try:
        # Try with temperature parameter  
        response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens})
    except TypeError:
        try:
            # Try with just basic parameters
            response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens})
        except Exception as e:
            print("RESPONSE_START")
            print(f"Error: Could not generate response - {{str(e)}}")
            print("RESPONSE_END")
            exit(1)

print("RESPONSE_START")
print(response)
print("RESPONSE_END")
'''
        ]
        
        # Run the inference
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for large models
        )
        
        if process.returncode != 0:
            logger.error(f"Base model test failed: {process.stderr}")
            raise HTTPException(status_code=500, detail=f"Base model inference failed: {process.stderr}")
        
        # Parse the response
        output = process.stdout
        if "RESPONSE_START" in output and "RESPONSE_END" in output:
            response_start = output.find("RESPONSE_START") + len("RESPONSE_START\n")
            response_end = output.find("RESPONSE_END")
            generated_text = output[response_start:response_end].strip()
        else:
            generated_text = output.strip()
        
        return {
            "success": True,
            "prompt": prompt,
            "response": generated_text,
            "model_info": {
                "base_model": model_path.split('/')[-1],
                "adapter": "none (base model)",
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Base model inference timed out")
    except Exception as e:
        logger.error(f"Base model test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/test")
async def test_model(request_data: dict):
    """Test the fine-tuned model with a prompt"""
    try:
        prompt = request_data.get("prompt", "")
        max_tokens = request_data.get("max_tokens", 1024)
        temperature = request_data.get("temperature", 0.7)
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Get the model and adapter paths from the latest training config
        if not training_manager.current_config:
            raise HTTPException(status_code=400, detail="No training session found. Please complete a training session first.")

        config = training_manager.current_config
        model_path = config.model_path

        # Use the stored adapter path if available, otherwise construct it
        if training_manager.current_adapter_path:
            adapter_path = training_manager.current_adapter_path
            adapter_name = "current"
        else:
            adapter_name = config.adapter_name
            # Check if this is a nested learning adapter
            nested_learning_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints"
            nested_adapter_path = os.path.join(nested_learning_dir, adapter_name, "checkpoints", "best")

            if os.path.exists(nested_adapter_path):
                # It's a nested learning adapter
                adapter_path = nested_adapter_path
            else:
                # Regular adapter
                adapter_path = os.path.join("/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters", adapter_name)

        # Verify adapter exists
        if not os.path.exists(adapter_path):
            raise HTTPException(status_code=404, detail=f"Fine-tuned adapter not found at {adapter_path}")

        # For regular training, check for best model
        best_adapter_file = os.path.join(adapter_path, "best_adapters.safetensors")
        latest_adapter_file = os.path.join(adapter_path, "adapters.safetensors")

        if os.path.exists(best_adapter_file) and not os.path.exists(latest_adapter_file):
            # Copy best model to adapters.safetensors so MLX can find it
            import shutil
            shutil.copy2(best_adapter_file, latest_adapter_file)
            model_type = "best"
        else:
            model_type = "latest"
        
        # Use MLX to generate text with the fine-tuned model
        # This is a simplified implementation - you might want to use a proper MLX inference script
        python_path = '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/.venv/bin/python'
        
        # Create a simple inference command using mlx-lm
        cmd = [
            python_path, '-c', f'''
import mlx.core as mx
from mlx_lm import load, generate

# Load the base model and adapter
model, tokenizer = load("{model_path}", adapter_path="{adapter_path}")

# Generate text
prompt = """{prompt}"""

# Try different parameter combinations based on MLX version
response = None
try:
    # Try with temp parameter
    response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens})
except TypeError:
    try:
        # Try with temperature parameter  
        response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens})
    except TypeError:
        try:
            # Try with just basic parameters
            response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens})
        except Exception as e:
            print("RESPONSE_START")
            print(f"Error: Could not generate response - {{str(e)}}")
            print("RESPONSE_END")
            exit(1)

print("RESPONSE_START")
print(response)
print("RESPONSE_END")
'''
        ]
        
        # Run the inference
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for large models
        )
        
        if process.returncode != 0:
            logger.error(f"Model test failed: {process.stderr}")
            raise HTTPException(status_code=500, detail=f"Model inference failed: {process.stderr}")
        
        # Parse the response
        output = process.stdout
        if "RESPONSE_START" in output and "RESPONSE_END" in output:
            response_start = output.find("RESPONSE_START") + len("RESPONSE_START\n")
            response_end = output.find("RESPONSE_END")
            generated_text = output[response_start:response_end].strip()
        else:
            generated_text = output.strip()
        
        return {
            "success": True,
            "prompt": prompt,
            "response": generated_text,
            "model_info": {
                "base_model": model_path.split('/')[-1],
                "adapter": adapter_name,
                "adapter_type": model_type,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Model inference timed out")
    except Exception as e:
        logger.error(f"Model test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.get("/models/available")
async def get_available_models():
    """Get all available models and their adapters"""
    try:
        base_model_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model"
        adapter_base_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters"
        
        models = []
        
        # Scan for available base models
        if os.path.exists(base_model_dir):
            for item in os.listdir(base_model_dir):
                model_path = os.path.join(base_model_dir, item)
                if os.path.isdir(model_path):
                    # Check for adapters for this model
                    adapters = []
                    if os.path.exists(adapter_base_dir):
                        for adapter_name in os.listdir(adapter_base_dir):
                            adapter_dir = os.path.join(adapter_base_dir, adapter_name)
                            if os.path.isdir(adapter_dir):
                                # Check if this adapter has weights
                                adapter_file = os.path.join(adapter_dir, "adapters.safetensors")
                                best_adapter_file = os.path.join(adapter_dir, "best_adapters.safetensors")
                                if os.path.exists(adapter_file) or os.path.exists(best_adapter_file):
                                    adapters.append({
                                        "name": adapter_name,
                                        "has_best": os.path.exists(best_adapter_file),
                                        "has_latest": os.path.exists(adapter_file),
                                        "path": adapter_dir
                                    })
                    
                    models.append({
                        "name": item,
                        "path": model_path,
                        "adapters": adapters
                    })
        
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/inference")
async def model_inference(request_data: dict):
    """Model-agnostic inference endpoint"""
    try:
        prompt = request_data.get("prompt", "").strip()
        model_name = request_data.get("model_name")
        adapter_name = request_data.get("adapter_name")  # Optional
        max_tokens = request_data.get("max_tokens", 100)
        temperature = request_data.get("temperature", 0.7)
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name is required")
        
        # Build model path
        base_model_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model"
        model_path = os.path.join(base_model_dir, model_name)
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        
        # Build adapter path if specified
        adapter_path = None
        adapter_type = "none"
        if adapter_name:
            # Check if this is a nested learning adapter
            if "(nested)" in adapter_name:
                # Extract the base name without the "(nested)" suffix
                base_name = adapter_name.replace(" (nested)", "")
                nested_learning_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/backend/nested_learning/checkpoints"
                adapter_dir = os.path.join(nested_learning_dir, base_name, "checkpoints", "best")

                if os.path.exists(adapter_dir):
                    adapter_file = os.path.join(adapter_dir, "adapters.safetensors")
                    if os.path.exists(adapter_file):
                        adapter_path = adapter_dir
                        adapter_type = "nested_learning"
            else:
                # Regular adapter
                adapter_base_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters"
                adapter_dir = os.path.join(adapter_base_dir, adapter_name)

                if os.path.exists(adapter_dir):
                    best_adapter_file = os.path.join(adapter_dir, "best_adapters.safetensors")
                    latest_adapter_file = os.path.join(adapter_dir, "adapters.safetensors")

                    # Use best model if available, otherwise latest
                    if os.path.exists(best_adapter_file):
                        import shutil
                        shutil.copy2(best_adapter_file, latest_adapter_file)
                        adapter_type = "best"
                    elif os.path.exists(latest_adapter_file):
                        adapter_type = "latest"

                    if os.path.exists(latest_adapter_file):
                        adapter_path = adapter_dir
        
        # Use MLX to generate text
        python_path = '/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/.venv/bin/python'
        
        if adapter_path:
            # Fine-tuned model inference
            cmd = [
                python_path, '-c', f'''
import mlx.core as mx
from mlx_lm import load, generate

try:
    model, tokenizer = load("{model_path}", adapter_path="{adapter_path}")
    prompt = """{prompt}"""
    
    response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens})
    print("RESPONSE_START")
    print(response)
    print("RESPONSE_END")
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
'''
            ]
        else:
            # Base model inference
            cmd = [
                python_path, '-c', f'''
import mlx.core as mx
from mlx_lm import load, generate

try:
    model, tokenizer = load("{model_path}")
    prompt = """{prompt}"""
    
    response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens})
    print("RESPONSE_START")
    print(response)
    print("RESPONSE_END")
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
'''
            ]
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for large models
        )
        
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {process.stderr}")
        
        # Extract the response between markers
        output = process.stdout
        if "RESPONSE_START" in output and "RESPONSE_END" in output:
            start_idx = output.find("RESPONSE_START") + len("RESPONSE_START")
            end_idx = output.find("RESPONSE_END")
            response_text = output[start_idx:end_idx].strip()
        else:
            response_text = output.strip()
        
        return {
            "success": True,
            "prompt": prompt,
            "response": response_text,
            "model_info": {
                "base_model": model_name,
                "adapter": adapter_name if adapter_name else "none (base model)",
                "adapter_type": adapter_type,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== On-Policy Distillation (OPD) Endpoints =====

@app.post("/opd/start")
async def opd_start(config_data: Dict[str, Any]):
    """
    Start On-Policy Distillation training.

    Request body:
    {
        "base_model_path": "/path/to/qwen2.5-7b",
        "teacher_model_path": "/path/to/qwen2.5-32b",
        "student_adapter_path": "/path/to/sft_adapter",
        "validation_prompts_path": "/path/to/val_prompts.jsonl",
        "num_steps": 1000,
        "batch_size": 4,
        "temperature": 2.0,
        "kl_weight": 0.8,
        "learning_rate": 0.00001
    }
    """
    try:
        result = await opd_manager.start_distillation(config_data)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"OPD start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/opd/status")
async def opd_status():
    """
    Get current OPD training status.

    Response:
    {
        "state": "running",  // idle, running, completed, error
        "run_id": "distill_20250128_143022",
        "metrics": {
            "step": 450,
            "total_steps": 1000,
            "progress_pct": 45.0,
            "kl_loss": 0.234,
            "token_agreement_pct": 78.5,
            "started_at": "2025-01-28T14:30:22"
        },
        "config": { ... }
    }
    """
    try:
        return opd_manager.get_status()
    except Exception as e:
        logger.error(f"OPD status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/opd/stop")
async def opd_stop():
    """
    Stop current OPD training.

    Response:
    {
        "status": "stopped",
        "final_step": 450,
        "checkpoint_path": "/path/to/checkpoint",
        "message": "Distillation stopped by user"
    }
    """
    try:
        return await opd_manager.stop_distillation()
    except Exception as e:
        logger.error(f"OPD stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/opd/metrics")
async def opd_metrics(run_id: Optional[str] = None):
    """
    Get metrics for a specific OPD run.

    Query params:
        run_id: Optional run ID. If not provided, uses current run.

    Response:
    {
        "run_id": "distill_20250128_143022",
        "total_steps": 1000,
        "metrics_history": [
            {
                "step": 0,
                "kl_loss": 1.234,
                "token_agreement_pct": 45.2,
                "timestamp": "2025-01-28T14:30:22"
            },
            ...
        ]
    }
    """
    try:
        return opd_manager.get_metrics(run_id)
    except Exception as e:
        logger.error(f"OPD metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/opd/runs")
async def opd_runs():
    """
    Get list of all OPD training runs.

    Response:
    {
        "runs": [
            {
                "run_id": "distill_20250128_143022",
                "status": "completed",
                "started_at": "2025-01-28T14:30:22",
                "completed_at": "2025-01-28T15:45:10",
                "teacher_model": "qwen2.5-32b",
                "student_model": "qwen2.5-7b",
                "config": { ... }
            },
            ...
        ]
    }
    """
    try:
        runs = opd_manager.get_all_runs()
        return {"runs": runs}
    except Exception as e:
        logger.error(f"OPD runs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluation/start")
async def start_evaluation(request_data: Dict[str, Any]):
    """Start model evaluation using LLM-as-judge (Cerebras-based system)"""
    try:
        adapter_name = request_data.get("adapter_name")
        training_data_path = request_data.get("training_data_path")
        num_questions = request_data.get("num_questions", 20)
        evaluate_base_model = request_data.get("evaluate_base_model", False)
        
        if not adapter_name:
            raise HTTPException(status_code=400, detail="adapter_name is required")
        
        success = await evaluation_manager.start_evaluation(
            adapter_name, training_data_path, num_questions, evaluate_base_model
        )
        
        if success:
            return {
                "success": True,
                "message": "Evaluation started",
                "adapter_name": adapter_name
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start evaluation")
            
    except Exception as e:
        logger.error(f"Evaluation start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluation/status")
async def get_evaluation_status():
    """Get evaluation status"""
    return evaluation_manager.get_status()

@app.get("/api/evaluation/result")
async def get_evaluation_result():
    """Get evaluation result"""
    result = evaluation_manager.get_result()
    if result:
        return {"success": True, "result": result}
    else:
        raise HTTPException(status_code=404, detail="No evaluation result available")

# New Tier 0+1 Evaluation Endpoints
@app.post("/api/evaluate/adapter")
async def evaluate_adapter(request_data: Dict[str, Any]):
    """Evaluate adapter using Tier 0 + Tier 1"""
    try:
        from combined_evaluator import CombinedEvaluator
        
        adapter_name = request_data.get("adapter_name")
        max_samples = request_data.get("max_samples", 20)
        
        if not adapter_name:
            raise HTTPException(status_code=400, detail="adapter_name is required")
        
        evaluator = CombinedEvaluator()
        try:
            report = evaluator.evaluate_adapter(
                adapter_name,
                include_base=False,
                max_samples=max_samples
            )
            return {"success": True, "result": report}
        finally:
            evaluator.cleanup()
            
    except Exception as e:
        logger.error(f"Adapter evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate/base-model")
async def evaluate_base_model(request_data: Dict[str, Any]):
    """Evaluate base model using Tier 1 only"""
    try:
        from combined_evaluator import CombinedEvaluator
        
        max_samples = request_data.get("max_samples", 20)
        
        evaluator = CombinedEvaluator()
        try:
            report = evaluator.evaluate_base_model(max_samples=max_samples)
            return {"success": True, "result": report}
        finally:
            evaluator.cleanup()
            
    except Exception as e:
        logger.error(f"Base model evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate/compare")
async def compare_adapters(request_data: Dict[str, Any]):
    """Compare two adapters using Tier 0 + Tier 1"""
    try:
        from combined_evaluator import CombinedEvaluator
        
        adapter1 = request_data.get("adapter1")
        adapter2 = request_data.get("adapter2")
        max_samples = request_data.get("max_samples", 20)
        
        if not adapter1 or not adapter2:
            raise HTTPException(status_code=400, detail="adapter1 and adapter2 are required")
        
        evaluator = CombinedEvaluator()
        try:
            comparison = evaluator.compare_adapters(
                adapter1,
                adapter2,
                include_base=False,
                max_samples=max_samples
            )
            return {"success": True, "result": comparison}
        finally:
            evaluator.cleanup()
            
    except Exception as e:
        logger.error(f"Adapter comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TINKER CLOUD FINE-TUNING ENDPOINTS
# ============================================================================

# Initialize Tinker client
tinker_client = None

def get_tinker_client():
    """Get or create Tinker client instance"""
    global tinker_client
    if tinker_client is None:
        from tinker_client import TinkerTrainingClient
        tinker_client = TinkerTrainingClient()
    return tinker_client

@app.post("/api/tinker/start-training")
async def start_tinker_training(request_data: Dict[str, Any]):
    """
    Start a Tinker cloud fine-tuning job.
    
    Request body:
    {
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "train_data_path": "/path/to/train.jsonl",
        "val_data_path": "/path/to/val.jsonl",  // optional
        "adapter_name": "my_adapter",
        "learning_rate": 1e-5,
        "batch_size": 1,
        "num_epochs": 3,
        "lora_rank": 64,
        "max_seq_length": 2048
    }
    """
    try:
        client = get_tinker_client()
        result = await client.start_training(**request_data)
        return result
    except Exception as e:
        logger.error(f"Tinker training start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tinker/status/{job_id}")
async def get_tinker_status(job_id: str):
    """
    Get status of a Tinker training job.
    
    Returns:
    {
        "status": "training" | "completed" | "error",
        "job_id": "...",
        "message": "...",
        "ready_for_download": true/false
    }
    """
    try:
        client = get_tinker_client()
        status = await client.get_training_status(job_id)
        return status
    except Exception as e:
        logger.error(f"Tinker status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tinker/download")
async def download_tinker_model(request_data: Dict[str, Any]):
    """
    Download trained model from Tinker to local artifacts.
    
    Request body:
    {
        "job_id": "tinker_...",
        "adapter_name": "my_adapter"
    }
    """
    try:
        job_id = request_data.get("job_id")
        adapter_name = request_data.get("adapter_name")
        
        if not job_id or not adapter_name:
            raise HTTPException(status_code=400, detail="job_id and adapter_name are required")
        
        client = get_tinker_client()
        result = await client.download_model(job_id, adapter_name)
        return result
    except Exception as e:
        logger.error(f"Tinker download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tinker/models")
async def list_tinker_models():
    """
    List all Tinker-trained models in artifacts directory.
    
    Returns:
    {
        "models": [
            {
                "adapter_name": "...",
                "base_model": "...",
                "training_source": "tinker",
                "completed_at": "...",
                ...
            }
        ]
    }
    """
    try:
        client = get_tinker_client()
        models = client.list_trained_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Tinker models list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time training updates"""
    await websocket.accept()
    await training_manager.add_websocket(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        training_manager.remove_websocket(websocket)

@app.get("/api/datasets")
async def list_datasets():
    """List available JSONL datasets."""
    try:
        datasets_dir = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/datasets"
        datasets = []
        
        if os.path.exists(datasets_dir):
            for file in os.listdir(datasets_dir):
                if file.endswith('.jsonl'):
                    full_path = os.path.join(datasets_dir, file)
                    datasets.append(full_path)
        
        return {"datasets": sorted(datasets)}
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return {"datasets": []}

# Register fusion API router
try:
    from fusion_api import router as fusion_router
    app.include_router(fusion_router)
    logger.info("Fusion API router registered")
except ImportError as e:
    logger.warning(f"Could not load fusion API: {e}")

# Register nested learning API router
try:
    from nested_learning_api import router as nested_learning_router
    app.include_router(nested_learning_router)
    logger.info("Nested Learning API router registered")
except ImportError as e:
    logger.warning(f"Could not load nested learning API: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
