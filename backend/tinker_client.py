#!/usr/bin/env python3
"""
Tinker Cloud Fine-Tuning Integration

INPUT FILES:
- Training datasets in JSONL format (from GUI selection)
- Validation datasets in JSONL format (optional)

OUTPUT FILES:
- Trained LoRA adapters downloaded to /artifacts/lora_adapters/tinker_<name>/
- model_info.json with metadata for each trained model
- Training logs and metrics

This module provides a clean interface between the Droid GUI and Tinker's
cloud fine-tuning API, handling training, monitoring, and model downloads.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import tinker
from tinker import types

logger = logging.getLogger(__name__)


class TinkerTrainingClient:
    """Manages Tinker cloud fine-tuning operations for the GUI"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tinker client.
        
        Args:
            api_key: Tinker API key (defaults to TINKER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('TINKER_API_KEY')
        if not self.api_key:
            raise ValueError("TINKER_API_KEY not found in environment or provided")
        
        os.environ['TINKER_API_KEY'] = self.api_key
        
        self.service_client = None
        self.training_client = None
        self.sampling_client = None
        self.current_job_id = None
        self.current_config = None
        self.status = "idle"
        self.message = ""
        self.ready_for_download = False
        
        # Output directory for downloaded models
        self.output_dir = Path("/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _ensure_service_client(self):
        """Ensure service client is initialized"""
        if not self.service_client:
            self.service_client = tinker.ServiceClient()
    
    async def start_training(
        self,
        base_model: str,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        adapter_name: str = "tinker_adapter",
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        num_epochs: int = 3,
        lora_rank: int = 64,
        max_seq_length: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Start a Tinker cloud fine-tuning job.
        
        Args:
            base_model: HuggingFace model ID (e.g., "Qwen/Qwen3-4B-Instruct-2507")
            train_data_path: Path to training JSONL file
            val_data_path: Path to validation JSONL file (optional)
            adapter_name: Name for the trained adapter
            learning_rate: Learning rate for training
            batch_size: Batch size
            num_epochs: Number of training epochs
            lora_rank: LoRA rank (8, 16, 32, 64, 128)
            max_seq_length: Maximum sequence length
            
        Returns:
            Dict with job_id and initial status
        """
        try:
            logger.info(f"Starting Tinker training: {adapter_name}")
            logger.info(f"Base model: {base_model}")
            logger.info(f"Training data: {train_data_path}")
            
            self._ensure_service_client()
            
            # Create training client
            self.training_client = self.service_client.create_lora_training_client(
                base_model=base_model,
                rank=lora_rank
            )
            
            # Get tokenizer
            tokenizer = self.training_client.get_tokenizer()
            
            # Load and process training data
            logger.info("Loading training data...")
            train_data = self._load_jsonl(train_data_path)
            logger.info(f"Loaded {len(train_data)} training examples")
            
            # Process conversations into Tinker format
            train_datums = []
            for i, raw_entry in enumerate(train_data):
                try:
                    conversation = self._normalize_entry(raw_entry)
                    datum = self._process_conversation(conversation, tokenizer)
                    train_datums.append(datum)
                except Exception as e:
                    logger.warning(f"Failed to process training example {i}: {e}")
            
            logger.info(f"Processed {len(train_datums)} training examples")
            
            # Store datums for training loop
            self.train_datums = train_datums
            self.val_datums = []
            
            # Load and process validation data if provided
            if val_data_path and os.path.exists(val_data_path):
                logger.info("Loading validation data...")
                val_data = self._load_jsonl(val_data_path)
                logger.info(f"Loaded {len(val_data)} validation examples")
                
                for i, raw_entry in enumerate(val_data):
                    try:
                        conversation = self._normalize_entry(raw_entry)
                        datum = self._process_conversation(conversation, tokenizer)
                        self.val_datums.append(datum)
                    except Exception as e:
                        logger.warning(f"Failed to process validation example {i}: {e}")
                
                logger.info(f"Processed {len(self.val_datums)} validation examples")
            
            # Store configuration
            self.current_config = {
                "adapter_name": adapter_name,
                "base_model": base_model,
                "train_data_path": train_data_path,
                "val_data_path": val_data_path,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "lora_rank": lora_rank,
                "max_seq_length": max_seq_length,
                "started_at": datetime.now().isoformat()
            }
            
            # Generate job ID
            self.current_job_id = f"tinker_{adapter_name}_{int(time.time())}"
            
            # Initialize status
            self.status = "starting"
            self.message = "Initializing training..."
            self.ready_for_download = False
            
            # Start training loop in background
            asyncio.create_task(self._training_loop(self.current_config))
            
            logger.info(f"Training job started: {self.current_job_id}")
            
            return {
                "success": True,
                "job_id": self.current_job_id,
                "adapter_name": adapter_name,
                "status": "training",
                "message": "Tinker training job started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start Tinker training: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to start training: {e}"
            }

    async def _training_loop(self, config):
        """Background training loop"""
        try:
            self.status = "training"
            self.message = "Training started..."
            
            batch_size = config["batch_size"]
            num_epochs = config["num_epochs"]
            learning_rate = config["learning_rate"]
            
            # Create batches
            batches = [self.train_datums[i:i + batch_size] for i in range(0, len(self.train_datums), batch_size)]
            total_steps = num_epochs * len(batches)
            current_step = 0
            
            logger.info(f"Starting training loop: {num_epochs} epochs, {len(batches)} batches/epoch")
            
            for epoch in range(num_epochs):
                for i, batch in enumerate(batches):
                    # Forward Backward
                    future_fb = await self.training_client.forward_backward_async(
                        data=batch,
                        loss_fn="cross_entropy",
                        loss_fn_config=None
                    )
                    fb_output = await future_fb
                    
                    # Log loss
                    # Calculate loss manually as per SDK example
                    import numpy as np
                    logprobs = np.concatenate([
                        output['logprobs'].tolist() 
                        for output in fb_output.loss_fn_outputs
                    ])
                    weights = np.concatenate([
                        d.loss_fn_inputs['weights'].tolist() 
                        for d in batch
                    ])
                    
                    if weights.sum() > 0:
                        loss = -np.dot(logprobs, weights) / weights.sum()
                    else:
                        loss = 0.0
                    current_step += 1
                    
                    if i % 10 == 0:  # Log every 10 batches
                        logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(batches)}, Loss: {loss:.4f}")
                        self.message = f"Training: Epoch {epoch+1}/{num_epochs}, Step {current_step}/{total_steps}, Loss: {loss:.4f}"
                    
                    # Optim Step
                    adam_params = types.AdamParams(
                        learning_rate=learning_rate,
                        beta1=0.9,
                        beta2=0.999,
                        eps=1e-8
                    )
                    future_optim = await self.training_client.optim_step_async(adam_params=adam_params)
                    await future_optim
            
            self.status = "completed"
            self.message = "Training completed successfully"
            self.ready_for_download = True
            logger.info("Training loop completed successfully")
            
        except Exception as e:
            logger.error(f"Training loop failed: {e}")
            self.status = "error"
            self.message = f"Training failed: {str(e)}"
            self.ready_for_download = False
    
    async def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a Tinker training job.
        
        Args:
            job_id: Job ID returned from start_training
            
        Returns:
            Dict with current status and metrics
        """
        try:
            if not self.training_client or self.current_job_id != job_id:
                return {
                    "status": "error",
                    "message": "Job not found or no active training"
                }
            
            return {
                "status": self.status,
                "job_id": job_id,
                "message": self.message,
                "ready_for_download": self.ready_for_download
            }
                
        except Exception as e:
            logger.error(f"Failed to get training status: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def list_cloud_models(self, base_model: str) -> List[Dict[str, Any]]:
        """
        List models stored in Tinker cloud for a given base model.
        
        Args:
            base_model: Base model ID to filter by
            
        Returns:
            List of checkpoint info dicts
        """
        try:
            self._ensure_service_client()
            
            # Access underlying client for AsyncWeightsResource
            from tinker.resources.weights import AsyncWeightsResource
            from tinker._client import AsyncTinker
            
            api_key = os.environ.get("TINKER_API_KEY")
            if not api_key:
                logger.warning("TINKER_API_KEY not found in environment")
                return []
                
            base_client = AsyncTinker(api_key=api_key)

            weights_resource = AsyncWeightsResource(base_client)
            
            # List checkpoints
            response = await weights_resource.list(model_id=base_model)
            
            checkpoints = []
            for cp in response.checkpoints:
                # Only include training checkpoints (not sampler ones if any)
                if getattr(cp, "checkpoint_type", "training") == "training":
                    checkpoints.append({
                        "checkpoint_id": cp.checkpoint_id,
                        "created_at": cp.time.isoformat() if hasattr(cp, "time") else None,
                        "tinker_path": cp.tinker_path,
                        "base_model": base_model
                    })
            
            # Sort by time, newest first
            checkpoints.sort(key=lambda x: x.get("created_at") or "", reverse=True)
            
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list cloud models: {e}")
            return []

    async def download_model(
        self,
        job_id: Optional[str] = None,
        adapter_name: str = "tinker_adapter",
        base_model_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            # 1. Determine ID to use (Training Run ID)
            target_id = checkpoint_id or job_id
            if not target_id:
                 raise ValueError("Either job_id or checkpoint_id must be provided")
            
            logger.info(f"Downloading Tinker model {target_id}...")
            
            # 2. Determine base model
            base_model = base_model_id or (self.current_config["base_model"] if self.current_config else None)
            if not base_model:
                logger.warning("Base model ID not provided. Metadata may be incomplete.")
                base_model = "unknown"

            # 3. Get download URL using SDK
            logger.info("Getting download URL via SDK...")
            
            # Ensure API key is set for the process
            if self.api_key:
                os.environ['TINKER_API_KEY'] = self.api_key
            
            def get_url():
                import tinker
                sc = tinker.ServiceClient()
                rc = sc.create_rest_client()
                # Construct path using the target ID (Run ID)
                tinker_path = f"tinker://{target_id}/sampler_weights/final"
                logger.info(f"Requesting URL for path: {tinker_path}")
                future = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path)
                return future.result()

            # Run blocking SDK call in thread
            checkpoint_archive_url_response = await asyncio.to_thread(get_url)
            download_url = checkpoint_archive_url_response.url
            
            logger.info(f"Got download URL: {download_url[:50]}...")
            
            # 4. Download archive
            # Create output directory with tinker_ prefix
            adapter_dir = self.output_dir / f"tinker_{adapter_name}"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            
            archive_path = adapter_dir / "adapter_archive.tar"
            
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(download_url, follow_redirects=True, timeout=300.0)
                resp.raise_for_status()
                with open(archive_path, "wb") as f:
                    f.write(resp.content)
            
            logger.info(f"Downloaded archive to {archive_path}")
            
            # 5. Extract archive
            import tarfile
            import zipfile
            
            if tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, 'r') as tar_ref:
                    tar_ref.extractall(adapter_dir)
                logger.info("Extracted tar archive")
                os.remove(archive_path)
            elif zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(adapter_dir)
                logger.info("Extracted zip archive")
                os.remove(archive_path)
            else:
                logger.warning("Downloaded file is not a recognized archive. Keeping as is.")
            
            # 6. Download Base Model (if needed)
            if base_model != "unknown":
                try:
                    logger.info(f"Ensuring base model {base_model} is available locally...")
                    from huggingface_hub import snapshot_download
                    # This will download to default HF cache
                    snapshot_download(repo_id=base_model)
                    logger.info(f"Base model {base_model} verified/downloaded.")
                except Exception as e:
                    logger.warning(f"Failed to download base model {base_model}: {e}. You may need to download it manually.")

            # 7. Create model info
            model_info = {
                "training_source": "tinker",
                "adapter_name": adapter_name,
                "base_model": base_model,
                "lora_rank": self.current_config.get("lora_rank") if self.current_config else 64, # Default or unknown
                "learning_rate": self.current_config.get("learning_rate") if self.current_config else 1e-5,
                "num_epochs": self.current_config.get("num_epochs") if self.current_config else 3,
                "batch_size": self.current_config.get("batch_size") if self.current_config else 1,
                "max_seq_length": self.current_config.get("max_seq_length") if self.current_config else 2048,
                "train_data_path": self.current_config.get("train_data_path") if self.current_config else "unknown",
                "val_data_path": self.current_config.get("val_data_path") if self.current_config else None,
                "started_at": self.current_config.get("started_at") if self.current_config else datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "tinker_model": True,
                "tinker_job_id": job_id,
                "local_path": str(adapter_dir),
                "checkpoint_id": target_id
            }
            
            with open(adapter_dir / "model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
                
            return {
                "success": True,
                "local_path": str(adapter_dir),
                "adapter_name": adapter_name
            }
            
        except Exception as e:
            logger.error(f"Failed to download Tinker model: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to download model: {e}"
            }
    
    def list_trained_models(self) -> List[Dict[str, Any]]:
        models = []
        if not self.output_dir.exists():
            return models
            
        for path in self.output_dir.iterdir():
            if path.is_dir():
                info_path = path / "model_info.json"
                if info_path.exists():
                    try:
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                            # Only include Tinker models
                            if info.get("tinker_model") or info.get("training_source") == "tinker":
                                models.append(info)
                    except Exception as e:
                        logger.warning(f"Failed to read model info at {info_path}: {e}")
        
        # Sort by completion time (newest first)
        models.sort(key=lambda x: x.get("completed_at", ""), reverse=True)
        return models

    def _load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _normalize_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Already in chat format
        if "messages" in entry:
            return entry
            
        # 2. Prompt/Completion pair
        if "prompt" in entry and "completion" in entry:
            return {
                "messages": [
                    {"role": "user", "content": entry["prompt"]},
                    {"role": "assistant", "content": entry["completion"]}
                ]
            }
            
        # 3. Alpaca format
        if "instruction" in entry and "output" in entry:
            instruction = entry["instruction"]
            if entry.get("input"):
                instruction += f"\nInput: {entry['input']}"
            return {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": entry["output"]}
                ]
            }
            
        # 4. Text format (MLX style)
        # This is tricky because it's usually pre-formatted.
        # We'll try to treat it as a single assistant message if we can't do better,
        # but Tinker requires a prompt.
        # If it's just "text", we might have to skip or error.
        if "text" in entry:
            # Heuristic: split on "Assistant:" if present?
            text = entry["text"]
            parts = text.split("Assistant:")
            if len(parts) >= 2:
                prompt = parts[0].replace("User:", "").strip()
                completion = "Assistant:".join(parts[1:]).strip()
                return {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion}
                    ]
                }
            
        raise ValueError(f"Unsupported data format: {list(entry.keys())}")
    
    def _process_conversation(self, conversation: Dict, tokenizer) -> types.Datum:
        try:
            # Try to use Tinker Cookbook if available
            from tinker_cookbook.renderers import get_renderer
            from tinker_cookbook.train_utils import conversation_to_datum
            
            renderer = get_renderer("role_colon")
            datum = conversation_to_datum(
                conversation=conversation,
                tokenizer=tokenizer,
                renderer=renderer,
                train_on_what="last_assistant_only"
            )
            return datum
            
        except ImportError:
            # Fallback to manual processing
            return self._manual_process_conversation(conversation, tokenizer)
    
    def _manual_process_conversation(self, conversation: Dict, tokenizer) -> types.Datum:
        messages = conversation['messages']
        
        # Extract messages by role
        system_msg = next((msg for msg in messages if msg['role'] == 'system'), None)
        user_msgs = [msg for msg in messages if msg['role'] == 'user']
        assistant_msgs = [msg for msg in messages if msg['role'] == 'assistant']
        
        if not assistant_msgs:
            raise ValueError("No assistant messages found in conversation")
        
        # Use the last assistant message for training
        last_assistant_msg = assistant_msgs[-1]
        
        # Build the prompt
        prompt_parts = []
        
        if system_msg:
            prompt_parts.append(f"System: {system_msg['content']}")
        
        for user_msg in user_msgs[:-1]:
            prompt_parts.append(f"User: {user_msg['content']}")
        
        if user_msgs:
            prompt_parts.append(f"User: {user_msgs[-1]['content']}")
        
        prompt_parts.append("Assistant:")
        
        prompt = "\n".join(prompt_parts)
        completion = last_assistant_msg['content']
        
        # Tokenize
        # Note: We need to handle special tokens carefully. 
        # For simplicity in manual mode, we assume standard concatenation.
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
        
        # Create full sequence and weights
        # 0 for prompt (masked), 1 for completion (trained)
        tokens = prompt_tokens + completion_tokens
        weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
        
        # Shift for next-token prediction (causal LM)
        # Input: [t0, t1, ..., tn-1]
        # Target: [t1, t2, ..., tn]
        # Weights: [w1, w2, ..., wn] (corresponding to targets)
        
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = weights[1:]
        
        # Create Datum matching the SDK expectation
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
        )
        
        return datum
    
    async def evaluate_model(self, tinker_path: str, base_model: str, dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            logger.info(f"Starting evaluation for {tinker_path} on {base_model}")
            self._ensure_service_client()
            
            # Import here to avoid hard dependency if not used
            from inspect_ai import Task, eval
            from inspect_ai.dataset import MemoryDataset, Sample
            from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
            from inspect_ai.model import Model as InspectAIModel
            from inspect_ai.scorer import model_graded_qa
            from inspect_ai.solver import generate
            from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
            
            # 1. Create Sampling Client
            # We need to use the underlying client to create a sampling client with the specific checkpoint
            # The user example showed: service_client.create_sampling_client(base_model=...)
            # But we need to specify the checkpoint. 
            # Tinker SDK create_sampling_client signature: (model_path, base_model, ...)
            
            sampling_client = self.service_client.create_sampling_client(
                model_path=tinker_path,
                base_model=base_model
            )
            
            # 2. Create Inspect API Adapter
            # The renderer_name needs to be inferred or passed. For now, assuming llama3/qwen style.
            # tinker_cookbook seems to use 'llama3' for many recent models.
            renderer_name = "llama3" 
            if "qwen" in base_model.lower():
                renderer_name = "qwen2" # or similar if supported, fallback to llama3 often works for chat
            
            api = InspectAPIFromTinkerSampling(
                renderer_name=renderer_name, 
                model_name=base_model,
                sampling_client=sampling_client,
                verbose=False,
            )
            
            # 3. Create Dataset
            samples = [
                Sample(input=d["input"], target=d["target"]) 
                for d in dataset
            ]
            qa_dataset = MemoryDataset(name="eval_dataset", samples=samples)
            
            # 4. Define Task
            # We use the model itself as the judge for simplicity, or we could use a standard judge if available.
            # The user example uses the model itself (GRADER_MODEL defaults to model being evaluated).
            
            grader_model = InspectAIModel(api=api, config=InspectAIGenerateConfig())
            
            task = Task(
                name="tinker_eval",
                dataset=qa_dataset,
                solver=generate(),
                scorer=model_graded_qa(
                    instructions="Grade strictly against the target text as general answer key and rubric. "
                    "Respond 'GRADE: C' if correct or 'GRADE: I' otherwise.",
                    partial_credit=False,
                    model=grader_model
                ),
            )
            
            # 5. Run Evaluation
            # eval returns a list of Log objects
            logs = eval(task, model=grader_model)
            
            # 6. Process Results
            results = []
            if logs and len(logs) > 0:
                log = logs[0]
                if log.results and log.results.scores:
                    # Extract scores
                    for score in log.results.scores:
                        results.append({
                            "metric": score.name,
                            "value": score.value,
                            "reducer": score.reducer
                        })
                
                # Extract samples with grades
                # This might require parsing the log samples
                sample_results = []
                if log.samples:
                    for s in log.samples:
                        sample_results.append({
                            "input": s.input,
                            "target": s.target,
                            "output": s.output.completion if s.output else "",
                            "score": s.score.value if s.score else None,
                            "explanation": s.score.explanation if s.score else None
                        })
                        
                return {
                    "success": True,
                    "metrics": results,
                    "samples": sample_results,
                    "log_id": log.eval.run_id
                }
            
            return {"success": False, "error": "No logs returned from evaluation"}

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
