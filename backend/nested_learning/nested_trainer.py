"""
Nested Learning Trainer

Main training orchestrator for Nested Learning with multi-frequency parameter updates.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
import numpy as np

from .config import NestedLearningConfig
from .nested_optimizer import NestedAdam, NestedAdamW
from .parameter_scheduler import ParameterTierScheduler

logger = logging.getLogger(__name__)


class NestedLoRATrainer:
    """
    Trainer for Nested Learning with LoRA fine-tuning.

    Implements multi-frequency parameter updates where different parameter tiers
    update at different rates (e.g., every 1, 2, 4 steps).

    Key Components:
    - ParameterTierScheduler: Assigns parameters to tiers
    - NestedAdam: Optimizer with tier-based gradient filtering
    - Training loop: Standard forward/backward with nested updates
    """

    def __init__(self, config: NestedLearningConfig):
        """
        Initialize Nested Learning trainer.

        Args:
            config: Nested learning configuration
        """
        self.config = config

        # Models and tokenizer (loaded in setup())
        self.model = None
        self.tokenizer = None

        # Nested learning components
        self.tier_scheduler = None
        self.optimizer = None
        self.parameter_tier_map = {}

        # Training state
        self.current_step = 0
        self.stop_requested = False  # FIX #6: Add stop flag

        # Callback for status updates (set by API if needed)
        self.step_callback = None
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.metrics_history = []

        # FIX #7: Add current metric fields for API
        self.current_train_loss = None
        self.current_val_loss = None

        # Early stopping state
        self.patience_counter = 0
        self.early_stopped = False

        # FIX #4: Gradient accumulation state
        self.accumulated_gradients = None
        self.accumulation_steps = 0

        # Data
        self.train_data = []
        self.val_data = []

        # Directories
        self.output_dir = Path(config.output_path) / config.experiment_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.metrics_dir = self.output_dir / "metrics"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Metrics files
        self.train_metrics_file = self.metrics_dir / "train_metrics.jsonl"
        self.eval_metrics_file = self.metrics_dir / "eval_metrics.jsonl"

        logger.info(f"NestedLoRATrainer initialized: {config.experiment_name}")

    def setup(self):
        """
        Set up all components for training.

        1. Load model with LoRA adapter
        2. Assign parameters to tiers
        3. Initialize nested optimizer
        4. Load dataset
        """
        logger.info("=" * 60)
        logger.info("Setting up Nested Learning training")
        logger.info("=" * 60)

        # 1. Load model with optional adapter
        logger.info(f"Loading base model: {self.config.base_model_path}")
        if self.config.adapter_path:
            logger.info(f"Loading adapter: {self.config.adapter_path}")
        else:
            logger.info("No adapter specified - will train from base model")
        start_time = time.time()

        # Load model with adapter
        if not self.config.adapter_path:
            raise ValueError(
                "Nested learning requires an existing LoRA adapter. "
                "Please train a regular LoRA adapter first using the 'Fine-Tuning' page, "
                "then use that adapter path here for nested learning. "
                "Nested learning trains the EXISTING adapter using multi-frequency updates."
            )

        logger.info(f"Loading model with adapter: {self.config.adapter_path}")
        self.model, self.tokenizer = load(
            self.config.base_model_path,
            adapter_path=self.config.adapter_path
        )

        # CRITICAL FIX: mlx_lm.load() makes EVERYTHING trainable (base model + adapters)
        # We need to ensure only LoRA adapters are trainable
        # In MLX, we control this through trainable_parameters() which already filters to LoRA
        # BUT we need gradients for the full model for backprop
        # Solution: Keep model as-is, but optimizer only updates LoRA parameters

        logger.info("Model loaded - will compute gradients for all parameters but only update LoRA adapters")

        # Count trainable parameters
        from mlx.utils import tree_flatten
        trainable_params = self.model.trainable_parameters()
        trainable_flat = tree_flatten(trainable_params, destination={})
        trainable_count = sum(v.size for v in trainable_flat.values())
        logger.info(f"Trainable LoRA parameters: {trainable_count:,}")

        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")

        # 2. Assign parameters to tiers
        logger.info("\nAssigning parameters to tiers...")
        self.tier_scheduler = ParameterTierScheduler(
            num_tiers=self.config.num_tiers,
            strategy=self.config.tier_assignment_strategy
        )

        self.parameter_tier_map = self.tier_scheduler.assign_tiers(
            model=self.model,
            gradient_history=None  # Will use layer_depth strategy initially
        )

        # Log tier summary
        tier_summary = self.tier_scheduler.get_tier_summary()
        logger.info(f"Tier assignment complete:")
        for tier_name, tier_info in tier_summary.get('tiers', {}).items():
            logger.info(f"  {tier_name}: {tier_info['parameter_count']} parameters")
            logger.info(f"    Frequency: every {self.config.tier_update_frequencies[int(tier_name.split('_')[1])]} steps")

        # 3. Initialize nested optimizer
        logger.info("\nInitializing nested optimizer...")
        self.optimizer = NestedAdam(
            learning_rate=self.config.learning_rate,
            tier_update_frequencies=self.config.tier_update_frequencies,
            parameter_tier_map=self.parameter_tier_map
        )

        logger.info(f"Optimizer: NestedAdam")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Tier frequencies: {self.config.tier_update_frequencies}")

        # 4. Load dataset
        logger.info("\nLoading dataset...")
        self.train_data = self._load_data(self.config.train_data_path)

        if self.config.val_data_path:
            self.val_data = self._load_data(self.config.val_data_path)
        else:
            # Split train data for validation
            split_idx = int(len(self.train_data) * 0.9)
            self.val_data = self.train_data[split_idx:]
            self.train_data = self.train_data[:split_idx]

        logger.info(f"Training samples: {len(self.train_data)}")
        logger.info(f"Validation samples: {len(self.val_data)}")

        # Save configuration
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"\nConfiguration saved to {config_file}")

        logger.info("=" * 60)
        logger.info("Setup complete! Ready to train.")
        logger.info("=" * 60)

    def stop_training(self):
        """
        Request training to stop gracefully.

        FIX #6: API stop semantics - sets flag that's checked in training loop.
        """
        self.stop_requested = True
        logger.info("Training stop requested by user")

    def _get_learning_rate(self, step: int) -> float:
        """
        Get learning rate for current step with warmup.

        FIX #4: Implement linear warmup schedule.

        Args:
            step: Current training step

        Returns:
            Learning rate for this step
        """
        if step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (step + 1) / self.config.warmup_steps
        else:
            # Base learning rate after warmup
            return self.config.learning_rate

    def train(self):
        """
        Main training loop with nested learning.

        Performs iterative training with tier-based parameter updates:
        1. Sample batch
        2. Forward pass
        3. Compute loss and gradients
        4. Filter gradients by active tiers
        5. Update parameters
        6. Log metrics
        7. Evaluate and checkpoint periodically
        """
        logger.info("\n" + "=" * 60)
        logger.info("Starting Nested Learning Training")
        logger.info("=" * 60)

        start_time = time.time()
        step_times = []

        try:
            for step in range(self.config.num_steps):
                # FIX #6: Check stop flag
                if self.stop_requested:
                    logger.info("Training stopped by user request")
                    break

                self.current_step = step
                step_start = time.time()

                # 1. Sample batch
                batch = self._sample_batch()

                # 2. Forward pass and compute loss/gradients
                loss, gradients = self._forward_backward(batch)

                # 3. CRITICAL: Force evaluation of loss BEFORE optimizer
                mx.eval(loss)

                # FIX #7: Store current train loss for API
                self.current_train_loss = float(loss)

                # FIX #4: Apply gradient clipping
                if self.config.max_grad_norm > 0:
                    # Flatten gradients for clipping
                    from mlx.utils import tree_flatten
                    flat_grads = tree_flatten(gradients, destination={})

                    # Compute global norm
                    grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in flat_grads.values()))
                    mx.eval(grad_norm)
                    grad_norm_value = float(grad_norm)

                    # Clip if necessary
                    if grad_norm_value > self.config.max_grad_norm:
                        scale = self.config.max_grad_norm / grad_norm_value
                        clipped_grads = {k: v * scale for k, v in flat_grads.items()}
                        gradients = clipped_grads
                        logger.debug(f"Gradient clipped: norm {grad_norm_value:.2f} -> {self.config.max_grad_norm}")

                # FIX #4: Gradient accumulation
                if self.config.gradient_accumulation_steps > 1:
                    # Accumulate gradients
                    if self.accumulated_gradients is None:
                        self.accumulated_gradients = gradients
                    else:
                        # Add to accumulated gradients
                        from mlx.utils import tree_flatten
                        flat_accumulated = tree_flatten(self.accumulated_gradients, destination={})
                        flat_new = tree_flatten(gradients, destination={})
                        for k in flat_new:
                            if k in flat_accumulated:
                                flat_accumulated[k] = flat_accumulated[k] + flat_new[k]
                        self.accumulated_gradients = flat_accumulated

                    self.accumulation_steps += 1

                    # Only update when we've accumulated enough steps
                    if self.accumulation_steps >= self.config.gradient_accumulation_steps:
                        # Average accumulated gradients
                        from mlx.utils import tree_flatten
                        flat_accumulated = tree_flatten(self.accumulated_gradients, destination={})
                        averaged_grads = {
                            k: v / self.config.gradient_accumulation_steps
                            for k, v in flat_accumulated.items()
                        }

                        # FIX #4: Update learning rate with warmup
                        current_lr = self._get_learning_rate(step)
                        self.optimizer.learning_rate = current_lr

                        # Update parameters
                        self.optimizer.apply_gradients(averaged_grads, self.model)

                        # Reset accumulation
                        self.accumulated_gradients = None
                        self.accumulation_steps = 0
                    else:
                        # Skip parameter update - still accumulating
                        pass
                else:
                    # No gradient accumulation - standard update
                    # FIX #4: Update learning rate with warmup
                    current_lr = self._get_learning_rate(step)
                    self.optimizer.learning_rate = current_lr

                    # 4. Update parameters (optimizer now handles gradient cleanup)
                    self.optimizer.apply_gradients(gradients, self.model)

                # 5. CRITICAL: Explicitly delete gradients dict to break references
                del gradients

                # 6. Force evaluation of updated parameters
                mx.eval(self.model.parameters())

                # 7. AGGRESSIVE memory cleanup
                mx.metal.clear_cache()
                import gc
                gc.collect()

                # 8. Periodic deep cleanup and memory logging every 10 steps
                if (step + 1) % 10 == 0:
                    import subprocess, os
                    result = subprocess.run(
                        ['ps', '-p', str(os.getpid()), '-o', 'rss='],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        memory_mb = int(result.stdout.strip()) / 1024
                        logger.info(f"Memory usage: {memory_mb:.1f} MB")

                # Track step time
                step_time = time.time() - step_start
                step_times.append(step_time)

                # 4. Collect metrics
                metrics = {
                    'step': step,
                    'loss': float(loss),
                    'step_time': step_time,
                    'learning_rate': self._get_learning_rate(step)  # FIX #4: Use actual LR with warmup
                }

                # Add tier statistics
                tier_stats = self.optimizer.get_tier_stats()
                metrics['tier_stats'] = tier_stats

                # 6. Log metrics
                self._log_step_metrics(metrics)

                # 7. Periodic evaluation
                if (step + 1) % self.config.eval_every == 0:
                    val_metrics = self.evaluate()
                    self._log_eval_metrics(step, val_metrics)

                    # FIX #7: Store current val loss for API and in metrics dict
                    self.current_val_loss = val_metrics['loss']
                    metrics['val_loss'] = val_metrics['loss']  # Add to metrics for callback

                    # Check for best model
                    if val_metrics['loss'] < self.best_val_loss - self.config.min_delta:
                        # Significant improvement
                        self.best_val_loss = val_metrics['loss']
                        self.patience_counter = 0
                        self.save_checkpoint(step, is_best=True)
                        logger.info(f"  🎉 New best model! Val loss: {self.best_val_loss:.4f}")
                    else:
                        # No improvement
                        self.patience_counter += 1
                        logger.info(f"  No improvement. Patience: {self.patience_counter}/{self.config.patience}")

                        # Check early stopping
                        if self.config.early_stop and self.patience_counter >= self.config.patience:
                            logger.info(f"\n⛔ Early stopping triggered after {self.patience_counter} evaluations without improvement")
                            logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
                            logger.info(f"  Stopping at step {step + 1}/{self.config.num_steps}")
                            self.early_stopped = True
                            break

                # 5. Call step callback if registered (for API status updates)
                # NOTE: Placed after validation so val_loss is included in metrics if available
                if self.step_callback:
                    try:
                        self.step_callback(step + 1, self.config.num_steps, metrics)
                    except Exception as e:
                        logger.warning(f"Step callback failed: {e}")

                # 8. Checkpoint (skip intermediate checkpoints to save disk space)
                # Only save best and final checkpoints
                # Uncomment below to enable intermediate checkpoints:
                # if (step + 1) % self.config.checkpoint_every == 0:
                #     self.save_checkpoint(step)

                # Progress logging
                if (step + 1) % 10 == 0:
                    avg_step_time = np.mean(step_times[-10:])
                    remaining_steps = self.config.num_steps - (step + 1)
                    eta_seconds = remaining_steps * avg_step_time
                    eta_minutes = eta_seconds / 60

                    # Get current active tiers
                    active_tiers = self.optimizer._get_active_tiers()

                    logger.info(
                        f"Step {step+1}/{self.config.num_steps} | "
                        f"Loss: {loss:.4f} | "
                        f"Active tiers: {active_tiers} | "
                        f"ETA: {eta_minutes:.1f}m"
                    )

            # Final checkpoint
            if self.early_stopped:
                logger.info("\n✅ Training stopped early! Best model already saved.")
            else:
                logger.info("\nTraining complete! Saving final checkpoint...")
                self.save_checkpoint(self.config.num_steps, is_final=True)

            # Training summary
            total_time = time.time() - start_time
            logger.info("=" * 60)
            logger.info("Training Summary")
            logger.info("=" * 60)
            logger.info(f"  Total time: {total_time/60:.2f} minutes")
            logger.info(f"  Steps: {self.config.num_steps}")
            logger.info(f"  Avg step time: {np.mean(step_times):.2f}s")
            logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
            logger.info(f"  Output dir: {self.output_dir}")
            logger.info("=" * 60)

            # Tier statistics
            final_stats = self.optimizer.get_tier_stats()
            logger.info("\nFinal Tier Statistics:")
            for tier_name, tier_info in final_stats['tier_parameters'].items():
                logger.info(f"  {tier_name}:")
                logger.info(f"    Update count: {tier_info['update_count']}")
                logger.info(f"    Frequency: every {tier_info['frequency']} steps")
                logger.info(f"    Parameters: {tier_info['parameter_count']}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _load_data(self, data_path: str) -> List[Dict]:
        """
        Load training data from JSONL file.

        Args:
            data_path: Path to JSONL file

        Returns:
            List of data samples
        """
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _sample_batch(self) -> List[Dict]:
        """Sample a random batch from training data."""
        import random
        batch_size = min(self.config.batch_size, len(self.train_data))
        return random.sample(self.train_data, batch_size)

    def _forward_backward(self, batch: List[Dict]) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Perform forward and backward pass.

        Args:
            batch: List of training samples

        Returns:
            (loss, gradients)
        """
        # Extract text from batch - handle multiple formats
        texts = []
        for sample in batch:
            # Format 1: Plain text field
            if 'text' in sample:
                texts.append(sample['text'])
            # Format 2: Prompt field
            elif 'prompt' in sample:
                texts.append(sample['prompt'])
            # Format 3: Messages format (standard SFT format)
            elif 'messages' in sample:
                # Convert messages to chat format text
                messages = sample['messages']
                text_parts = []
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                texts.append('\n'.join(text_parts))
            else:
                # Fallback to empty string
                texts.append('')

        # Tokenize
        tokens_list = []
        for i, text in enumerate(texts):
            tokens = self.tokenizer.encode(text)
            # Truncate to max length
            if len(tokens) > self.config.max_seq_length:
                tokens = tokens[:self.config.max_seq_length]
            # Skip empty sequences (need at least 2 tokens for next-token prediction)
            if len(tokens) >= 2:
                tokens_list.append(tokens)

        # If no valid samples in batch, return zero loss
        if not tokens_list:
            logger.warning("Batch has no valid samples (all empty or too short)")
            return mx.array(0.0), {}

        # Pad to same length
        max_len = max(len(t) for t in tokens_list)
        padded_tokens = []
        for tokens in tokens_list:
            padded = tokens + [self.tokenizer.eos_token_id] * (max_len - len(tokens))
            padded_tokens.append(padded)

        # Convert to MLX array - must be int32 for embedding layer
        input_ids = mx.array(padded_tokens, dtype=mx.int32)

        # Define loss function
        def loss_fn(model):
            # Forward pass
            logits = model(input_ids)

            # Compute cross-entropy loss
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            # Flatten
            batch_size, seq_len, vocab_size = shift_logits.shape
            shift_logits_flat = mx.reshape(shift_logits, (-1, vocab_size))
            shift_labels_flat = mx.reshape(shift_labels, (-1,))

            # Ensure labels are int32 for cross_entropy
            shift_labels_flat = shift_labels_flat.astype(mx.int32)

            # Cross-entropy loss
            loss = nn.losses.cross_entropy(shift_logits_flat, shift_labels_flat, reduction='mean')

            return loss

        # Compute loss and gradients
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss, gradients = loss_and_grad_fn(self.model)

        # CRITICAL: Force evaluation to free memory from computation graph
        mx.eval(loss)
        mx.eval(gradients)

        return loss, gradients

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Returns:
            Dictionary with validation metrics
        """
        logger.info(f"\nEvaluating on validation set ({len(self.val_data)} samples)...")

        # Sample subset if val set is large
        eval_samples = self.val_data
        if len(eval_samples) > 100:
            import random
            eval_samples = random.sample(eval_samples, 100)

        # Compute loss on validation set
        val_losses = []

        for sample in eval_samples:
            text = sample.get('text', sample.get('prompt', ''))
            tokens = self.tokenizer.encode(text)

            if len(tokens) > self.config.max_seq_length:
                tokens = tokens[:self.config.max_seq_length]

            # Skip empty or too-short sequences
            if len(tokens) < 2:
                logger.warning(f"Skipping validation sample with only {len(tokens)} tokens")
                continue

            # Must be int32 for embedding layer
            input_ids = mx.array([tokens], dtype=mx.int32)

            # Forward pass (no gradients needed for evaluation)
            logits = self.model(input_ids)

            # Compute loss
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            vocab_size = shift_logits.shape[-1]
            shift_logits_flat = mx.reshape(shift_logits, (-1, vocab_size))
            shift_labels_flat = mx.reshape(shift_labels, (-1,))

            # Ensure labels are int32 for cross_entropy
            shift_labels_flat = shift_labels_flat.astype(mx.int32)

            loss = nn.losses.cross_entropy(shift_logits_flat, shift_labels_flat, reduction='mean')
            mx.eval(loss)  # Force evaluation
            val_losses.append(float(loss))

        # Handle case where no valid validation samples
        if not val_losses:
            logger.warning("  No valid validation samples - skipping evaluation")
            return {'loss': float('inf')}

        avg_loss = np.mean(val_losses)

        logger.info(f"  Validation loss: {avg_loss:.4f}")

        return {'loss': avg_loss}

    def _log_step_metrics(self, metrics: Dict):
        """Log training step metrics to file."""
        with open(self.train_metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

        self.metrics_history.append(metrics)

    def _log_eval_metrics(self, step: int, metrics: Dict):
        """Log evaluation metrics to file."""
        metrics['step'] = step

        with open(self.eval_metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def save_checkpoint(
        self,
        step: int,
        is_best: bool = False,
        is_final: bool = False
    ):
        """
        Save checkpoint.

        Args:
            step: Current training step
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint
        """
        if is_final:
            checkpoint_name = "final"
        elif is_best:
            checkpoint_name = "best"
        else:
            checkpoint_name = f"checkpoint_{step:07d}"

        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        adapter_file = checkpoint_path / "adapters.safetensors"

        # Get trainable parameters (LoRA adapters)
        trainable_params = self.model.trainable_parameters()

        # Flatten the nested dict structure using tree_flatten
        from mlx.utils import tree_flatten
        flattened = tree_flatten(trainable_params, destination={})

        # IMPORTANT: Only save LoRA adapter weights, not the full model
        # LoRA parameters have .lora_a and .lora_b suffixes
        adapter_weights = {
            k: v for k, v in flattened.items()
            if '.lora_a' in k or '.lora_b' in k or 'adapter' in k.lower()
        }

        # If no LoRA weights found, log warning and save all trainable params
        # (this happens when model.train() makes everything trainable)
        if not adapter_weights:
            logger.warning("No LoRA-specific weights found in trainable parameters!")
            logger.warning(f"Sample keys: {list(flattened.keys())[:5]}")
            logger.warning("This will save the full model (~14GB) instead of just adapters (~100MB)")
            adapter_weights = flattened

        # Evaluate all arrays
        mx.eval(adapter_weights)

        # Save using MLX
        mx.save_safetensors(str(adapter_file), adapter_weights)

        # Log what was saved
        total_params = sum(v.size for v in adapter_weights.values())
        logger.info(f"  Saved {len(adapter_weights)} parameter groups ({total_params:,} total parameters)")

        # Save checkpoint metadata
        metadata = {
            'step': step,
            'experiment_name': self.config.experiment_name,
            'is_best': is_best,
            'is_final': is_final,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'tier_stats': self.optimizer.get_tier_stats(),
            'timestamp': datetime.now().isoformat()
        }

        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save adapter_config.json required by MLX for loading
        # Need to get num_layers from model config
        num_layers = getattr(self.model.model.args, 'num_hidden_layers',
                             len(self.model.model.layers) if hasattr(self.model.model, 'layers') else 24)

        adapter_config = {
            "adapter_type": "lora",
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": 0.0,
            "lora_parameters": {
                "rank": self.config.lora_rank,
                "dropout": 0.0,
                "scale": 10.0
            },
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "num_layers": num_layers,
            "model": self.config.base_model_path
        }
        adapter_config_file = checkpoint_path / "adapter_config.json"
        with open(adapter_config_file, 'w') as f:
            json.dump(adapter_config, f, indent=2)

        logger.info(f"  ✓ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)

        # Load metadata
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Update training state
        self.current_step = metadata['step']
        self.best_val_loss = metadata.get('best_val_loss', float('inf'))

        logger.info(f"Loaded checkpoint from step {self.current_step}")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
