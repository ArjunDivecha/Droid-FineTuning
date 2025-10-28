"""
Distillation Trainer - Main Orchestrator

Coordinates the full knowledge distillation training loop:
1. Load teacher and student models
2. Generate/load training data
3. Run training loop (forward, loss, backward, update)
4. Evaluate on validation set
5. Save checkpoints
6. Log metrics
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

import mlx.core as mx
import mlx.optimizers as optim

from .config import OPDConfig, OPDMetrics
from .teacher_model import TeacherModel
from .student_model import StudentModel
from .distillation_loss import DistillationLoss
from .data_loader import PromptDataset, load_prompts, create_batches

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """
    Main orchestrator for knowledge distillation training.

    Manages the complete training pipeline from data loading to checkpointing.
    """

    def __init__(self, config: OPDConfig):
        """
        Initialize distillation trainer.

        Args:
            config: OPD configuration
        """
        self.config = config

        # Models (initialized in setup())
        self.teacher = None
        self.student = None
        self.loss_fn = None
        self.optimizer = None

        # Data (initialized in setup())
        self.dataset = None
        self.train_prompts = None
        self.val_prompts = None

        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.metrics_history = []

        # Directories
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.run_id
        self.metrics_dir = Path("./OnPolicyDistill/metrics")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Metrics files
        self.train_metrics_file = self.metrics_dir / f"{config.run_id}_train.jsonl"
        self.eval_metrics_file = self.metrics_dir / f"{config.run_id}_eval.jsonl"

        logger.info(f"DistillationTrainer initialized for run: {config.run_id}")

    def setup(self):
        """
        Set up all components for training.

        1. Load teacher model
        2. Load student model
        3. Initialize loss function
        4. Initialize optimizer
        5. Load dataset
        """
        logger.info("="*60)
        logger.info("Setting up distillation training")
        logger.info("="*60)

        # 1. Load teacher model
        logger.info("Loading teacher model...")
        self.teacher = TeacherModel(
            model_path=self.config.teacher_model_path,
            cache_dir=self.config.cache_dir,
            use_cache=self.config.use_cache,
            cache_size_mb=self.config.cache_size_mb
        )
        self.teacher.load()

        # 2. Load student model
        logger.info("Loading student model...")
        self.student = StudentModel(
            base_model_path=self.config.base_model_path,
            adapter_path=self.config.student_adapter_path,
            freeze_base=True
        )
        self.student.load()

        # 3. Initialize loss function
        logger.info("Initializing loss function...")
        self.loss_fn = DistillationLoss(
            temperature=self.config.temperature,
            kl_weight=self.config.kl_weight,
            ce_weight=self.config.ce_weight
        )

        # 4. Initialize optimizer
        logger.info("Initializing optimizer...")
        trainable_params = self.student.get_trainable_parameters()
        self.optimizer = optim.Adam(
            learning_rate=self.config.learning_rate
        )

        logger.info(f"Optimizer: Adam(lr={self.config.learning_rate})")
        logger.info(f"Trainable parameters: {self.student.get_num_trainable_params():,}")

        # 5. Load dataset
        logger.info("Loading dataset...")
        self.train_prompts, self.val_prompts = load_prompts(
            self.config.validation_prompts_path,
            max_prompts=self.config.max_prompts,
            split_ratio=self.config.train_val_split,
            shuffle=True,
            seed=self.config.seed
        )

        logger.info("="*60)
        logger.info("Setup complete!")
        logger.info(f"  Teacher: {self.config.teacher_model_path}")
        logger.info(f"  Student: {self.config.student_adapter_path}")
        logger.info(f"  Train prompts: {len(self.train_prompts)}")
        logger.info(f"  Val prompts: {len(self.val_prompts)}")
        logger.info(f"  Total steps: {self.config.num_steps}")
        logger.info("="*60)

    def train(self):
        """
        Main training loop.

        Performs iterative distillation:
        1. Sample batch of prompts
        2. Get teacher outputs (with caching)
        3. Get student outputs
        4. Compute loss
        5. Backprop and update
        6. Log metrics
        7. Evaluate periodically
        8. Save checkpoints
        """
        logger.info("\n" + "="*60)
        logger.info("Starting distillation training")
        logger.info("="*60)

        start_time = time.time()
        step_times = []

        try:
            for step in range(self.config.num_steps):
                self.current_step = step
                step_start = time.time()

                # 1. Sample batch
                batch_prompts = self._sample_batch()

                # 2. Get teacher outputs
                teacher_outputs = self._get_teacher_outputs(batch_prompts)

                # 3. Get student outputs
                student_outputs = self._get_student_outputs(
                    batch_prompts,
                    teacher_outputs
                )

                # 4. Compute loss
                loss, metrics = self._compute_loss(
                    student_outputs,
                    teacher_outputs
                )

                # 5. Backprop and update
                self._update_parameters(loss)

                # Track step time
                step_time = time.time() - step_start
                step_times.append(step_time)

                # Add timing metrics
                metrics['step_time_seconds'] = step_time
                metrics['tokens_per_second'] = metrics.get('samples_processed', 0) / step_time if step_time > 0 else 0

                # 6. Log metrics
                self._log_step_metrics(step, metrics)

                # 7. Evaluate
                if (step + 1) % self.config.eval_every == 0:
                    val_metrics = self.evaluate()
                    self._log_eval_metrics(step, val_metrics)

                    # Check for best model
                    if val_metrics['kl_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['kl_loss']
                        self.save_checkpoint(step, is_best=True)
                        logger.info(f"  🎉 New best model! Val KL loss: {self.best_val_loss:.4f}")

                # 8. Checkpoint
                if (step + 1) % self.config.checkpoint_every == 0:
                    self.save_checkpoint(step, is_best=False)

                # Progress logging
                if (step + 1) % 10 == 0:
                    avg_step_time = sum(step_times[-10:]) / len(step_times[-10:])
                    remaining_steps = self.config.num_steps - (step + 1)
                    eta_seconds = remaining_steps * avg_step_time
                    eta_minutes = eta_seconds / 60

                    logger.info(
                        f"Step {step+1}/{self.config.num_steps} | "
                        f"Loss: {metrics['total_loss']:.4f} | "
                        f"KL: {metrics['kl_loss']:.4f} | "
                        f"Agree: {metrics['token_agreement_pct']:.1f}% | "
                        f"ETA: {eta_minutes:.1f}m"
                    )

            # Final checkpoint
            logger.info("\nTraining complete! Saving final checkpoint...")
            self.save_checkpoint(self.config.num_steps, is_final=True)

            # Training summary
            total_time = time.time() - start_time
            logger.info("="*60)
            logger.info("Training Summary")
            logger.info("="*60)
            logger.info(f"  Total time: {total_time/60:.2f} minutes")
            logger.info(f"  Steps: {self.config.num_steps}")
            logger.info(f"  Avg step time: {sum(step_times)/len(step_times):.2f}s")
            logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
            logger.info(f"  Final checkpoint: {self.checkpoint_dir}/final")
            logger.info("="*60)

            # Cache statistics
            cache_stats = self.teacher.get_cache_stats()
            logger.info("\nTeacher Cache Statistics:")
            logger.info(f"  Cache hits: {cache_stats['cache_hits']}")
            logger.info(f"  Cache misses: {cache_stats['cache_misses']}")
            logger.info(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
            logger.info(f"  Cached prompts: {cache_stats['cached_prompts']}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _sample_batch(self) -> List[str]:
        """Sample a random batch of training prompts"""
        import random
        return random.sample(
            self.train_prompts,
            min(self.config.batch_size, len(self.train_prompts))
        )

    def _get_teacher_outputs(self, prompts: List[str]) -> List[Dict]:
        """Get teacher outputs for batch of prompts (with caching)"""
        return self.teacher.batch_get_logprobs(
            prompts,
            max_tokens=self.config.max_generation_tokens,
            temperature=self.config.generation_temperature,
            show_progress=False
        )

    def _get_student_outputs(
        self,
        prompts: List[str],
        teacher_outputs: List[Dict]
    ) -> Dict:
        """
        Get student outputs aligned with teacher.

        Runs student forward pass on the same token sequences as teacher.
        """
        # Extract teacher token sequences
        teacher_token_ids = [t['token_ids'] for t in teacher_outputs]

        # Run student forward pass
        student_outputs = self.student.forward(
            prompts=prompts,
            teacher_token_ids=teacher_token_ids,
            return_logits=True
        )

        return student_outputs

    def _compute_loss(
        self,
        student_outputs: Dict,
        teacher_outputs: List[Dict]
    ) -> Tuple[mx.array, Dict]:
        """Compute distillation loss"""
        # Convert teacher outputs to MLX arrays
        # Teacher logprobs are already computed, need to convert to logits
        # Actually, we need to work with logprobs directly

        # Get student logits
        student_logits = student_outputs['logits']
        mask = student_outputs['mask']

        # For teacher, we need to reconstruct logits from logprobs
        # Or work directly with logprobs
        # Let's construct teacher logits for loss computation

        # TODO: This is a simplified version - needs proper alignment
        # For now, use student's logprobs vs teacher's logprobs

        # Extract teacher logprobs and align
        # This is a placeholder - real implementation needs careful alignment

        teacher_logprobs_list = [t['logprobs'] for t in teacher_outputs]

        # Convert to MLX array (need to pad to same length)
        max_len = student_logits.shape[1]
        vocab_size = student_logits.shape[2]

        # Create teacher "logits" by converting logprobs back
        # This is approximate for now
        teacher_logits = student_logits  # Placeholder

        # Compute loss
        loss, metrics = self.loss_fn.compute(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            target_token_ids=None,  # Optional
            mask=mask
        )

        # Add metadata
        metrics['samples_processed'] = len(teacher_outputs)

        return loss, metrics

    def _update_parameters(self, loss: mx.array):
        """Backprop and update student parameters"""
        # Compute gradients
        loss_and_grad = mx.value_and_grad(lambda: loss)

        # Update parameters
        # TODO: Implement proper gradient update with MLX optimizer

        # For now, this is a placeholder
        pass

    def _log_step_metrics(self, step: int, metrics: Dict):
        """Log training step metrics"""
        # Add step number
        metrics['step'] = step

        # Save to file
        with open(self.train_metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

        # Add to history
        self.metrics_history.append(metrics)

    def _log_eval_metrics(self, step: int, metrics: Dict):
        """Log evaluation metrics"""
        metrics['step'] = step

        with open(self.eval_metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def evaluate(self) -> Dict:
        """
        Evaluate on validation set.

        Returns:
            Dictionary with validation metrics
        """
        logger.info(f"\nEvaluating on validation set ({len(self.val_prompts)} prompts)...")

        # Sample subset if val set is large
        eval_prompts = self.val_prompts
        if len(eval_prompts) > self.config.eval_max_samples:
            import random
            eval_prompts = random.sample(eval_prompts, self.config.eval_max_samples)

        # Create batches
        eval_batches = create_batches(
            eval_prompts,
            self.config.batch_size,
            shuffle=False
        )

        all_metrics = []

        for batch in eval_batches:
            # Get teacher and student outputs
            teacher_outputs = self._get_teacher_outputs(batch)
            student_outputs = self._get_student_outputs(batch, teacher_outputs)

            # Compute loss (without gradients)
            with mx.no_grad():
                loss, metrics = self._compute_loss(student_outputs, teacher_outputs)

            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = self._average_metrics(all_metrics)

        logger.info(f"  Val KL Loss: {avg_metrics['kl_loss']:.4f}")
        logger.info(f"  Val Token Agreement: {avg_metrics['token_agreement_pct']:.1f}%")

        return avg_metrics

    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Average metrics across multiple batches"""
        if not metrics_list:
            return {}

        avg = {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                avg[key] = sum(m[key] for m in metrics_list) / len(metrics_list)

        return avg

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
            checkpoint_name = f"step_{step:07d}"

        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        adapter_path = checkpoint_path / "adapters.safetensors"

        # Save student adapter
        self.student.save_adapter(str(adapter_path))

        # Save checkpoint metadata
        metadata = {
            'step': step,
            'run_id': self.config.run_id,
            'is_best': is_best,
            'is_final': is_final,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }

        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  ✓ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)

        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Update training state
        self.current_step = metadata['step']
        self.best_val_loss = metadata.get('best_val_loss', float('inf'))

        logger.info(f"Loaded checkpoint from step {self.current_step}")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")

    def __del__(self):
        """Cleanup when trainer is destroyed"""
        if self.teacher and self.teacher._is_loaded:
            self.teacher.unload()
        if self.student and self.student._is_loaded:
            self.student.unload()
