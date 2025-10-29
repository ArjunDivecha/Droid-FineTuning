"""
Student Model for On-Policy Distillation

This module handles loading and training the student model (Qwen 7B with LoRA adapters).
Manages gradient tracking for LoRA weights while keeping base model frozen.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import numpy as np

logger = logging.getLogger(__name__)


class StudentModel:
    """
    Student model manager for knowledge distillation.

    Responsibilities:
    - Load student base model (Qwen 7B)
    - Load and manage LoRA adapters
    - Forward pass with gradient tracking
    - Freeze base model, enable LoRA gradients only
    - Generate rollouts for on-policy data
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        freeze_base: bool = True
    ):
        """
        Initialize student model manager.

        Args:
            base_model_path: Path to base model (Qwen 7B)
            adapter_path: Path to LoRA adapters (from SFT)
            freeze_base: Whether to freeze base model weights
        """
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.freeze_base = freeze_base

        # Model and tokenizer (loaded on demand)
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        # Track which parameters are trainable
        self.trainable_params = []
        self.frozen_params = []

        logger.info(f"StudentModel initialized with adapter at {self.adapter_path}")

    def load(self):
        """Load student model with LoRA adapters"""
        if self._is_loaded:
            logger.info("Student model already loaded")
            return

        logger.info(f"Loading student base model from {self.base_model_path}")
        logger.info(f"Loading adapter from {self.adapter_path}")
        start_time = time.time()

        try:
            # Load base model + LoRA adapter
            self.model, self.tokenizer = load(
                self.base_model_path,
                adapter_path=self.adapter_path
            )

            # Freeze base model parameters, keep LoRA trainable
            if self.freeze_base:
                self._setup_parameter_freezing()

            # Set to train mode for LoRA
            self.model.train()

            load_time = time.time() - start_time
            logger.info(f"Student model loaded in {load_time:.2f}s")
            logger.info(f"Trainable params: {len(self.trainable_params)}, Frozen: {len(self.frozen_params)}")

            self._is_loaded = True

        except Exception as e:
            logger.error(f"Failed to load student model: {e}")
            raise

    def _setup_parameter_freezing(self):
        """
        Freeze base model parameters, enable gradients for LoRA only.

        MLX handles LoRA parameters automatically - when adapters are loaded,
        only the adapter parameters are trainable by default.
        """
        # MLX models with loaded adapters automatically have only adapter parameters trainable
        # We just need to identify them for logging purposes
        
        # Get all trainable parameters (MLX provides this via trainable_parameters())
        if hasattr(self.model, 'trainable_parameters'):
            trainable = self.model.trainable_parameters()
            self.trainable_params = list(trainable.items()) if isinstance(trainable, dict) else []
        else:
            # Fallback: assume all parameters in model are trainable (adapter-only)
            self.trainable_params = []
        
        self.frozen_params = []  # MLX handles freezing internally
        
        logger.info(f"Configured gradient tracking:")
        logger.info(f"  MLX adapter model loaded - only LoRA parameters are trainable")
        logger.info(f"  Base model parameters are frozen by default")

    def forward(
        self,
        prompts: List[str],
        teacher_token_ids: Optional[List[List[int]]] = None,
        return_logits: bool = True
    ) -> Dict[str, mx.array]:
        """
        Forward pass through student model.

        Args:
            prompts: Input prompts
            teacher_token_ids: If provided, run forward on these token sequences
                              (for aligned comparison with teacher)
            return_logits: Whether to return raw logits (True) or logprobs (False)

        Returns:
            Dictionary with:
                - logits: (batch, seq_len, vocab_size) - raw logits
                - logprobs: (batch, seq_len, vocab_size) - log probabilities
                - token_ids: List of token ID sequences
                - mask: (batch, seq_len) - mask for valid positions
        """
        if not self._is_loaded:
            raise RuntimeError("Student model not loaded. Call load() first.")

        batch_size = len(prompts)

        # If teacher tokens provided, use those; otherwise encode prompts
        if teacher_token_ids is not None:
            token_sequences = teacher_token_ids
        else:
            token_sequences = [self.tokenizer.encode(p) for p in prompts]

        # Find max length for padding
        max_len = max(len(seq) for seq in token_sequences)

        # Pad sequences and create mask
        padded_sequences = []
        masks = []

        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0

        for seq in token_sequences:
            seq_len = len(seq)
            padding_len = max_len - seq_len

            # Pad sequence
            padded_seq = seq + [pad_token_id] * padding_len
            padded_sequences.append(padded_seq)

            # Create mask (1 for real tokens, 0 for padding)
            mask = [1] * seq_len + [0] * padding_len
            masks.append(mask)

        # Convert to MLX arrays
        input_tensor = mx.array(padded_sequences)  # (batch, max_len)
        mask_tensor = mx.array(masks)  # (batch, max_len)

        # Forward pass
        outputs = self.model(input_tensor)

        # Extract logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits  # (batch, seq_len, vocab_size)
        else:
            logits = outputs  # Assume outputs are logits directly

        # Compute log probabilities
        # Use log_softmax for numerical stability
        probs = mx.softmax(logits, axis=-1)
        logprobs = mx.log(probs + 1e-10)

        result = {
            'logits': logits,
            'logprobs': logprobs,
            'token_ids': token_sequences,
            'mask': mask_tensor
        }

        return result

    def generate_rollouts(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 1.0,
        num_samples_per_prompt: int = 1
    ) -> List[Dict]:
        """
        Generate completions from student model (for on-policy data).

        Args:
            prompts: Input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            num_samples_per_prompt: Number of completions per prompt

        Returns:
            List of rollout dictionaries with:
                - prompt: Input prompt
                - generated_text: Generated completion
                - token_ids: Token IDs of completion
                - sample_id: Sample index (0 to num_samples_per_prompt-1)
        """
        if not self._is_loaded:
            raise RuntimeError("Student model not loaded. Call load() first.")

        rollouts = []

        for prompt_idx, prompt in enumerate(prompts):
            for sample_idx in range(num_samples_per_prompt):
                try:
                    # Generate completion
                    # Note: MLX's generate() handles temperature internally
                    response = generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temp=temperature,  # MLX uses 'temp' parameter
                        verbose=False
                    )

                    # Tokenize the generated text
                    token_ids = self.tokenizer.encode(response)

                    rollout = {
                        'prompt_id': prompt_idx,
                        'sample_id': sample_idx,
                        'prompt': prompt,
                        'generated_text': response,
                        'token_ids': token_ids,
                        'num_tokens': len(token_ids)
                    }

                    rollouts.append(rollout)

                except Exception as e:
                    logger.error(f"Failed to generate rollout for prompt {prompt_idx}, sample {sample_idx}: {e}")

        return rollouts

    def save_adapter(self, save_path: str):
        """
        Save current LoRA adapter weights.

        Args:
            save_path: Path where adapter will be saved
        """
        if not self._is_loaded:
            raise RuntimeError("Student model not loaded")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save adapter weights
            # MLX models typically have a save_weights method
            if hasattr(self.model, 'save_weights'):
                self.model.save_weights(str(save_path))
            else:
                # Fallback: save trainable parameters only
                adapter_weights = {
                    name: param for name, param in self.trainable_params
                }
                mx.save_safetensors(str(save_path), adapter_weights)

            logger.info(f"Adapter saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save adapter: {e}")
            raise

    def get_trainable_parameters(self) -> List[Tuple[str, mx.array]]:
        """Get list of trainable parameters (LoRA only)"""
        return self.trainable_params

    def get_num_trainable_params(self) -> int:
        """Get total number of trainable parameters"""
        total = 0
        for name, param in self.trainable_params:
            if hasattr(param, 'size'):
                total += param.size
            elif hasattr(param, 'shape'):
                import numpy as np
                total += np.prod(param.shape)
        return total

    def get_num_frozen_params(self) -> int:
        """Get total number of frozen parameters"""
        total = 0
        for name, param in self.frozen_params:
            if hasattr(param, 'size'):
                total += param.size
            elif hasattr(param, 'shape'):
                import numpy as np
                total += np.prod(param.shape)
        return total

    def unload(self):
        """Unload student model to free memory"""
        if not self._is_loaded:
            return

        logger.info("Unloading student model")
        self.model = None
        self.tokenizer = None
        self.trainable_params = []
        self.frozen_params = []
        self._is_loaded = False

        # Clear MLX cache
        mx.clear_cache()

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        trainable = len(self.trainable_params) if self._is_loaded else 0
        return f"StudentModel(base={self.base_model_path}, adapter={self.adapter_path}, status={status}, trainable_params={trainable})"
