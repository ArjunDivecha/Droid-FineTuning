"""
Teacher Model for On-Policy Distillation

This module handles loading and inference with the teacher model (Qwen 32B 4-bit).
Provides token-level logprobs extraction and caching.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import numpy as np

from .config import TeacherCacheEntry

logger = logging.getLogger(__name__)


class TeacherModel:
    """
    Teacher model manager for knowledge distillation.

    Responsibilities:
    - Load teacher model (Qwen 32B 4-bit)
    - Generate text with token-level logprobs
    - Cache teacher outputs to disk
    - Batch processing for efficiency
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str = "./OnPolicyDistill/teacher_cache",
        use_cache: bool = True,
        cache_size_mb: int = 4096
    ):
        """
        Initialize teacher model manager.

        Args:
            model_path: Path to teacher model (Qwen 32B)
            cache_dir: Directory for caching teacher outputs
            use_cache: Whether to use caching
            cache_size_mb: Maximum in-memory cache size (MB)
        """
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_size_mb = cache_size_mb

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache index (maps prompt hash -> cache file)
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()

        # Model and tokenizer (loaded on demand)
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_inference_time = 0.0

        logger.info(f"TeacherModel initialized with cache at {self.cache_dir}")

    def load(self):
        """Load teacher model and tokenizer"""
        if self._is_loaded:
            logger.info("Teacher model already loaded")
            return

        logger.info(f"Loading teacher model from {self.model_path}")
        start_time = time.time()

        try:
            self.model, self.tokenizer = load(self.model_path)

            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Set to eval mode
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(f"Teacher model loaded in {load_time:.2f}s")

            self._is_loaded = True

        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            raise

    def unload(self):
        """Unload teacher model to free memory"""
        if not self._is_loaded:
            return

        logger.info("Unloading teacher model")
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        # Clear MLX cache
        mx.clear_cache()

    def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0
    ) -> Dict:
        """
        Generate text and extract token-level logprobs.

        This uses the manual generation loop approach (Option C) that was
        validated in Phase 0 testing.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = no scaling)

        Returns:
            Dictionary with:
                - prompt: Input prompt
                - generated_text: Generated completion
                - tokens: List of token strings
                - token_ids: List of token IDs
                - logprobs: List of log probabilities (per token)
                - full_logprobs: Full distribution logprobs (optional, if needed)
                - generation_time: Time taken (seconds)
        """
        if not self._is_loaded:
            raise RuntimeError("Teacher model not loaded. Call load() first.")

        start_time = time.time()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        tokens = input_ids.copy()

        generated_tokens = []
        generated_token_ids = []
        token_logprobs = []

        # Get EOS token
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None

        try:
            for step in range(max_tokens):
                # Forward pass
                input_tensor = mx.array([tokens])
                outputs = self.model(input_tensor)

                # Get logits for last position (next token prediction)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits[0, -1, :]  # (vocab_size,)
                else:
                    logits = outputs[0, -1, :]  # Assume outputs are logits

                # Apply temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature

                # Compute probabilities and log probabilities
                probs = mx.softmax(logits, axis=-1)
                logprobs = mx.log(probs + 1e-10)  # Add small epsilon for numerical stability

                # Sample next token (greedy for reproducibility in distillation)
                # Use argmax for deterministic generation
                next_token_id = int(mx.argmax(probs).item())

                # Get logprob of selected token
                token_logprob = float(logprobs[next_token_id].item())

                # Decode token
                next_token = self.tokenizer.decode([next_token_id])

                # Store
                generated_token_ids.append(next_token_id)
                generated_tokens.append(next_token)
                token_logprobs.append(token_logprob)
                tokens.append(next_token_id)

                # Check for EOS
                if eos_token_id and next_token_id == eos_token_id:
                    break

            generation_time = time.time() - start_time
            generated_text = ''.join(generated_tokens)

            result = {
                'prompt': prompt,
                'generated_text': generated_text,
                'tokens': generated_tokens,
                'token_ids': generated_token_ids,
                'logprobs': token_logprobs,
                'generation_time': generation_time,
                'num_tokens': len(generated_tokens)
            }

            self.total_inference_time += generation_time

            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def get_logprobs(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        use_cache: bool = None
    ) -> Dict:
        """
        Get teacher logprobs for a prompt, with caching.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cache: Override instance cache setting

        Returns:
            Dictionary with teacher outputs and logprobs
        """
        # Determine if caching is enabled
        cache_enabled = use_cache if use_cache is not None else self.use_cache

        # Check cache first
        if cache_enabled:
            cache_key = self._get_cache_key(prompt, max_tokens, temperature)
            cached = self._load_from_cache(cache_key)

            if cached is not None:
                self.cache_hits += 1
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached

            self.cache_misses += 1

        # Generate with teacher
        logger.debug(f"Generating teacher output for: {prompt[:50]}...")
        result = self.generate_with_logprobs(prompt, max_tokens, temperature)

        # Cache result
        if cache_enabled:
            self._save_to_cache(cache_key, result)

        return result

    def batch_get_logprobs(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 1.0,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Get teacher logprobs for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per prompt
            temperature: Sampling temperature
            show_progress: Whether to log progress

        Returns:
            List of teacher outputs (one per prompt)
        """
        results = []

        for i, prompt in enumerate(prompts):
            if show_progress and i % 10 == 0:
                logger.info(f"Processing prompt {i+1}/{len(prompts)}")

            try:
                result = self.get_logprobs(prompt, max_tokens, temperature)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process prompt {i}: {e}")
                # Add None for failed prompts
                results.append(None)

        return results

    def _get_cache_key(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Generate deterministic cache key for a prompt.

        Uses SHA256 hash of (prompt + max_tokens + temperature).
        """
        content = f"{prompt}||{max_tokens}||{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached teacher output"""
        if cache_key not in self.cache_index:
            return None

        cache_file = self.cache_dir / self.cache_index[cache_key]

        if not cache_file.exists():
            # Cache index is stale, remove entry
            del self.cache_index[cache_key]
            self._save_cache_index()
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save teacher output to cache"""
        cache_filename = f"{cache_key}.json"
        cache_file = self.cache_dir / cache_filename

        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)

            # Update cache index
            self.cache_index[cache_key] = cache_filename
            self._save_cache_index()

        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def _load_cache_index(self) -> Dict[str, str]:
        """Load cache index from disk"""
        if not self.cache_index_path.exists():
            return {}

        try:
            with open(self.cache_index_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
            return {}

    def _save_cache_index(self):
        """Save cache index to disk"""
        try:
            with open(self.cache_index_path, 'w') as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'total_inference_time': self.total_inference_time,
            'cached_prompts': len(self.cache_index)
        }

    def clear_cache(self):
        """Clear all cached teacher outputs"""
        logger.info("Clearing teacher cache")

        # Delete all cache files
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file != self.cache_index_path:
                cache_file.unlink()

        # Reset cache index
        self.cache_index = {}
        self._save_cache_index()

        # Reset statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"TeacherModel(path={self.model_path}, status={status})"
