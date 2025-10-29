"""
Distillation Loss Computation

Implements reverse KL divergence loss and related metrics for knowledge distillation.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DistillationLoss:
    """
    Computes distillation loss for knowledge transfer.

    Uses reverse KL divergence: D_KL(Student || Teacher)
    This makes the student "mode-seeking" - focuses on teacher's high-probability regions.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        kl_weight: float = 0.8,
        ce_weight: float = 0.2
    ):
        """
        Initialize distillation loss calculator.

        Args:
            temperature: Temperature for softening distributions
            kl_weight: Weight for KL divergence loss
            ce_weight: Weight for cross-entropy loss (optional)
        """
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight

        # Validate weights sum to 1.0
        total_weight = kl_weight + ce_weight
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Loss weights sum to {total_weight}, not 1.0. Normalizing.")
            self.kl_weight = kl_weight / total_weight
            self.ce_weight = ce_weight / total_weight

        logger.info(f"DistillationLoss: T={temperature}, KL={self.kl_weight}, CE={self.ce_weight}")

    def compute_kl_divergence(
        self,
        student_logits: mx.array,
        teacher_logits: mx.array,
        mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, Dict]:
        """
        Compute reverse KL divergence: D_KL(Student || Teacher)

        Formula:
            KL(P_student || P_teacher) = sum_v [ P_student(v) * log(P_student(v) / P_teacher(v)) ]

        Args:
            student_logits: (batch, seq_len, vocab_size) - student model logits
            teacher_logits: (batch, seq_len, vocab_size) - teacher model logits
            mask: (batch, seq_len) - binary mask (1 for valid, 0 for padding)

        Returns:
            (kl_loss, metrics_dict)
        """
        # Apply temperature scaling
        student_logits_scaled = student_logits / self.temperature
        teacher_logits_scaled = teacher_logits / self.temperature

        # Convert to probabilities
        student_probs = mx.softmax(student_logits_scaled, axis=-1)
        teacher_probs = mx.softmax(teacher_logits_scaled, axis=-1)

        # Compute KL divergence per position
        # KL(P||Q) = sum[ P * (log(P) - log(Q)) ]
        log_student = mx.log(student_probs + 1e-10)
        log_teacher = mx.log(teacher_probs + 1e-10)

        kl_per_token = mx.sum(
            student_probs * (log_student - log_teacher),
            axis=-1  # Sum over vocabulary dimension
        )  # Shape: (batch, seq_len)

        # Apply mask if provided
        if mask is not None:
            kl_per_token = kl_per_token * mask
            num_tokens = mx.sum(mask)
        else:
            num_tokens = mx.prod(mx.array(kl_per_token.shape))

        # Mean KL divergence
        kl_loss = mx.sum(kl_per_token) / num_tokens

        # Scale by temperature^2 (standard practice in distillation)
        kl_loss = kl_loss * (self.temperature ** 2)

        # Compute additional metrics
        metrics = self._compute_kl_metrics(
            student_probs,
            teacher_probs,
            kl_per_token,
            mask
        )
        metrics['kl_loss'] = float(kl_loss)

        return kl_loss, metrics

    def compute_cross_entropy(
        self,
        student_logits: mx.array,
        target_token_ids: mx.array,
        mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, Dict]:
        """
        Compute standard cross-entropy loss.

        Args:
            student_logits: (batch, seq_len, vocab_size)
            target_token_ids: (batch, seq_len) - target token indices
            mask: (batch, seq_len) - binary mask

        Returns:
            (ce_loss, metrics_dict)
        """
        # Compute cross-entropy loss
        # MLX's cross_entropy expects logits and targets
        ce_per_token = nn.losses.cross_entropy(
            student_logits,
            target_token_ids,
            reduction='none'
        )  # Shape: (batch, seq_len)

        # Apply mask
        if mask is not None:
            ce_per_token = ce_per_token * mask
            num_tokens = mx.sum(mask)
        else:
            num_tokens = mx.prod(mx.array(ce_per_token.shape))

        # Mean cross-entropy
        ce_loss = mx.sum(ce_per_token) / num_tokens

        metrics = {
            'ce_loss': float(ce_loss),
            'perplexity': float(mx.exp(ce_loss))
        }

        return ce_loss, metrics

    def compute(
        self,
        student_logits: mx.array,
        teacher_logits: mx.array,
        target_token_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, Dict]:
        """
        Compute total distillation loss.

        Args:
            student_logits: (batch, seq_len, vocab_size)
            teacher_logits: (batch, seq_len, vocab_size)
            target_token_ids: (batch, seq_len) - optional for CE loss
            mask: (batch, seq_len) - binary mask

        Returns:
            (total_loss, metrics_dict)
        """
        # Compute KL divergence
        kl_loss, kl_metrics = self.compute_kl_divergence(
            student_logits,
            teacher_logits,
            mask
        )

        # Start with weighted KL loss
        total_loss = self.kl_weight * kl_loss

        # Combine metrics
        metrics = kl_metrics.copy()
        metrics['kl_weight'] = self.kl_weight

        # Add cross-entropy if enabled and targets provided
        if self.ce_weight > 0 and target_token_ids is not None:
            ce_loss, ce_metrics = self.compute_cross_entropy(
                student_logits,
                target_token_ids,
                mask
            )

            total_loss = total_loss + self.ce_weight * ce_loss

            # Add CE metrics
            metrics.update(ce_metrics)
            metrics['ce_weight'] = self.ce_weight

        metrics['total_loss'] = float(total_loss)

        return total_loss, metrics

    def _compute_kl_metrics(
        self,
        student_probs: mx.array,
        teacher_probs: mx.array,
        kl_per_token: mx.array,
        mask: Optional[mx.array]
    ) -> Dict:
        """
        Compute additional metrics for analysis.

        Returns:
            Dictionary with:
                - kl_mean, kl_std, kl_max, kl_min
                - token_agreement_pct
                - top5_agreement_pct
                - student_entropy, teacher_entropy
                - js_divergence
        """
        # KL statistics
        if mask is not None:
            # Only compute over valid positions
            valid_kl = kl_per_token * mask
            kl_mean = float(mx.sum(valid_kl) / mx.sum(mask))
            # For std, need to mask properly - use mx.where instead of boolean indexing
            kl_values = mx.where(mask == 1, kl_per_token, 0.0)
            # Only compute std over non-zero (valid) positions
            num_valid = int(mx.sum(mask))
            if num_valid > 0:
                kl_std = float(mx.std(kl_values))
            else:
                kl_std = 0.0
        else:
            kl_mean = float(mx.mean(kl_per_token))
            kl_std = float(mx.std(kl_per_token))

        kl_max = float(mx.max(kl_per_token))
        kl_min = float(mx.min(kl_per_token))

        # Token agreement (argmax matches)
        student_preds = mx.argmax(student_probs, axis=-1)  # (batch, seq_len)
        teacher_preds = mx.argmax(teacher_probs, axis=-1)  # (batch, seq_len)

        agreement = (student_preds == teacher_preds).astype(mx.float32)

        if mask is not None:
            agreement = agreement * mask
            token_agreement_pct = float(mx.sum(agreement) / mx.sum(mask)) * 100
        else:
            token_agreement_pct = float(mx.mean(agreement)) * 100

        # Top-5 agreement (teacher's top-1 in student's top-5)
        # Get top-5 student predictions
        student_top5_indices = mx.argsort(student_probs, axis=-1)[:, :, -5:]  # Top 5
        # Check if teacher's argmax is in student's top-5
        teacher_in_top5 = mx.any(
            student_top5_indices == teacher_preds[:, :, None],
            axis=-1
        ).astype(mx.float32)

        if mask is not None:
            teacher_in_top5 = teacher_in_top5 * mask
            top5_agreement_pct = float(mx.sum(teacher_in_top5) / mx.sum(mask)) * 100
        else:
            top5_agreement_pct = float(mx.mean(teacher_in_top5)) * 100

        # Entropy
        student_entropy = -mx.sum(
            student_probs * mx.log(student_probs + 1e-10),
            axis=-1
        )  # (batch, seq_len)

        teacher_entropy = -mx.sum(
            teacher_probs * mx.log(teacher_probs + 1e-10),
            axis=-1
        )  # (batch, seq_len)

        if mask is not None:
            student_entropy_mean = float(mx.sum(student_entropy * mask) / mx.sum(mask))
            teacher_entropy_mean = float(mx.sum(teacher_entropy * mask) / mx.sum(mask))
        else:
            student_entropy_mean = float(mx.mean(student_entropy))
            teacher_entropy_mean = float(mx.mean(teacher_entropy))

        # Jensen-Shannon divergence (symmetric measure)
        m = 0.5 * (student_probs + teacher_probs)
        js_div_per_token = 0.5 * (
            mx.sum(student_probs * mx.log((student_probs + 1e-10) / (m + 1e-10)), axis=-1) +
            mx.sum(teacher_probs * mx.log((teacher_probs + 1e-10) / (m + 1e-10)), axis=-1)
        )

        if mask is not None:
            js_divergence = float(mx.sum(js_div_per_token * mask) / mx.sum(mask))
        else:
            js_divergence = float(mx.mean(js_div_per_token))

        return {
            'kl_mean': kl_mean,
            'kl_std': kl_std,
            'kl_max': kl_max,
            'kl_min': kl_min,
            'token_agreement_pct': token_agreement_pct,
            'top5_agreement_pct': top5_agreement_pct,
            'student_entropy': student_entropy_mean,
            'teacher_entropy': teacher_entropy_mean,
            'js_divergence': js_divergence
        }

    def update_temperature(self, new_temperature: float):
        """Update temperature (for curriculum learning)"""
        logger.info(f"Updating temperature: {self.temperature} -> {new_temperature}")
        self.temperature = new_temperature

    def update_weights(self, kl_weight: float, ce_weight: float):
        """Update loss weights (for curriculum learning)"""
        # Normalize
        total = kl_weight + ce_weight
        self.kl_weight = kl_weight / total
        self.ce_weight = ce_weight / total
        logger.info(f"Updated weights: KL={self.kl_weight}, CE={self.ce_weight}")
