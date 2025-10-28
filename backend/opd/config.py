"""
Configuration classes for On-Policy Distillation (OPD)
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
import uuid
from datetime import datetime


@dataclass
class OPDConfig:
    """Configuration for knowledge distillation training"""

    # === MODEL PATHS ===
    base_model_path: str
    """Path to base model (Qwen 7B)"""

    teacher_model_path: str
    """Path to teacher model (Qwen 32B)"""

    student_adapter_path: str
    """Path to student's fine-tuned LoRA adapters (from SFT)"""

    output_adapter_path: str
    """Path where distilled adapters will be saved"""

    # === DATA ===
    validation_prompts_path: str
    """Path to validation prompts file (JSONL)"""

    max_prompts: int = 1000
    """Maximum number of prompts to use (memory constraint)"""

    train_val_split: float = 0.8
    """Split ratio for train/validation (0.8 = 80% train, 20% val)"""

    # === TRAINING ===
    num_steps: int = 1000
    """Total number of training steps"""

    batch_size: int = 4
    """Batch size (optimized for 128GB RAM)"""

    gradient_accumulation_steps: int = 2
    """Gradient accumulation steps (effective batch = batch_size * this)"""

    learning_rate: float = 1e-5
    """Learning rate for LoRA adapters"""

    warmup_steps: int = 100
    """Number of warmup steps for learning rate"""

    max_grad_norm: float = 1.0
    """Gradient clipping threshold"""

    # === DISTILLATION ===
    temperature: float = 2.0
    """Temperature for softening probability distributions"""

    kl_weight: float = 0.8
    """Weight for KL divergence loss"""

    ce_weight: float = 0.2
    """Weight for cross-entropy loss (optional)"""

    use_teacher_targets: bool = True
    """Whether to use teacher's generated text as targets for CE loss"""

    # === GENERATION ===
    max_generation_tokens: int = 512
    """Maximum tokens to generate per prompt"""

    generation_temperature: float = 1.0
    """Temperature for sampling during generation"""

    logprob_method: Literal["manual_loop", "forward_pass", "callback"] = "manual_loop"
    """Method for extracting logprobs from teacher model"""

    # === CHECKPOINTING ===
    checkpoint_every: int = 100
    """Save checkpoint every N steps"""

    eval_every: int = 100
    """Evaluate on validation set every N steps"""

    save_best_only: bool = False
    """Only save checkpoint when validation loss improves"""

    keep_last_n_checkpoints: int = 5
    """Number of recent checkpoints to keep (0 = keep all)"""

    # === TEACHER CACHE ===
    use_cache: bool = True
    """Enable caching of teacher outputs"""

    cache_dir: str = "./OnPolicyDistill/teacher_cache"
    """Directory for caching teacher outputs"""

    cache_size_mb: int = 4096
    """Maximum cache size in memory (MB) before writing to disk"""

    keep_teacher_loaded: bool = True
    """Keep teacher model loaded in memory (recommended for 128GB RAM)"""

    # === SYSTEM ===
    seed: int = 42
    """Random seed for reproducibility"""

    mixed_precision: bool = True
    """Use mixed precision (float16) training"""

    num_workers: int = 0
    """Number of data loading workers (0 = main thread)"""

    # === RUN METADATA ===
    run_id: Optional[str] = None
    """Unique identifier for this training run"""

    session_id: Optional[str] = None
    """Link to parent SFT session ID"""

    experiment_name: Optional[str] = None
    """Optional experiment name for organization"""

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Generate run_id if not provided
        if self.run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"distill_{timestamp}"

        # Validate numeric ranges
        if not 0 < self.train_val_split < 1:
            raise ValueError("train_val_split must be between 0 and 1")

        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

        if not 0 <= self.kl_weight <= 1:
            raise ValueError("kl_weight must be between 0 and 1")

        if not 0 <= self.ce_weight <= 1:
            raise ValueError("ce_weight must be between 0 and 1")

        if abs((self.kl_weight + self.ce_weight) - 1.0) > 1e-6:
            raise ValueError("kl_weight + ce_weight must sum to 1.0")

        # Ensure directories exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        checkpoint_base = Path(self.output_adapter_path).parent
        checkpoint_base.mkdir(parents=True, exist_ok=True)

        # Validate paths exist
        if not Path(self.base_model_path).exists():
            raise FileNotFoundError(f"Base model not found: {self.base_model_path}")

        if not Path(self.teacher_model_path).exists():
            raise FileNotFoundError(f"Teacher model not found: {self.teacher_model_path}")

        if not Path(self.student_adapter_path).exists():
            raise FileNotFoundError(f"Student adapter not found: {self.student_adapter_path}")

        if not Path(self.validation_prompts_path).exists():
            raise FileNotFoundError(f"Validation prompts not found: {self.validation_prompts_path}")

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }

    def save(self, path: str):
        """Save configuration to YAML file"""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> 'OPDConfig':
        """Load configuration from YAML file"""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


@dataclass
class OPDMetrics:
    """Metrics tracked during distillation training"""

    step: int
    """Current training step"""

    # === LOSS COMPONENTS ===
    kl_loss: float
    """Reverse KL divergence loss"""

    ce_loss: Optional[float] = None
    """Cross-entropy loss (if used)"""

    total_loss: float = 0.0
    """Combined total loss"""

    # === KL STATISTICS ===
    kl_mean: float = 0.0
    """Mean KL divergence across tokens"""

    kl_std: float = 0.0
    """Standard deviation of KL divergence"""

    kl_max: float = 0.0
    """Maximum KL divergence"""

    kl_min: float = 0.0
    """Minimum KL divergence"""

    # === ALIGNMENT METRICS ===
    token_agreement_pct: float = 0.0
    """Percentage of tokens where student's argmax matches teacher's"""

    top5_agreement_pct: float = 0.0
    """Percentage where teacher's top-1 is in student's top-5"""

    # === DISTRIBUTION METRICS ===
    student_entropy: float = 0.0
    """Average entropy of student's output distribution"""

    teacher_entropy: float = 0.0
    """Average entropy of teacher's output distribution"""

    js_divergence: float = 0.0
    """Jensen-Shannon divergence (symmetric measure)"""

    # === PERFORMANCE ===
    tokens_per_second: float = 0.0
    """Throughput in tokens per second"""

    samples_processed: int = 0
    """Number of samples processed in this step"""

    step_time_seconds: float = 0.0
    """Time taken for this step"""

    # === TEACHER INFERENCE ===
    teacher_inference_ms: float = 0.0
    """Teacher model inference time (milliseconds)"""

    teacher_cache_hit_rate: float = 0.0
    """Percentage of teacher outputs retrieved from cache"""

    # === VALIDATION METRICS (populated during eval) ===
    val_kl_loss: Optional[float] = None
    """Validation KL loss"""

    val_token_agreement: Optional[float] = None
    """Validation token agreement percentage"""

    val_student_entropy: Optional[float] = None
    """Validation student entropy"""

    # === CHECKPOINTING ===
    checkpoint_path: Optional[str] = None
    """Path to checkpoint saved at this step"""

    is_best: bool = False
    """Whether this is the best checkpoint so far"""

    # === MEMORY ===
    memory_used_gb: Optional[float] = None
    """Peak memory usage in GB"""

    def to_dict(self) -> dict:
        """Convert metrics to dictionary"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def to_jsonl_line(self) -> str:
        """Convert to JSONL line for logging"""
        import json
        return json.dumps(self.to_dict())


@dataclass
class TeacherCacheEntry:
    """Cached teacher model output"""

    prompt: str
    """Input prompt"""

    generated_text: str
    """Teacher's generated completion"""

    tokens: list[str]
    """Token strings"""

    token_ids: list[int]
    """Token IDs"""

    logprobs: list[float]
    """Log probabilities for each token (shape: seq_len,)"""

    full_logprobs: Optional[list[list[float]]] = None
    """Full distribution logprobs (shape: seq_len, vocab_size) - optional"""

    timestamp: Optional[str] = None
    """When this was cached"""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


# === PRESET CONFIGURATIONS ===

def get_fast_iteration_config(**overrides) -> OPDConfig:
    """
    Fast iteration preset for testing and debugging.

    Reduced steps and batch size for quick experiments.
    """
    config = OPDConfig(
        num_steps=100,
        batch_size=2,
        checkpoint_every=25,
        eval_every=25,
        max_prompts=50,
        gradient_accumulation_steps=1,
        **overrides
    )
    return config


def get_high_quality_config(**overrides) -> OPDConfig:
    """
    High quality preset for production training.

    More steps, higher temperature, better knowledge transfer.
    """
    config = OPDConfig(
        num_steps=2000,
        batch_size=4,
        gradient_accumulation_steps=2,
        checkpoint_every=200,
        eval_every=100,
        temperature=3.0,  # Higher temperature for more knowledge
        max_prompts=2000,
        **overrides
    )
    return config


def get_memory_efficient_config(**overrides) -> OPDConfig:
    """
    Memory efficient preset for systems with less RAM.

    Smaller batch size, aggressive caching, teacher unloading.
    """
    config = OPDConfig(
        batch_size=1,
        gradient_accumulation_steps=8,
        max_prompts=500,
        cache_size_mb=2048,
        keep_teacher_loaded=False,  # Unload teacher between batches
        **overrides
    )
    return config
