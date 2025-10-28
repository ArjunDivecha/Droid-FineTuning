# On-Policy Distillation (OPD) Implementation Plan v3.0 FINAL
## Knowledge Distillation: Qwen 32B → 7B for Droid-FineTuning

**Status**: FINAL - Focused on Knowledge Distillation with local Qwen 32B teacher

**Key Approach**: Use user's existing Qwen 32B model as teacher to distill knowledge into smaller, faster 7B student model.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Core Approach: Knowledge Distillation](#core-approach-knowledge-distillation)
4. [Directory Structure](#directory-structure)
5. [Component Design](#component-design)
6. [Implementation Phases](#implementation-phases)
7. [Configuration](#configuration)
8. [API Design](#api-design)
9. [UI Design](#ui-design)
10. [Testing Strategy](#testing-strategy)
11. [Timeline](#timeline)
12. [Success Criteria](#success-criteria)

---

## Executive Summary

### The Goal
Enable users to distill knowledge from a large teacher model (Qwen 32B) into their fine-tuned smaller student model (Qwen 7B), producing a compact model with improved quality while maintaining fast inference.

### The Approach
**Classic Knowledge Distillation** via reverse KL divergence:
- Teacher: User's Qwen 32B (frozen, inference only)
- Student: User's fine-tuned Qwen 7B with LoRA adapters (trainable)
- Loss: Minimize KL(Student || Teacher) on validation prompts
- Result: Compact 7B model that mimics 32B behavior

### Why This Works
- ✅ User already has 32B model (no new downloads)
- ✅ 100% local, no API costs
- ✅ Proven distillation technique
- ✅ Fits MLX architecture perfectly
- ✅ Produces faster inference model
- ✅ Clear quality improvement metrics

### Key Benefits
1. **Compress Knowledge**: 32B intelligence → 7B model
2. **Faster Inference**: Deploy 7B instead of 32B
3. **Lower Memory**: Run distilled model on more devices
4. **Better Quality**: Student learns from superior teacher
5. **Full Control**: All local, no external dependencies

---

## Architecture Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE DISTILLATION PIPELINE               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Validation Prompts (from SFT training data)             │
│         ↓                                                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │  TEACHER MODEL (Qwen 32B)                        │          │
│  │  - Frozen weights                                │          │
│  │  - Inference only                                │          │
│  │  - High quality responses                        │          │
│  └──────────────────────────────────────────────────┘          │
│         ↓                                                        │
│  Teacher Logprobs: P_teacher(token | prompt)                    │
│         ↓                                                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │  STUDENT MODEL (Qwen 7B)                         │          │
│  │  - Base model + Fine-tuned LoRA adapters         │          │
│  │  - LoRA weights trainable                        │          │
│  │  - Learning to match teacher                     │          │
│  └──────────────────────────────────────────────────┘          │
│         ↓                                                        │
│  Student Logprobs: P_student(token | prompt)                    │
│         ↓                                                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │  DISTILLATION LOSS                               │          │
│  │  KL(Student || Teacher)                          │          │
│  │  = Σ P_student * log(P_student / P_teacher)      │          │
│  └──────────────────────────────────────────────────┘          │
│         ↓                                                        │
│                                                                  │
│  Backprop → Update LoRA adapters                                │
│         ↓                                                        │
│                                                                  │
│  OUTPUT: Distilled LoRA Adapters                                │
│          (7B model with 32B knowledge)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                     DROID FINETUNING WORKFLOW                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Setup Page                                                   │
│     └─> Select base model (Qwen 7B)                             │
│     └─> Upload training data                                     │
│                                                                  │
│  2. Training Page (SFT)                                          │
│     └─> Fine-tune with LoRA adapters                            │
│     └─> Result: task-adapted model                              │
│                                                                  │
│  3. [NEW] OPD Page (Knowledge Distillation)                     │
│     └─> Select teacher (Qwen 32B)                               │
│     └─> Select student (fine-tuned 7B)                          │
│     └─> Result: distilled adapters (7B w/ 32B knowledge)        │
│                                                                  │
│  4. Results Page                                                 │
│     └─> View metrics, compare models                            │
│                                                                  │
│  5. Compare Page                                                 │
│     └─> Test: Base vs SFT vs Distilled                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Approach: Knowledge Distillation

### Mathematical Foundation

**Objective**: Train student to match teacher's output distribution

**Standard Cross-Entropy Loss**:
```
L_CE = -Σ y_true * log(P_student)
```

**Distillation Loss** (Hinton et al., 2015):
```
L_KL = KL(P_teacher || P_student)
     = Σ P_teacher(t) * log(P_teacher(t) / P_student(t))
```

**Temperature Scaling** (softens probability distributions):
```
P(t) = exp(logit(t) / T) / Σ exp(logit(t) / T)

where T = temperature (typically 1.0 - 4.0)
```

**Combined Loss**:
```
L_total = α * L_KL + (1-α) * L_CE

where:
- α = distillation weight (typically 0.7 - 0.9)
- Higher α = more teacher influence
```

### Why Reverse KL (Student || Teacher)?

We use **reverse KL divergence**: `KL(Student || Teacher)` instead of forward KL.

**Reverse KL Properties**:
- **Mode-seeking**: Student focuses on teacher's high-probability regions
- **Avoids over-coverage**: Student doesn't waste capacity on low-prob regions
- **Better for distillation**: Matches primary behaviors, not full distribution

**Forward vs Reverse KL**:
```
Forward KL(Teacher || Student):
- "Cover all teacher modes"
- Student tries to match everywhere
- Can lead to over-smoothing

Reverse KL(Student || Teacher):
- "Focus on main teacher modes"
- Student matches where teacher is confident
- Better compression
```

### Temperature Tuning

**Effect of Temperature**:
```
T = 1.0:  Sharp distributions (high confidence)
T = 2.0:  Softer distributions (moderate uncertainty)
T = 4.0:  Very soft (explores more of distribution)

Higher T → More "dark knowledge" transferred
Lower T  → More focused on top predictions
```

**Recommended**: Start with T=2.0, tune based on validation performance.

---

## Directory Structure

```
Droid-FineTuning/
├── OnPolicyDistill/                         # OPD root directory
│   ├── configs/                             # Run configurations
│   │   └── distill_{run_id}.yaml
│   │
│   ├── teacher_cache/                       # Cached teacher outputs
│   │   ├── {prompt_hash}_logprobs.npz      # Teacher logprobs per prompt
│   │   └── cache_index.json                 # Cache metadata
│   │
│   ├── student_rollouts/                    # Student generation samples
│   │   └── {run_id}_rollouts.jsonl
│   │
│   ├── checkpoints/                         # Distilled adapters
│   │   └── {run_id}/
│   │       ├── step_000100.safetensors
│   │       ├── step_000200.safetensors
│   │       └── best_adapters.safetensors    # Best validation loss
│   │
│   └── metrics/                             # Training metrics
│       ├── {run_id}_train.jsonl             # Step-by-step metrics
│       └── {run_id}_eval.jsonl              # Eval checkpoints
│
├── backend/
│   ├── main.py                              # MODIFY: Add /opd endpoints
│   │
│   └── opd/                                 # NEW: OPD backend module
│       ├── __init__.py
│       │
│       ├── config.py                        # OPDConfig dataclass
│       │   - DistillationConfig
│       │   - OPDMetrics
│       │
│       ├── teacher_model.py                 # Teacher (32B) loader & inference
│       │   - TeacherModel class
│       │   - load_teacher()
│       │   - get_teacher_logprobs()
│       │   - cache management
│       │
│       ├── student_model.py                 # Student (7B) with LoRA
│       │   - StudentModel class
│       │   - load_student()
│       │   - forward_with_logits()
│       │
│       ├── distillation_loss.py             # KL divergence computation
│       │   - DistillationLoss class
│       │   - compute_kl_divergence()
│       │   - temperature scaling
│       │
│       ├── distillation_trainer.py          # Main training orchestrator
│       │   - DistillationTrainer class
│       │   - train_loop()
│       │   - checkpoint management
│       │   - evaluation
│       │
│       ├── data_loader.py                   # Load validation prompts
│       │   - load_prompts()
│       │   - create_batches()
│       │
│       └── utils.py                         # Helper functions
│           - compute_alignment_score()
│           - cache helpers
│           - logging utilities
│
└── frontend/src/
    ├── pages/
    │   └── OPDPage.tsx                      # NEW: Main OPD page
    │
    ├── components/
    │   └── opd/                             # NEW: OPD components
    │       ├── DistillationSetup.tsx        # Teacher/student selection
    │       ├── DistillationProgress.tsx     # Real-time training view
    │       ├── DistillationMetrics.tsx      # KL loss charts
    │       └── DistillationResults.tsx      # Final results summary
    │
    ├── store/slices/
    │   └── opdSlice.ts                      # NEW: OPD Redux state
    │
    └── App.tsx                              # MODIFY: Add /opd route
```

---

## Component Design

### 1. Configuration (`backend/opd/config.py`)

**OPDConfig Dataclass**:
```python
@dataclass
class OPDConfig:
    """Configuration for knowledge distillation"""

    # === MODEL PATHS ===
    base_model_path: str              # Base Qwen 7B
    teacher_model_path: str           # Qwen 32B teacher
    student_adapter_path: str         # Input: fine-tuned LoRA from SFT
    output_adapter_path: str          # Output: distilled LoRA

    # === DATA ===
    validation_prompts_path: str      # Prompts for distillation
    max_prompts: int = 1000           # Limit prompts (memory constraint)

    # === TRAINING ===
    num_steps: int = 1000             # Total training steps
    batch_size: int = 2               # Small batch (memory intensive)
    learning_rate: float = 1e-5       # LoRA learning rate
    gradient_accumulation_steps: int = 4  # Effective batch_size *= 4

    # === DISTILLATION ===
    temperature: float = 2.0          # Softening temperature
    kl_weight: float = 0.8            # Weight for KL loss
    ce_weight: float = 0.2            # Weight for standard CE loss
    use_teacher_targets: bool = True  # Use teacher's generated text as targets

    # === GENERATION (for rollouts) ===
    max_generation_tokens: int = 512  # Max tokens to generate
    generation_temperature: float = 1.0

    # === CHECKPOINTING ===
    checkpoint_every: int = 100       # Save checkpoint interval
    eval_every: int = 100             # Evaluation interval
    save_best_only: bool = False      # Save all checkpoints or best only

    # === TEACHER CACHE ===
    use_cache: bool = True            # Cache teacher outputs
    cache_dir: str = "./OnPolicyDistill/teacher_cache"

    # === SYSTEM ===
    seed: int = 42
    mixed_precision: bool = True      # Use float16 where possible

    # === RUN METADATA ===
    run_id: Optional[str] = None
    session_id: Optional[str] = None  # Link to SFT session

@dataclass
class OPDMetrics:
    """Metrics tracked during distillation"""
    step: int

    # Loss components
    kl_loss: float                    # Reverse KL divergence
    ce_loss: Optional[float] = None   # Cross-entropy (if used)
    total_loss: float = 0.0

    # KL statistics
    kl_mean: float = 0.0
    kl_std: float = 0.0
    kl_max: float = 0.0
    kl_min: float = 0.0

    # Alignment metrics
    token_agreement_pct: float = 0.0  # % tokens where argmax matches
    top5_agreement_pct: float = 0.0   # % tokens where teacher top-5 includes student top-1

    # Distribution metrics
    student_entropy: float = 0.0      # Student output entropy
    teacher_entropy: float = 0.0      # Teacher output entropy
    js_divergence: float = 0.0        # Jensen-Shannon divergence

    # Performance
    tokens_per_second: float = 0.0
    samples_processed: int = 0
    teacher_inference_ms: float = 0.0
    student_inference_ms: float = 0.0

    # Validation (if eval step)
    val_kl_loss: Optional[float] = None
    val_token_agreement: Optional[float] = None

    # Checkpointing
    checkpoint_path: Optional[str] = None
    is_best: bool = False
```

---

### 2. Teacher Model (`backend/opd/teacher_model.py`)

**Purpose**: Load Qwen 32B teacher, run inference, extract logprobs, cache results

**Key Responsibilities**:
- Load teacher model (frozen weights)
- Generate completions for prompts
- Extract per-token logprobs from model's forward pass
- Cache logprobs to disk (avoid recomputation)
- Memory-efficient batch processing

**Class Structure**:
```python
class TeacherModel:
    def __init__(self, model_path: str, cache_dir: str):
        """Initialize teacher model manager"""

    def load(self):
        """Load Qwen 32B model (frozen)"""
        # Load with MLX
        # Freeze all parameters
        # Set to eval mode

    def get_logprobs(
        self,
        prompt: str,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Get teacher's token-level logprobs for a prompt.

        Returns:
            {
                'prompt': str,
                'generated_text': str,
                'tokens': List[str],           # Token strings
                'token_ids': List[int],        # Token IDs
                'logprobs': np.ndarray,        # (seq_len, vocab_size)
                'token_logprobs': List[float], # log P(token_i | context)
            }
        """
        # Check cache first
        # If not cached, run inference
        # Extract logprobs from forward pass
        # Cache result
        # Return

    def batch_get_logprobs(
        self,
        prompts: List[str],
        max_tokens: int = 512
    ) -> List[Dict]:
        """Get logprobs for batch of prompts"""

    def _get_cache_key(self, prompt: str) -> str:
        """Generate deterministic cache key"""
        # Use SHA256 hash of prompt

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached teacher output"""

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save teacher output to cache"""

    def unload(self):
        """Free teacher model memory"""
```

**Critical Implementation Detail**: Extracting logprobs from MLX model

MLX's `generate()` function may not expose per-token logprobs directly. We need to:

**Option A**: Use model's forward pass
```python
# Run forward pass to get logits
logits = model(input_ids)  # (batch, seq_len, vocab_size)
logprobs = mx.log_softmax(logits, axis=-1)
```

**Option B**: Hook into generation loop
```python
# If MLX generate() supports callback
def logprob_callback(token_id, logits):
    logprobs = mx.log_softmax(logits, axis=-1)
    # Store logprobs[token_id]

response = generate(model, tokenizer, prompt,
                   callback=logprob_callback)
```

**Option C**: Manual generation loop
```python
# Implement our own generation with logprob tracking
def generate_with_logprobs(model, tokenizer, prompt, max_tokens):
    tokens = tokenizer.encode(prompt)
    logprobs_list = []

    for _ in range(max_tokens):
        logits = model(tokens)  # Forward pass
        logprobs = mx.log_softmax(logits[-1], axis=-1)  # Last position

        # Sample next token
        next_token = sample(logprobs, temperature=1.0)
        tokens.append(next_token)
        logprobs_list.append(logprobs)

        if next_token == eos_token:
            break

    return tokens, logprobs_list
```

**Recommended**: Use Option C for full control and guaranteed logprob access.

---

### 3. Student Model (`backend/opd/student_model.py`)

**Purpose**: Load Qwen 7B + fine-tuned LoRA, run forward pass with gradient tracking

**Key Responsibilities**:
- Load base model + LoRA adapters
- Enable gradient tracking for LoRA weights only
- Compute forward pass with logits output
- Support training mode (gradients) and eval mode

**Class Structure**:
```python
class StudentModel:
    def __init__(
        self,
        base_model_path: str,
        adapter_path: str
    ):
        """Initialize student model with LoRA"""

    def load(self):
        """Load base + adapter with gradient tracking"""
        # Load Qwen 7B base
        # Load LoRA adapters
        # Freeze base model
        # Enable gradients for LoRA only

    def forward(
        self,
        prompts: List[str],
        teacher_token_ids: List[List[int]] = None
    ) -> Dict[str, mx.array]:
        """
        Forward pass through student model.

        Args:
            prompts: Input prompts
            teacher_token_ids: If provided, run forward on these tokens
                              (for aligned comparison with teacher)

        Returns:
            {
                'logits': mx.array,      # (batch, seq_len, vocab_size)
                'logprobs': mx.array,    # log_softmax of logits
                'token_ids': List[List[int]],
                'loss_mask': mx.array    # Mask for valid positions
            }
        """

    def generate_rollouts(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 1.0
    ) -> List[Dict]:
        """
        Generate completions from student (for on-policy data).

        Returns list of:
            {
                'prompt': str,
                'generated_text': str,
                'token_ids': List[int]
            }
        """

    def save_adapter(self, path: str):
        """Save current LoRA adapter weights"""

    def get_trainable_params(self) -> List[mx.array]:
        """Get list of trainable parameters (LoRA only)"""
```

---

### 4. Distillation Loss (`backend/opd/distillation_loss.py`)

**Purpose**: Compute reverse KL divergence between student and teacher

**Key Responsibilities**:
- Apply temperature scaling
- Compute KL divergence loss
- Optionally mix in cross-entropy loss
- Compute alignment metrics
- Handle sequence masking (ignore padding)

**Class Structure**:
```python
class DistillationLoss:
    def __init__(
        self,
        temperature: float = 2.0,
        kl_weight: float = 0.8,
        ce_weight: float = 0.2
    ):
        """Initialize distillation loss calculator"""

    def compute(
        self,
        student_logits: mx.array,     # (batch, seq_len, vocab_size)
        teacher_logits: mx.array,     # (batch, seq_len, vocab_size)
        target_token_ids: mx.array = None,  # (batch, seq_len)
        mask: mx.array = None         # (batch, seq_len) - 1 for valid, 0 for pad
    ) -> Tuple[mx.array, Dict]:
        """
        Compute distillation loss.

        Returns:
            (total_loss, metrics_dict)
        """

        # 1. Apply temperature scaling
        student_logits_scaled = student_logits / self.temperature
        teacher_logits_scaled = teacher_logits / self.temperature

        # 2. Convert to probabilities
        student_probs = mx.softmax(student_logits_scaled, axis=-1)
        teacher_probs = mx.softmax(teacher_logits_scaled, axis=-1)

        # 3. Compute KL divergence: KL(Student || Teacher)
        # KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
        kl_div = student_probs * (
            mx.log(student_probs + 1e-10) - mx.log(teacher_probs + 1e-10)
        )
        kl_div = mx.sum(kl_div, axis=-1)  # Sum over vocabulary

        # 4. Apply mask (ignore padding)
        if mask is not None:
            kl_div = kl_div * mask
            num_tokens = mx.sum(mask)
        else:
            num_tokens = kl_div.size

        # 5. Mean KL loss
        kl_loss = mx.sum(kl_div) / num_tokens

        # 6. Scale by temperature^2 (standard practice)
        kl_loss = kl_loss * (self.temperature ** 2)

        # 7. Optional: Add cross-entropy loss
        ce_loss = 0.0
        if self.ce_weight > 0 and target_token_ids is not None:
            ce_loss = self._compute_ce_loss(
                student_logits,
                target_token_ids,
                mask
            )

        # 8. Combine losses
        total_loss = self.kl_weight * kl_loss + self.ce_weight * ce_loss

        # 9. Compute metrics
        metrics = self._compute_metrics(
            student_probs,
            teacher_probs,
            kl_div,
            mask
        )
        metrics['kl_loss'] = float(kl_loss)
        metrics['ce_loss'] = float(ce_loss) if ce_loss != 0 else None
        metrics['total_loss'] = float(total_loss)

        return total_loss, metrics

    def _compute_ce_loss(
        self,
        logits: mx.array,
        targets: mx.array,
        mask: mx.array
    ) -> mx.array:
        """Compute cross-entropy loss"""

    def _compute_metrics(
        self,
        student_probs: mx.array,
        teacher_probs: mx.array,
        kl_div: mx.array,
        mask: mx.array
    ) -> Dict:
        """Compute additional metrics"""
        # Token agreement (argmax matches)
        student_preds = mx.argmax(student_probs, axis=-1)
        teacher_preds = mx.argmax(teacher_probs, axis=-1)
        agreement = (student_preds == teacher_preds)

        if mask is not None:
            agreement = agreement * mask
            agreement_pct = mx.sum(agreement) / mx.sum(mask)
        else:
            agreement_pct = mx.mean(agreement)

        # Entropy
        student_entropy = -mx.sum(
            student_probs * mx.log(student_probs + 1e-10),
            axis=-1
        )
        teacher_entropy = -mx.sum(
            teacher_probs * mx.log(teacher_probs + 1e-10),
            axis=-1
        )

        # Jensen-Shannon divergence (symmetric)
        m = 0.5 * (student_probs + teacher_probs)
        js_div = 0.5 * (
            mx.sum(student_probs * mx.log(student_probs / m + 1e-10), axis=-1) +
            mx.sum(teacher_probs * mx.log(teacher_probs / m + 1e-10), axis=-1)
        )

        return {
            'kl_mean': float(mx.mean(kl_div)),
            'kl_std': float(mx.std(kl_div)),
            'kl_max': float(mx.max(kl_div)),
            'kl_min': float(mx.min(kl_div)),
            'token_agreement_pct': float(agreement_pct) * 100,
            'student_entropy': float(mx.mean(student_entropy)),
            'teacher_entropy': float(mx.mean(teacher_entropy)),
            'js_divergence': float(mx.mean(js_div))
        }
```

---

### 5. Distillation Trainer (`backend/opd/distillation_trainer.py`)

**Purpose**: Orchestrate full distillation training loop

**Key Responsibilities**:
- Load teacher and student models
- Load validation prompts
- Create training batches
- Run training loop (forward, loss, backward, update)
- Checkpoint management
- Evaluation on validation set
- Metrics logging and broadcasting

**Class Structure**:
```python
class DistillationTrainer:
    def __init__(self, config: OPDConfig):
        """Initialize distillation trainer"""
        self.config = config
        self.teacher = None
        self.student = None
        self.loss_fn = None
        self.optimizer = None
        self.current_step = 0
        self.best_val_loss = float('inf')

    def setup(self):
        """Load models and prepare training"""
        # 1. Load teacher (Qwen 32B)
        self.teacher = TeacherModel(
            self.config.teacher_model_path,
            self.config.cache_dir
        )
        self.teacher.load()

        # 2. Load student (Qwen 7B + LoRA)
        self.student = StudentModel(
            self.config.base_model_path,
            self.config.student_adapter_path
        )
        self.student.load()

        # 3. Initialize loss function
        self.loss_fn = DistillationLoss(
            temperature=self.config.temperature,
            kl_weight=self.config.kl_weight,
            ce_weight=self.config.ce_weight
        )

        # 4. Initialize optimizer (Adam for LoRA)
        self.optimizer = create_optimizer(
            params=self.student.get_trainable_params(),
            lr=self.config.learning_rate
        )

        # 5. Load validation prompts
        self.prompts = load_prompts(
            self.config.validation_prompts_path,
            max_prompts=self.config.max_prompts
        )

    def train(self):
        """Main training loop"""
        logging.info(f"Starting distillation training: {self.config.run_id}")
        logging.info(f"Teacher: {self.config.teacher_model_path}")
        logging.info(f"Student: {self.config.student_adapter_path}")
        logging.info(f"Prompts: {len(self.prompts)}")

        for step in range(self.config.num_steps):
            self.current_step = step

            # 1. Sample batch of prompts
            batch_prompts = self._sample_batch(self.config.batch_size)

            # 2. Get teacher outputs (with caching)
            teacher_outputs = self._get_teacher_outputs(batch_prompts)

            # 3. Get student outputs (forward pass)
            student_outputs = self._get_student_outputs(
                batch_prompts,
                teacher_outputs
            )

            # 4. Compute loss
            loss, metrics = self.loss_fn.compute(
                student_outputs['logits'],
                teacher_outputs['logits'],
                target_token_ids=teacher_outputs['token_ids'],
                mask=student_outputs['mask']
            )

            # 5. Backward pass and update
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 6. Log metrics
            self._log_metrics(step, metrics)
            self._broadcast_progress(step, metrics)

            # 7. Evaluate
            if step % self.config.eval_every == 0:
                val_metrics = self.evaluate()
                self._log_metrics(step, val_metrics, prefix='val')

                # Check for best model
                if val_metrics['kl_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['kl_loss']
                    self.save_checkpoint(step, is_best=True)

            # 8. Checkpoint
            if step % self.config.checkpoint_every == 0:
                self.save_checkpoint(step)

        # Final checkpoint
        self.save_checkpoint(self.config.num_steps, is_final=True)
        logging.info("Distillation training completed!")

    def _get_teacher_outputs(self, prompts: List[str]) -> Dict:
        """Get teacher logprobs for batch of prompts"""
        outputs = []

        for prompt in prompts:
            teacher_output = self.teacher.get_logprobs(
                prompt,
                max_tokens=self.config.max_generation_tokens
            )
            outputs.append(teacher_output)

        # Convert to batch tensors
        # Pad sequences to same length
        # Return unified dict

    def _get_student_outputs(
        self,
        prompts: List[str],
        teacher_outputs: Dict
    ) -> Dict:
        """Run student forward pass on teacher's token sequences"""
        # Important: Run student on same token sequence as teacher
        # This ensures aligned logit comparison

        student_outputs = self.student.forward(
            prompts,
            teacher_token_ids=[t['token_ids'] for t in teacher_outputs]
        )

        return student_outputs

    def evaluate(self) -> Dict:
        """Evaluate on held-out validation set"""
        # Sample validation prompts (different from training)
        # Run teacher and student
        # Compute metrics without gradients
        # Return validation metrics

    def save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        """Save adapter checkpoint"""
        # Save LoRA adapters
        # Save optimizer state
        # Save config and metrics

    def _log_metrics(self, step: int, metrics: Dict, prefix: str = 'train'):
        """Log metrics to file and console"""

    def _broadcast_progress(self, step: int, metrics: Dict):
        """Broadcast progress via WebSocket"""

    def _sample_batch(self, batch_size: int) -> List[str]:
        """Sample random batch of prompts"""
```

---

### 6. Data Loader (`backend/opd/data_loader.py`)

**Purpose**: Load and prepare validation prompts

**Functions**:
```python
def load_prompts(
    prompts_path: str,
    max_prompts: int = 1000,
    split_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    """
    Load prompts from JSONL file and split into train/val.

    Args:
        prompts_path: Path to JSONL file
        max_prompts: Maximum number of prompts to load
        split_ratio: Train/val split (0.8 = 80% train, 20% val)

    Returns:
        (train_prompts, val_prompts)
    """

def create_batches(
    prompts: List[str],
    batch_size: int,
    shuffle: bool = True
) -> List[List[str]]:
    """Create batched prompts"""

def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file"""

def save_jsonl(data: List[Dict], path: str):
    """Save to JSONL file"""
```

---

## Implementation Phases

### Phase 0: Setup & Preparation (2-3 hours)

**Goals**:
- Create directory structure
- Set up configuration system
- Test model loading

**Tasks**:
1. ✅ Create `OnPolicyDistill/` directory tree
2. ✅ Create `backend/opd/` module structure
3. ✅ Implement `OPDConfig` dataclass
4. ✅ Test loading Qwen 32B model (verify memory usage)
5. ✅ Test loading Qwen 7B + user's fine-tuned adapters
6. ✅ Document memory requirements

**Deliverables**:
- Directory structure created
- Config module complete
- Memory profiling report

---

### Phase 1: Core Components (12-16 hours)

#### Task 1.1: Teacher Model (4-5 hours)
- ✅ Implement `TeacherModel` class
- ✅ Implement logprob extraction from MLX model
- ✅ Implement caching system (SHA256 keys, NPZ format)
- ✅ Test: Load 32B, generate completion, extract logprobs
- ✅ Test: Cache hit/miss behavior

#### Task 1.2: Student Model (3-4 hours)
- ✅ Implement `StudentModel` class
- ✅ Load base + LoRA with gradient tracking
- ✅ Implement forward pass with logits output
- ✅ Test: Forward pass produces correct shape logits
- ✅ Test: Gradients flow only through LoRA

#### Task 1.3: Distillation Loss (3-4 hours)
- ✅ Implement `DistillationLoss` class
- ✅ Temperature scaling
- ✅ Reverse KL divergence calculation
- ✅ Metrics computation
- ✅ Test: Loss backward produces gradients
- ✅ Test: Validate against reference KL implementation

#### Task 1.4: Data Loader (2-3 hours)
- ✅ Implement prompt loading from JSONL
- ✅ Train/val split
- ✅ Batch creation
- ✅ Test: Load user's validation data

**Deliverables**:
- All core components implemented
- Unit tests passing
- Integration test: teacher → student → loss pipeline

---

### Phase 2: Training Loop (10-14 hours)

#### Task 2.1: Distillation Trainer (6-8 hours)
- ✅ Implement `DistillationTrainer` class
- ✅ Training loop with gradient accumulation
- ✅ Checkpoint saving/loading
- ✅ Metrics logging to JSONL
- ✅ Test: Run 10 training steps on small data

#### Task 2.2: Evaluation (2-3 hours)
- ✅ Implement validation evaluation
- ✅ Compute validation KL loss
- ✅ Best model tracking
- ✅ Test: Evaluation produces correct metrics

#### Task 2.3: Optimizer Integration (2-3 hours)
- ✅ Set up Adam optimizer for LoRA
- ✅ Learning rate scheduling (optional)
- ✅ Gradient clipping
- ✅ Test: Optimizer updates weights correctly

**Deliverables**:
- Complete training pipeline
- End-to-end test: 50 steps on real data
- Checkpoints saved correctly
- Metrics logged

---

### Phase 3: Backend API (6-8 hours)

#### Task 3.1: FastAPI Endpoints (4-5 hours)
- ✅ Add `/opd/start` endpoint
- ✅ Add `/opd/status` endpoint
- ✅ Add `/opd/stop` endpoint
- ✅ Add `/opd/metrics` endpoint
- ✅ Subprocess management for training

#### Task 3.2: WebSocket Integration (2-3 hours)
- ✅ Broadcast training progress
- ✅ Real-time metrics updates
- ✅ Error handling and reporting

**Deliverables**:
- Backend API complete
- Postman/curl tests pass
- WebSocket events firing correctly

---

### Phase 4: Frontend Implementation (12-16 hours)

#### Task 4.1: Redux Slice (2-3 hours)
- ✅ Create `opdSlice.ts`
- ✅ Define state interface
- ✅ Actions: updateConfig, updateMetrics, setStatus
- ✅ Integrate with store

#### Task 4.2: OPD Page (3-4 hours)
- ✅ Create `OPDPage.tsx`
- ✅ Layout: setup, progress, results views
- ✅ State-based view switching

#### Task 4.3: Setup Component (3-4 hours)
- ✅ Teacher model selector (detect Qwen 32B)
- ✅ Student adapter selector (list available adapters)
- ✅ Validation prompts file picker
- ✅ Configuration form (temperature, steps, etc.)
- ✅ "Start Distillation" button

#### Task 4.4: Progress Monitor (3-4 hours)
- ✅ Real-time metrics display
- ✅ KL loss chart (Chart.js)
- ✅ Token agreement chart
- ✅ Progress bar (steps / total_steps)
- ✅ Log viewer

#### Task 4.5: Results Display (1-2 hours)
- ✅ Final metrics summary
- ✅ Best checkpoint info
- ✅ Download distilled adapter button
- ✅ "Test Model" link to Compare page

**Deliverables**:
- Complete OPD UI
- Integrated with backend
- Real-time updates working

---

### Phase 5: Integration & Polish (6-8 hours)

#### Task 5.1: Navigation Integration (1-2 hours)
- ✅ Add "Distillation" to sidebar
- ✅ Update routing in App.tsx
- ✅ Post-SFT suggestion to run distillation

#### Task 5.2: Session Management (2-3 hours)
- ✅ Link distillation runs to SFT sessions
- ✅ Save distillation config and metrics to session JSON
- ✅ Load previous runs

#### Task 5.3: Compare Page Extension (2-3 hours)
- ✅ Add distilled model to comparison dropdown
- ✅ 3-way compare: Base vs SFT vs Distilled
- ✅ Display model size/memory usage

**Deliverables**:
- Seamless workflow: Setup → SFT → Distillation → Compare
- Session persistence working
- Multi-model comparison

---

### Phase 6: Testing & Validation (8-12 hours)

#### Task 6.1: Unit Tests (3-4 hours)
- ✅ Test teacher model caching
- ✅ Test student model gradient flow
- ✅ Test loss computation
- ✅ Test data loading

#### Task 6.2: Integration Tests (3-4 hours)
- ✅ End-to-end: API call → training → checkpoint
- ✅ Resume from checkpoint
- ✅ Cache hit behavior
- ✅ Memory usage validation

#### Task 6.3: Manual Testing (2-4 hours)
- ✅ Run full distillation on user's dataset
- ✅ Verify KL loss decreases
- ✅ Test distilled model quality (compare outputs)
- ✅ Measure inference speedup (32B vs 7B distilled)

**Deliverables**:
- All tests passing
- Quality validation report
- Performance benchmarks

---

## Configuration

### Example Configuration File

```yaml
# OnPolicyDistill/configs/distill_20250128_143022.yaml

run_id: distill_20250128_143022
session_id: sft_session_abc123  # Link to SFT session

# === MODELS ===
base_model_path: /Users/macbook2024/Dropbox/mlx/base_model/qwen2.5-7b
teacher_model_path: /Users/macbook2024/Dropbox/mlx/base_model/qwen2.5-32b
student_adapter_path: /Users/macbook2024/Dropbox/mlx/lora_adapters/my_finetuned_adapter
output_adapter_path: /Users/macbook2024/Dropbox/mlx/OnPolicyDistill/checkpoints/distill_20250128_143022

# === DATA ===
validation_prompts_path: /Users/macbook2024/Dropbox/mlx/data/validation_prompts.jsonl
max_prompts: 1000

# === TRAINING ===
num_steps: 1000
batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.00001

# === DISTILLATION ===
temperature: 2.0
kl_weight: 0.8
ce_weight: 0.2
use_teacher_targets: true

# === GENERATION ===
max_generation_tokens: 512
generation_temperature: 1.0

# === CHECKPOINTING ===
checkpoint_every: 100
eval_every: 100
save_best_only: false

# === CACHE ===
use_cache: true
cache_dir: ./OnPolicyDistill/teacher_cache

# === SYSTEM ===
seed: 42
mixed_precision: true
```

### Configuration Presets

**Fast Iteration** (for testing):
```yaml
num_steps: 100
batch_size: 1
checkpoint_every: 25
eval_every: 25
max_prompts: 50
```

**High Quality** (production):
```yaml
num_steps: 2000
batch_size: 2
gradient_accumulation_steps: 8
checkpoint_every: 200
eval_every: 100
temperature: 3.0  # Higher temp for more knowledge transfer
```

**Memory Efficient**:
```yaml
batch_size: 1
gradient_accumulation_steps: 8
max_prompts: 500
mixed_precision: true
use_cache: true  # Critical - avoids recomputing teacher
```

---

## API Design

### POST /opd/start

**Request**:
```json
{
  "base_model_path": "/path/to/qwen2.5-7b",
  "teacher_model_path": "/path/to/qwen2.5-32b",
  "student_adapter_path": "/path/to/sft_adapter",
  "validation_prompts_path": "/path/to/val_prompts.jsonl",
  "num_steps": 1000,
  "batch_size": 2,
  "temperature": 2.0,
  "kl_weight": 0.8,
  "learning_rate": 0.00001
}
```

**Response**:
```json
{
  "status": "success",
  "run_id": "distill_20250128_143022",
  "message": "Distillation training started",
  "estimated_duration_minutes": 45,
  "memory_required_gb": 48
}
```

---

### GET /opd/status

**Response**:
```json
{
  "state": "running",
  "run_id": "distill_20250128_143022",
  "metrics": {
    "step": 450,
    "total_steps": 1000,
    "progress_pct": 45.0,
    "kl_loss": 0.234,
    "token_agreement_pct": 78.5,
    "student_entropy": 3.42,
    "teacher_entropy": 3.89,
    "tokens_per_second": 125.3
  },
  "estimated_time_remaining_seconds": 1350,
  "current_checkpoint": "/path/to/checkpoint/step_000400.safetensors"
}
```

---

### POST /opd/stop

**Response**:
```json
{
  "status": "stopped",
  "final_step": 450,
  "checkpoint_path": "/path/to/checkpoint/step_000450.safetensors",
  "message": "Training stopped by user"
}
```

---

### GET /opd/metrics?run_id={run_id}

**Response**:
```json
{
  "run_id": "distill_20250128_143022",
  "total_steps": 1000,
  "metrics_history": [
    {
      "step": 0,
      "kl_loss": 1.234,
      "token_agreement_pct": 45.2,
      "student_entropy": 4.12,
      "timestamp": "2025-01-28T14:30:22Z"
    },
    {
      "step": 100,
      "kl_loss": 0.876,
      "token_agreement_pct": 62.1,
      "student_entropy": 3.85,
      "timestamp": "2025-01-28T14:35:15Z"
    },
    ...
  ],
  "best_checkpoint": {
    "step": 800,
    "val_kl_loss": 0.189,
    "path": "/path/to/best_adapters.safetensors"
  }
}
```

---

### GET /opd/runs

**Response**:
```json
{
  "runs": [
    {
      "run_id": "distill_20250128_143022",
      "status": "completed",
      "started_at": "2025-01-28T14:30:22Z",
      "completed_at": "2025-01-28T15:45:10Z",
      "final_kl_loss": 0.189,
      "teacher_model": "qwen2.5-32b",
      "student_adapter": "my_finetuned_adapter"
    },
    ...
  ]
}
```

---

## UI Design

### OPD Page Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  Navigation: [Setup] [Training] [Results] [→ Distillation] [Compare] │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  KNOWLEDGE DISTILLATION                                          │
│  Compress your fine-tuned model using a larger teacher          │
└──────────────────────────────────────────────────────────────────┘

[When idle - Setup View]
┌──────────────────────────────────────────────────────────────────┐
│  Model Configuration                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Teacher Model (32B)                                       │ │
│  │  [Dropdown: Qwen2.5-32B ▼]                                │ │
│  │  Memory Required: ~40GB                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Student Model (7B + LoRA)                                 │ │
│  │  [Dropdown: my_finetuned_adapter ▼]                       │ │
│  │  From SFT session: Jan 28, 2025 2:30 PM                   │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Data Configuration                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Validation Prompts                                        │ │
│  │  [📁 Select File...]                                       │ │
│  │  Selected: validation_data.jsonl (500 prompts)            │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Training Parameters                                             │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐ │
│  │ Steps       │ Batch Size  │ Temperature │ Learning Rate   │ │
│  │ [1000  ▼]  │ [2     ▼]  │ [2.0   ▼]  │ [0.00001   ▼]  │ │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘ │
│                                                                  │
│  Advanced:                                                       │
│  ☑ Use teacher cache (recommended)                              │
│  ☑ Mixed precision training                                     │
│  ☐ Save all checkpoints (default: best only)                    │
└──────────────────────────────────────────────────────────────────┘

                    [Start Distillation Training]


[When running - Progress View]
┌──────────────────────────────────────────────────────────────────┐
│  Training Progress                                               │
│  ═════════════════════════════════════════════ 45%              │
│  Step 450 / 1000  •  ~22 minutes remaining                      │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Live Metrics                                                    │
│  ┌─────────────────────┬─────────────────────┬────────────────┐ │
│  │  KL Loss            │  Token Agreement    │  Entropy       │ │
│  │  0.234              │  78.5%              │  3.42          │ │
│  │  ↓ -0.012           │  ↑ +2.3%            │  ↓ -0.08       │ │
│  └─────────────────────┴─────────────────────┴────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  KL Divergence Over Time                                         │
│                                                                  │
│  1.5┤                                                            │
│     │●                                                           │
│  1.0┤ ●                                                          │
│     │  ●●                                                        │
│  0.5┤    ●●                                                      │
│     │      ●●●                                                   │
│  0.0┤         ●●●●●●●●●●●●●●●●●●●────                           │
│     └──────────────────────────────────────                     │
│     0        250        500        750       1000                │
│                         Steps                                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Teacher Cache Stats                                             │
│  Cache Hits: 423 / 450 (94%)  •  Time Saved: ~18 minutes        │
└──────────────────────────────────────────────────────────────────┘

                          [Stop Training]


[When complete - Results View]
┌──────────────────────────────────────────────────────────────────┐
│  ✓ Distillation Complete!                                        │
│  Duration: 42 minutes  •  Final KL Loss: 0.189                   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Final Metrics                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  KL Loss:               0.189  (↓ 84% from start)           ││
│  │  Token Agreement:       89.2%  (↑ 44% from start)           ││
│  │  Student Entropy:       3.15   (closer to teacher: 3.42)    ││
│  │  Best Checkpoint:       Step 800                             ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Distilled Model                                                 │
│  Location: /OnPolicyDistill/checkpoints/.../best_adapters.safetensors │
│                                                                  │
│  [Download Adapter]  [Test in Compare Page]  [View Metrics]     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Improvement Summary                                             │
│  • Model size: 7B parameters (vs 32B teacher)                   │
│  • Inference speed: ~4.2x faster than teacher                   │
│  • Quality: 89% agreement with teacher outputs                  │
│  • Ready for deployment: ✓                                       │
└──────────────────────────────────────────────────────────────────┘

                      [Run Another Distillation]
```

---

## Testing Strategy

### Unit Tests

**Backend Tests** (`backend/tests/opd/`):

```
test_teacher_model.py
  ✓ test_load_qwen_32b()
  ✓ test_get_logprobs_shape()
  ✓ test_cache_hit()
  ✓ test_cache_miss()
  ✓ test_cache_key_deterministic()

test_student_model.py
  ✓ test_load_with_lora()
  ✓ test_forward_pass_shape()
  ✓ test_gradients_only_lora()
  ✓ test_base_model_frozen()

test_distillation_loss.py
  ✓ test_kl_divergence_computation()
  ✓ test_temperature_scaling()
  ✓ test_token_agreement_metric()
  ✓ test_entropy_calculation()
  ✓ test_loss_backward()

test_data_loader.py
  ✓ test_load_jsonl_prompts()
  ✓ test_train_val_split()
  ✓ test_batch_creation()

test_trainer.py
  ✓ test_trainer_initialization()
  ✓ test_training_step()
  ✓ test_checkpoint_save_load()
  ✓ test_evaluation()
```

**Frontend Tests** (`frontend/src/__tests__/opd/`):

```
DistillationSetup.test.tsx
  ✓ test_teacher_model_selector()
  ✓ test_student_adapter_selector()
  ✓ test_validation_file_picker()
  ✓ test_start_button_validation()

DistillationProgress.test.tsx
  ✓ test_metrics_display()
  ✓ test_progress_bar_update()
  ✓ test_chart_rendering()

opdSlice.test.ts
  ✓ test_updateMetrics_action()
  ✓ test_setStatus_action()
  ✓ test_updateConfig_action()
```

---

### Integration Tests

```
test_end_to_end_distillation.py
  Setup:
    - Create small test dataset (10 prompts)
    - Use small models (for speed)

  Test:
    1. Load teacher and student models
    2. Run 10 training steps
    3. Verify:
       - KL loss decreases
       - Checkpoint saved
       - Metrics logged
       - Cache used correctly
    4. Load checkpoint and resume
    5. Run 5 more steps
    6. Verify training continues correctly

test_api_integration.py
  Test:
    1. POST /opd/start → returns run_id
    2. GET /opd/status → returns running state
    3. Wait for completion
    4. GET /opd/metrics → returns full history
    5. Verify checkpoint files exist
```

---

### Manual Testing Checklist

#### Pre-Testing Setup
- [ ] Ensure Qwen 32B model is downloaded
- [ ] Have fine-tuned 7B adapter from previous SFT
- [ ] Prepare validation prompts file (100-500 prompts)
- [ ] Free up ~48GB RAM

#### Test 1: Quick Distillation Run (30 minutes)
- [ ] Start OPD with config: 100 steps, batch_size=2
- [ ] Verify teacher cache creates entries
- [ ] Monitor memory usage (should stay < 48GB)
- [ ] Check KL loss decreases over steps
- [ ] Verify checkpoint saved at step 100

#### Test 2: Cache Effectiveness
- [ ] Run distillation for 50 steps
- [ ] Stop and restart with same prompts
- [ ] Verify cache hit rate > 90%
- [ ] Compare runtime: should be ~50% faster

#### Test 3: Quality Validation
- [ ] Complete full distillation (1000 steps)
- [ ] Load distilled adapter
- [ ] Test on 20 held-out prompts
- [ ] Compare outputs:
   - Base model (7B no adapters)
   - SFT model (7B + SFT adapter)
   - Teacher model (32B)
   - Distilled model (7B + distilled adapter)
- [ ] Verify: Distilled closer to Teacher than SFT

#### Test 4: UI Integration
- [ ] Navigate: Setup → SFT → Complete → [Suggested] Run Distillation
- [ ] Select models via dropdowns
- [ ] Start training, monitor live updates
- [ ] Charts update in real-time
- [ ] Stop and resume training
- [ ] View results page
- [ ] Test distilled model in Compare page

---

## Timeline

### Week 1: Core Implementation

**Days 1-2**: Setup + Core Components (Phase 0 + Phase 1)
- Create directory structure
- Implement TeacherModel, StudentModel, DistillationLoss
- Write unit tests
- **Milestone**: Core pipeline working (teacher → student → loss)

**Days 3-4**: Training Loop (Phase 2)
- Implement DistillationTrainer
- Add checkpoint management
- Add evaluation logic
- **Milestone**: Can run 100 training steps end-to-end

**Day 5**: Backend API (Phase 3)
- Add FastAPI endpoints
- Subprocess management
- WebSocket integration
- **Milestone**: Can trigger training via API

---

### Week 2: Frontend + Integration

**Days 1-2**: Frontend Core (Phase 4.1-4.3)
- Redux slice
- OPD page layout
- Setup component
- **Milestone**: UI can configure and start training

**Days 3-4**: Frontend Polish (Phase 4.4-4.5)
- Progress monitor with charts
- Results display
- Real-time updates
- **Milestone**: Full UI working

**Day 5**: Integration (Phase 5)
- Connect all pieces
- Session management
- Compare page extension
- **Milestone**: Complete workflow functional

---

### Week 3: Testing + Launch

**Days 1-2**: Testing (Phase 6)
- Write all tests
- Run integration tests
- Manual QA
- **Milestone**: All tests passing

**Days 3-4**: Optimization + Documentation
- Performance tuning
- Memory optimization
- User documentation
- **Milestone**: Production-ready

**Day 5**: Launch
- Final testing on real datasets
- Demo video
- Release notes
- **Milestone**: Launch to users

---

**Total Timeline**: 3 weeks (15 working days)

---

## Success Criteria

### Technical Success
✅ **Distillation Works**: KL loss decreases over training
✅ **Quality Improvement**: Distilled model outputs closer to teacher than SFT alone
✅ **Token Agreement**: >85% agreement between student and teacher by end of training
✅ **Stable Training**: No OOM errors, training completes successfully
✅ **Cache Effective**: >90% cache hit rate on repeated runs
✅ **Checkpoints Valid**: Can load and resume from checkpoints

### Performance Success
✅ **Memory Usage**: Peak RAM < 48GB on 64GB system
✅ **Speed**: 1000 steps completes in < 1 hour
✅ **Inference Speedup**: Distilled model 3-4x faster than teacher
✅ **Model Size**: Distilled adapters < 100MB

### User Experience Success
✅ **Easy Setup**: Can configure distillation in < 2 minutes
✅ **Clear Progress**: Real-time metrics update every 5 seconds
✅ **Helpful UI**: Charts show clear training progress
✅ **Reliable**: Training completes without errors
✅ **Integrated**: Seamless flow from SFT → Distillation → Compare

### Quality Benchmarks
✅ **Output Similarity**: Distilled model outputs semantically similar to teacher
✅ **Task Performance**: Distilled model maintains >95% of teacher's task accuracy
✅ **Reduced Size**: 7B distilled model performs close to 32B teacher
✅ **Deployment Ready**: Distilled model suitable for production use

---

## Open Questions & Decisions Needed

### Critical Questions

1. **MLX Logprob Extraction**
   - **Question**: Does MLX's `generate()` expose per-token logprobs?
   - **Action**: Test with current MLX version, document approach
   - **Fallback**: Implement manual generation loop with forward passes

2. **Memory Profiling**
   - **Question**: What's actual peak memory with 32B teacher + 7B student?
   - **Action**: Run memory profiler on user's system
   - **Decision**: May need to unload teacher between batches

3. **Validation Data Source**
   - **Question**: Where do validation prompts come from?
   - **Options**:
     - A. Auto-split user's SFT training data (80/20)
     - B. Require separate validation file
     - C. Generate synthetic prompts
   - **Recommendation**: Option A (auto-split) for ease of use

4. **Temperature Tuning**
   - **Question**: What's optimal temperature for Qwen distillation?
   - **Action**: Run grid search (T ∈ {1.0, 2.0, 3.0, 4.0})
   - **Default**: Start with T=2.0 (standard practice)

---

### Implementation Details

5. **Batch Processing Strategy**
   - **Question**: Process teacher in batch or sequential?
   - **Trade-off**: Batch faster but more memory
   - **Decision**: Start sequential, optimize to batch if memory allows

6. **Cache Format**
   - **Question**: Use NPZ (NumPy) or SafeTensors for cache?
   - **Recommendation**: NPZ for simplicity, easy to load/save

7. **LoRA Rank**
   - **Question**: Does distillation work better with higher rank LoRA?
   - **Action**: Test with user's existing LoRA config first
   - **Note**: May need to increase rank for better capacity

8. **Evaluation Metric**
   - **Question**: Primary metric for "best model" selection?
   - **Options**: Validation KL loss, token agreement, task accuracy
   - **Recommendation**: Validation KL loss (most direct)

---

### User Experience

9. **Progress Estimation**
   - **Question**: How to estimate time remaining accurately?
   - **Approach**: Track avg time per step after first 10 steps
   - **Note**: Account for cache hits speeding up later steps

10. **Error Recovery**
    - **Question**: What if training OOMs mid-run?
    - **Strategy**: Save emergency checkpoint, reduce batch size, resume
    - **UI**: Show "Memory pressure high" warning proactively

---

## Risk Mitigation

### Risk 1: Out of Memory
**Likelihood**: Medium
**Impact**: High
**Mitigation**:
- Monitor memory usage during first 10 steps
- Auto-reduce batch size if approaching limit
- Option to unload teacher between batches
- Clear user guidance on memory requirements

### Risk 2: Slow Training
**Likelihood**: Medium
**Impact**: Medium
**Mitigation**:
- Implement teacher caching (critical!)
- Use mixed precision training
- Optimize batch processing
- Provide fast-iteration preset (100 steps)

### Risk 3: Poor Distillation Quality
**Likelihood**: Low
**Impact**: High
**Mitigation**:
- Start with proven temperature (T=2.0)
- Validate with token agreement metric
- Allow temperature tuning in UI
- Document expected quality ranges

### Risk 4: MLX API Limitations
**Likelihood**: Low
**Impact**: Medium
**Mitigation**:
- Test logprob extraction early (Phase 0)
- Have manual generation fallback ready
- Engage with MLX community if needed

---

## Next Steps

1. ✅ **Review this plan** - Approve approach and timeline
2. ✅ **Verify Qwen 32B availability** - Confirm model location and can load
3. ✅ **Test memory requirements** - Load 32B + 7B simultaneously, measure RAM
4. ✅ **Clarify open questions** - Especially validation data source
5. ✅ **Begin Phase 0** - Set up directories, test model loading
6. 🚀 **Start implementation** - Follow phase-by-phase plan

---

## Summary

This plan provides a **complete, production-ready implementation** of knowledge distillation for Droid-FineTuning using local Qwen models.

**Key Advantages**:
- ✅ Uses your existing Qwen 32B model
- ✅ 100% local, no API costs
- ✅ Proven distillation technique
- ✅ Clear quality metrics
- ✅ Fits seamlessly into existing workflow
- ✅ 3-week implementation timeline

**Expected Outcome**:
A compact 7B model with LoRA adapters that performs close to the 32B teacher, suitable for fast deployment while maintaining high quality.

**Ready to build!** 🚀

---

**Document Version**: 3.0 FINAL
**Date**: January 28, 2025
**Status**: AWAITING APPROVAL
**Focus**: Knowledge Distillation with Qwen 32B → 7B
