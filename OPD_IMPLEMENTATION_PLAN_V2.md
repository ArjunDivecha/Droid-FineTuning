# On-Policy Distillation (OPD) Implementation Plan v2.0
## Local-Only Solution for Droid-FineTuning

**Status**: REVISED - Removes Claude API dependency, uses local models only

**Key Change**: Replaced Claude Sonnet 4.5 API with **local teacher models** and **self-improvement** approaches.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Revised Approach](#revised-approach)
3. [Implementation Strategy](#implementation-strategy)
4. [Detailed Design](#detailed-design)
5. [Step-by-Step Implementation](#step-by-step-implementation)
6. [Testing & Validation](#testing--validation)
7. [Timeline & Rollout](#timeline--rollout)

---

## Architecture Overview

### Previous Problem (v1.0)
- ❌ Relied on Claude API for teacher logprobs
- ❌ API doesn't expose per-token logprobs
- ❌ High costs for iterative training
- ❌ Network latency bottleneck

### New Solution (v2.0)
- ✅ **100% local** - All models run via MLX
- ✅ **Two-tier system** - Adapts to user's hardware
- ✅ **Zero API costs** - No external dependencies
- ✅ **Fast iteration** - No network calls

---

## Revised Approach

### Two Operating Modes

#### **Mode 1: Knowledge Distillation** (for high-memory systems)

**Requirements**: 64GB+ unified memory

**Architecture**:
```
┌─────────────────────────────────────────────────┐
│  TEACHER MODEL (Larger)                         │
│  - Qwen2.5-32B / Qwen2.5-72B                   │
│  - Frozen (inference only)                      │
│  - Provides token-level logprobs                │
└─────────────────────────────────────────────────┘
                    ↓ KL divergence loss
┌─────────────────────────────────────────────────┐
│  STUDENT MODEL (Smaller)                        │
│  - Qwen2.5-7B / Qwen2.5-3B                     │
│  - LoRA adapters (trainable)                    │
│  - Learns to match teacher distribution         │
└─────────────────────────────────────────────────┘
```

**Process**:
1. Load large teacher model (frozen)
2. Load smaller student model (with LoRA)
3. Generate prompts from validation set
4. Get teacher logprobs for each prompt
5. Train student to minimize KL(student || teacher)
6. Result: Compact model with large model's knowledge

**Benefits**:
- True knowledge distillation
- Compress 32B knowledge into 7B model
- Faster inference after distillation
- Better quality than student alone

---

#### **Mode 2: Self-Improvement** (default, works on all systems)

**Requirements**: 16GB+ unified memory

**Architecture**:
```
┌─────────────────────────────────────────────────┐
│  FINE-TUNED MODEL (from SFT)                    │
│  - User's trained adapter                       │
│  - Generates N candidate responses              │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  QUALITY SCORER (Local)                         │
│  - Task completion check                        │
│  - Coherence (perplexity)                       │
│  - Length appropriateness                       │
│  - Diversity                                    │
└─────────────────────────────────────────────────┘
                    ↓ Filter top-k%
┌─────────────────────────────────────────────────┐
│  BEST EXAMPLES (High Quality Only)              │
│  - Top 20-30% of generations                    │
│  - Used as new training data                    │
└─────────────────────────────────────────────────┘
                    ↓ Fine-tune
┌─────────────────────────────────────────────────┐
│  REFINED MODEL                                   │
│  - Same model, improved via self-selection      │
│  - Iteratively gets better                      │
└─────────────────────────────────────────────────┘
```

**Process** (Best-of-N Sampling):
1. Start with fine-tuned model from SFT
2. For each validation prompt:
   - Generate N responses (e.g., N=5)
   - Score each response with quality metrics
   - Keep only best response
3. Create new training dataset from best responses
4. Fine-tune model on filtered data
5. Repeat for K iterations
6. Result: Model refined via self-selection

**Benefits**:
- Works on standard Apple Silicon (M1/M2 16GB)
- No extra memory for teacher model
- Proven approach (used in RLHF, Constitutional AI)
- Continuous improvement possible

---

## Quality Scoring System (for Mode 2)

### Automatic Quality Metrics

#### 1. **Task Completion Score**
```python
def task_completion_score(response: str, task_type: str) -> float:
    """
    Check if response follows expected format/structure

    Examples:
    - QA: Has direct answer
    - Code: Contains code block
    - Summary: Within length bounds
    """
    if task_type == "qa":
        # Check for answer indicators
        has_answer = any(marker in response.lower()
                        for marker in ["answer:", "the answer is", "="])
        return 1.0 if has_answer else 0.3

    elif task_type == "code":
        # Check for code block
        has_code = "```" in response or "def " in response
        return 1.0 if has_code else 0.2

    # Generic: check not empty
    return 0.5 if len(response.strip()) > 10 else 0.0
```

#### 2. **Coherence Score** (Perplexity-based)
```python
def coherence_score(model, tokenizer, response: str) -> float:
    """
    Lower perplexity = more coherent
    """
    tokens = tokenizer.encode(response)
    logprobs = model.get_logprobs(tokens)
    perplexity = mx.exp(-mx.mean(logprobs))

    # Normalize to 0-1 scale
    # Good: perplexity < 20, Bad: perplexity > 100
    normalized = 1.0 / (1.0 + perplexity / 20.0)
    return float(normalized)
```

#### 3. **Length Appropriateness**
```python
def length_score(response: str, target_range: tuple) -> float:
    """
    Penalize too short or too long responses
    """
    min_len, max_len = target_range
    length = len(response.split())

    if min_len <= length <= max_len:
        return 1.0
    elif length < min_len:
        return length / min_len  # Partial credit
    else:
        return max(0.5, 1.0 - (length - max_len) / max_len)
```

#### 4. **Diversity Score**
```python
def diversity_score(response: str) -> float:
    """
    Penalize repetitive responses
    """
    words = response.split()
    unique_ratio = len(set(words)) / max(len(words), 1)

    # Check for repeated phrases
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_ratio = len(set(bigrams)) / max(len(bigrams), 1)

    return (unique_ratio + bigram_ratio) / 2.0
```

#### 5. **Composite Quality Score**
```python
def overall_quality(response: str, config: dict, model=None) -> float:
    """
    Weighted combination of all metrics
    """
    scores = {
        'task_completion': task_completion_score(response, config['task_type']),
        'coherence': coherence_score(model, tokenizer, response) if model else 0.5,
        'length': length_score(response, config['length_range']),
        'diversity': diversity_score(response)
    }

    weights = config.get('score_weights', {
        'task_completion': 0.4,
        'coherence': 0.3,
        'length': 0.2,
        'diversity': 0.1
    })

    total = sum(scores[k] * weights[k] for k in scores)
    return total
```

---

## Implementation Strategy

### Architecture Decision Tree

```
User starts OPD
    ↓
Check system memory
    ├─> ≥64GB RAM
    │   ↓
    │   Ask: "Load larger teacher model?"
    │   ├─> Yes: Mode 1 (Knowledge Distillation)
    │   └─> No: Mode 2 (Self-Improvement)
    │
    └─> <64GB RAM
        ↓
        Mode 2 (Self-Improvement) only
```

### Directory Structure

```
OnPolicyDistill/
├── configs/
│   └── {run_id}.yaml                    # Run configuration
├── mode1_distillation/                   # Knowledge distillation mode
│   ├── teacher_cache/
│   │   └── {prompt_hash}_logprobs.npz   # Cached teacher outputs
│   ├── rollouts/
│   │   └── {run_id}_rollouts.jsonl      # Student samples
│   └── checkpoints/
│       └── {run_id}/
│           ├── step_000100.safetensors
│           └── best.safetensors
├── mode2_self_improvement/               # Self-improvement mode
│   ├── generations/
│   │   └── {run_id}_candidates.jsonl    # N samples per prompt
│   ├── filtered/
│   │   └── {run_id}_best.jsonl          # Top-k filtered
│   ├── scores/
│   │   └── {run_id}_scores.jsonl        # Quality scores
│   └── checkpoints/
│       └── {run_id}/
│           ├── iteration_001.safetensors
│           └── iteration_002.safetensors
└── metrics/
    └── {run_id}_metrics.jsonl            # Training metrics
```

---

## Detailed Design

### Mode 1: Knowledge Distillation

#### Components

**1. Teacher Model Loader** (`backend/opd/teacher_model.py`)
```python
class TeacherModel:
    """Manages large local teacher model"""

    def __init__(self, model_path: str, cache_dir: str):
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load teacher model (frozen)"""
        self.model, self.tokenizer = load(self.model_path)
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def get_logprobs(self, prompt: str, max_tokens: int = 512) -> Dict:
        """
        Generate and return token-level logprobs

        Returns:
            {
                'tokens': List[str],
                'token_ids': List[int],
                'logprobs': np.ndarray,  # (seq_len, vocab_size)
                'text': str
            }
        """
        # Check cache first
        cache_key = self._get_cache_key(prompt)
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        # Generate with teacher
        output = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=1.0
        )

        # Extract logprobs (requires MLX modification or workaround)
        # For now, we'll compute them manually
        tokens = self.tokenizer.encode(output)
        logprobs = self._compute_logprobs(tokens)

        result = {
            'tokens': [self.tokenizer.decode([t]) for t in tokens],
            'token_ids': tokens,
            'logprobs': logprobs,
            'text': output
        }

        # Save to cache
        self._save_cache(cache_key, result)

        return result

    def _compute_logprobs(self, token_ids: List[int]) -> np.ndarray:
        """Compute logprobs for sequence"""
        # Run forward pass through model
        # Return logprobs for each position
        # This requires access to model's internal states
        pass  # Implementation depends on MLX API
```

**2. Distillation Loss** (`backend/opd/distillation_loss.py`)
```python
class DistillationLoss:
    """Compute KL divergence loss for distillation"""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def compute_kl_loss(
        self,
        student_logits: mx.array,  # (batch, seq_len, vocab)
        teacher_logits: mx.array,  # (batch, seq_len, vocab)
        mask: mx.array = None
    ) -> Tuple[mx.array, Dict]:
        """
        KL divergence: D_KL(Teacher || Student)

        Loss = sum_t [ Teacher(t) * log(Teacher(t) / Student(t)) ]
        """
        # Apply temperature scaling
        student_probs = mx.softmax(student_logits / self.temperature, axis=-1)
        teacher_probs = mx.softmax(teacher_logits / self.temperature, axis=-1)

        # KL divergence
        kl = teacher_probs * (mx.log(teacher_probs) - mx.log(student_probs))
        kl = mx.sum(kl, axis=-1)  # Sum over vocab

        # Apply mask
        if mask is not None:
            kl = kl * mask
            num_tokens = mx.sum(mask)
        else:
            num_tokens = kl.size

        # Mean over sequence
        loss = mx.sum(kl) / num_tokens

        # Scale back by temperature^2 (standard practice)
        loss = loss * (self.temperature ** 2)

        metrics = {
            'kl_loss': float(loss),
            'kl_mean': float(mx.mean(kl)),
            'kl_max': float(mx.max(kl))
        }

        return loss, metrics
```

**3. Distillation Trainer** (`backend/opd/distillation_trainer.py`)
```python
class DistillationTrainer:
    """Orchestrates knowledge distillation training"""

    def __init__(self, config: OPDConfig):
        self.config = config
        self.teacher = TeacherModel(config.teacher_model_path)
        self.student = StudentModel(
            config.base_model_path,
            config.student_adapter_path
        )
        self.loss_fn = DistillationLoss(temperature=config.temperature)
        self.optimizer = create_optimizer(config)

    def train(self):
        """Main distillation loop"""
        # Load models
        self.teacher.load()
        self.student.load()

        # Load validation prompts
        prompts = self.load_prompts(self.config.validation_prompts_path)

        for step in range(self.config.num_steps):
            # Sample batch of prompts
            batch = random.sample(prompts, self.config.batch_size)

            # Get teacher outputs (cached if available)
            teacher_outputs = [
                self.teacher.get_logprobs(prompt)
                for prompt in batch
            ]

            # Get student outputs
            student_outputs = self.student.forward(batch)

            # Compute loss
            loss, metrics = self.loss_fn.compute_kl_loss(
                student_outputs['logits'],
                teacher_outputs['logprobs']
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Log metrics
            self.log_metrics(step, metrics)

            # Checkpoint
            if step % self.config.checkpoint_every == 0:
                self.save_checkpoint(step)

            # Evaluate
            if step % self.config.eval_every == 0:
                self.evaluate()
```

---

### Mode 2: Self-Improvement

#### Components

**1. Best-of-N Sampler** (`backend/opd/best_of_n_sampler.py`)
```python
class BestOfNSampler:
    """Generate N candidates and select best"""

    def __init__(
        self,
        model_path: str,
        adapter_path: str,
        n_samples: int = 5
    ):
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        self.n_samples = n_samples

    def generate_candidates(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 512
    ) -> List[Dict]:
        """
        Generate N candidate responses for a prompt

        Returns:
            [
                {'text': str, 'tokens': List[int], 'logprobs': List[float]},
                ...
            ]
        """
        candidates = []

        for i in range(self.n_samples):
            # Generate with sampling
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=None  # Different samples
            )

            # Tokenize
            tokens = self.tokenizer.encode(response)

            # Store candidate
            candidates.append({
                'sample_id': i,
                'text': response,
                'tokens': tokens,
                'token_count': len(tokens)
            })

        return candidates

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[List[Dict]]:
        """Generate candidates for multiple prompts"""
        return [
            self.generate_candidates(prompt, **kwargs)
            for prompt in prompts
        ]
```

**2. Quality Scorer** (`backend/opd/quality_scorer.py`)
```python
class QualityScorer:
    """Score response quality with multiple metrics"""

    def __init__(self, config: Dict):
        self.config = config
        self.task_type = config.get('task_type', 'generic')
        self.length_range = config.get('length_range', (10, 500))
        self.score_weights = config.get('score_weights', {
            'task_completion': 0.4,
            'coherence': 0.3,
            'length': 0.2,
            'diversity': 0.1
        })

    def score_response(self, response: str, **kwargs) -> Dict:
        """
        Compute all quality metrics for a response

        Returns:
            {
                'task_completion': float,
                'coherence': float,
                'length': float,
                'diversity': float,
                'overall': float  # Weighted sum
            }
        """
        scores = {}

        # Task completion
        scores['task_completion'] = self._task_completion_score(response)

        # Coherence (placeholder - would need model for perplexity)
        scores['coherence'] = self._coherence_score(response)

        # Length appropriateness
        scores['length'] = self._length_score(response)

        # Diversity
        scores['diversity'] = self._diversity_score(response)

        # Overall weighted score
        scores['overall'] = sum(
            scores[k] * self.score_weights[k]
            for k in self.score_weights
        )

        return scores

    def _task_completion_score(self, response: str) -> float:
        """Check task-specific completion criteria"""
        # Implement task-specific logic
        # For now, generic checks
        if len(response.strip()) < 10:
            return 0.0
        if self.task_type == 'qa' and '?' in response and not any(
            marker in response.lower() for marker in ['answer', 'is', 'the']
        ):
            return 0.3
        return 1.0

    def _coherence_score(self, response: str) -> float:
        """Estimate coherence without full model inference"""
        # Simple heuristics (can be improved)
        words = response.split()

        # Check for very short responses
        if len(words) < 5:
            return 0.3

        # Check for excessive punctuation (incoherent rambling)
        punct_ratio = sum(c in '!?.,' for c in response) / len(response)
        if punct_ratio > 0.1:
            return 0.5

        # Default: assume coherent
        return 0.8

    def _length_score(self, response: str) -> float:
        """Score based on length appropriateness"""
        min_len, max_len = self.length_range
        word_count = len(response.split())

        if min_len <= word_count <= max_len:
            return 1.0
        elif word_count < min_len:
            return word_count / min_len
        else:
            return max(0.5, 1.0 - (word_count - max_len) / max_len)

    def _diversity_score(self, response: str) -> float:
        """Penalize repetitive responses"""
        words = response.split()
        if len(words) == 0:
            return 0.0

        unique_ratio = len(set(words)) / len(words)

        # Check for repeated bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        if bigrams:
            bigram_ratio = len(set(bigrams)) / len(bigrams)
        else:
            bigram_ratio = 1.0

        return (unique_ratio * 0.6 + bigram_ratio * 0.4)

    def select_best(
        self,
        candidates: List[Dict],
        top_k: int = 1
    ) -> List[Dict]:
        """
        Score all candidates and return top-k

        Args:
            candidates: List of candidate responses
            top_k: Number of best to return

        Returns:
            Top-k candidates with scores
        """
        # Score each candidate
        for candidate in candidates:
            scores = self.score_response(candidate['text'])
            candidate['scores'] = scores
            candidate['quality'] = scores['overall']

        # Sort by quality
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x['quality'],
            reverse=True
        )

        return sorted_candidates[:top_k]
```

**3. Self-Improvement Trainer** (`backend/opd/self_improvement_trainer.py`)
```python
class SelfImprovementTrainer:
    """Iterative self-improvement via best-of-N"""

    def __init__(self, config: OPDConfig):
        self.config = config
        self.sampler = BestOfNSampler(
            config.base_model_path,
            config.student_adapter_path,
            n_samples=config.n_samples
        )
        self.scorer = QualityScorer(config.scorer_config)

    def train(self):
        """
        Main self-improvement loop:

        1. Generate N responses per prompt
        2. Score and filter to best
        3. Create training dataset from best
        4. Fine-tune on best examples
        5. Repeat for K iterations
        """
        prompts = self.load_prompts(self.config.validation_prompts_path)

        for iteration in range(self.config.num_iterations):
            logging.info(f"Iteration {iteration + 1}/{self.config.num_iterations}")

            # Step 1: Generate candidates
            all_candidates = []
            for prompt in prompts:
                candidates = self.sampler.generate_candidates(
                    prompt,
                    temperature=self.config.temperature
                )

                # Add prompt to each candidate
                for c in candidates:
                    c['prompt'] = prompt

                all_candidates.extend(candidates)

            # Step 2: Score and filter
            best_examples = []
            for prompt in prompts:
                # Get candidates for this prompt
                prompt_candidates = [
                    c for c in all_candidates if c['prompt'] == prompt
                ]

                # Select best
                best = self.scorer.select_best(
                    prompt_candidates,
                    top_k=self.config.top_k_per_prompt
                )

                best_examples.extend(best)

            # Step 3: Create training dataset
            train_data_path = self.save_training_data(
                best_examples,
                iteration
            )

            # Step 4: Fine-tune on best examples
            self.fine_tune(train_data_path, iteration)

            # Step 5: Evaluate
            metrics = self.evaluate(iteration)
            self.log_metrics(iteration, metrics)

            # Reload model for next iteration
            self.sampler = BestOfNSampler(
                self.config.base_model_path,
                self.get_latest_checkpoint(),
                n_samples=self.config.n_samples
            )

    def fine_tune(self, train_data_path: str, iteration: int):
        """Fine-tune model on filtered examples"""
        # Use existing SFT training pipeline
        # Just point it to new filtered data
        # Save checkpoint for next iteration
        pass

    def evaluate(self, iteration: int) -> Dict:
        """Evaluate current model quality"""
        # Generate samples on held-out set
        # Measure average quality score
        # Compare to previous iteration
        pass
```

---

## Step-by-Step Implementation

### Phase 0: Setup (1-2 hours)

**0.1 Create Directory Structure**
```bash
mkdir -p OnPolicyDistill/{configs,mode1_distillation/{teacher_cache,rollouts,checkpoints},mode2_self_improvement/{generations,filtered,scores,checkpoints},metrics}
mkdir -p backend/opd
```

**0.2 Update Dependencies**
```bash
cd backend
# No new dependencies needed - all local!
```

**0.3 Create Base Configuration**
```python
# backend/opd/config.py
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class OPDConfig:
    """Configuration for OPD training"""

    # Mode selection
    mode: Literal["distillation", "self_improvement"] = "self_improvement"

    # Model paths
    base_model_path: str = ""
    student_adapter_path: str = ""
    teacher_model_path: Optional[str] = None  # Only for distillation mode

    # Data
    validation_prompts_path: str = ""

    # Common parameters
    num_steps: int = 1000  # For distillation
    num_iterations: int = 5  # For self-improvement
    batch_size: int = 4
    learning_rate: float = 1e-5
    seed: int = 42

    # Self-improvement specific
    n_samples: int = 5  # N in best-of-N
    top_k_per_prompt: int = 1  # How many best to keep
    temperature: float = 0.8

    # Distillation specific
    distillation_temperature: float = 2.0
    kl_weight: float = 1.0

    # Quality scoring
    scorer_config: dict = None

    # Checkpointing
    checkpoint_dir: str = "./OnPolicyDistill"
    checkpoint_every: int = 100
    eval_every: int = 100

    # Session
    run_id: Optional[str] = None
```

---

### Phase 1: Mode 2 Implementation (Self-Improvement) - 12-16 hours

**Priority**: Implement Mode 2 first (works on all systems)

**1.1 Best-of-N Sampler** (3-4 hours)
- [ ] Implement `BestOfNSampler` class
- [ ] Add generation caching
- [ ] Test with existing fine-tuned model
- [ ] Verify deterministic seeding works

**1.2 Quality Scorer** (4-5 hours)
- [ ] Implement task completion checker
- [ ] Implement coherence heuristics
- [ ] Implement length scorer
- [ ] Implement diversity scorer
- [ ] Add configurable weights
- [ ] Test on sample responses

**1.3 Self-Improvement Trainer** (5-7 hours)
- [ ] Implement training loop
- [ ] Add dataset creation from filtered examples
- [ ] Integrate with existing SFT training
- [ ] Add checkpoint management
- [ ] Implement evaluation logic
- [ ] Add metrics logging

---

### Phase 2: Mode 1 Implementation (Knowledge Distillation) - 16-20 hours

**Priority**: Implement after Mode 2 is working

**2.1 Teacher Model Loader** (4-5 hours)
- [ ] Implement model loading with freezing
- [ ] Add logprob extraction (requires MLX investigation)
- [ ] Implement caching system
- [ ] Test with large model (32B)
- [ ] Memory profiling

**2.2 Distillation Loss** (3-4 hours)
- [ ] Implement KL divergence calculation
- [ ] Add temperature scaling
- [ ] Test gradient flow
- [ ] Validate against reference implementation

**2.3 Distillation Trainer** (5-6 hours)
- [ ] Implement training loop
- [ ] Add student/teacher orchestration
- [ ] Implement checkpointing
- [ ] Add evaluation
- [ ] Metrics logging

**2.4 Memory Detection** (2-3 hours)
- [ ] Detect system unified memory
- [ ] Auto-select mode based on RAM
- [ ] Add UI for mode selection
- [ ] Validate memory estimates

---

### Phase 3: Backend API Integration (6-8 hours)

**3.1 FastAPI Endpoints**
```python
# backend/main.py additions

from opd.config import OPDConfig
from opd.self_improvement_trainer import SelfImprovementTrainer
from opd.distillation_trainer import DistillationTrainer

@dataclass
class OPDStartRequest:
    mode: str  # "distillation" or "self_improvement"
    base_model_path: str
    student_adapter_path: str
    validation_prompts_path: str
    teacher_model_path: Optional[str] = None
    num_steps: int = 1000
    num_iterations: int = 5
    n_samples: int = 5
    temperature: float = 0.8

@app.post("/opd/start")
async def start_opd_training(request: OPDStartRequest):
    """Start OPD training (either mode)"""
    global opd_process

    # Create config
    config = OPDConfig(
        mode=request.mode,
        base_model_path=request.base_model_path,
        student_adapter_path=request.student_adapter_path,
        validation_prompts_path=request.validation_prompts_path,
        teacher_model_path=request.teacher_model_path,
        num_steps=request.num_steps,
        num_iterations=request.num_iterations,
        n_samples=request.n_samples,
        temperature=request.temperature
    )

    # Generate run ID
    run_id = f"opd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.run_id = run_id

    # Save config
    config_path = f"./OnPolicyDistill/configs/{run_id}.yaml"
    # ... save config ...

    # Start training subprocess
    if config.mode == "self_improvement":
        trainer_class = "SelfImprovementTrainer"
    else:
        trainer_class = "DistillationTrainer"

    # Spawn training process
    opd_process = subprocess.Popen([
        sys.executable,
        "-c",
        f"""
import sys
sys.path.append('backend')
from opd.config import OPDConfig
from opd.{trainer_class.lower()} import {trainer_class}

config = OPDConfig.load('{config_path}')
trainer = {trainer_class}(config)
trainer.train()
        """
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return {"status": "success", "run_id": run_id}

@app.get("/opd/status")
async def get_opd_status():
    """Get current OPD status"""
    # Similar to existing training status endpoint
    # Read from metrics file
    pass

@app.post("/opd/stop")
async def stop_opd_training():
    """Stop OPD training"""
    global opd_process
    if opd_process:
        opd_process.terminate()
        return {"status": "stopped"}
    return {"status": "not_running"}
```

---

### Phase 4: Frontend Implementation (12-16 hours)

**4.1 Redux Slice** (2-3 hours)
```typescript
// frontend/src/store/slices/opdSlice.ts

interface OPDState {
  mode: 'distillation' | 'self_improvement';
  config: {
    studentAdapterPath: string;
    validationPromptsPath: string;
    teacherModelPath?: string;
    nSamples: number;
    numIterations: number;
    temperature: number;
  };
  trainingState: 'idle' | 'running' | 'completed' | 'error';
  metrics: {
    iteration?: number;
    step?: number;
    avgQualityScore?: number;
    klLoss?: number;
  } | null;
  systemMemory: number;  // GB
  recommendedMode: 'distillation' | 'self_improvement';
}

// ... slice implementation ...
```

**4.2 OPD Page** (4-5 hours)
- [ ] Create `OPDPage.tsx`
- [ ] Add mode selection UI
- [ ] System memory display
- [ ] Mode recommendation logic

**4.3 Setup Panels** (3-4 hours)
- [ ] Self-Improvement setup form
- [ ] Knowledge Distillation setup form
- [ ] Validation prompts file selector
- [ ] Teacher model selector (Mode 1)

**4.4 Progress Monitor** (3-4 hours)
- [ ] Mode-specific metrics display
- [ ] Quality score charts (Mode 2)
- [ ] KL loss charts (Mode 1)
- [ ] Iteration progress (Mode 2)
- [ ] Step progress (Mode 1)

---

### Phase 5: Testing (8-12 hours)

**5.1 Unit Tests**
- [ ] Test BestOfNSampler determinism
- [ ] Test QualityScorer metrics
- [ ] Test mode auto-selection
- [ ] Test checkpoint save/load

**5.2 Integration Tests**
- [ ] End-to-end Mode 2 (self-improvement)
- [ ] End-to-end Mode 1 (distillation) if 64GB available
- [ ] Mode switching
- [ ] Training resume

**5.3 Manual Testing**
- [ ] Run Mode 2 on small dataset (10 prompts, 2 iterations)
- [ ] Verify quality improvement across iterations
- [ ] Check memory usage stays within bounds
- [ ] Test UI responsiveness

---

## Configuration Examples

### Mode 2: Self-Improvement Config

```yaml
# OnPolicyDistill/configs/self_improvement_example.yaml

mode: self_improvement

# Models
base_model_path: /path/to/qwen2.5-7b
student_adapter_path: /path/to/sft_adapters  # From previous training

# Data
validation_prompts_path: /path/to/val_prompts.jsonl

# Training
num_iterations: 5
batch_size: 4
learning_rate: 0.00001

# Sampling
n_samples: 5
top_k_per_prompt: 1
temperature: 0.8

# Quality Scoring
scorer_config:
  task_type: qa  # or "code", "summary", "generic"
  length_range: [20, 200]
  score_weights:
    task_completion: 0.4
    coherence: 0.3
    length: 0.2
    diversity: 0.1

# Checkpointing
checkpoint_dir: ./OnPolicyDistill/mode2_self_improvement/checkpoints
checkpoint_every: 1  # Save after each iteration
eval_every: 1

seed: 42
```

### Mode 1: Knowledge Distillation Config

```yaml
# OnPolicyDistill/configs/distillation_example.yaml

mode: distillation

# Models
base_model_path: /path/to/qwen2.5-7b
student_adapter_path: /path/to/sft_adapters
teacher_model_path: /path/to/qwen2.5-32b

# Data
validation_prompts_path: /path/to/val_prompts.jsonl

# Training
num_steps: 1000
batch_size: 2  # Smaller due to teacher memory
learning_rate: 0.00001

# Distillation
distillation_temperature: 2.0
kl_weight: 1.0

# Checkpointing
checkpoint_dir: ./OnPolicyDistill/mode1_distillation/checkpoints
checkpoint_every: 100
eval_every: 100

seed: 42
```

---

## API Reference

### POST /opd/start

**Request**:
```json
{
  "mode": "self_improvement",
  "base_model_path": "/path/to/model",
  "student_adapter_path": "/path/to/adapter",
  "validation_prompts_path": "/path/to/prompts.jsonl",
  "n_samples": 5,
  "num_iterations": 5,
  "temperature": 0.8
}
```

**Response**:
```json
{
  "status": "success",
  "run_id": "opd_20250128_143022",
  "mode": "self_improvement",
  "estimated_duration_minutes": 30
}
```

### GET /opd/status

**Response (Mode 2)**:
```json
{
  "state": "running",
  "mode": "self_improvement",
  "run_id": "opd_20250128_143022",
  "metrics": {
    "iteration": 3,
    "total_iterations": 5,
    "current_avg_quality": 0.78,
    "improvement_vs_iteration_1": 0.15
  }
}
```

**Response (Mode 1)**:
```json
{
  "state": "running",
  "mode": "distillation",
  "run_id": "opd_20250128_143022",
  "metrics": {
    "step": 450,
    "total_steps": 1000,
    "kl_loss": 0.234,
    "student_teacher_alignment": 0.82
  }
}
```

---

## Timeline & Rollout

### Week 1: Mode 2 (Self-Improvement)
- Days 1-2: Sampler + Scorer implementation
- Days 3-4: Trainer implementation
- Day 5: Testing & bug fixes

### Week 2: Backend Integration + Frontend
- Days 1-2: FastAPI endpoints
- Days 3-4: Frontend UI (Redux + components)
- Day 5: Integration testing

### Week 3: Mode 1 (Knowledge Distillation) - Optional
- Days 1-2: Teacher model loader
- Days 3-4: Distillation trainer
- Day 5: Testing on high-memory system

### Week 4: Polish & Launch
- Days 1-2: Documentation
- Days 3-4: Performance optimization
- Day 5: User testing & launch

**Total**: 3-4 weeks for full implementation

---

## Key Advantages Over v1.0

| Aspect | v1.0 (Claude API) | v2.0 (Local) |
|--------|-------------------|--------------|
| **Dependencies** | Anthropic API | None |
| **Cost** | $$ per run | $0 |
| **Speed** | Network latency | Fast (local) |
| **Flexibility** | API limits | Full control |
| **Memory** | Low | Medium-High |
| **Quality** | High (Sonnet 4.5) | Good (configurable) |
| **Reliability** | Internet required | Fully offline |

---

## Success Criteria

### Mode 2 (Self-Improvement)
✅ Quality scores improve across iterations
✅ Model produces better responses on validation set
✅ Works on standard Apple Silicon (16GB RAM)
✅ Completes 5 iterations in <1 hour for 100 prompts
✅ UI shows clear iteration progress

### Mode 1 (Knowledge Distillation)
✅ Student KL divergence decreases over training
✅ Student matches teacher outputs on validation
✅ Compressed model (7B) performs close to teacher (32B)
✅ Training completes without OOM on 64GB system
✅ UI shows KL loss charts

---

## Open Questions

1. **MLX Logprob Extraction**
   - Does MLX expose per-token logprobs during generation?
   - If not, can we compute them via forward pass?
   - Alternative: use final layer activations as proxy?

2. **Quality Metric Tuning**
   - What task types are most common for users?
   - Should we add custom metric plugins?
   - How to calibrate score thresholds?

3. **Memory Limits**
   - What's minimum RAM for Mode 1 with 32B teacher?
   - Should we add batch size auto-tuning?
   - Support model quantization for teacher?

4. **Validation Data**
   - Auto-split user's training data?
   - Require separate validation file?
   - How many prompts needed for good results?

---

## Next Steps

1. ✅ **Approve this revised plan**
2. **Answer open questions** (especially MLX logprobs)
3. **Start Phase 0** (setup)
4. **Implement Mode 2** (self-improvement) first
5. **Test on real fine-tuned model**
6. **Iterate based on results**

---

**Document Version**: 2.0
**Date**: January 28, 2025
**Status**: AWAITING APPROVAL
**Key Change**: Removed Claude API dependency, 100% local implementation

