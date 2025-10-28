# On-Policy Distillation (OPD) Implementation Plan

## Executive Summary

This document provides a comprehensive step-by-step plan to implement On-Policy Distillation (OPD) in the Droid-FineTuning application. The implementation will add a new "OPD Distillation" tab to the existing GUI, enabling users to perform teacher-student distillation using Claude Sonnet 4.5 as the teacher model.

**Implementation Status**: NOT STARTED (Awaiting approval)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Architectural Decisions](#key-architectural-decisions)
3. [Directory Structure](#directory-structure)
4. [Implementation Phases](#implementation-phases)
5. [Detailed Implementation Steps](#detailed-implementation-steps)
6. [File-by-File Implementation Guide](#file-by-file-implementation-guide)
7. [Integration Points](#integration-points)
8. [Testing Strategy](#testing-strategy)
9. [Configuration Reference](#configuration-reference)
10. [API Reference](#api-reference)
11. [Rollout Plan](#rollout-plan)

---

## Architecture Overview

### Current Architecture
```
Electron App (Desktop)
├── Frontend (React + TypeScript + Redux)
│   ├── Pages: Setup → Training → Results → Compare
│   ├── State: Redux Toolkit (trainingSlice, modelsSlice, uiSlice)
│   └── Communication: WebSocket + HTTP polling
├── Backend (FastAPI + Python)
│   ├── Training Manager (subprocess orchestration)
│   ├── Model Loading (MLX framework)
│   └── Metrics Parsing (regex-based)
└── Storage
    ├── Base Models (MLX format)
    ├── LoRA Adapters (SafeTensors)
    └── Sessions (JSON)
```

### New OPD Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      OPD WORKFLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Student Rollouts                                        │
│     └─> Generate on-policy samples from student model      │
│                                                              │
│  2. Teacher Scoring (Claude Sonnet 4.5)                    │
│     └─> Get token-level logprobs via Anthropic API         │
│                                                              │
│  3. Loss Computation                                         │
│     └─> Reverse KL divergence (student || teacher)         │
│                                                              │
│  4. LoRA Update                                             │
│     └─> Backprop through student adapters only             │
│                                                              │
│  5. Evaluation                                              │
│     └─> Metrics on held-out validation set                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Architectural Decisions

### Decision 1: Teacher Model Architecture

**Option A: Claude Sonnet 4.5 via API (PRD Specification)**
- ✅ Pro: Matches PRD exactly
- ✅ Pro: State-of-the-art teacher quality
- ❌ Con: API costs per token
- ❌ Con: Network latency
- ❌ Con: Requires Anthropic API key

**Option B: Hybrid (Local + API)**
- ✅ Pro: Use fine-tuned local model as initial teacher
- ✅ Pro: Falls back to Sonnet for disagreements
- ⚠️ Con: More complex orchestration

**Option C: Local Fine-tuned Model Only**
- ✅ Pro: No API costs
- ✅ Pro: Faster iteration
- ❌ Con: Deviates from PRD

**RECOMMENDATION**: **Option A (API-based)** to match PRD specification. We'll implement with:
- Teacher client for Anthropic API
- Token-level logprob extraction via Messages API
- Cost tracking and budget guardrails
- Caching layer to minimize redundant calls

### Decision 2: Integration Pattern

**Chosen Approach**: New page in existing flow
```
Setup → Training → Results → [NEW] OPD Distillation → Compare
```

**Rationale**:
- Follows existing page-based architecture
- OPD runs AFTER initial SFT training completes
- Can be optional enhancement step
- Reuses existing Redux/WebSocket patterns

### Decision 3: Data Flow

```
[SFT Training Completes]
         ↓
[Fine-tuned LoRA Adapters Created] ← Input to OPD
         ↓
[OPD Setup: Select Teacher + Config]
         ↓
[Generate Student Rollouts] → prompts from validation set
         ↓
[Call Teacher API] → get logprobs for each token
         ↓
[Compute Reverse-KL Loss]
         ↓
[Update Student LoRA] → new adapter weights
         ↓
[Evaluate on Val Set]
         ↓
[OPD Results: Distilled Model]
```

---

## Directory Structure

### New Directories to Create

```
Droid-FineTuning/
├── OnPolicyDistill/                    # NEW: Root OPD data directory
│   ├── configs/                        # OPD run configurations
│   │   └── {run_id}.yaml
│   ├── rollouts/                       # Student-generated samples
│   │   ├── train/
│   │   │   └── {run_id}_rollouts.jsonl
│   │   └── val/
│   │       └── {run_id}_val_rollouts.jsonl
│   ├── teacher_completions/            # Raw teacher outputs
│   │   └── {run_id}_completions.jsonl
│   ├── teacher_logprobs/               # Cached teacher logprobs
│   │   └── {cache_key}.json
│   ├── checkpoints/                    # OPD adapter snapshots
│   │   ├── {run_id}/
│   │   │   ├── step_000100_adapters.safetensors
│   │   │   ├── step_000200_adapters.safetensors
│   │   │   └── best_adapters.safetensors
│   └── metrics/                        # OPD-specific metrics
│       ├── {run_id}_train.jsonl
│       └── {run_id}_eval.jsonl
│
├── backend/
│   ├── main.py                         # MODIFY: Add OPD endpoints
│   ├── opd/                            # NEW: OPD backend module
│   │   ├── __init__.py
│   │   ├── config.py                   # OPDConfig dataclass
│   │   ├── teacher_client.py           # Anthropic API client
│   │   ├── rollout_generator.py        # Student sampling
│   │   ├── loss_calculator.py          # Reverse-KL computation
│   │   ├── trainer.py                  # OPD training loop
│   │   ├── evaluator.py                # Validation metrics
│   │   ├── extractor.py                # Extract f(y_teacher)
│   │   └── utils.py                    # Caching, alignment, etc.
│   └── requirements.txt                # MODIFY: Add anthropic SDK
│
└── frontend/src/
    ├── pages/
    │   └── OPDPage.tsx                 # NEW: Main OPD page
    ├── components/
    │   ├── opd/                        # NEW: OPD-specific components
    │   │   ├── OPDSetupPanel.tsx       # Config form
    │   │   ├── OPDProgressMonitor.tsx  # Real-time progress
    │   │   ├── OPDMetricsChart.tsx     # KL divergence charts
    │   │   ├── OPDResultsPanel.tsx     # Final metrics display
    │   │   └── TeacherCostTracker.tsx  # API spend tracking
    │   └── Sidebar.tsx                 # MODIFY: Add OPD nav item
    └── store/slices/
        └── opdSlice.ts                 # NEW: OPD Redux state
```

---

## Implementation Phases

### Phase 0: Preparation (1-2 hours)
- [ ] Set up development environment
- [ ] Create directory structure
- [ ] Install dependencies
- [ ] Set up Anthropic API key

### Phase 1: Backend Core (8-12 hours)
- [ ] Teacher client implementation
- [ ] Rollout generator
- [ ] Loss calculation
- [ ] Data structures and serialization

### Phase 2: Backend Training Loop (8-12 hours)
- [ ] OPD trainer orchestration
- [ ] Checkpoint management
- [ ] Metrics collection
- [ ] Evaluator implementation

### Phase 3: Backend API (4-6 hours)
- [ ] FastAPI endpoints
- [ ] WebSocket integration
- [ ] Session management
- [ ] Error handling

### Phase 4: Frontend State & Components (8-12 hours)
- [ ] Redux slice for OPD
- [ ] OPD page layout
- [ ] Setup panel
- [ ] Progress monitor
- [ ] Results display

### Phase 5: Integration & Polish (6-8 hours)
- [ ] Connect frontend ↔ backend
- [ ] Chart integration
- [ ] Cost tracking UI
- [ ] Navigation flow

### Phase 6: Testing & Validation (8-12 hours)
- [ ] Unit tests
- [ ] Integration tests
- [ ] End-to-end workflow test
- [ ] Budget guardrail validation

**Total Estimated Time**: 40-60 hours

---

## Detailed Implementation Steps

### STEP 1: Environment Setup

#### 1.1 Install Dependencies

**Backend (backend/requirements.txt)**
```bash
# Add to existing requirements.txt
anthropic>=0.18.0  # Claude API client
tiktoken>=0.5.0    # Tokenizer for alignment checks
tenacity>=8.2.0    # Retry logic with backoff
```

**Commands**:
```bash
cd backend
source /path/to/mlx/.venv/bin/activate
pip install anthropic tiktoken tenacity
```

#### 1.2 Environment Variables

Create `.env` file in root:
```bash
# Anthropic API Configuration
ANTHROPIC_API_KEY=your_api_key_here

# OPD Configuration
OPD_DEFAULT_TEACHER=claude-sonnet-4-5-20250929
OPD_MAX_BUDGET_USD=100.0
OPD_RATE_LIMIT_TPM=10000  # Tokens per minute
OPD_CACHE_DIR=./OnPolicyDistill/teacher_logprobs
```

#### 1.3 Create Directory Structure

```bash
# Run from project root
mkdir -p OnPolicyDistill/{configs,rollouts/{train,val},teacher_completions,teacher_logprobs,checkpoints,metrics}
mkdir -p backend/opd
mkdir -p frontend/src/pages
mkdir -p frontend/src/components/opd
```

---

### STEP 2: Backend Implementation

#### 2.1 Configuration Module (`backend/opd/config.py`)

**Purpose**: Define data structures for OPD configuration

**Implementation**:
```python
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

@dataclass
class OPDConfig:
    """Configuration for On-Policy Distillation training"""

    # Model paths
    base_model_path: str
    student_adapter_path: str  # Input: fine-tuned adapter from SFT
    teacher_model_id: str = "claude-sonnet-4-5-20250929"

    # Data paths
    validation_prompts_path: str
    rollout_output_dir: str = "./OnPolicyDistill/rollouts/train"
    val_rollout_dir: str = "./OnPolicyDistill/rollouts/val"

    # Training parameters
    num_rollouts_per_prompt: int = 3
    rollout_max_tokens: int = 512
    batch_size: int = 4
    num_training_steps: int = 1000
    learning_rate: float = 1e-5

    # Loss configuration
    kl_weight: float = 1.0
    kl_schedule: Literal["constant", "warmup", "cosine"] = "warmup"
    kl_warmup_steps: int = 100
    use_sft_mixup: bool = False
    sft_weight: float = 0.1

    # Teacher API configuration
    teacher_max_concurrent: int = 5
    teacher_rate_limit_tpm: int = 10000
    teacher_timeout_seconds: int = 30
    max_budget_usd: float = 100.0

    # Checkpointing
    checkpoint_dir: str = "./OnPolicyDistill/checkpoints"
    checkpoint_every_steps: int = 100
    save_best_only: bool = False

    # Evaluation
    eval_every_steps: int = 100
    eval_max_samples: int = 100

    # Reproducibility
    seed: int = 42

    # Session metadata
    run_id: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.kl_weight < 0 or self.kl_weight > 1:
            raise ValueError("kl_weight must be between 0 and 1")
        if self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive")

        # Ensure directories exist
        Path(self.rollout_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.val_rollout_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class OPDMetrics:
    """Metrics tracked during OPD training"""
    step: int

    # Loss components
    kl_loss: float
    sft_loss: Optional[float] = None
    total_loss: float = 0.0

    # KL statistics
    kl_divergence_mean: float = 0.0
    kl_divergence_max: float = 0.0
    kl_divergence_min: float = 0.0

    # Model statistics
    student_entropy: float = 0.0
    teacher_alignment_pct: float = 0.0  # % tokens where student matches teacher

    # Performance
    tokens_per_second: float = 0.0
    samples_processed: int = 0

    # Teacher API
    teacher_api_latency_ms: float = 0.0
    teacher_api_calls: int = 0
    cumulative_cost_usd: float = 0.0

    # Checkpoint info
    checkpoint_path: Optional[str] = None
    is_best: bool = False
```

---

#### 2.2 Teacher Client (`backend/opd/teacher_client.py`)

**Purpose**: Interface with Claude Sonnet 4.5 API for token-level logprobs

**Key Features**:
- Batched API calls with rate limiting
- Token alignment validation
- Cost tracking
- Response caching
- Retry with exponential backoff

**Implementation Outline**:
```python
import anthropic
from anthropic import Anthropic
from typing import List, Dict, Tuple
import hashlib
import json
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

class TeacherClient:
    """Client for getting token-level logprobs from Claude Sonnet 4.5"""

    def __init__(
        self,
        api_key: str,
        model_id: str = "claude-sonnet-4-5-20250929",
        max_concurrent: int = 5,
        rate_limit_tpm: int = 10000,
        cache_dir: str = "./OnPolicyDistill/teacher_logprobs",
        max_budget_usd: float = 100.0
    ):
        self.client = Anthropic(api_key=api_key)
        self.model_id = model_id
        self.max_concurrent = max_concurrent
        self.rate_limit_tpm = rate_limit_tpm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_budget_usd = max_budget_usd
        self.cumulative_cost_usd = 0.0

    def _get_cache_key(self, prompt: str, continuation_tokens: List[int]) -> str:
        """Generate cache key for teacher responses"""
        content = f"{prompt}||{','.join(map(str, continuation_tokens))}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached teacher logprobs"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save teacher logprobs to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def get_token_logprobs(
        self,
        prompt: str,
        continuation: str,
        max_tokens: int = 512
    ) -> Dict:
        """
        Get token-level logprobs from teacher for a given prompt + continuation.

        Returns:
            {
                'logprobs': List[float],  # Log probability for each token
                'tokens': List[str],       # Token strings
                'token_ids': List[int],    # Token IDs
                'alignment_map': Dict,     # Maps student token idx -> teacher token idx
                'cost_usd': float         # API call cost
            }
        """
        # Check cache first
        cache_key = self._get_cache_key(prompt, continuation)
        cached = self._load_from_cache(cache_key)
        if cached:
            logging.info(f"Cache hit for key {cache_key[:8]}...")
            return cached

        # Check budget
        if self.cumulative_cost_usd >= self.max_budget_usd:
            raise RuntimeError(
                f"Budget exceeded: ${self.cumulative_cost_usd:.2f} >= ${self.max_budget_usd:.2f}"
            )

        # Call Anthropic API
        # NOTE: As of Jan 2025, Claude API doesn't expose per-token logprobs directly
        # This is a MOCK implementation - real implementation would use:
        # 1. Prompt Caching for efficiency
        # 2. Extended API features when available
        # 3. Alternative: use completion likelihood scoring

        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": f"{prompt}\n\n{continuation}"
                }],
                # In real implementation, request logprobs via API parameter
                # For now, we'll use a placeholder
            )

            # Extract logprobs (PLACEHOLDER - actual API may differ)
            logprobs = self._extract_logprobs_from_response(response)

            # Calculate cost (approximate)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost_usd = self._calculate_cost(input_tokens, output_tokens)
            self.cumulative_cost_usd += cost_usd

            result = {
                'logprobs': logprobs,
                'tokens': self._extract_tokens(response),
                'token_ids': self._extract_token_ids(response),
                'alignment_map': {},  # TODO: implement alignment
                'cost_usd': cost_usd,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }

            # Cache result
            self._save_to_cache(cache_key, result)

            return result

        except Exception as e:
            logging.error(f"Teacher API call failed: {e}")
            raise

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API call cost based on token counts"""
        # Sonnet 4.5 pricing (as of Jan 2025, approximate)
        input_cost_per_mtok = 3.00  # $3 per million input tokens
        output_cost_per_mtok = 15.00  # $15 per million output tokens

        input_cost = (input_tokens / 1_000_000) * input_cost_per_mtok
        output_cost = (output_tokens / 1_000_000) * output_cost_per_mtok

        return input_cost + output_cost

    def _extract_logprobs_from_response(self, response) -> List[float]:
        """Extract per-token logprobs from API response"""
        # PLACEHOLDER: Real implementation depends on API format
        # For now, return dummy data
        logging.warning("Using placeholder logprobs - implement actual extraction")
        return [0.0] * 10  # Dummy

    def _extract_tokens(self, response) -> List[str]:
        """Extract token strings from API response"""
        # PLACEHOLDER
        return response.content[0].text.split()[:10] if response.content else []

    def _extract_token_ids(self, response) -> List[int]:
        """Extract token IDs from API response"""
        # PLACEHOLDER
        return list(range(10))

    def batch_get_logprobs(
        self,
        prompts: List[str],
        continuations: List[str],
        max_tokens: int = 512
    ) -> List[Dict]:
        """
        Batch process multiple prompt-continuation pairs.
        Handles rate limiting and concurrency.
        """
        results = []
        for prompt, continuation in zip(prompts, continuations):
            try:
                result = self.get_token_logprobs(prompt, continuation, max_tokens)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to process prompt: {e}")
                results.append(None)

        return results

    def get_cumulative_cost(self) -> float:
        """Get total API spend so far"""
        return self.cumulative_cost_usd
```

**IMPORTANT NOTE**: The Claude API (as of January 2025) does not natively expose per-token logprobs. This implementation requires either:
1. **Wait for API feature**: Anthropic may add logprob support in future
2. **Alternative approach**: Use likelihood scoring on student-generated tokens
3. **Hybrid**: Use teacher completions for SFT-style targets instead of pure KL

**Recommended Interim Solution**: Generate teacher completions and use them as targets for SFT loss, with optional preference ranking between student and teacher outputs.

---

#### 2.3 Rollout Generator (`backend/opd/rollout_generator.py`)

**Purpose**: Generate on-policy samples from the student model

**Implementation Outline**:
```python
import mlx.core as mx
from mlx_lm import load, generate
from typing import List, Dict
import json
from pathlib import Path
import logging

class RolloutGenerator:
    """Generates on-policy rollouts from student model"""

    def __init__(
        self,
        base_model_path: str,
        student_adapter_path: str,
        seed: int = 42
    ):
        self.base_model_path = base_model_path
        self.student_adapter_path = student_adapter_path
        self.seed = seed
        self.model = None
        self.tokenizer = None

    def load_student_model(self):
        """Load student model with LoRA adapters"""
        logging.info(f"Loading student model from {self.base_model_path}")
        logging.info(f"With adapters from {self.student_adapter_path}")

        self.model, self.tokenizer = load(
            self.base_model_path,
            adapter_path=self.student_adapter_path
        )

        logging.info("Student model loaded successfully")

    def generate_rollouts(
        self,
        prompts: List[str],
        num_samples_per_prompt: int = 3,
        max_tokens: int = 512,
        temperature: float = 1.0,
        output_path: str = None
    ) -> List[Dict]:
        """
        Generate on-policy rollouts for given prompts.

        Args:
            prompts: List of input prompts
            num_samples_per_prompt: Number of completions per prompt
            max_tokens: Max tokens to generate per sample
            temperature: Sampling temperature
            output_path: Where to save rollouts (JSONL)

        Returns:
            List of rollout records with structure:
            {
                'rollout_id': str,
                'prompt': str,
                'completion': str,
                'token_ids': List[int],
                'logprobs': List[float],  # Student's own logprobs
                'temperature': float,
                'max_tokens': int,
                'timestamp': str
            }
        """
        if self.model is None:
            self.load_student_model()

        rollouts = []

        for prompt_idx, prompt in enumerate(prompts):
            for sample_idx in range(num_samples_per_prompt):
                try:
                    # Generate completion
                    completion = generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        verbose=False
                    )

                    # Tokenize for IDs
                    token_ids = self.tokenizer.encode(completion)

                    # Create rollout record
                    rollout = {
                        'rollout_id': f"{prompt_idx:04d}_{sample_idx:02d}",
                        'prompt': prompt,
                        'completion': completion,
                        'token_ids': token_ids,
                        'logprobs': [],  # TODO: extract from model
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'timestamp': datetime.utcnow().isoformat()
                    }

                    rollouts.append(rollout)
                    logging.info(f"Generated rollout {rollout['rollout_id']}")

                except Exception as e:
                    logging.error(f"Failed to generate rollout for prompt {prompt_idx}: {e}")

        # Save to file if path provided
        if output_path:
            self._save_rollouts(rollouts, output_path)

        return rollouts

    def _save_rollouts(self, rollouts: List[Dict], output_path: str):
        """Save rollouts to JSONL file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + '\n')

        logging.info(f"Saved {len(rollouts)} rollouts to {output_path}")

    @staticmethod
    def load_rollouts(input_path: str) -> List[Dict]:
        """Load rollouts from JSONL file"""
        rollouts = []
        with open(input_path, 'r') as f:
            for line in f:
                rollouts.append(json.loads(line))
        return rollouts
```

---

#### 2.4 Loss Calculator (`backend/opd/loss_calculator.py`)

**Purpose**: Compute reverse KL divergence and optional SFT loss

**Implementation Outline**:
```python
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Tuple
import logging

class OPDLossCalculator:
    """Calculates reverse KL divergence loss for OPD"""

    def __init__(
        self,
        kl_weight: float = 1.0,
        sft_weight: float = 0.0,
        token_weighting: str = "uniform"
    ):
        self.kl_weight = kl_weight
        self.sft_weight = sft_weight
        self.token_weighting = token_weighting

    def compute_reverse_kl(
        self,
        student_logprobs: mx.array,
        teacher_logprobs: mx.array,
        mask: mx.array = None
    ) -> Tuple[mx.array, Dict]:
        """
        Compute reverse KL divergence: D_KL(P_student || P_teacher)

        Formula: sum_tokens [ log(P_student) - log(P_teacher) ]

        Args:
            student_logprobs: (batch_size, seq_len, vocab_size) log probabilities
            teacher_logprobs: (batch_size, seq_len, vocab_size) log probabilities
            mask: (batch_size, seq_len) binary mask for valid tokens

        Returns:
            loss: scalar tensor
            metrics: dict with detailed statistics
        """
        # Get token-level KL divergence
        # KL(student || teacher) = sum_v [ P_s(v) * (log P_s(v) - log P_t(v)) ]

        # Convert logprobs to probs
        student_probs = mx.exp(student_logprobs)

        # KL divergence per position
        kl_per_token = mx.sum(
            student_probs * (student_logprobs - teacher_logprobs),
            axis=-1  # sum over vocabulary
        )

        # Apply mask if provided
        if mask is not None:
            kl_per_token = kl_per_token * mask
            num_tokens = mx.sum(mask)
        else:
            num_tokens = mx.prod(mx.array(kl_per_token.shape))

        # Compute mean KL
        kl_loss = mx.sum(kl_per_token) / num_tokens

        # Compute statistics
        metrics = {
            'kl_loss': float(kl_loss),
            'kl_mean': float(mx.mean(kl_per_token)),
            'kl_max': float(mx.max(kl_per_token)),
            'kl_min': float(mx.min(kl_per_token)),
            'kl_std': float(mx.std(kl_per_token))
        }

        return kl_loss, metrics

    def compute_sft_loss(
        self,
        student_logprobs: mx.array,
        target_token_ids: mx.array,
        mask: mx.array = None
    ) -> Tuple[mx.array, Dict]:
        """
        Compute standard cross-entropy loss for SFT mix-in

        Args:
            student_logprobs: (batch_size, seq_len, vocab_size)
            target_token_ids: (batch_size, seq_len) target token indices
            mask: (batch_size, seq_len) binary mask

        Returns:
            loss: scalar tensor
            metrics: dict with statistics
        """
        # Cross-entropy loss
        ce_loss = nn.losses.cross_entropy(
            student_logprobs,
            target_token_ids,
            reduction='none'
        )

        # Apply mask
        if mask is not None:
            ce_loss = ce_loss * mask
            num_tokens = mx.sum(mask)
        else:
            num_tokens = mx.prod(mx.array(ce_loss.shape))

        # Mean loss
        sft_loss = mx.sum(ce_loss) / num_tokens

        metrics = {
            'sft_loss': float(sft_loss),
            'perplexity': float(mx.exp(sft_loss))
        }

        return sft_loss, metrics

    def compute_total_loss(
        self,
        student_logprobs: mx.array,
        teacher_logprobs: mx.array,
        target_token_ids: mx.array = None,
        mask: mx.array = None,
        current_step: int = 0,
        warmup_steps: int = 0
    ) -> Tuple[mx.array, Dict]:
        """
        Compute weighted combination of reverse KL and optional SFT loss

        Returns:
            total_loss: scalar tensor
            metrics: dict with all loss components
        """
        # Compute reverse KL
        kl_loss, kl_metrics = self.compute_reverse_kl(
            student_logprobs,
            teacher_logprobs,
            mask
        )

        # Apply KL weight with optional warmup
        if warmup_steps > 0 and current_step < warmup_steps:
            kl_weight_scheduled = self.kl_weight * (current_step / warmup_steps)
        else:
            kl_weight_scheduled = self.kl_weight

        total_loss = kl_weight_scheduled * kl_loss

        metrics = {
            **kl_metrics,
            'kl_weight': kl_weight_scheduled,
            'total_loss': float(total_loss)
        }

        # Add SFT loss if enabled
        if self.sft_weight > 0 and target_token_ids is not None:
            sft_loss, sft_metrics = self.compute_sft_loss(
                student_logprobs,
                target_token_ids,
                mask
            )
            total_loss = total_loss + self.sft_weight * sft_loss
            metrics.update(sft_metrics)
            metrics['total_loss'] = float(total_loss)

        return total_loss, metrics
```

---

#### 2.5 OPD Trainer (`backend/opd/trainer.py`)

**Purpose**: Main training loop orchestration

**High-level structure**:
```python
class OPDTrainer:
    """Main OPD training orchestrator"""

    def __init__(self, config: OPDConfig):
        self.config = config
        self.teacher_client = TeacherClient(...)
        self.rollout_generator = RolloutGenerator(...)
        self.loss_calculator = OPDLossCalculator(...)
        self.optimizer = ...
        self.current_step = 0

    def train(self):
        """
        Main training loop:
        1. Generate rollouts from current student
        2. Get teacher logprobs
        3. Compute losses
        4. Update student adapters
        5. Evaluate periodically
        6. Save checkpoints
        """
        for step in range(self.config.num_training_steps):
            # 1. Generate on-policy rollouts
            rollouts = self.rollout_generator.generate_rollouts(...)

            # 2. Get teacher scores
            teacher_logprobs = self.teacher_client.batch_get_logprobs(...)

            # 3. Compute loss
            loss, metrics = self.loss_calculator.compute_total_loss(...)

            # 4. Backprop and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 5. Log metrics
            self.log_metrics(step, metrics)

            # 6. Evaluate
            if step % self.config.eval_every_steps == 0:
                self.evaluate()

            # 7. Checkpoint
            if step % self.config.checkpoint_every_steps == 0:
                self.save_checkpoint(step)
```

---

#### 2.6 FastAPI Endpoints (`backend/main.py` modifications)

**New endpoints to add**:

```python
# OPD Configuration
@dataclass
class OPDStartRequest:
    base_model_path: str
    student_adapter_path: str
    validation_prompts_path: str
    teacher_model_id: str = "claude-sonnet-4-5-20250929"
    num_rollouts_per_prompt: int = 3
    num_training_steps: int = 1000
    kl_weight: float = 1.0
    max_budget_usd: float = 100.0
    # ... other config fields

# Endpoints
@app.post("/opd/start")
async def start_opd_training(request: OPDStartRequest):
    """Start OPD training process"""
    # Create OPDConfig from request
    # Spawn OPD training subprocess
    # Return session ID

@app.post("/opd/stop")
async def stop_opd_training():
    """Stop running OPD training"""

@app.get("/opd/status")
async def get_opd_status():
    """Get current OPD training status and metrics"""
    return {
        "state": "running|completed|error",
        "metrics": {...},
        "current_step": int,
        "total_steps": int,
        "cumulative_cost_usd": float
    }

@app.get("/opd/metrics")
async def get_opd_metrics(run_id: str):
    """Get detailed metrics for a specific OPD run"""

@app.get("/opd/runs")
async def list_opd_runs():
    """List all OPD training runs"""

# WebSocket messages for OPD
async def broadcast_opd_progress(metrics: Dict):
    await manager.broadcast({
        "type": "opd_progress",
        "data": metrics
    })
```

---

### STEP 3: Frontend Implementation

#### 3.1 Redux Slice (`frontend/src/store/slices/opdSlice.ts`)

**Purpose**: Manage OPD state in Redux

```typescript
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface OPDMetrics {
  step: number;
  total_steps: number;
  kl_loss: number;
  kl_divergence_mean: number;
  student_entropy: number;
  teacher_alignment_pct: number;
  cumulative_cost_usd: number;
  teacher_api_latency_ms: number;
}

interface OPDConfig {
  baseModelPath: string;
  studentAdapterPath: string;
  validationPromptsPath: string;
  teacherModelId: string;
  numRolloutsPerPrompt: number;
  numTrainingSteps: number;
  klWeight: number;
  maxBudgetUsd: number;
  learningRate: number;
  batchSize: number;
  seed: number;
}

interface OPDState {
  // Configuration
  config: OPDConfig;

  // Training state
  trainingState: 'idle' | 'preparing' | 'running' | 'paused' | 'completed' | 'error';

  // Metrics
  metrics: OPDMetrics | null;
  metricsHistory: OPDMetrics[];

  // Logs
  logs: string[];

  // Run metadata
  currentRunId: string | null;
  availableAdapters: string[];

  // Budget tracking
  budgetWarning: boolean;
  budgetExceeded: boolean;
}

const initialState: OPDState = {
  config: {
    baseModelPath: '',
    studentAdapterPath: '',
    validationPromptsPath: '',
    teacherModelId: 'claude-sonnet-4-5-20250929',
    numRolloutsPerPrompt: 3,
    numTrainingSteps: 1000,
    klWeight: 1.0,
    maxBudgetUsd: 100.0,
    learningRate: 1e-5,
    batchSize: 4,
    seed: 42
  },
  trainingState: 'idle',
  metrics: null,
  metricsHistory: [],
  logs: [],
  currentRunId: null,
  availableAdapters: [],
  budgetWarning: false,
  budgetExceeded: false
};

const opdSlice = createSlice({
  name: 'opd',
  initialState,
  reducers: {
    updateConfig: (state, action: PayloadAction<Partial<OPDConfig>>) => {
      state.config = { ...state.config, ...action.payload };
    },
    setTrainingState: (state, action: PayloadAction<OPDState['trainingState']>) => {
      state.trainingState = action.payload;
    },
    updateMetrics: (state, action: PayloadAction<OPDMetrics>) => {
      state.metrics = action.payload;
      state.metricsHistory.push(action.payload);

      // Check budget warnings
      const budgetUsagePercent = (action.payload.cumulative_cost_usd / state.config.maxBudgetUsd) * 100;
      state.budgetWarning = budgetUsagePercent > 80;
      state.budgetExceeded = budgetUsagePercent >= 100;
    },
    addLog: (state, action: PayloadAction<string>) => {
      state.logs.push(action.payload);
      if (state.logs.length > 1000) {
        state.logs = state.logs.slice(-1000);
      }
    },
    setRunId: (state, action: PayloadAction<string>) => {
      state.currentRunId = action.payload;
    },
    setAvailableAdapters: (state, action: PayloadAction<string[]>) => {
      state.availableAdapters = action.payload;
    },
    resetOPD: (state) => {
      return initialState;
    }
  }
});

export const {
  updateConfig,
  setTrainingState,
  updateMetrics,
  addLog,
  setRunId,
  setAvailableAdapters,
  resetOPD
} = opdSlice.actions;

export default opdSlice.reducer;
```

#### 3.2 Update Store Configuration (`frontend/src/store/store.ts`)

```typescript
// Add to existing store configuration
import opdReducer from './slices/opdSlice';

export const store = configureStore({
  reducer: {
    training: trainingReducer,
    models: modelsReducer,
    ui: uiReducer,
    opd: opdReducer,  // NEW
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

---

#### 3.3 OPD Page (`frontend/src/pages/OPDPage.tsx`)

**Purpose**: Main OPD page with setup, monitoring, and results

```typescript
import React, { useEffect, useState } from 'react';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import OPDSetupPanel from '../components/opd/OPDSetupPanel';
import OPDProgressMonitor from '../components/opd/OPDProgressMonitor';
import OPDResultsPanel from '../components/opd/OPDResultsPanel';
import { setTrainingState } from '../store/slices/opdSlice';

const OPDPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const trainingState = useAppSelector(state => state.opd.trainingState);

  return (
    <div className="flex flex-col h-full p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">On-Policy Distillation</h1>
          <p className="text-sm text-gray-500 mt-1">
            Distill your fine-tuned model using Claude Sonnet 4.5 as teacher
          </p>
        </div>

        {/* Status Badge */}
        <div className={`px-4 py-2 rounded-lg font-semibold ${
          trainingState === 'running' ? 'bg-blue-100 text-blue-700' :
          trainingState === 'completed' ? 'bg-green-100 text-green-700' :
          trainingState === 'error' ? 'bg-red-100 text-red-700' :
          'bg-gray-100 text-gray-700'
        }`}>
          {trainingState.toUpperCase()}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        {trainingState === 'idle' || trainingState === 'preparing' ? (
          <OPDSetupPanel />
        ) : trainingState === 'running' || trainingState === 'paused' ? (
          <OPDProgressMonitor />
        ) : (
          <OPDResultsPanel />
        )}
      </div>
    </div>
  );
};

export default OPDPage;
```

---

#### 3.4 OPD Setup Panel (`frontend/src/components/opd/OPDSetupPanel.tsx`)

**Purpose**: Configuration form for OPD training

```typescript
import React from 'react';
import { useAppSelector, useAppDispatch } from '../../store/hooks';
import { updateConfig, setTrainingState } from '../../store/slices/opdSlice';

const OPDSetupPanel: React.FC = () => {
  const dispatch = useAppDispatch();
  const config = useAppSelector(state => state.opd.config);
  const availableAdapters = useAppSelector(state => state.opd.availableAdapters);

  const handleStartTraining = async () => {
    // Validate config
    if (!config.studentAdapterPath || !config.validationPromptsPath) {
      alert('Please select a student adapter and validation prompts');
      return;
    }

    // Start OPD training
    dispatch(setTrainingState('preparing'));

    try {
      const response = await fetch('http://localhost:8000/opd/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (response.ok) {
        const data = await response.json();
        dispatch(setRunId(data.run_id));
        dispatch(setTrainingState('running'));
      } else {
        throw new Error('Failed to start OPD training');
      }
    } catch (error) {
      console.error(error);
      dispatch(setTrainingState('error'));
    }
  };

  return (
    <div className="space-y-6">
      {/* Model Selection */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Model Configuration</h2>

        {/* Student Adapter Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Student Model (Fine-tuned Adapter)
          </label>
          <select
            value={config.studentAdapterPath}
            onChange={(e) => dispatch(updateConfig({ studentAdapterPath: e.target.value }))}
            className="w-full p-2 border rounded"
          >
            <option value="">Select adapter...</option>
            {availableAdapters.map(adapter => (
              <option key={adapter} value={adapter}>{adapter}</option>
            ))}
          </select>
        </div>

        {/* Teacher Model */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Teacher Model
          </label>
          <select
            value={config.teacherModelId}
            onChange={(e) => dispatch(updateConfig({ teacherModelId: e.target.value }))}
            className="w-full p-2 border rounded"
          >
            <option value="claude-sonnet-4-5-20250929">Claude Sonnet 4.5</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            Teacher model provides token-level guidance for distillation
          </p>
        </div>
      </div>

      {/* Training Parameters */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Training Parameters</h2>

        <div className="grid grid-cols-2 gap-4">
          {/* KL Weight */}
          <div>
            <label className="block text-sm font-medium mb-2">
              KL Weight
            </label>
            <input
              type="number"
              step="0.1"
              value={config.klWeight}
              onChange={(e) => dispatch(updateConfig({ klWeight: parseFloat(e.target.value) }))}
              className="w-full p-2 border rounded"
            />
          </div>

          {/* Training Steps */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Training Steps
            </label>
            <input
              type="number"
              value={config.numTrainingSteps}
              onChange={(e) => dispatch(updateConfig({ numTrainingSteps: parseInt(e.target.value) }))}
              className="w-full p-2 border rounded"
            />
          </div>

          {/* Learning Rate */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Learning Rate
            </label>
            <input
              type="number"
              step="0.00001"
              value={config.learningRate}
              onChange={(e) => dispatch(updateConfig({ learningRate: parseFloat(e.target.value) }))}
              className="w-full p-2 border rounded"
            />
          </div>

          {/* Batch Size */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Batch Size
            </label>
            <input
              type="number"
              value={config.batchSize}
              onChange={(e) => dispatch(updateConfig({ batchSize: parseInt(e.target.value) }))}
              className="w-full p-2 border rounded"
            />
          </div>
        </div>
      </div>

      {/* Budget Configuration */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Budget & Safety</h2>

        <div>
          <label className="block text-sm font-medium mb-2">
            Maximum Budget (USD)
          </label>
          <input
            type="number"
            step="10"
            value={config.maxBudgetUsd}
            onChange={(e) => dispatch(updateConfig({ maxBudgetUsd: parseFloat(e.target.value) }))}
            className="w-full p-2 border rounded"
          />
          <p className="text-xs text-gray-500 mt-1">
            Training will stop if teacher API costs exceed this amount
          </p>
        </div>
      </div>

      {/* Start Button */}
      <div className="flex justify-end">
        <button
          onClick={handleStartTraining}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700"
        >
          Start OPD Training
        </button>
      </div>
    </div>
  );
};

export default OPDSetupPanel;
```

---

#### 3.5 OPD Progress Monitor (`frontend/src/components/opd/OPDProgressMonitor.tsx`)

**Purpose**: Real-time training progress display

```typescript
import React, { useEffect } from 'react';
import { useAppSelector } from '../../store/hooks';
import { Line } from 'react-chartjs-2';
import TeacherCostTracker from './TeacherCostTracker';

const OPDProgressMonitor: React.FC = () => {
  const metrics = useAppSelector(state => state.opd.metrics);
  const metricsHistory = useAppSelector(state => state.opd.metricsHistory);
  const logs = useAppSelector(state => state.opd.logs);

  // Chart data
  const chartData = {
    labels: metricsHistory.map(m => m.step),
    datasets: [
      {
        label: 'KL Loss',
        data: metricsHistory.map(m => m.kl_loss),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
      },
      {
        label: 'Teacher Alignment %',
        data: metricsHistory.map(m => m.teacher_alignment_pct),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
      }
    ]
  };

  return (
    <div className="space-y-6">
      {/* Progress Bar */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium">Training Progress</span>
          <span className="text-sm text-gray-500">
            Step {metrics?.step || 0} / {metrics?.total_steps || 0}
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all"
            style={{ width: `${((metrics?.step || 0) / (metrics?.total_steps || 1)) * 100}%` }}
          />
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <div className="text-sm text-gray-500">KL Loss</div>
          <div className="text-2xl font-bold">{metrics?.kl_loss.toFixed(4) || '-'}</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <div className="text-sm text-gray-500">Teacher Alignment</div>
          <div className="text-2xl font-bold">{metrics?.teacher_alignment_pct.toFixed(1) || '-'}%</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <div className="text-sm text-gray-500">Student Entropy</div>
          <div className="text-2xl font-bold">{metrics?.student_entropy.toFixed(2) || '-'}</div>
        </div>
      </div>

      {/* Cost Tracker */}
      <TeacherCostTracker />

      {/* Charts */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Training Metrics</h3>
        <Line data={chartData} options={{ responsive: true }} />
      </div>

      {/* Logs */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Training Logs</h3>
        <div className="bg-gray-900 text-green-400 p-4 rounded font-mono text-xs h-64 overflow-y-auto">
          {logs.slice(-50).map((log, idx) => (
            <div key={idx}>{log}</div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default OPDProgressMonitor;
```

---

#### 3.6 Teacher Cost Tracker (`frontend/src/components/opd/TeacherCostTracker.tsx`)

```typescript
import React from 'react';
import { useAppSelector } from '../../store/hooks';
import { AlertTriangle, DollarSign } from 'lucide-react';

const TeacherCostTracker: React.FC = () => {
  const metrics = useAppSelector(state => state.opd.metrics);
  const maxBudget = useAppSelector(state => state.opd.config.maxBudgetUsd);
  const budgetWarning = useAppSelector(state => state.opd.budgetWarning);
  const budgetExceeded = useAppSelector(state => state.opd.budgetExceeded);

  const currentCost = metrics?.cumulative_cost_usd || 0;
  const usagePercent = (currentCost / maxBudget) * 100;

  return (
    <div className={`bg-white rounded-lg shadow p-6 ${
      budgetExceeded ? 'border-2 border-red-500' :
      budgetWarning ? 'border-2 border-yellow-500' : ''
    }`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center">
          <DollarSign className="w-5 h-5 mr-2" />
          Teacher API Cost
        </h3>
        {(budgetWarning || budgetExceeded) && (
          <AlertTriangle className={`w-5 h-5 ${budgetExceeded ? 'text-red-500' : 'text-yellow-500'}`} />
        )}
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Current Spend</span>
          <span className="font-semibold">${currentCost.toFixed(2)}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span>Budget Limit</span>
          <span>${maxBudget.toFixed(2)}</span>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
          <div
            className={`h-2 rounded-full transition-all ${
              budgetExceeded ? 'bg-red-600' :
              budgetWarning ? 'bg-yellow-600' : 'bg-green-600'
            }`}
            style={{ width: `${Math.min(usagePercent, 100)}%` }}
          />
        </div>

        <div className="text-xs text-gray-500 text-right">
          {usagePercent.toFixed(1)}% used
        </div>
      </div>

      {budgetExceeded && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
          <strong>Budget Exceeded!</strong> Training has been stopped to prevent further costs.
        </div>
      )}
    </div>
  );
};

export default TeacherCostTracker;
```

---

#### 3.7 Update Navigation (`frontend/src/components/Sidebar.tsx`)

```typescript
// Add to navItems array
const navItems = [
  { icon: Wrench, label: 'Setup', path: '/setup' },
  { icon: Play, label: 'Training', path: '/training' },
  { icon: BarChart3, label: 'Results', path: '/results' },
  { icon: Layers, label: 'OPD Distillation', path: '/opd' },  // NEW
  { icon: FileText, label: 'Compare', path: '/compare' },
];
```

**Update App Router** (`frontend/src/App.tsx`):
```typescript
import OPDPage from './pages/OPDPage';

// In Routes
<Route path="/opd" element={<OPDPage />} />
```

---

## Integration Points

### 1. Post-SFT Training Hook
When SFT training completes, offer OPD as next step:

```typescript
// In TrainingPage.tsx completion handler
if (trainingState === 'completed') {
  showNotification({
    title: 'Training Complete!',
    message: 'Would you like to run On-Policy Distillation?',
    actions: [
      { label: 'Start OPD', onClick: () => navigate('/opd') },
      { label: 'View Results', onClick: () => navigate('/results') }
    ]
  });
}
```

### 2. Model Comparison Extension
Update ComparePage to support 3-way comparison:
- Base Model
- Fine-tuned Model (SFT)
- Distilled Model (OPD)

### 3. Session Management
Extend session storage to include OPD runs:

```json
{
  "session_id": "uuid",
  "sft_run": { ... },
  "opd_runs": [
    {
      "run_id": "opd_uuid",
      "config": { ... },
      "metrics": { ... },
      "final_adapter_path": "..."
    }
  ]
}
```

---

## Testing Strategy

### Unit Tests

**Backend Tests** (`backend/tests/`):
```
test_teacher_client.py
  ✓ test_cache_hit_no_api_call()
  ✓ test_budget_enforcement()
  ✓ test_retry_on_rate_limit()
  ✓ test_token_alignment()

test_rollout_generator.py
  ✓ test_deterministic_sampling()
  ✓ test_rollout_serialization()

test_loss_calculator.py
  ✓ test_reverse_kl_computation()
  ✓ test_warmup_schedule()
  ✓ test_sft_mixup()

test_trainer.py
  ✓ test_checkpoint_save_load()
  ✓ test_early_stopping()
```

**Frontend Tests** (`frontend/src/__tests__/`):
```
opd/OPDSetupPanel.test.tsx
  ✓ test_config_validation()
  ✓ test_adapter_selection()

opd/TeacherCostTracker.test.tsx
  ✓ test_budget_warning_threshold()
  ✓ test_budget_exceeded_alert()
```

### Integration Tests

```
test_end_to_end_opd_workflow.py
  1. Start with fine-tuned adapter
  2. Configure OPD with small budget
  3. Run 10 training steps
  4. Verify metrics logged
  5. Verify checkpoint saved
  6. Verify cost tracking accurate
```

### Manual Testing Checklist

- [ ] Load existing fine-tuned adapter
- [ ] Configure OPD with test budget ($1)
- [ ] Start training and verify:
  - [ ] Rollouts generated correctly
  - [ ] Teacher API called successfully
  - [ ] Metrics updated in real-time
  - [ ] Cost tracker accurate
  - [ ] Budget limit enforced
  - [ ] Checkpoints saved
  - [ ] Training can be stopped/resumed
- [ ] Verify distilled model loads correctly
- [ ] Compare base vs SFT vs OPD outputs

---

## Configuration Reference

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional (with defaults)
OPD_DEFAULT_TEACHER=claude-sonnet-4-5-20250929
OPD_MAX_BUDGET_USD=100.0
OPD_RATE_LIMIT_TPM=10000
OPD_CACHE_DIR=./OnPolicyDistill/teacher_logprobs
OPD_CHECKPOINT_DIR=./OnPolicyDistill/checkpoints
```

### OPD Config File Format (YAML)

```yaml
# OnPolicyDistill/configs/{run_id}.yaml
run_id: "opd_20250128_123456"
session_id: "sft_session_uuid"

models:
  base_model_path: "/path/to/base/model"
  student_adapter_path: "/path/to/fine-tuned/adapter"
  teacher_model_id: "claude-sonnet-4-5-20250929"

data:
  validation_prompts_path: "/path/to/val_prompts.jsonl"
  rollout_output_dir: "./OnPolicyDistill/rollouts/train"
  val_rollout_dir: "./OnPolicyDistill/rollouts/val"

training:
  num_rollouts_per_prompt: 3
  rollout_max_tokens: 512
  batch_size: 4
  num_training_steps: 1000
  learning_rate: 0.00001

loss:
  kl_weight: 1.0
  kl_schedule: "warmup"
  kl_warmup_steps: 100
  use_sft_mixup: false
  sft_weight: 0.1

teacher:
  max_concurrent: 5
  rate_limit_tpm: 10000
  timeout_seconds: 30
  max_budget_usd: 100.0

checkpointing:
  checkpoint_dir: "./OnPolicyDistill/checkpoints"
  checkpoint_every_steps: 100
  save_best_only: false

evaluation:
  eval_every_steps: 100
  eval_max_samples: 100

reproducibility:
  seed: 42
```

---

## API Reference

### POST /opd/start

**Request**:
```json
{
  "base_model_path": "/path/to/model",
  "student_adapter_path": "/path/to/adapter",
  "validation_prompts_path": "/path/to/prompts.jsonl",
  "teacher_model_id": "claude-sonnet-4-5-20250929",
  "num_training_steps": 1000,
  "kl_weight": 1.0,
  "max_budget_usd": 100.0
}
```

**Response**:
```json
{
  "status": "success",
  "run_id": "opd_20250128_123456",
  "message": "OPD training started"
}
```

### GET /opd/status

**Response**:
```json
{
  "state": "running",
  "run_id": "opd_20250128_123456",
  "metrics": {
    "step": 250,
    "total_steps": 1000,
    "kl_loss": 0.1234,
    "cumulative_cost_usd": 5.67
  },
  "estimated_time_remaining_seconds": 1200
}
```

### POST /opd/stop

**Response**:
```json
{
  "status": "success",
  "message": "OPD training stopped",
  "final_step": 250,
  "checkpoint_path": "./OnPolicyDistill/checkpoints/step_250"
}
```

### GET /opd/metrics?run_id={run_id}

**Response**:
```json
{
  "run_id": "opd_20250128_123456",
  "metrics_history": [
    {
      "step": 0,
      "kl_loss": 0.5,
      "teacher_alignment_pct": 45.2,
      ...
    },
    ...
  ],
  "final_metrics": { ... }
}
```

---

## Rollout Plan

### Phase 0: Local Testing (Week 1)
- [ ] Set up dev environment
- [ ] Implement core backend modules
- [ ] Add unit tests
- [ ] Test with cached teacher responses (no API)

### Phase 1: API Integration (Week 2)
- [ ] Integrate Anthropic API
- [ ] Test with small budget ($1-5)
- [ ] Validate cost tracking
- [ ] Measure API latency

### Phase 2: Frontend Development (Week 2-3)
- [ ] Implement Redux slice
- [ ] Build OPD page and components
- [ ] Add charts and monitoring
- [ ] Test navigation flow

### Phase 3: Integration Testing (Week 3)
- [ ] End-to-end workflow test
- [ ] Budget guardrail validation
- [ ] Resume training test
- [ ] Model comparison test

### Phase 4: Production Deployment (Week 4)
- [ ] Documentation
- [ ] User guide
- [ ] Performance tuning
- [ ] Launch to users

---

## Open Questions & Decisions Needed

### Critical Decisions

1. **Teacher API Logprobs Access**
   - ❓ Does Anthropic Claude API support token-level logprobs?
   - ⚠️ **BLOCKER**: If not available, need alternative approach:
     - Option A: Use completion quality scoring instead
     - Option B: Use preference ranking (pairwise)
     - Option C: Wait for API feature

2. **Validation Prompts Source**
   - Where do validation prompts come from?
   - Should we auto-split user's training data?
   - Or require separate validation file?

3. **Default Teacher Prompt Template**
   - What system prompt to use for teacher?
   - Should it match the student's fine-tuning task?
   - Configurable per run or global default?

4. **Extractor Function `f`**
   - How to extract "final answer" from teacher output?
   - Regex patterns? XML tags? JSON?
   - Task-specific or generic?

5. **Primary Success Metric**
   - What's the canonical metric to gate success?
   - KL divergence reduction?
   - Task-specific accuracy?
   - User preference rating?

### Implementation Details

6. **Tokenizer Alignment**
   - How to handle tokenization differences between MLX and Claude?
   - Build alignment map? Re-tokenize? Skip mismatches?

7. **Checkpoint Format**
   - Same as SFT (SafeTensors)?
   - Additional metadata needed?

8. **Cost Estimation**
   - Provide pre-run cost estimate?
   - Based on prompt count × avg tokens × API pricing?

9. **Error Handling**
   - What happens if teacher API fails mid-training?
   - Resume from last checkpoint? Skip failed samples?

10. **Multi-GPU Support**
    - Should OPD support multi-GPU like SFT?
    - Or single GPU sufficient given API bottleneck?

---

## Dependencies

### New Python Packages
```
anthropic>=0.18.0
tiktoken>=0.5.0
tenacity>=8.2.0
```

### New Node Packages
```
# None - existing packages sufficient
```

### System Requirements
- Anthropic API key with credits
- Stable internet connection for API calls
- Existing MLX environment from SFT

---

## File Checklist

### Backend Files (New)
- [ ] `backend/opd/__init__.py`
- [ ] `backend/opd/config.py`
- [ ] `backend/opd/teacher_client.py`
- [ ] `backend/opd/rollout_generator.py`
- [ ] `backend/opd/loss_calculator.py`
- [ ] `backend/opd/trainer.py`
- [ ] `backend/opd/evaluator.py`
- [ ] `backend/opd/extractor.py`
- [ ] `backend/opd/utils.py`

### Backend Files (Modified)
- [ ] `backend/main.py` - Add OPD endpoints
- [ ] `backend/requirements.txt` - Add dependencies

### Frontend Files (New)
- [ ] `frontend/src/pages/OPDPage.tsx`
- [ ] `frontend/src/components/opd/OPDSetupPanel.tsx`
- [ ] `frontend/src/components/opd/OPDProgressMonitor.tsx`
- [ ] `frontend/src/components/opd/OPDMetricsChart.tsx`
- [ ] `frontend/src/components/opd/OPDResultsPanel.tsx`
- [ ] `frontend/src/components/opd/TeacherCostTracker.tsx`
- [ ] `frontend/src/store/slices/opdSlice.ts`

### Frontend Files (Modified)
- [ ] `frontend/src/store/store.ts` - Add opdReducer
- [ ] `frontend/src/components/Sidebar.tsx` - Add nav item
- [ ] `frontend/src/App.tsx` - Add route

### Documentation Files
- [ ] `docs/OPD_USER_GUIDE.md`
- [ ] `docs/OPD_API_REFERENCE.md`
- [ ] `docs/OPD_TROUBLESHOOTING.md`

### Test Files
- [ ] `backend/tests/test_teacher_client.py`
- [ ] `backend/tests/test_rollout_generator.py`
- [ ] `backend/tests/test_loss_calculator.py`
- [ ] `backend/tests/test_trainer.py`
- [ ] `backend/tests/test_end_to_end_opd.py`
- [ ] `frontend/src/__tests__/opd/OPDSetupPanel.test.tsx`
- [ ] `frontend/src/__tests__/opd/TeacherCostTracker.test.tsx`

---

## Success Criteria

### Must Have (MVP)
✅ User can start OPD from GUI
✅ Teacher API integration works
✅ Budget tracking and enforcement
✅ Training produces distilled adapter
✅ Metrics logged and displayed
✅ Cost under control (no runaway spend)

### Should Have (v1.1)
⭐ Resume training from checkpoint
⭐ Comparison page shows 3 models
⭐ Validation metrics on held-out set
⭐ Export training report

### Nice to Have (Future)
💡 Multiple teacher models
💡 Curriculum learning schedules
💡 Automated hyperparameter tuning
💡 A/B testing between distilled models

---

## Estimated Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 0: Prep | 1-2 hours | Environment, directories, dependencies |
| Phase 1: Backend Core | 8-12 hours | Teacher client, rollout gen, loss calc |
| Phase 2: Backend Training | 8-12 hours | Trainer, evaluator, checkpointing |
| Phase 3: Backend API | 4-6 hours | FastAPI endpoints, WebSocket |
| Phase 4: Frontend State | 4-6 hours | Redux slice, hooks |
| Phase 5: Frontend UI | 8-12 hours | Pages, components, charts |
| Phase 6: Integration | 6-8 hours | Connect frontend ↔ backend |
| Phase 7: Testing | 8-12 hours | Unit, integration, E2E tests |
| **Total** | **47-70 hours** | **~1-2 weeks full-time** |

---

## Next Steps

1. **Review and Approve This Plan**
   - Validate architectural decisions
   - Answer open questions (especially teacher API logprobs)
   - Confirm budget and timeline

2. **Set Up Development Environment**
   - Get Anthropic API key
   - Install dependencies
   - Create directory structure

3. **Start Implementation**
   - Begin with Phase 1 (Backend Core)
   - Implement teacher client with mock/cache first
   - Build incrementally with tests

4. **Iterative Development**
   - Complete each phase before moving to next
   - Test continuously
   - Document as you go

---

## Contact & Support

**Implementation Questions**: Refer to this document

**API Issues**: https://docs.anthropic.com/claude/reference

**MLX Documentation**: https://ml-explore.github.io/mlx/

---

**Document Version**: 1.0
**Date**: January 28, 2025
**Status**: AWAITING APPROVAL

---

# Summary

This implementation plan provides:

1. ✅ **Complete architecture** aligned with existing Droid-FineTuning patterns
2. ✅ **Step-by-step implementation guide** for all components
3. ✅ **Detailed code outlines** for key modules
4. ✅ **Integration strategy** with existing SFT workflow
5. ✅ **Testing plan** covering unit, integration, and E2E
6. ✅ **Configuration reference** for all settings
7. ✅ **Rollout plan** with phases and timeline
8. ✅ **Open questions** that need decisions before implementation

**Ready to proceed upon approval!**
