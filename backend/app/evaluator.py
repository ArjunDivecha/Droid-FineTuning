"""
=============================================================================
SCRIPT NAME: evaluator.py
=============================================================================

INPUT FILES:
- <root>/adapters/<name>/adapters.safetensors : trained LoRA weights (Tier 0)
- <root>/adapters/<name>/adapter_config.json  : LoRA config (rank/scale)
- <root>/data/sft_b_train_split/valid.jsonl   : validation samples (Tier 1)
- <root>/.env                                  : DEEPSEEK_API_KEY (LLM judge)
- any LM Studio / HF MLX model dir            : base model + tokenizer

OUTPUT FILES:
- (none written; returns dicts to the FastAPI layer)

VERSION: 1.0
LAST UPDATED: 2026-06-21
AUTHOR: Droid-FineTuning

DESCRIPTION:
Self-contained evaluation engine for the Compare tab, reimplemented for the
v2 (mlx-lm-lora) backend. Replaces the legacy combined_evaluator.py, which
relied on hardcoded developer paths and an external Cerebras judge.

Three capabilities, matching the frontend's existing contract:

  Tier 1 (Perplexity)  — load model (+ optional adapter), compute average
                         cross-entropy loss + perplexity on validation text.
                         Works for BOTH base model and adapter. Lower
                         perplexity -> higher quality_score.

  Tier 0 (Mathematical) — analyze LoRA adapter weight matrices (spectral
                          norm, effective rank, L2 norm, sparsity,
                          concentration). Adapter-only. Maps to quality_score.

  LLM Judge            — generate N QA responses from the model, then ask
                         DeepSeek (deepseek-v4-flash) to score each on
                         faithfulness / fact_recall / consistency /
                         hallucination. Returns the `scores` object the
                         frontend's "Evaluate (LLM Judge)" button renders.

Scoring scales are 0-100 with letter grades A/B/C/D/F, matching the
EvaluationCard component.

DEPENDENCIES:
- mlx, mlx_lm, numpy, safetensors (already in backend venv)
- urllib.request (stdlib) for DeepSeek calls — no new deps
=============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from . import paths

logger = logging.getLogger("droid.evaluator")

# --------------------------------------------------------------------------- #
# Env loading (no python-dotenv dependency)
# --------------------------------------------------------------------------- #
def _load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (best-effort)."""
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(paths.PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Validation text source for Tier 1 perplexity. Prefer the train/valid split;
# fall back to the generated sample data; finally to a tiny built-in corpus.
_VALID_CANDIDATES = [
    paths.DATA_DIR / "sft_b_train_split" / "valid.jsonl",
    paths.DATA_DIR / "sft_b_train_split" / "train.jsonl",
    paths.DATA_DIR / "sample_20260620_233351.jsonl",
]


# --------------------------------------------------------------------------- #
# Helpers: grading, validation text, model resolution
# --------------------------------------------------------------------------- #
def _sanitize_text(text: str) -> str:
    """Strip control characters that would break JSON transport.

    Model generations can contain raw control chars (e.g. vertical tabs) that
    some strict JSON parsers reject. Keep newlines/tabs (valid in JSON strings
    when escaped by the encoder) but drop other C0 control chars.
    """
    if not text:
        return ""
    return "".join(
        ch for ch in text if ch in "\n\t" or (ord(ch) >= 0x20 and ord(ch) != 0x7F)
    ).strip()


def _grade(score: float) -> str:
    """Map a 0-100 score to a letter grade."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _load_validation_texts(max_samples: int) -> List[str]:
    """Return up to `max_samples` text strings for perplexity evaluation."""
    for cand in _VALID_CANDIDATES:
        if not cand.is_file():
            continue
        texts: List[str] = []
        try:
            with open(cand) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    # Support both {"text": "..."} and {"messages": [...]}
                    if "text" in rec and isinstance(rec["text"], str):
                        texts.append(rec["text"])
                    elif "messages" in rec and isinstance(rec["messages"], list):
                        rendered = _render_messages(rec["messages"])
                        if rendered:
                            texts.append(rendered)
                    if len(texts) >= max_samples:
                        break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed reading validation file %s: %s", cand, exc)
            continue
        if texts:
            return texts[:max_samples]
    # Last-resort fallback so Tier 1 always returns something meaningful.
    return [
        "User: What is 2 plus 2?\nAssistant: 2 plus 2 is 4.",
        "User: Say hello.\nAssistant: Hello! How can I help you today?",
        "User: What is the capital of France?\nAssistant: The capital of France is Paris.",
    ][:max_samples]


def _render_messages(messages: List[Dict[str, Any]]) -> str:
    """Render a chat-style {messages:[...]} record into a single text string."""
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.capitalize()}: {content}")
    return "\n".join(parts)


def _resolve_adapter_path(adapter_name: Optional[str]) -> Optional[str]:
    """Resolve an adapter name to an on-disk adapter directory."""
    if not adapter_name:
        return None
    p = paths.ADAPTERS_DIR / adapter_name
    return str(p) if p.is_dir() else None


def _resolve_model_and_adapter(
    adapter_name: Optional[str], include_adapter: bool
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve model + adapter for evaluation.

    Mirrors main._resolve_model_and_adapter but kept local so the evaluator
    stays self-contained. Falls back to the latest run's config.yaml on disk.
    """
    adapter_path = _resolve_adapter_path(adapter_name) if include_adapter else None

    # Try the latest run's config.yaml for the base model path (durable).
    model_path: Optional[str] = None
    try:
        import yaml  # type: ignore

        if paths.RUNS_DIR.is_dir():
            for run_dir in sorted(
                (p for p in paths.RUNS_DIR.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                cfg = run_dir / "config.yaml"
                if not cfg.is_file():
                    continue
                with open(cfg) as f:
                    data = yaml.safe_load(f) or {}
                m = data.get("model") or data.get("model_path")
                if m:
                    model_path = m
                    # Prefer the run whose adapter matches, if any.
                    ra = data.get("adapter_path") or data.get("adapter_name")
                    if ra and include_adapter and not adapter_path:
                        resolved = str(ra) if Path(str(ra)).is_dir() else str(paths.ADAPTERS_DIR / str(ra))
                        if Path(resolved).is_dir():
                            adapter_path = resolved
                    break
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read latest run config for evaluation: %s", exc)

    return model_path, adapter_path


# --------------------------------------------------------------------------- #
# Tier 1: Perplexity / average loss
# --------------------------------------------------------------------------- #
def _compute_loss(model, tokenizer, text: str) -> Tuple[float, int]:
    """Return (summed_cross_entropy_loss, num_predicted_tokens) for one text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 2:
        return 0.0, 0
    x = mx.array([tokens])
    logits = model(x)  # (1, seq, vocab)
    # Predict token t+1 from logits at position t.
    logit_seq = logits[0, :-1, :].astype(mx.float32)
    target = x[0, 1:]

    # Cross-entropy: log_softmax then gather, numerically stable on MLX.
    # mlx.core has no log_softmax in this version; build from softmax+log.
    log_probs = mx.log(mx.softmax(logit_seq, axis=-1) + 1e-12)
    chosen = mx.take_along_axis(
        log_probs, target.reshape(-1, 1), axis=-1
    ).squeeze(-1)
    loss_sum = float(mx.sum(chosen))
    n = int(target.shape[0])
    return -loss_sum, n


def evaluate_tier1(
    model_path: str,
    adapter_path: Optional[str] = None,
    max_samples: int = 20,
) -> Dict[str, Any]:
    """Run Tier 1 perplexity evaluation.

    Returns: {quality_score, grade, perplexity, avg_loss, time_seconds}
    """
    from mlx_lm import load

    t0 = time.perf_counter()
    logger.info("Tier 1: loading %s (adapter=%s)", model_path, adapter_path)
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    texts = _load_validation_texts(max_samples)
    logger.info("Tier 1: evaluating %d validation samples", len(texts))

    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        try:
            ls, nt = _compute_loss(model, tokenizer, text)
            total_loss += ls
            total_tokens += nt
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tier 1: skipped sample (%s)", exc)
            continue

    elapsed = time.perf_counter() - t0
    if total_tokens == 0:
        avg_loss = float("nan")
        perplexity = float("nan")
        quality_score = 0.0
    else:
        avg_loss = total_loss / total_tokens
        perplexity = float(mx.exp(mx.array(avg_loss)))
        # Map perplexity to a 0-100 score.
        # ppl=1 -> 100 (perfect); ppl~6 -> ~67; ppl>30 -> low.
        # Score = 100 * exp(-(ppl-1)/k), k chosen so ppl=10 -> ~50.
        quality_score = 100.0 * float(mx.exp(-mx.array((perplexity - 1.0) / 12.0)))
        quality_score = max(0.0, min(100.0, quality_score))

    return {
        "quality_score": round(quality_score, 1),
        "grade": _grade(quality_score),
        "perplexity": float(perplexity) if perplexity == perplexity else 0.0,
        "avg_loss": float(avg_loss) if avg_loss == avg_loss else 0.0,
        "time_seconds": round(elapsed, 2),
    }


# --------------------------------------------------------------------------- #
# Tier 0: Mathematical analysis of LoRA weights
# --------------------------------------------------------------------------- #
def _load_lora_updates(adapter_path: str) -> List[np.ndarray]:
    """Load every (lora_b @ lora_a) effective update matrix from a LoRA dir.

    Returns a list of 2D numpy arrays. Falls back to the raw lora_a / lora_b
    matrices if pairing fails.
    """
    # Prefer the final adapters.safetensors; else the newest checkpoint.
    safetensors = sorted(
        Path(adapter_path).glob("*.safetensors"),
        key=lambda p: (p.name != "adapters.safetensors", -p.stat().st_mtime),
    )
    if not safetensors:
        raise FileNotFoundError(f"No .safetensors in {adapter_path}")
    path = safetensors[0]

    a_mats: Dict[str, np.ndarray] = {}
    b_mats: Dict[str, np.ndarray] = {}
    singles: List[np.ndarray] = []
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            arr = np.asarray(f.get_tensor(key))
            if arr.ndim != 2:
                continue
            if key.endswith(".lora_a"):
                a_mats[key[: -len(".lora_a")]] = arr
            elif key.endswith(".lora_b"):
                b_mats[key[: -len(".lora_b")]] = arr
            else:
                singles.append(arr)

    updates: List[np.ndarray] = []
    for stem in a_mats.keys() & b_mats.keys():
        a = a_mats[stem]  # (in, r)
        b = b_mats[stem]  # (r, out)
        # LoRA forward is scale * (x @ A) @ B, so the effective weight delta
        # (matching the base weight's (in, out) shape) is A @ B.
        updates.append(a @ b)  # (in, out) effective update
    if not updates:
        updates = singles or list(a_mats.values()) + list(b_mats.values())
    return updates


def _spectral_stats(matrices: List[np.ndarray]) -> Dict[str, float]:
    """Compute spectral/L2/sparsity/effective-rank stats over a set of matrices."""
    all_singular: List[np.ndarray] = []
    per_matrix_effrank: List[float] = []
    l2_total = 0.0
    n_elements = 0
    n_nearzero = 0
    for m in matrices:
        # Singular values (clamped matrix size for speed on huge layers).
        if m.shape[0] > 512 or m.shape[1] > 512:
            # Subsample rows/cols for spectral estimate on very large matrices.
            idx = np.random.default_rng(0).choice(
                m.shape[0], size=min(512, m.shape[0]), replace=False
            )
            sub = m[idx[:], : min(512, m.shape[1])]
            sv = np.linalg.svd(sub, compute_uv=False)
        else:
            sv = np.linalg.svd(m, compute_uv=False)
        all_singular.append(sv)
        l2_total += float(np.linalg.norm(m))
        n_elements += m.size
        n_nearzero += int(np.sum(np.abs(m) < 1e-6))

        # Per-matrix effective rank (entropy of normalized singular values),
        # then averaged across matrices so stacking many layers doesn't inflate it.
        s_pos = sv[sv > 0]
        if s_pos.size:
            s_norm = s_pos / (s_pos.sum() + 1e-12)
            per_matrix_effrank.append(
                float(np.exp(-np.sum(s_norm * np.log(s_norm))))
            )

    svs = np.concatenate(all_singular) if all_singular else np.array([0.0])
    spectral_norm = float(np.max(svs)) if svs.size else 0.0
    l2_norm = l2_total
    sparsity = (n_nearzero / n_elements) if n_elements else 0.0

    # Average per-matrix effective rank; for rank-r LoRA this is <= r.
    eff_rank = (
        float(np.mean(per_matrix_effrank)) if per_matrix_effrank else 0.0
    )

    # Concentration: share of energy in the top singular value.
    concentration = float(svs[0] / (np.sum(svs) + 1e-12)) if svs.size else 0.0

    return {
        "spectral_norm": spectral_norm,
        "effective_rank": eff_rank,
        "concentration": concentration,
        "l2_norm": l2_norm,
        "sparsity": sparsity,
    }


def evaluate_tier0(adapter_path: str) -> Dict[str, Any]:
    """Run Tier 0 mathematical analysis on a LoRA adapter.

    Returns: {quality_score, grade, spectral_norm, effective_rank,
              concentration, l2_norm, sparsity, warnings, time_seconds}
    """
    t0 = time.perf_counter()
    updates = _load_lora_updates(adapter_path)
    stats = _spectral_stats(updates)

    # Score composition (0-100). These heuristics reward healthy, non-degenerate,
    # well-distributed LoRA updates. They are intentionally transparent.
    eff_rank = stats["effective_rank"]
    concentration = stats["concentration"]
    spectral = stats["spectral_norm"]
    l2 = stats["l2_norm"]

    # Effective rank component: more ranks used -> better (cap ~32 for r=8 stacks).
    rank_score = min(100.0, (eff_rank / 32.0) * 100.0)
    # Concentration component: moderate concentration is healthy; near 1.0 means
    # one direction dominates (degenerate); near 0 means flat/noise.
    conc_score = 100.0 * (1.0 - abs(concentration - 0.4) / 0.6)
    conc_score = max(0.0, min(100.0, conc_score))
    # Magnitude component: non-trivial L2 means the adapter learned something.
    mag_score = min(100.0, (l2 / 50.0) * 100.0)
    # Spectral component: penalize exploding or vanishing top singular value.
    if spectral > 0:
        spec_score = 100.0 * float(np.exp(-((spectral - 2.0) ** 2) / 8.0))
    else:
        spec_score = 0.0

    quality_score = 0.30 * rank_score + 0.25 * conc_score + 0.25 * mag_score + 0.20 * spec_score
    quality_score = max(0.0, min(100.0, quality_score))

    warnings: List[str] = []
    if concentration > 0.85:
        warnings.append("High spectral concentration — update may be rank-deficient.")
    if l2 < 1e-3:
        warnings.append("Very low weight magnitude — adapter may be near-untrained.")
    if eff_rank < 1.5:
        warnings.append("Low effective rank — update collapses to few directions.")

    elapsed = time.perf_counter() - t0
    return {
        "quality_score": round(quality_score, 1),
        "grade": _grade(quality_score),
        "spectral_norm": stats["spectral_norm"],
        "effective_rank": stats["effective_rank"],
        "concentration": stats["concentration"],
        "l2_norm": stats["l2_norm"],
        "sparsity": stats["sparsity"],
        "warnings": warnings,
        "time_seconds": round(elapsed, 2),
    }


# --------------------------------------------------------------------------- #
# Combined report builders (match the frontend's expected result shape)
# --------------------------------------------------------------------------- #
def evaluate_base_model(max_samples: int = 20) -> Dict[str, Any]:
    """Tier 1 only for the base model (Tier 0 needs adapter weights)."""
    model_path, _ = _resolve_model_and_adapter(None, include_adapter=False)
    if not model_path:
        raise ValueError(
            "No model path available for base-model evaluation. "
            "Train or load a run first."
        )
    tier1 = evaluate_tier1(model_path, adapter_path=None, max_samples=max_samples)
    return {
        "model_name": "base_model",
        "adapter_name": "base_model",
        "is_base_model": True,
        "evaluation_method": "tier1_only",
        "timestamp": _now_iso(),
        "tier0": None,
        "tier1": tier1,
        "total_time_seconds": tier1["time_seconds"],
    }


def evaluate_adapter(
    adapter_name: str, max_samples: int = 20
) -> Dict[str, Any]:
    """Tier 0 + Tier 1 for a LoRA adapter."""
    model_path, adapter_path = _resolve_model_and_adapter(adapter_name, include_adapter=True)
    if not adapter_path:
        raise ValueError(f"Adapter '{adapter_name}' not found in {paths.ADAPTERS_DIR}")
    if not model_path:
        raise ValueError("No base model path available to evaluate the adapter against.")

    tier0 = evaluate_tier0(adapter_path)
    tier1 = evaluate_tier1(model_path, adapter_path=adapter_path, max_samples=max_samples)

    return {
        "adapter_name": adapter_name,
        "is_base_model": False,
        "evaluation_method": "tier0_tier1_separate",
        "timestamp": _now_iso(),
        "tier0": tier0,
        "tier1": tier1,
        "total_time_seconds": round(tier0["time_seconds"] + tier1["time_seconds"], 2),
    }


# --------------------------------------------------------------------------- #
# LLM Judge (Function 3): DeepSeek-scored QA evaluation
# --------------------------------------------------------------------------- #
_JUDGE_PROMPT = """You are a strict but fair evaluator grading an AI assistant's answer to a user question.

User question:
{question}

Assistant's answer:
{answer}

Grade the answer on four dimensions, each from 0 to 100:
- faithfulness: how well the answer stays truthful to the question without inventing unsupported claims
- fact_recall: factual correctness and completeness for what is asked
- consistency: internal logical consistency and coherence
- hallucination: INVERTED — 100 means no hallucination, 0 means heavily fabricated

Respond with ONLY a JSON object, no prose, in exactly this shape:
{{"faithfulness": <int>, "fact_recall": <int>, "consistency": <int>, "hallucination": <int>}}
"""

_QA_PROMPTS = [
    "What is 2 plus 2?",
    "Explain what a LoRA adapter is in one sentence.",
    "What is the capital of France?",
    "Write one sentence in a direct, analytical, decision-oriented writing voice.",
    "What does perplexity measure in language models?",
    "Name one benefit of fine-tuning a small model over using a large one.",
    "What is the result of 7 multiplied by 6?",
    "Define 'effective rank' of a weight matrix in one sentence.",
]


def _deepseek_chat(messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
    """Call DeepSeek chat completions. Returns the assistant content string.

    deepseek-v4-flash is a reasoning model: it spends tokens on internal
    reasoning before emitting the final answer. Callers that need a
    structured answer (like the judge) MUST pass a large max_tokens so the
    model has room to finish reasoning and actually emit its answer.
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Add it to .env to use the LLM Judge."
        )
    payload = json.dumps(
        {"model": DEEPSEEK_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choice = data["choices"][0]["message"]
    # Prefer `content` (final answer); fall back to `reasoning_content` if the
    # model spent its whole budget reasoning and never produced a final answer.
    content = (choice.get("content") or "").strip()
    if not content:
        content = (choice.get("reasoning_content") or "").strip()
    return content


def _parse_judge_json(raw: str) -> Optional[Dict[str, int]]:
    """Extract the grading JSON from a (possibly noisy) model response."""
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except Exception:  # noqa: BLE001
        return None
    out: Dict[str, int] = {}
    for k in ("faithfulness", "fact_recall", "consistency", "hallucination"):
        try:
            out[k] = max(0, min(100, int(round(float(obj.get(k, 0))))))
        except Exception:  # noqa: BLE001
            out[k] = 0
    return out


def llm_judge_evaluate(
    adapter_name: Optional[str],
    num_questions: int = 20,
    evaluate_base_model: bool = False,
    on_progress=None,
) -> Dict[str, Any]:
    """Generate QA answers from the model and DeepSeek-grade them.

    Returns the result shape the frontend's LLM-judge flow expects:
      {adapter_name, is_base_model, scores:{overall,faithfulness,
       fact_recall,consistency,hallucination}, num_questions, detailed_results}

    `on_progress(done, total)` is called after each question for status polling.
    """
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    include_adapter = not evaluate_base_model
    model_path, adapter_path = _resolve_model_and_adapter(
        adapter_name, include_adapter=include_adapter
    )
    if not model_path:
        raise ValueError("No model path available for LLM-judge evaluation.")

    logger.info(
        "LLM Judge: loading %s (adapter=%s, base_mode=%s)",
        model_path, adapter_path, evaluate_base_model,
    )
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    sampler = make_sampler(temp=0.3)

    questions = (_QA_PROMPTS * ((num_questions // len(_QA_PROMPTS)) + 1))[:num_questions]
    detailed: List[Dict[str, Any]] = []
    accum = {"faithfulness": 0, "fact_recall": 0, "consistency": 0, "hallucination": 0}
    graded = 0

    for i, q in enumerate(questions):
        answer = _sanitize_text(
            generate(
                model, tokenizer, prompt=f"User: {q}\nAssistant:", max_tokens=128, sampler=sampler
            )
        )
        scores = None
        try:
            raw = _deepseek_chat(
                [{"role": "user", "content": _JUDGE_PROMPT.format(question=q, answer=answer[:1500])}],
                max_tokens=4000,
            )
            scores = _parse_judge_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM Judge: grader call failed for Q%d: %s", i, exc)

        if scores:
            graded += 1
            for k in accum:
                accum[k] += scores[k]
        detailed.append({"question": q, "answer": answer, "scores": scores})
        if on_progress:
            on_progress(i + 1, len(questions))

    if graded == 0:
        raise RuntimeError(
            "LLM Judge completed no graded questions. Check DEEPSEEK_API_KEY and quota."
        )

    avg = {k: round(accum[k] / graded, 1) for k in accum}
    overall = round(sum(avg.values()) / 4.0, 1)

    return {
        "adapter_name": adapter_name or "base_model",
        "is_base_model": evaluate_base_model,
        "scores": {
            "overall": overall,
            "faithfulness": avg["faithfulness"],
            "fact_recall": avg["fact_recall"],
            "consistency": avg["consistency"],
            "hallucination": avg["hallucination"],
        },
        # Report total questions attempted (matches frontend's num_questions label);
        # detailed_results carries per-question scores, including any nulls.
        "num_questions": len(questions),
        "detailed_results": detailed,
    }


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat()
