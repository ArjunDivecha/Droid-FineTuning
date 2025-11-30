"""Style rewards for voice training (sequence-level).

Provide simple, offline rewards:
- lexical_f1_reward: token F1 between completion and reference
- length_window_reward: 1.0 inside [min_len, max_len], decays outside
- keyword_coverage_reward: fraction of keywords present in completion

Register functions for mlx_lm_lora via register_reward_function.
"""
import re
from typing import List, Optional

from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function

def _tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"\w+", s.lower()) if t]

@register_reward_function()
def lexical_f1_reward(*, prompts: List[str], completions: List[str], answer: List[str], types: Optional[List[str]] = None) -> List[float]:
    rewards: List[float] = []
    for comp, ref in zip(completions, answer):
        p = _tokens(comp)
        r = _tokens(ref)
        if not p or not r:
            rewards.append(0.0)
            continue
        from collections import Counter
        cp = Counter(p); cr = Counter(r)
        overlap = sum((cp & cr).values())
        prec = overlap / max(1, sum(cp.values()))
        rec = overlap / max(1, sum(cr.values()))
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rewards.append(float(f1))
    return rewards

@register_reward_function()
def length_window_reward(*, prompts: List[str], completions: List[str], answer: List[str], types: Optional[List[str]] = None, min_len: int = 200, max_len: int = 400) -> List[float]:
    out: List[float] = []
    for comp in completions:
        n = len(_tokens(comp))
        if min_len <= n <= max_len:
            out.append(1.0)
        else:
            # linear decay outside window, floor at 0
            if n < min_len:
                gap = min_len - n
                out.append(max(0.0, 1.0 - gap / max(min_len, 1)))
            else:
                gap = n - max_len
                out.append(max(0.0, 1.0 - gap / max(max_len, 1)))
    return out

@register_reward_function()
def keyword_coverage_reward(*, prompts: List[str], completions: List[str], answer: List[str], types: Optional[List[str]] = None, keywords: Optional[List[str]] = None) -> List[float]:
    # If no keywords, derive a light set from reference tokens (top few unique tokens)
    out: List[float] = []
    for comp, ref in zip(completions, answer):
        kw = keywords
        if not kw:
            toks = _tokens(ref)
            # pick up to 5 salient tokens (longer words)
            kw = [t for t in toks if len(t) >= 6][:5]
        comp_set = set(_tokens(comp))
        if not kw:
            out.append(0.0)
            continue
        covered = sum(1 for k in kw if k in comp_set)
        out.append(float(covered) / float(len(kw)))
    return out

