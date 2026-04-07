"""
evaluation/metrics.py
=====================
Shared metric functions used by both graph and LLM evaluators.

Metric origins
--------------
* Exact Match / Token F1         — Yang et al. 2018, HotpotQA official eval script
* Answer Precision / Recall      — token-overlap diagnostics used in this project
* Substring Match                — HotpotQA-style string containment check
* Evidence Recall@k              — HotpotQA-style retrieval/evidence check
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from typing import List, Dict, Tuple, Optional


# ===========================================================================
# Question Sampling (shared so graph & LLM use identical question sets)
# ===========================================================================

def sample_questions(
    all_data: List[Dict],
    num_questions: int,
    seed: int = 42,
) -> List[Dict]:
    """
    Stratified sample of `num_questions` questions from HotpotQA.

    - Split equally across easy / medium / hard.
    - Within each level split equally between bridge and comparison.
    - Uses `seed` for reproducibility.

    If `num_questions == 0` the entire dataset is returned.
    """
    if num_questions == 0:
        return all_data

    levels = ["easy", "medium", "hard"]
    pools: Dict[str, List] = {}
    rng = random.Random(seed)

    for lvl in levels:
        lvl_items = [item for item in all_data if item.get("level") == lvl]
        bridge     = [i for i in lvl_items if i.get("type") == "bridge"]
        comparison = [i for i in lvl_items if i.get("type") == "comparison"]
        bridge_rng = random.Random(seed)
        bridge_rng.shuffle(bridge)
        comparison_rng = random.Random(seed)
        comparison_rng.shuffle(comparison)
        pools[f"{lvl}_bridge"]     = bridge
        pools[f"{lvl}_comparison"] = comparison

    per_level = num_questions // 3
    remainder = num_questions % 3
    selected: List[Dict] = []

    for i, lvl in enumerate(levels):
        level_count  = per_level + (1 if i < remainder else 0)
        bridge_count = level_count // 2
        comp_count   = level_count - bridge_count
        bk, ck       = f"{lvl}_bridge", f"{lvl}_comparison"
        tb = min(bridge_count, len(pools[bk]))
        tc = min(comp_count,   len(pools[ck]))
        selected.extend(pools[bk][:tb])
        selected.extend(pools[ck][:tc])

    rng.shuffle(selected)
    return selected


# ===========================================================================
# Text Normalisation (HotpotQA official)
# ===========================================================================

def normalize_answer(s: str) -> str:
    """HotpotQA official answer normalisation."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split() if s else []


# ===========================================================================
# Exact Match
# ===========================================================================

def exact_match(prediction: str, ground_truth: str) -> float:
    """
    HotpotQA Exact Match (EM).
    Returns 1.0 if normalised strings are identical, else 0.0.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# ===========================================================================
# Token F1
# ===========================================================================

def substring_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 when the normalized answer strings contain each other."""
    pred = normalize_answer(prediction)
    truth = normalize_answer(ground_truth)
    if pred in truth or truth in pred:
        return 1.0
    return 0.0


def precision_recall_f1(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Token overlap precision / recall / F1 using normalized HotpotQA tokens."""
    pred_tokens  = get_tokens(prediction)
    truth_tokens = get_tokens(ground_truth)

    common   = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall    = num_same / len(truth_tokens) if truth_tokens else 0.0
    if precision + recall == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1        = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def evidence_recall_at_k(
    evidence_texts: List[str],
    gold_answer: str,
    k: int = 5,
) -> float:
    """
    HotpotQA-style evidence recall@k.

    Returns 1.0 when either the gold answer appears in the evidence span or the
    evidence span is a substring of the gold answer.
    """
    if not evidence_texts:
        return 0.0

    norm_gold = normalize_answer(gold_answer)
    for text in evidence_texts[:k]:
        norm_text = normalize_answer(text)
        if norm_gold in norm_text or norm_text in norm_gold:
            return 1.0
    return 0.0


# ===========================================================================
# Aggregate helper
# ===========================================================================

def aggregate(
    records: List[Dict],
    key: str,
    default: float = 0.0,
) -> float:
    """Mean of `key` across `records`, or `default` if empty."""
    vals = [r[key] for r in records if key in r and r[key] is not None]
    return round(sum(vals) / len(vals), 4) if vals else default
