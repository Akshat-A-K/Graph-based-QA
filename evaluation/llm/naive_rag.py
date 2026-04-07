"""
evaluation/llm/naive_rag.py
============================
Naive RAG baseline: BM25 top-k sentence retrieval → Llama3.

Purpose
-------
Isolates the value of the graph layer.  This baseline has EXACTLY the same
retrieval quality as BM25-only (no graph, no semantic embeddings), then
hands the retrieved sentences to Llama3 as context.

Directly answers: "Does the DRG/KG graph add anything over BM25 + LLM?"

Reference: Lewis et al. 2021, Retrieval-Augmented Generation (RAG), NeurIPS.
"""

from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np

from evaluation import config
from evaluation.metrics import (
    sample_questions,
    exact_match,
    precision_recall_f1,
    substring_match,
    evidence_recall_at_k,
    aggregate,
)
from evaluation.llm.ollama_client import _generate, _throttle, _format_context


# ---------------------------------------------------------------------------
# tqdm with graceful fallback
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm

    def make_progress_bar(total: int, desc: str):
        return _tqdm(total=total, desc=desc, unit="q",
                     ncols=90, dynamic_ncols=False,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
except ImportError:
    class _FallbackBar:
        def __init__(self, total, desc): self.n = 0; self.total = total; self.desc = desc
        def update(self, n=1):
            self.n += n
            pct = self.n / max(self.total, 1) * 100
            print(f"\r{self.desc}: {self.n}/{self.total} ({pct:.1f}%)", end="", flush=True)
        def close(self): print()
        def set_postfix_str(self, s): pass
        def __enter__(self): return self
        def __exit__(self, *a): self.close()
    def make_progress_bar(total, desc): return _FallbackBar(total, desc)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Minimal BM25 implementation (no external dependency)
# ---------------------------------------------------------------------------

def _bm25_scores(
    query_tokens: List[str],
    corpus: List[List[str]],
    k1: float = 1.5,
    b: float  = 0.75,
) -> List[float]:
    """
    BM25 Okapi scoring.  Returns a score per document in `corpus`.

    Reference: Robertson & Zaragoza 2009, "The Probabilistic Relevance
    Framework: BM25 and Beyond".
    """
    from math import log

    N      = len(corpus)
    avgdl  = sum(len(d) for d in corpus) / max(N, 1)
    scores = []

    for doc in corpus:
        doc_len = len(doc)
        score   = 0.0
        for term in query_tokens:
            tf  = doc.count(term)
            df  = sum(1 for d in corpus if term in d)
            idf = log((N - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
            score  += idf * tf_norm
        scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# Flatten context into (sentence, source_title) pairs
# ---------------------------------------------------------------------------

def _flatten_context(context_paragraphs: List) -> List[Tuple[str, str]]:
    """
    Returns list of (sentence_text, title) from HotpotQA context.
    Works with both [(title, [s1, s2, ...]), ...] and plain string lists.
    """
    flat = []
    for item in context_paragraphs:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            title, sents = item
            if isinstance(sents, (list, tuple)):
                for s in sents:
                    flat.append((str(s), str(title)))
            else:
                flat.append((str(sents), str(title)))
        else:
            flat.append((str(item), "unknown"))
    return flat


# ---------------------------------------------------------------------------
# Naive RAG answer
# ---------------------------------------------------------------------------

def naive_rag_answer(
    question: str,
    context_paragraphs: List,
    top_k: int = None,
) -> Tuple[str, float, List[str]]:
    """
    BM25-retrieve top-k sentences → feed to Llama3 as context.

    Returns
    -------
    (answer_text, latency_ms, retrieved_sentences)

    `retrieved_sentences` is returned so the caller can compute
    Retrieval Recall@k against the gold answer.
    """
    if top_k is None:
        top_k = config.NAIVE_RAG_TOP_K

    # Flatten all sentences
    flat = _flatten_context(context_paragraphs)
    if not flat:
        return "No context available", 0.0, []

    # Tokenise
    q_tokens   = question.lower().split()
    corpus_tok = [s.lower().split() for s, _ in flat]

    # BM25 scoring
    scores = _bm25_scores(q_tokens, corpus_tok)

    # Top-k by score
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices  = ranked[:top_k]
    retrieved    = [flat[i][0] for i in top_indices]

    # Build prompt with only retrieved sentences
    ctx_str = "\n".join(f"- {s}" for s in retrieved)
    prompt  = (
        f"Answer the question using ONLY the following retrieved sentences.\n"
        f"Give a short, direct answer (1–5 words). Do not explain.\n\n"
        f"Retrieved context:\n{ctx_str}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    start = time.time()
    try:
        answer, llm_latency = _generate(prompt)
    except RuntimeError:
        answer, llm_latency = "ERROR", 0.0
    _throttle()

    # Total latency = BM25 retrieval (already done) + LLM call
    total_latency = llm_latency

    return answer, round(total_latency, 1), retrieved


def _score_answer(prediction: str, ground_truth: str) -> Dict[str, float]:
    prf = precision_recall_f1(prediction, ground_truth)
    return {
        "exact_match": exact_match(prediction, ground_truth),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "substring_match": substring_match(prediction, ground_truth),
    }


def _analyse(records: List[Dict], system_name: str, total_time: float) -> Dict:
    total = len(records)
    levels = ["easy", "medium", "hard"]
    types = ["bridge", "comparison"]

    def _avg(recs, key):
        return sum(r[key] for r in recs) / max(len(recs), 1)

    em_by_level = {lvl: round(_avg([r for r in records if r["level"] == lvl], "exact_match"), 4) for lvl in levels}
    f1_by_level = {lvl: round(_avg([r for r in records if r["level"] == lvl], "f1"), 4) for lvl in levels}
    precision_by_level = {lvl: round(_avg([r for r in records if r["level"] == lvl], "precision"), 4) for lvl in levels}
    recall_by_level = {lvl: round(_avg([r for r in records if r["level"] == lvl], "recall"), 4) for lvl in levels}
    substring_by_level = {lvl: round(_avg([r for r in records if r["level"] == lvl], "substring_match"), 4) for lvl in levels}
    evidence_recall_at_5_by_level = {
        lvl: round(_avg([r for r in records if r["level"] == lvl], "evidence_recall_at_5"), 4)
        for lvl in levels
    }

    questions_by_level = {lvl: len([r for r in records if r["level"] == lvl]) for lvl in levels}
    answered_by_level = {lvl: len([r for r in records if r["level"] == lvl and r["predicted_answer"] != "No answer found"]) for lvl in levels}
    bridge_by_level = {lvl: len([r for r in records if r["level"] == lvl and r["type"] == "bridge"]) for lvl in levels}
    comparison_by_level = {lvl: len([r for r in records if r["level"] == lvl and r["type"] == "comparison"]) for lvl in levels}

    em_by_type = {qt: round(_avg([r for r in records if r["type"] == qt], "exact_match"), 4) for qt in types}
    f1_by_type = {qt: round(_avg([r for r in records if r["type"] == qt], "f1"), 4) for qt in types}

    answered = [r for r in records if r["predicted_answer"] != "No answer found"]

    return {
        "system": system_name,
        "total_questions": total,
        "answered": len(answered),
        "unanswered": total - len(answered),
        "answer_rate_pct": round(len(answered) / max(total, 1) * 100, 1),
        "exact_match": round(_avg(records, "exact_match"), 4),
        "precision": round(_avg(records, "precision"), 4),
        "recall": round(_avg(records, "recall"), 4),
        "f1": round(_avg(records, "f1"), 4),
        "substring_match": round(_avg(records, "substring_match"), 4),
        "em_by_level": em_by_level,
        "f1_by_level": f1_by_level,
        "precision_by_level": precision_by_level,
        "recall_by_level": recall_by_level,
        "substring_by_level": substring_by_level,
        "questions_by_level": questions_by_level,
        "answered_by_level": answered_by_level,
        "bridge_by_level": bridge_by_level,
        "comparison_by_level": comparison_by_level,
        "em_by_type": em_by_type,
        "f1_by_type": f1_by_type,
        "evidence_recall_at_5": round(_avg(records, "evidence_recall_at_5"), 4),
        "avg_reasoning_depth": None,
        "avg_time_per_question_s": round(_avg(records, "time_s"), 3),
        "total_pipeline_time_s": round(total_time, 2),
    }


def run_naive_rag_eval() -> Optional[Dict]:
    """Run the naive RAG baseline on the shared HotpotQA sample."""
    results_dir = config.get_results_dir()
    hotpot_path = config.get_hotpot_path()
    if not os.path.exists(hotpot_path):
        print(f"[ERROR] Dataset not found: {hotpot_path}")
        return None

    print("\n" + "=" * 72)
    print("  NAIVE RAG EVALUATION  (BM25 + Llama3)")
    print("=" * 72)

    print(f"  Loading dataset : {hotpot_path}")
    with open(hotpot_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    questions = sample_questions(all_data, config.NUM_QUESTIONS, config.SEED)
    total_q = len(questions)

    print(f"  Questions selected : {total_q}  (seed={config.SEED})")
    print(f"  Results dir        : {results_dir}")
    print(f"  Top-k sentences    : {config.NAIVE_RAG_TOP_K}")
    print("=" * 72 + "\n")

    log_path = os.path.join(results_dir, "naive_rag_eval_log.txt")
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write("NAIVE RAG EVALUATION LOG\n")
    log_f.write(f"Generated  : {datetime.now().isoformat(timespec='seconds')}\n")
    log_f.write(f"Questions  : {total_q}  (seed={config.SEED})\n")
    log_f.write("=" * 80 + "\n\n")

    records: List[Dict] = []
    sep = "-" * 80
    pipeline_start = time.time()

    with make_progress_bar(total_q, "Naive RAG") as pbar:
        for q_idx, item in enumerate(questions, start=1):
            q_id = item.get("_id", f"q{q_idx}")
            question = item["question"]
            gold_answer = item["answer"]
            q_type = item.get("type", "unknown")
            q_level = item.get("level", "unknown")
            context = item["context"]

            log_f.write(f"\n{sep}\n")
            log_f.write(f"Q{q_idx:04d}/{total_q}  [{q_type}|{q_level}]  id={q_id}\n")
            log_f.write(f"Question : {question}\n")
            log_f.write(f"Gold     : {gold_answer}\n")

            q_start = time.time()
            answer, latency_ms, retrieved = naive_rag_answer(question, context)
            elapsed = time.time() - q_start
            score = _score_answer(answer, gold_answer)
            r5 = evidence_recall_at_k(retrieved, gold_answer, k=5)

            log_f.write(f"Predicted: {answer}\n")
            log_f.write(
                f"EM={score['exact_match']:.0f}  F1={score['f1']:.4f}  P={score['precision']:.4f}"
                f"  R={score['recall']:.4f}  Sub={score['substring_match']:.0f}"
                f"  R@5={r5:.0f}  Latency={latency_ms:.0f}ms  Time={elapsed:.2f}s\n"
            )
            log_f.write(sep + "\n")
            log_f.flush()

            records.append({
                "q_num": q_idx,
                "_id": q_id,
                "type": q_type,
                "level": q_level,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": answer,
                "exact_match": score["exact_match"],
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1"],
                "substring_match": score["substring_match"],
                "evidence_recall_at_5": r5,
                "latency_ms": latency_ms,
                "time_s": round(elapsed, 3),
                "internal_eval": {
                    "exact_match": round(score["exact_match"], 4),
                    "precision": round(score["precision"], 4),
                    "recall": round(score["recall"], 4),
                    "f1": round(score["f1"], 4),
                    "substring_match": round(score["substring_match"], 4),
                    "evidence_recall_at_5": round(r5, 4),
                },
            })

            pbar.set_postfix_str(f"RAG-EM={score['exact_match']:.0f}")
            pbar.update(1)

            gc.collect()

    log_f.close()
    total_time = time.time() - pipeline_start
    analysis = _analyse(records, "Naive RAG", total_time)

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": config.OLLAMA_MODEL,
        "total_time_s": round(total_time, 1),
        "naive_rag": {"analysis": analysis, "qa": records},
    }

    json_path = os.path.join(results_dir, "naive_rag_eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\n  JSON results → {json_path}")

    txt_path = os.path.join(results_dir, "naive_rag_eval_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("NAIVE RAG EVALUATION REPORT\n")
        f.write(f"Generated : {output['generated_at']}\n")
        f.write(f"Model     : {config.OLLAMA_MODEL}\n\n")
        f.write("ANALYSIS SUMMARY\n" + "=" * 72 + "\n")
        for k, v in analysis.items():
            f.write(f"  {k:<40}: {v}\n")
    print(f"  Text report  → {txt_path}")
    print(f"  Per-Q log    → {log_path}\n")

    return {"naive_rag": analysis}
