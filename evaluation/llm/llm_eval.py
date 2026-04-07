"""
evaluation/llm/llm_eval.py
===========================
Full LLM evaluation runner — mirrors graph/hotpot_dataset.py structure.

Runs the LLM baseline on the SAME question set as the graph evaluator
(identical seed + num_questions = deterministic fairness).

Baseline
--------
1. LLM           — question + full HotpotQA context paragraphs

Output
------
  results/llm_eval_log.txt        — per-question detailed log
  results/llm_eval_results.json   — full records + per-system analysis
  results/llm_eval_results.txt    — human-readable summary
  Terminal                        — single tqdm progress bar + final table
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

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

# ---------------------------------------------------------------------------
# Numpy JSON encoder
# ---------------------------------------------------------------------------
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_llm_eval() -> Optional[Dict]:
    """
    Run LLM baseline evaluation.
    Returns the combined analysis dict (same schema as graph eval).
    """
    # Late imports (users may run graph-only without needing these)
    from evaluation import config
    from evaluation.metrics import (
        sample_questions, exact_match,
        precision_recall_f1, substring_match,
        aggregate,
    )
    from evaluation.llm.ollama_client import (
        check_ollama_health, ask_llm_baseline,
    )

    results_dir = config.get_results_dir()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  LLM EVALUATION  (Llama3 via Ollama)")
    print("=" * 72)

    if not check_ollama_health():
        print(
            "\n[ERROR] Ollama is not running or llama3 model not found.\n"
            "  → Start Ollama:  ollama serve\n"
            "  → Pull model:    ollama pull llama3\n"
            "  → Then re-run.\n"
        )
        return None

    print(f"  [OK] Ollama is running | model: {config.OLLAMA_MODEL}")

    # ------------------------------------------------------------------
    # Load dataset (lazy load avoids 500MB RAM hit when graph-only)
    # ------------------------------------------------------------------
    hotpot_path = config.get_hotpot_path()
    if not os.path.exists(hotpot_path):
        print(f"[ERROR] Dataset not found: {hotpot_path}")
        return None

    print(f"  Loading dataset: {hotpot_path}")
    with open(hotpot_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    questions = sample_questions(all_data, config.NUM_QUESTIONS, config.SEED)
    total_q   = len(questions)

    print(f"  Questions selected : {total_q}  (seed={config.SEED})")
    print(f"  Model              : {config.OLLAMA_MODEL}")
    print(f"  Results dir        : {results_dir}")
    print("=" * 72 + "\n")

    # ------------------------------------------------------------------
    # Open log file
    # ------------------------------------------------------------------
    log_path = os.path.join(results_dir, "llm_eval_log.txt")
    log_f    = open(log_path, "w", encoding="utf-8")
    log_f.write("LLM EVALUATION LOG  (Llama3 via Ollama)\n")
    log_f.write(f"Generated  : {datetime.now().isoformat(timespec='seconds')}\n")
    log_f.write(f"Model      : {config.OLLAMA_MODEL}\n")
    log_f.write(f"Questions  : {total_q}  (seed={config.SEED})\n")
    log_f.write("=" * 80 + "\n\n")

    def _score_answer(prediction: str, ground_truth: str) -> Dict[str, float]:
        prf = precision_recall_f1(prediction, ground_truth)
        return {
            "exact_match": exact_match(prediction, ground_truth),
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "substring_match": substring_match(prediction, ground_truth),
        }

    # ------------------------------------------------------------------
    # Per-question records
    # ------------------------------------------------------------------
    llm_records: List[Dict] = []

    sep = "-" * 80
    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Main evaluation loop — tqdm in terminal, detail to log file
    # ------------------------------------------------------------------
    with make_progress_bar(total_q, "LLM Eval") as pbar:
        for q_idx, item in enumerate(questions, start=1):
            q_id        = item.get("_id", f"q{q_idx}")
            question    = item["question"]
            gold_answer = item["answer"]
            q_type      = item.get("type", "unknown")
            q_level     = item.get("level", "unknown")
            context     = item["context"]  # list of [title, [sentences]]
            # ── Write header to log ─────────────────────────────────────
            log_f.write(f"\n{sep}\n")
            log_f.write(f"Q{q_idx:04d}/{total_q}  [{q_type}|{q_level}]  id={q_id}\n")
            log_f.write(f"Question  : {question}\n")
            log_f.write(f"Gold      : {gold_answer}\n\n")

            # ── LLM baseline ───────────────────────────────────────────
            llm_start = time.time()
            llm_answer, llm_latency = ask_llm_baseline(question, context)
            llm_time_s = time.time() - llm_start
            llm_eval = _score_answer(llm_answer, gold_answer)

            log_f.write(f"  [LLM]\n")
            log_f.write(f"    Predicted : {llm_answer}\n")
            log_f.write(
                f"    EM={llm_eval['exact_match']:.0f}  P={llm_eval['precision']:.4f}"
                f"  R={llm_eval['recall']:.4f}  F1={llm_eval['f1']:.4f}"
                f"  Sub={llm_eval['substring_match']:.0f}"
                f"  Latency={llm_latency:.0f}ms  Time={llm_time_s:.2f}s\n\n"
            )
            log_f.write(sep + "\n")
            log_f.flush()

            # ── Store records ───────────────────────────────────────────
            base = {"q_num": q_idx, "_id": q_id, "type": q_type,
                    "level": q_level, "question": question,
                    "gold_answer": gold_answer}

            llm_records.append({**base,
                "predicted_answer": llm_answer,
                "exact_match": llm_eval["exact_match"],
                "precision": llm_eval["precision"],
                "recall": llm_eval["recall"],
                "f1": llm_eval["f1"],
                "substring_match": llm_eval["substring_match"],
                "latency_ms": llm_latency,
                "time_s": round(llm_time_s, 3),
                "internal_eval": llm_eval,
            })

            pbar.set_postfix_str(
                f"LLM-EM={llm_eval['exact_match']:.0f}"
            )
            pbar.update(1)

    log_f.close()
    total_time = time.time() - pipeline_start

    # ------------------------------------------------------------------
    # Aggregate metrics per system
    # ------------------------------------------------------------------
    def _analyse(records: List[Dict], system_name: str) -> Dict:
        """Build analysis dict for one system."""
        total   = len(records)
        levels  = ["easy", "medium", "hard"]
        types   = ["bridge", "comparison"]

        answered = [r for r in records if r.get("predicted_answer") not in ("", "ERROR")]

        em_by_level = {}
        f1_by_level = {}
        precision_by_level = {}
        recall_by_level = {}
        substring_by_level = {}
        for lvl in levels:
            recs = [r for r in records if r["level"] == lvl]
            em_by_level[lvl] = round(aggregate(recs, "exact_match"), 4)
            f1_by_level[lvl] = round(aggregate(recs, "f1"), 4)
            precision_by_level[lvl] = round(aggregate(recs, "precision"), 4)
            recall_by_level[lvl] = round(aggregate(recs, "recall"), 4)
            substring_by_level[lvl] = round(aggregate(recs, "substring_match"), 4)

        em_by_type = {}
        f1_by_type = {}
        for qt in types:
            recs = [r for r in records if r["type"] == qt]
            em_by_type[qt] = round(aggregate(recs, "exact_match"), 4)
            f1_by_type[qt] = round(aggregate(recs, "f1"), 4)

        questions_by_level = {}
        answered_by_level = {}
        bridge_by_level = {}
        comparison_by_level = {}
        for lvl in levels:
            lvl_recs = [r for r in records if r["level"] == lvl]
            questions_by_level[lvl] = len(lvl_recs)
            answered_by_level[lvl] = len([
                r for r in lvl_recs if r.get("predicted_answer") not in ("", "ERROR")
            ])
            bridge_by_level[lvl] = len([r for r in lvl_recs if r["type"] == "bridge"])
            comparison_by_level[lvl] = len([r for r in lvl_recs if r["type"] == "comparison"])

        analysis = {
            "system":                system_name,
            "total_questions":       total,
            "answered":              len(answered),
            "unanswered":            total - len(answered),
            "answer_rate_pct":       round(len(answered) / max(total, 1) * 100, 1),
            "exact_match":           round(aggregate(records, "exact_match"), 4),
            "precision":             round(aggregate(records, "precision"), 4),
            "recall":                round(aggregate(records, "recall"), 4),
            "f1":                    round(aggregate(records, "f1"), 4),
            "substring_match":       round(aggregate(records, "substring_match"), 4),
            "avg_time_per_question_s": round(aggregate(records, "time_s"), 3),
            "em_by_level":           em_by_level,
            "f1_by_level":           f1_by_level,
            "precision_by_level":    precision_by_level,
            "recall_by_level":       recall_by_level,
            "substring_by_level":    substring_by_level,
            "questions_by_level":    questions_by_level,
            "answered_by_level":     answered_by_level,
            "bridge_by_level":       bridge_by_level,
            "comparison_by_level":   comparison_by_level,
            "em_by_type":            em_by_type,
            "f1_by_type":            f1_by_type,
            "evidence_recall_at_5_by_level": {lvl: None for lvl in levels},
            "evidence_recall_at_5":  None,
            "avg_reasoning_depth":   None,
            "total_pipeline_time_s":  round(total_time, 2),
        }
        return analysis

    llm_analysis = _analyse(llm_records, "LLM")

    # ------------------------------------------------------------------
    # Terminal summary table per system
    # ------------------------------------------------------------------
    _print_llm_summary(llm_analysis, total_time)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "model":          config.OLLAMA_MODEL,
        "total_time_s":   round(total_time, 1),
        "llm":            {"analysis": llm_analysis,  "qa": llm_records},
    }

    json_path = os.path.join(results_dir, "llm_eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\n  JSON results → {json_path}")

    txt_path = os.path.join(results_dir, "llm_eval_results.txt")
    _save_txt_report(txt_path, llm_analysis, llm_records, config.OLLAMA_MODEL)
    print(f"  Text report  → {txt_path}")
    print(f"  Per-Q log    → {log_path}\n")

    return {
        "llm":         llm_analysis,
    }


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def _print_llm_summary(llm, total_time: float) -> None:
    W = 72
    print("\n" + "=" * W)
    print("  LLM BASELINE SUMMARY")
    print("=" * W)
    print(f"  {'Metric':<35} {'LLM':>11}")
    print("-" * W)

    def _p(v):
        return f"{v*100:>9.1f}%" if v is not None else "       N/A"

    rows = [
        ("Exact Match (EM)",        "exact_match"),
        ("Precision",               "precision"),
        ("Recall",                  "recall"),
        ("Token F1",                "f1"),
        ("Substring Match",         "substring_match"),
        ("Substring / R@5",         "evidence_recall_at_5"),
    ]
    for label, key in rows:
        ov = llm.get(key)
        ov_s = f"{ov:>9.4f}" if ov is not None else "       N/A"
        if label == "Substring / R@5" and ov is not None:
            ov_s = f"{llm.get('substring_match', 0.0):>9.4f} / {ov:>9.4f}"
        print(f"  {label:<35} {ov_s}")

    print("-" * W)
    print(f"  Total pipeline time : {total_time/60:.1f} min ({total_time:.0f}s)")
    print("=" * W)

    # Level breakdown
    print("\n" + "=" * W)
    print("  LEVEL-WISE BREAKDOWN")
    print("=" * W)
    print(f"  {'Level / Metric':<35} {'LLM':>11}")
    print("-" * W)
    for lvl in ("easy", "medium", "hard"):
        print(f"  {lvl.upper()}")
        llm_em  = llm.get("em_by_level", {}).get(lvl)
        llm_precision = llm.get("precision_by_level", {}).get(lvl)
        llm_recall = llm.get("recall_by_level", {}).get(lvl)
        llm_f1  = llm.get("f1_by_level", {}).get(lvl)
        llm_sub = llm.get("substring_by_level", {}).get(lvl)
        print(f"    {'  EM':<33} {_p(llm_em)}")
        print(f"    {'  Precision':<33} {llm_precision:>9.4f}")
        print(f"    {'  Recall':<33} {llm_recall:>9.4f}")
        print(f"    {'  F1':<33} {_p(llm_f1)}")
        print(f"    {'  Substring':<33} {llm_sub:>9.4f}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Text report saver
# ---------------------------------------------------------------------------

def _save_txt_report(path, llm, llm_recs, model):
    with open(path, "w", encoding="utf-8") as f:
        f.write("LLM EVALUATION REPORT\n")
        f.write(f"Generated : {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Model     : {model}\n\n")
        f.write(f"\n{'='*80}\nLLM\n{'='*80}\n")
        for k, v in llm.items():
            f.write(f"  {k:<40}: {v}\n")
        f.write("\nQA RECORDS\n" + "-"*80 + "\n")
        for r in llm_recs:
            f.write(f"\nQ{r['q_num']:04d} [{r['type']}|{r['level']}] id={r['_id']}\n")
            f.write(f"  Question  : {r['question']}\n")
            f.write(f"  Gold      : {r['gold_answer']}\n")
            f.write(f"  Predicted : {r['predicted_answer']}\n")
            f.write(
                f"  EM={r['exact_match']:.0f}  P={r['precision']:.4f}"
                f"  R={r['recall']:.4f}  F1={r['f1']:.4f}"
                f"  Sub={r['substring_match']:.0f}"
                f"  Latency={r['latency_ms']:.0f}ms\n"
            )
            f.write("-"*80 + "\n")


def _parse_args() -> Optional[object]:
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM evaluation on HotpotQA")
    parser.add_argument("--num", type=int, help="Number of sampled questions")
    parser.add_argument("--seed", type=int, help="Sampling seed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.num is not None or args.seed is not None:
        from evaluation import config
        if args.num is not None:
            config.NUM_QUESTIONS = args.num
        if args.seed is not None:
            config.SEED = args.seed

    run_llm_eval()


if __name__ == "__main__":
    main()
