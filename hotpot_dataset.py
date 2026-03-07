"""
hotpot_dataset.py
=================
Runs the Graph-based QA pipeline on HotpotQA *easy* questions.

Usage
-----
    python hotpot_dataset.py                          # 50 questions, default dataset
    python hotpot_dataset.py --num 200                # 200 easy questions
    python hotpot_dataset.py --num 0                  # ALL easy questions (~17 972)
    python hotpot_dataset.py --dataset my_hotpot.json # custom path
    python hotpot_dataset.py --num 100 --seed 99      # reproducible shuffle

The script converts every HotpotQA context block
    [[title, [sent0, sent1, ...]], ...]
into the internal "pages" format expected by the pipeline, then runs the
identical DRG → SpanGraph → EnhancedHybridReasoner → select_answer chain
used by run_custom_qa.py.

Outputs (written next to the dataset file):
    hotpot_easy_qa_output.json
    hotpot_easy_qa_output.txt
"""

import sys
import os
import gc
import re
import json
import time
import random
import string
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Optional colour support
# ---------------------------------------------------------------------------
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _HAS_COLOR = True
except Exception:
    Fore = Style = None
    _HAS_COLOR = False


def _cprint(msg: str, color: str = "cyan") -> None:
    if _HAS_COLOR:
        codes = {"cyan": Fore.CYAN, "yellow": Fore.YELLOW,
                 "green": Fore.GREEN, "red": Fore.RED, "white": Fore.WHITE}
        print(codes.get(color, "") + msg + Style.RESET_ALL)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Numpy JSON encoder
# ---------------------------------------------------------------------------
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Official HotpotQA normalisation / evaluation helpers
# (mirrors the official eval script: hotpotqa/hotpot_evaluate_v1.py)
# ---------------------------------------------------------------------------
def _normalize_answer(s: str) -> str:
    """Lower-case, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _get_tokens(s: str) -> List[str]:
    return _normalize_answer(s).split() if s else []


def compute_exact(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens  = _get_tokens(prediction)
    truth_tokens = _get_tokens(ground_truth)
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    num_same  = sum(min(pred_tokens.count(t), truth_tokens.count(t)) for t in common)
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Convert a HotpotQA context block → internal "pages" list
# ---------------------------------------------------------------------------
def hotpot_context_to_pages(context: List) -> List[Dict]:
    """
    context = [[title, [sent0, sent1, ...]], ...]
    → [{"page": 1, "text": "Title\\nsent0 sent1 ..."}, ...]
    """
    pages = []
    for page_idx, (title, sentences) in enumerate(context, start=1):
        text = title.strip() + "\n" + " ".join(s.strip() for s in sentences)
        pages.append({"page": page_idx, "text": text})
    return pages


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Graph-based QA on HotpotQA easy questions")
    parser.add_argument("--dataset", default="hotpot_train_v1.1.json",
                        help="Path to the HotpotQA JSON file")
    parser.add_argument("--num", type=int, default=50,
                        help="How many easy questions to evaluate (0 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling the question sample")
    parser.add_argument("--embed-model",
                        default="sentence-transformers/all-mpnet-base-v2",
                        help="Sentence-transformer model to use")
    parser.add_argument("--drg-threshold", type=float, default=0.75,
                        help="Cosine similarity threshold for DRG semantic edges")
    parser.add_argument("--span-threshold", type=float, default=0.70,
                        help="Cosine similarity threshold for Span Graph edges")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Resolve dataset path
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), dataset_path)
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Lazy-import pipeline components (after args are validated)
    # ------------------------------------------------------------------
    from parser.drg_nodes import build_nodes
    from parser.drg_graph import DocumentReasoningGraph
    from parser.span_extractor import SpanExtractor
    from parser.span_graph import SpanGraph
    from parser.enhanced_reasoner import EnhancedHybridReasoner
    from parser.answer_selector import select_answer
    from parser.evaluator import QAEvaluator

    EMBED_MODEL = args.embed_model

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    easy_questions = [item for item in all_data if item.get("level") == "easy"]
    print(f"Total easy questions in dataset: {len(easy_questions)}")

    random.seed(args.seed)
    random.shuffle(easy_questions)

    num_to_run = (len(easy_questions) if args.num == 0
                  else min(args.num, len(easy_questions)))
    easy_questions = easy_questions[:num_to_run]
    print(f"Running evaluation on {num_to_run} easy questions  (seed={args.seed})\n")

    # ------------------------------------------------------------------
    # Per-question evaluation loop
    # ------------------------------------------------------------------
    qa_records    = []
    sep           = "-" * 72
    pipeline_start = time.time()

    print("=" * 72)
    print("  HOTPOTQA EASY — QA RESULTS")
    print("=" * 72)

    for q_idx, item in enumerate(easy_questions, start=1):
        q_id        = item.get("_id", f"q{q_idx}")
        question    = item["question"]
        gold_answer = item["answer"]
        q_type      = item.get("type", "unknown")
        context     = item["context"]           # [[title, [sents]], ...]

        print(f"\n{sep}")
        _cprint(f"Q{q_idx:04d}/{num_to_run}  [{q_type}]  id={q_id}", "cyan")
        print(f"Question : {question}")
        print(f"Gold     : {gold_answer}")

        q_start = time.time()

        # ── 1. Convert context → pages ─────────────────────────────────
        pages = hotpot_context_to_pages(context)
        if not pages:
            print("  SKIP: empty context")
            continue

        # ── 2. Build sentence nodes ────────────────────────────────────
        sentence_nodes = build_nodes(pages)
        if not sentence_nodes:
            print("  SKIP: no sentence nodes")
            continue

        # ── 3. Document Reasoning Graph ────────────────────────────────
        drg = DocumentReasoningGraph(model_name=EMBED_MODEL)
        drg.add_nodes(sentence_nodes)
        drg.compute_embeddings()
        drg.add_semantic_edges(threshold=args.drg_threshold)
        drg.compute_graph_metrics()

        if drg.graph.number_of_nodes() == 0:
            print("  SKIP: DRG empty")
            gc.collect()
            continue

        # ── 4. Span extraction + Span Graph ───────────────────────────
        span_extractor     = SpanExtractor()
        spans              = span_extractor.extract_spans_from_nodes(sentence_nodes)

        span_graph_builder = SpanGraph(model_name=EMBED_MODEL)
        span_graph_builder.add_nodes(spans)
        span_graph_builder.compute_embeddings()
        span_graph_builder.add_semantic_edges(threshold=args.span_threshold)
        span_graph_builder.add_discourse_edges()
        span_graph_builder.add_entity_overlap_edges()
        span_graph_builder.compute_graph_metrics()

        if span_graph_builder.graph.number_of_nodes() == 0:
            print("  SKIP: Span Graph empty")
            gc.collect()
            continue

        # ── 5. Reasoning ───────────────────────────────────────────────
        reasoner = EnhancedHybridReasoner(
            sentence_graph=drg.graph,
            span_graph=span_graph_builder.graph,
            model_name=EMBED_MODEL,
        )

        results      = reasoner.enhanced_reasoning(question, k=5)
        final_spans  = results.get("final_spans", [])

        answer           = "No answer found"
        evidence_spans   = []
        confidence       = results.get("confidence", 0.0)
        confidence_label = results.get("confidence_label", "Low")

        if final_spans:
            answer, evidence_spans, confidence, confidence_label = select_answer(
                results,
                span_graph_builder,
                question,
                reasoner=reasoner,
                max_length=220,
            )

        elapsed = time.time() - q_start

        # ── 6. Evaluation against gold answer ─────────────────────────
        em = compute_exact(answer, gold_answer)
        f1 = compute_f1(answer, gold_answer)

        # Pipeline's internal EM/F1 vs top evidence (diagnostic only)
        internal_eval = {"exact_match": 0.0, "f1": 0.0}
        recall_at_5   = 0.0
        if answer != "No answer found" and evidence_spans:
            internal_eval = QAEvaluator.evaluate(answer, evidence_spans[0])
            recall_at_5   = QAEvaluator.evidence_recall_at_k(
                evidence_spans, evidence_spans[0], k=5)

        reasoning_depth = QAEvaluator.reasoning_depth(
            results.get("traversal_results", []),
            results.get("expansion_results", []),
        )

        _cprint(f"Predicted: {answer}", "green" if em == 1.0 else "yellow")
        print(f"EM={em:.0f}  F1={f1:.2f}  Conf={confidence_label}({confidence:.2%})"
              f"  Depth={reasoning_depth}  Time={elapsed:.2f}s")
        print(sep)

        qa_records.append({
            "q_num":            q_idx,
            "_id":              q_id,
            "type":             q_type,
            "question":         question,
            "gold_answer":      gold_answer,
            "predicted_answer": answer,
            "exact_match":      em,
            "f1":               f1,
            "confidence":       round(confidence, 4),
            "confidence_label": confidence_label,
            "time_s":           round(elapsed, 3),
            "evidence_count":   len(evidence_spans),
            "evidence_spans":   evidence_spans[:3],
            "reasoning_depth":  reasoning_depth,
            "internal_eval": {
                "exact_match": round(internal_eval["exact_match"], 4),
                "f1":          round(internal_eval["f1"], 4),
                "recall_at_5": round(recall_at_5, 4),
            },
            "retrieval": {
                "hybrid":    len(results.get("hybrid_results", [])),
                "traversal": len(results.get("traversal_results", [])),
                "expansion": len(results.get("expansion_results", [])),
            },
            "graph_stats": {
                "drg_nodes":  drg.graph.number_of_nodes(),
                "drg_edges":  drg.graph.number_of_edges(),
                "span_nodes": span_graph_builder.graph.number_of_nodes(),
                "span_edges": span_graph_builder.graph.number_of_edges(),
            },
        })

        # ── 7. Free memory between questions ──────────────────────────
        del drg, span_graph_builder, span_extractor, reasoner, results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    total_time = time.time() - pipeline_start
    answered   = [r for r in qa_records if r["predicted_answer"] != "No answer found"]
    total_q    = len(qa_records)

    avg_em   = sum(r["exact_match"] for r in qa_records) / max(total_q, 1)
    avg_f1   = sum(r["f1"] for r in qa_records) / max(total_q, 1)
    avg_conf = sum(r["confidence"] for r in answered) / max(len(answered), 1)
    high_conf = sum(1 for r in answered if r["confidence"] >= 0.70)
    med_conf  = sum(1 for r in answered if 0.45 <= r["confidence"] < 0.70)
    low_conf  = sum(1 for r in answered if r["confidence"] < 0.45)
    avg_time  = sum(r["time_s"] for r in qa_records) / max(total_q, 1)
    avg_depth = sum(r["reasoning_depth"] for r in answered) / max(len(answered), 1)

    # By question type
    bridge_recs = [r for r in qa_records if r["type"] == "bridge"]
    comp_recs   = [r for r in qa_records if r["type"] == "comparison"]
    type_em = {
        "bridge":     sum(r["exact_match"] for r in bridge_recs) / max(len(bridge_recs), 1),
        "comparison": sum(r["exact_match"] for r in comp_recs)   / max(len(comp_recs), 1),
    }
    type_f1 = {
        "bridge":     sum(r["f1"] for r in bridge_recs) / max(len(bridge_recs), 1),
        "comparison": sum(r["f1"] for r in comp_recs)   / max(len(comp_recs), 1),
    }

    analysis = {
        "dataset":                 os.path.basename(dataset_path),
        "embed_model":             EMBED_MODEL,
        "total_questions":         total_q,
        "answered":                len(answered),
        "unanswered":              total_q - len(answered),
        "answer_rate_pct":         round(len(answered) / max(total_q, 1) * 100, 1),
        "exact_match":             round(avg_em, 4),
        "f1":                      round(avg_f1, 4),
        "avg_confidence":          round(avg_conf, 4),
        "confidence_distribution": {"high": high_conf, "medium": med_conf, "low": low_conf},
        "em_by_type":              {k: round(v, 4) for k, v in type_em.items()},
        "f1_by_type":              {k: round(v, 4) for k, v in type_f1.items()},
        "avg_reasoning_depth":     round(avg_depth, 2),
        "avg_time_per_question_s": round(avg_time, 3),
        "total_pipeline_time_s":   round(total_time, 2),
    }

    print("\n" + "=" * 72)
    _cprint("  ANALYSIS SUMMARY — HOTPOTQA EASY", "cyan")
    print("=" * 72)
    print(f"  Questions evaluated  : {total_q}")
    print(f"  Answered             : {len(answered)}  ({analysis['answer_rate_pct']}%)")
    print(f"  Exact Match (EM)     : {avg_em:.4f}  ({avg_em*100:.1f}%)")
    print(f"  F1 Score             : {avg_f1:.4f}  ({avg_f1*100:.1f}%)")
    print(f"  EM bridge / compare  : {type_em['bridge']:.4f} / {type_em['comparison']:.4f}")
    print(f"  F1 bridge / compare  : {type_f1['bridge']:.4f} / {type_f1['comparison']:.4f}")
    print(f"  Avg confidence       : {avg_conf:.2%}")
    print(f"  Confidence dist      : High={high_conf}  Med={med_conf}  Low={low_conf}")
    print(f"  Avg reasoning depth  : {avg_depth:.2f}")
    print(f"  Avg time / question  : {avg_time:.2f}s")
    print(f"  Total pipeline time  : {total_time/60:.1f}min  ({total_time:.0f}s)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    out_dir   = os.path.dirname(os.path.abspath(dataset_path))
    json_path = os.path.join(out_dir, "hotpot_easy_qa_output.json")
    txt_path  = os.path.join(out_dir, "hotpot_easy_qa_output.txt")

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis":     analysis,
        "qa":           qa_records,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\nJSON output saved → {json_path}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("HOTPOTQA EASY QA OUTPUT\n")
        f.write(f"Generated : {output['generated_at']}\n")
        f.write(f"Model     : {EMBED_MODEL}\n\n")

        f.write("ANALYSIS SUMMARY\n" + "=" * 72 + "\n")
        for k, v in analysis.items():
            f.write(f"  {k:<35}: {v}\n")
        f.write("\n")

        f.write("QA RESULTS\n" + "=" * 72 + "\n")
        for r in qa_records:
            f.write(f"\nQ{r['q_num']:04d}  [{r['type']}]  id={r['_id']}\n")
            f.write(f"  Question : {r['question']}\n")
            f.write(f"  Gold     : {r['gold_answer']}\n")
            f.write(f"  Predicted: {r['predicted_answer']}\n")
            f.write(f"  EM={r['exact_match']:.0f}  F1={r['f1']:.4f}"
                    f"  Conf={r['confidence_label']}({r['confidence']:.2%})"
                    f"  Time={r['time_s']}s\n")
            if r["evidence_spans"]:
                f.write(f"  Top evid : {r['evidence_spans'][0][:120].strip()}...\n")
            f.write("-" * 72 + "\n")

    print(f"Text report saved → {txt_path}\n")


if __name__ == "__main__":
    main()