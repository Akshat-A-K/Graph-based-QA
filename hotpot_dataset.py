"""
hotpot_dataset.py
Runs the Graph-based QA pipeline on HotpotQA questions.

This script evaluates bridge/comparison questions and stores detailed
per-question graph statistics and QA metrics for reporting.

Usage
    python hotpot_dataset.py
    python hotpot_dataset.py --num 200
    python hotpot_dataset.py --num 0
    python hotpot_dataset.py --num 100 --seed 99
    python hotpot_dataset.py --level easy --num 50
    python hotpot_dataset.py --level all  --num 90   # 30 per level
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
from parser.config import HOTPOT_EMBED_MODEL

# Handle Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    print("WARNING: torch not available. CUDA acceleration will be disabled.")

from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Optional color support for readable terminal summaries.
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _HAS_COLOR = True
except Exception:
    Fore = Style = None
    _HAS_COLOR = False


def _cprint(msg: str, color: str = "cyan") -> None:
    if _HAS_COLOR:
        codes = {
            "cyan":   Fore.CYAN,
            "yellow": Fore.YELLOW,
            "green":  Fore.GREEN,
            "red":    Fore.RED,
            "white":  Fore.WHITE,
        }
        print(codes.get(color, "") + msg + Style.RESET_ALL)
    else:
        print(msg)


# Convert numpy types so json.dump can serialize report payloads.
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# Official HotpotQA normalization helpers used for EM/F1.
def _normalize_answer(s: str) -> str:
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


# Convert Hotpot context entries into page dictionaries used by the pipeline.
def hotpot_context_to_pages(context: List) -> List[Dict]:
    pages = []
    for page_idx, (title, sentences) in enumerate(context, start=1):
        text = title.strip() + "\n" + " ".join(s.strip() for s in sentences)
        pages.append({"page": page_idx, "text": text})
    return pages


# Build additional KG evidence used by bridge/comparison question types.
def _kg_evidence_for_bridge(
    kg,
    question: str,
    sentence_nodes: List[Dict],
) -> List[str]:
    """
    For bridge questions: use the KG to find multi-hop paths between
    named entities that appear in the question.
    Returns a list of natural-language evidence strings.
    """
    evidence: List[str] = []
    if kg is None or kg.graph.number_of_nodes() == 0:
        return evidence

    # Extract noun phrases / entities from the question (simple heuristic)
    # We use the top-PageRank entities that overlap with question words
    q_words = set(_normalize_answer(question).split())
    top_ents = kg.top_entities(n=20)
    q_entities = [
        e["entity"] for e in top_ents
        if any(w in e["entity"] for w in q_words)
    ][:6]

    # Try all pairs for shortest path
    for i in range(len(q_entities)):
        for j in range(i + 1, len(q_entities)):
            path = kg.shortest_path_evidence(q_entities[i], q_entities[j])
            if path:
                evidence.extend(path[:3])  # max 3 hops per pair

    return evidence[:10]  # cap total


def _kg_evidence_for_entity(
    kg,
    entity: str,
    max_hops: int = 2,
) -> List[str]:
    """Return fact strings for *entity* from the KG."""
    triples = kg.query_entity(entity, max_hops=max_hops)
    facts = [
        f"{t['subject']} {t['relation']} {t['object']}"
        for t in triples[:8]
    ]
    return facts


def _kg_boolean_vote(
    kg,
    entity1: str,
    entity2: str,
    question: str,
) -> Optional[str]:
    """
    Use KG triples for both entities to cast a yes/no vote.
    Returns "yes", "no", or None (abstain).
    """
    if kg is None or kg.graph.number_of_nodes() == 0:
        return None

    facts1 = _kg_evidence_for_entity(kg, entity1)
    facts2 = _kg_evidence_for_entity(kg, entity2)
    all_facts = " ".join(facts1 + facts2).lower()

    neg_patterns = [
        r'\bnot\b', r'\bnever\b', r'\bno\b', r'\bneither\b',
        r"n't\b", r'\bdifferent\b', r'\bvaries\b',
    ]
    neg_count = sum(
        len(re.findall(p, all_facts)) for p in neg_patterns
    )

    q_lower = question.lower()
    both_kw = ["both", "same", "also", "either", "share", "similar"]

    e1_present = entity1.lower() in all_facts
    e2_present = entity2.lower() in all_facts

    if neg_count >= 2:
        return "no"
    if any(w in q_lower for w in both_kw) and e1_present and e2_present and neg_count == 0:
        return "yes"
    return None


# Parse command line options for evaluation runs.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Graph-based QA on HotpotQA questions")
    parser.add_argument("--dataset", default="hotpot_train_v1.1.json")
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed-model",
                        default=HOTPOT_EMBED_MODEL)
    parser.add_argument("--level", default="all",
                        choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--drg-threshold", type=float, default=0.75)
    parser.add_argument("--span-threshold", type=float, default=0.70)
    parser.add_argument("--kg-model", default="en_core_web_trf",
                        help="spaCy model for Knowledge Graph (en_core_web_trf / sm / lg)")
    parser.add_argument("--kg-hops", type=int, default=2,
                        help="Max KG hops for bridge evidence retrieval")
    return parser.parse_args()


# Run the end-to-end HotpotQA evaluation pipeline.
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

    # Import heavy modules only when the script runs.
    from parser.drg_nodes import build_nodes
    from parser.drg_graph import DocumentReasoningGraph
    from parser.span_extractor import SpanExtractor
    from parser.span_graph import SpanGraph
    from parser.knowledge_graph import KnowledgeGraph
    from parser.enhanced_reasoner import EnhancedHybridReasoner
    from parser.answer_selector import select_answer
    from parser.evaluator import QAEvaluator

    EMBED_MODEL = args.embed_model

    # Load dataset JSON once before sampling questions.
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    levels = ["easy", "medium", "hard"]

    if args.level == "all" and args.num > 0:
        pools: Dict[str, List] = {}
        for lvl in levels:
            lvl_items = [item for item in all_data if item.get("level") == lvl]
            pools[f"{lvl}_bridge"]     = [i for i in lvl_items if i.get("type") == "bridge"]
            pools[f"{lvl}_comparison"] = [i for i in lvl_items if i.get("type") == "comparison"]
        for key in pools:
            random.seed(args.seed)
            random.shuffle(pools[key])

        per_level = args.num // 3
        rem = args.num % 3
        selected_questions: List[Dict] = []
        for i, lvl in enumerate(levels):
            level_count  = per_level + (1 if i < rem else 0)
            bridge_count = level_count // 2
            comp_count   = level_count - bridge_count
            bk, ck       = f"{lvl}_bridge", f"{lvl}_comparison"
            tb = min(bridge_count, len(pools[bk]))
            tc = min(comp_count,   len(pools[ck]))
            selected_questions.extend(pools[bk][:tb])
            selected_questions.extend(pools[ck][:tc])
            print(f"  {lvl.capitalize()}: {tb} bridge + {tc} comparison = {tb+tc} questions")

        num_to_run = len(selected_questions)
        random.shuffle(selected_questions)
        print(f"Running evaluation on {num_to_run} total questions (seed={args.seed})\n")
    else:
        if args.level == "all":
            selected_questions = all_data
        else:
            selected_questions = [i for i in all_data if i.get("level") == args.level]
        print(f"Total '{args.level}' questions in dataset: {len(selected_questions)}")
        random.seed(args.seed)
        random.shuffle(selected_questions)
        num_to_run = (len(selected_questions) if args.num == 0
                      else min(args.num, len(selected_questions)))
        selected_questions = selected_questions[:num_to_run]
        print(f"Running evaluation on {num_to_run} '{args.level}' questions (seed={args.seed})\n")

    # Evaluate each sampled question and collect report records.
    qa_records: List[Dict] = []
    sep = "-" * 72
    pipeline_start = time.time()

    print("=" * 72)
    print("  HOTPOTQA - QA RESULTS (with Knowledge Graph)")
    print("=" * 72)

    for q_idx, item in enumerate(selected_questions, start=1):
        q_id        = item.get("_id", f"q{q_idx}")
        question    = item["question"]
        gold_answer = item["answer"]
        q_type      = item.get("type", "unknown")
        q_level     = item.get("level", "unknown")
        context     = item["context"]

        print(f"\n{sep}")
        _cprint(f"Q{q_idx:04d}/{num_to_run}  [{q_type}|{q_level}]  id={q_id}", "cyan")
        print(f"Question : {question}")
        print(f"Gold     : {gold_answer}")

        q_start = time.time()

        # Prepare context pages for downstream graph construction.
        pages = hotpot_context_to_pages(context)
        if not pages:
            print("  SKIP: empty context")
            continue

        # Build sentence-level nodes from context pages.
        sentence_nodes = build_nodes(pages)
        if not sentence_nodes:
            print("  SKIP: no sentence nodes")
            continue

        sentence_texts = [node["text"] for node in sentence_nodes]

        # Build the question-specific Knowledge Graph before DRG reasoning.
        kg = KnowledgeGraph(model_name=args.kg_model)
        kg.build_graph(sentence_texts)
        kg_stats = kg.get_stats()
        _cprint(
            f"  [KG] nodes={kg_stats['nodes']}  edges={kg_stats['edges']}"
            f"  communities={kg_stats['communities']}"
            f"  density={kg_stats['density']:.4f}",
            "white",
        )

        # Build the document reasoning graph.
        drg = DocumentReasoningGraph(model_name=EMBED_MODEL)
        drg.add_nodes(sentence_nodes)
        drg.compute_embeddings()
        drg.add_semantic_edges(threshold=args.drg_threshold)
        drg.compute_graph_metrics()

        if drg.graph.number_of_nodes() == 0:
            print("  SKIP: DRG empty")
            gc.collect()
            continue

        # Build span nodes and the span graph.
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

        # Run hybrid reasoning over DRG, span graph, and KG.
        reasoner = EnhancedHybridReasoner(
            sentence_graph=drg.graph,
            span_graph=span_graph_builder.graph,
            kg_graph=kg.graph,
            model_name=EMBED_MODEL,
        )
        results     = reasoner.enhanced_reasoning(question, k=5)
        final_spans = results.get("final_spans", [])

        answer           = "No answer found"
        evidence_spans   = []
        confidence       = 0.0

        if final_spans:
            answer, evidence_spans, confidence = select_answer(
                results,
                span_graph_builder,
                question,
                reasoner=reasoner,
                max_length=220,
            )

        # Add bridge-specific evidence mined from KG paths.
        kg_evidence: List[str] = []
        if q_type == "bridge":
            kg_evidence = _kg_evidence_for_bridge(
                kg, question, sentence_nodes
            )
            if kg_evidence:
                _cprint(f"  [KG bridge evidence] {kg_evidence[0]}", "white")

        # Refine comparison answers with entity-aware and KG-aware checks.
        if q_type == "comparison":
            try:
                from parser.comparison_utils import (
                    extract_comparison_entities,
                    classify_comparison_type,
                )

                e1, e2       = extract_comparison_entities(question)
                comp_subtype = classify_comparison_type(question)

                def _retrieve(entity: str) -> List[str]:
                    hits: List[str] = []
                    for n in spans:
                        text = n.get("text", "") if isinstance(n, dict) else str(n)
                        if entity.lower() in text.lower():
                            hits.append(text)
                        if len(hits) >= 5:
                            break
                    if len(hits) < 5:
                        for p in pages:
                            if entity.lower() in p.get("text", "").lower():
                                hits.append(p["text"])
                            if len(hits) >= 5:
                                break
                    # augment with KG facts
                    hits.extend(_kg_evidence_for_entity(kg, entity))
                    return hits

                def _get_year(texts: List[str]) -> Optional[int]:
                    for t in texts:
                        years = re.findall(r'\b(1[89]\d{2}|20\d{2})\b', t)
                        if years:
                            return int(years[0])
                    return None

                def _clean_entity(ent: str) -> str:
                    noise = r'^(are\s+the|is\s+the|are\s+|is\s+|the\s+|both\s+)'
                    return re.sub(noise, '', ent, flags=re.IGNORECASE).strip()

                if e1 and e2:
                    e1 = _clean_entity(e1)
                    e2 = _clean_entity(e2)
                    e1_texts = _retrieve(e1)
                    e2_texts = _retrieve(e2)
                    all_evid = " ".join(e1_texts + e2_texts).lower()

                    print(f"  [comp:{comp_subtype}] e1='{e1}' e2='{e2}'")

                    if comp_subtype == "boolean":
                        # First: ask the KG
                        kg_vote = _kg_boolean_vote(kg, e1, e2, question)

                        strong_neg = [
                            "neither", "only one", "not both",
                            "different countr", "different locat",
                            "not the same"
                        ]
                        soft_neg = any(
                            re.search(
                                r'\bnot\s+(?:located|headquartered|based'
                                r'|available|a\s+\w+|in\s+the)',
                                " ".join(_retrieve(ent)).lower()
                            )
                            for ent in [e1, e2]
                        )
                        q_lower    = question.lower()
                        both_kw    = ["both", "same", "also", "either", "share"]
                        is_both_q  = any(w in q_lower for w in both_kw)

                        if kg_vote is not None:
                            answer = kg_vote
                            _cprint(f"  [KG boolean vote] -> {answer}", "white")
                        elif any(n in all_evid for n in strong_neg) or soft_neg:
                            answer = "no"
                        elif is_both_q and e1.lower() in all_evid and e2.lower() in all_evid:
                            answer = "yes"
                        elif e1.lower() in all_evid and e2.lower() in all_evid:
                            answer = "yes"

                    elif answer and answer.lower() not in ("yes", "no", "no answer found"):
                        for entity in [e1, e2]:
                            if (answer.lower() in entity.lower()
                                    and len(entity) > len(answer)):
                                answer = entity
                                break

                    if comp_subtype == "comparative":
                        q_lower   = question.lower()
                        is_first  = any(w in q_lower for w in ["first", "earlier", "oldest"])
                        is_recent = any(w in q_lower for w in ["recent", "latest", "newest"])
                        if is_first or is_recent:
                            y1 = _get_year(e1_texts)
                            y2 = _get_year(e2_texts)
                            if y1 and y2:
                                answer = e1 if (
                                    (is_first  and y1 < y2) or
                                    (is_recent and y1 > y2)
                                ) else e2

                    # Add a short KG path between compared entities when available.
                    bridge_path = kg.shortest_path_evidence(e1, e2)
                    if bridge_path:
                        kg_evidence.extend(bridge_path[:2])
                        _cprint(
                            f"  [KG comp path] {' | '.join(bridge_path[:2])}",
                            "white",
                        )

            except Exception as exc:
                print(f"  [comp error] {exc}")

        elapsed = time.time() - q_start

        # Evaluate answer quality and evidence quality for reporting.
        em = compute_exact(answer, gold_answer)
        f1 = compute_f1(answer, gold_answer)

        qa_eval = {
            "exact_match": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "substring_match": 0.0,
        }
        recall_at_5 = 0.0
        if answer != "No answer found":
            qa_eval = QAEvaluator.evaluate(answer, gold_answer)
            if evidence_spans:
                recall_at_5 = QAEvaluator.evidence_recall_at_k(
                    evidence_spans, gold_answer, k=5)

        reasoning_depth = QAEvaluator.reasoning_depth(
            results.get("traversal_results", []),
            results.get("expansion_results", []),
        )

        _cprint(
            f"Predicted: {answer}",
            "green" if em == 1.0 else "yellow",
        )

        confidence_label = "Low"
        if confidence >= 0.8: confidence_label = "High"
        elif confidence >= 0.45: confidence_label = "Med"

        metrics_line = (
            f"Eval(P={qa_eval['precision']:.2f}, R={qa_eval['recall']:.2f}, "
            f"F1={qa_eval['f1']:.2f}, Sub={qa_eval['substring_match']:.0f}, "
            f"R@5={recall_at_5:.2f})"
        )
        print(
            f"EM={em:.0f}  F1={f1:.2f}  {metrics_line}"
            f"  Conf={confidence_label}({confidence:.2%})"
            f"  Depth={reasoning_depth}  Time={elapsed:.2f}s"
        )
        print(sep)

        qa_records.append({
            "q_num":            q_idx,
            "_id":              q_id,
            "type":             q_type,
            "level":            q_level,
            "question":         question,
            "gold_answer":      gold_answer,
            "predicted_answer": answer,
            "exact_match":      em,
            "f1":               f1,
            "confidence":       confidence,
            "time_s":           round(elapsed, 3),
            "evidence_count":   len(evidence_spans),
            "evidence_spans":   evidence_spans[:3],
            "kg_evidence":      kg_evidence[:5],
            "reasoning_depth":  reasoning_depth,
            "internal_eval": {
                "exact_match":      round(qa_eval["exact_match"], 4),
                "precision":        round(qa_eval["precision"], 4),
                "recall":           round(qa_eval["recall"], 4),
                "f1":               round(qa_eval["f1"], 4),
                "substring_match":  round(qa_eval["substring_match"], 4),
                "recall_at_5":      round(recall_at_5, 4),
                "evidence_recall_at_5": round(recall_at_5, 4),
            },
            "retrieval": {
                "hybrid":    len(results.get("hybrid_results", [])),
                "traversal": len(results.get("traversal_results", [])),
                "expansion": len(results.get("expansion_results", [])),
                "kg_guided": len(results.get("kg_results", [])),
            },
            "graph_stats": {
                "drg_nodes":        drg.graph.number_of_nodes(),
                "drg_edges":        drg.graph.number_of_edges(),
                "span_nodes":       span_graph_builder.graph.number_of_nodes(),
                "span_edges":       span_graph_builder.graph.number_of_edges(),
                # NEW: KG stats
                "kg_nodes":         kg_stats["nodes"],
                "kg_edges":         kg_stats["edges"],
                "kg_communities":   kg_stats["communities"],
                "kg_density":       kg_stats["density"],
                "kg_entity_labels": kg_stats["entity_labels"],
            },
        })

        # Release graph/reasoner objects to reduce memory growth.
        del drg, span_graph_builder, span_extractor, reasoner, results, kg
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute aggregate metrics across all processed questions.
    total_time = time.time() - pipeline_start
    answered   = [r for r in qa_records if r["predicted_answer"] != "No answer found"]
    total_q    = len(qa_records)

    avg_em    = sum(r["exact_match"] for r in qa_records) / max(total_q, 1)
    avg_f1    = sum(r["f1"] for r in qa_records) / max(total_q, 1)
    avg_precision = sum(r["internal_eval"]["precision"] for r in qa_records) / max(total_q, 1)
    avg_recall = sum(r["internal_eval"]["recall"] for r in qa_records) / max(total_q, 1)
    avg_substring = sum(r["internal_eval"]["substring_match"] for r in qa_records) / max(total_q, 1)
    avg_evidence_recall_at_5 = sum(
        r["internal_eval"]["evidence_recall_at_5"] for r in qa_records
    ) / max(total_q, 1)
    avg_conf  = sum(r["confidence"] for r in answered) / max(len(answered), 1)
    high_conf = sum(1 for r in answered if r["confidence"] >= 0.70)
    med_conf  = sum(1 for r in answered if 0.45 <= r["confidence"] < 0.70)
    low_conf  = sum(1 for r in answered if r["confidence"] < 0.45)
    avg_time  = sum(r["time_s"] for r in qa_records) / max(total_q, 1)
    avg_depth = sum(r["reasoning_depth"] for r in answered) / max(len(answered), 1)

    bridge_recs = [r for r in qa_records if r["type"] == "bridge"]
    comp_recs   = [r for r in qa_records if r["type"] == "comparison"]
    type_em = {
        "bridge":     sum(r["exact_match"] for r in bridge_recs) / max(len(bridge_recs), 1),
        "comparison": sum(r["exact_match"] for r in comp_recs)   / max(len(comp_recs),   1),
    }
    type_f1 = {
        "bridge":     sum(r["f1"] for r in bridge_recs) / max(len(bridge_recs), 1),
        "comparison": sum(r["f1"] for r in comp_recs)   / max(len(comp_recs),   1),
    }

    # Aggregate graph size statistics for all three graph layers.
    drg_nodes_avg = sum(
        r["graph_stats"].get("drg_nodes", 0) for r in qa_records
    ) / max(total_q, 1)
    drg_edges_avg = sum(
        r["graph_stats"].get("drg_edges", 0) for r in qa_records
    ) / max(total_q, 1)
    span_nodes_avg = sum(
        r["graph_stats"].get("span_nodes", 0) for r in qa_records
    ) / max(total_q, 1)
    span_edges_avg = sum(
        r["graph_stats"].get("span_edges", 0) for r in qa_records
    ) / max(total_q, 1)
    kg_nodes_avg = sum(
        r["graph_stats"].get("kg_nodes", 0) for r in qa_records
    ) / max(total_q, 1)
    kg_edges_avg = sum(
        r["graph_stats"].get("kg_edges", 0) for r in qa_records
    ) / max(total_q, 1)

    analysis = {
        "dataset":                 os.path.basename(dataset_path),
        "embed_model":             EMBED_MODEL,
        "kg_model":                args.kg_model,
        "total_questions":         total_q,
        "answered":                len(answered),
        "unanswered":              total_q - len(answered),
        "answer_rate_pct":         round(len(answered) / max(total_q, 1) * 100, 1),
        "exact_match":             round(avg_em, 4),
        "f1":                      round(avg_f1, 4),
        "precision":               round(avg_precision, 4),
        "recall":                  round(avg_recall, 4),
        "substring_match":         round(avg_substring, 4),
        "evidence_recall_at_5":    round(avg_evidence_recall_at_5, 4),
        "em_by_type":              {k: round(v, 4) for k, v in type_em.items()},
        "f1_by_type":              {k: round(v, 4) for k, v in type_f1.items()},
        "avg_reasoning_depth":     round(avg_depth, 2),
        "avg_time_per_question_s": round(avg_time, 3),
        "total_pipeline_time_s":   round(total_time, 2),
        "drg_avg_nodes":           round(drg_nodes_avg, 1),
        "drg_avg_edges":           round(drg_edges_avg, 1),
        "span_avg_nodes":          round(span_nodes_avg, 1),
        "span_avg_edges":          round(span_edges_avg, 1),
        "kg_avg_nodes":            round(kg_nodes_avg, 1),
        "kg_avg_edges":            round(kg_edges_avg, 1),
    }

    print("\n" + "=" * 72)
    _cprint("  ANALYSIS SUMMARY - HOTPOTQA ", "cyan")
    print("=" * 72)
    print(f"  Questions evaluated  : {total_q}")
    print(f"  Answered             : {len(answered)}  ({analysis['answer_rate_pct']}%)")
    print(f"  Exact Match (EM)     : {avg_em:.4f}  ({avg_em*100:.1f}%)")
    print(f"  F1 Score             : {avg_f1:.4f}  ({avg_f1*100:.1f}%)")
    print(f"  Precision / Recall   : {avg_precision:.4f} / {avg_recall:.4f}")
    print(f"  Substring / R@5      : {avg_substring:.4f} / {avg_evidence_recall_at_5:.4f}")
    print(f"  EM bridge / compare  : {type_em['bridge']:.4f} / {type_em['comparison']:.4f}")
    print(f"  F1 bridge / compare  : {type_f1['bridge']:.4f} / {type_f1['comparison']:.4f}")
    print(f"  Avg reasoning depth  : {avg_depth:.2f}")
    print(f"  Avg DRG nodes/edges  : {drg_nodes_avg:.0f} / {drg_edges_avg:.0f}")
    print(f"  Avg Span nodes/edges : {span_nodes_avg:.0f} / {span_edges_avg:.0f}")
    print(f"  Avg KG nodes/edges   : {kg_nodes_avg:.0f} / {kg_edges_avg:.0f}")
    print(f"  Avg time / question  : {avg_time:.2f}s")
    print(f"  Total pipeline time  : {total_time/60:.1f}min  ({total_time:.0f}s)")
    print("=" * 72)

    # Show level-wise summary when all difficulty levels are included.
    if args.level == "all":
        print("\n" + "=" * 72)
        _cprint("  LEVEL-WISE BREAKDOWN", "cyan")
        print("=" * 72)
        for lvl in levels:
            lvl_recs = [r for r in qa_records if r.get("level") == lvl]
            if not lvl_recs:
                continue
            lvl_answered  = [r for r in lvl_recs if r["predicted_answer"] != "No answer found"]
            lvl_total     = len(lvl_recs)
            lvl_bridge    = [r for r in lvl_recs if r["type"] == "bridge"]
            lvl_comp      = [r for r in lvl_recs if r["type"] == "comparison"]
            lvl_em        = sum(r["exact_match"] for r in lvl_recs) / max(lvl_total, 1)
            lvl_f1        = sum(r["f1"] for r in lvl_recs) / max(lvl_total, 1)
            lvl_bridge_em = sum(r["exact_match"] for r in lvl_bridge) / max(len(lvl_bridge), 1)
            lvl_comp_em   = sum(r["exact_match"] for r in lvl_comp)   / max(len(lvl_comp),   1)
            lvl_bridge_f1 = sum(r["f1"] for r in lvl_bridge) / max(len(lvl_bridge), 1)
            lvl_comp_f1   = sum(r["f1"] for r in lvl_comp)   / max(len(lvl_comp),   1)
            lvl_precision = sum(r["internal_eval"]["precision"] for r in lvl_recs) / max(lvl_total, 1)
            lvl_recall = sum(r["internal_eval"]["recall"] for r in lvl_recs) / max(lvl_total, 1)
            lvl_substring = sum(r["internal_eval"]["substring_match"] for r in lvl_recs) / max(lvl_total, 1)
            lvl_r5 = sum(r["internal_eval"]["evidence_recall_at_5"] for r in lvl_recs) / max(lvl_total, 1)
            print(f"\n{lvl.upper()}:")
            print(f"  Questions evaluated  : {lvl_total}")
            print(f"  Answered             : {len(lvl_answered)}  ({len(lvl_answered)/max(lvl_total,1)*100:.1f}%)")
            print(f"  Question types       : {len(lvl_bridge)} bridge / {len(lvl_comp)} comparison")
            print(f"  Exact Match (EM)     : {lvl_em:.4f}  ({lvl_em*100:.1f}%)")
            print(f"  F1 Score             : {lvl_f1:.4f}  ({lvl_f1*100:.1f}%)")
            print(f"  Precision / Recall   : {lvl_precision:.4f} / {lvl_recall:.4f}")
            print(f"  Substring / R@5      : {lvl_substring:.4f} / {lvl_r5:.4f}")
            print(f"  EM bridge / compare  : {lvl_bridge_em:.4f} / {lvl_comp_em:.4f}")
            print(f"  F1 bridge / compare  : {lvl_bridge_f1:.4f} / {lvl_comp_f1:.4f}")
        print("\n" + "=" * 72)

    # Save JSON and text reports for the final submission/report.
    out_dir   = os.path.dirname(os.path.abspath(dataset_path))
    json_path = os.path.join(out_dir, "hotpot_all_qa_output.json")
    txt_path  = os.path.join(out_dir, "hotpot_all_qa_output.txt")

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis":     analysis,
        "qa":           qa_records,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\nJSON output saved -> {json_path}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("HOTPOTQA ALL LEVELS QA OUTPUT  (with Knowledge Graph)\n")
        f.write(f"Generated : {output['generated_at']}\n")
        f.write(f"Model     : {EMBED_MODEL}\n")
        f.write(f"KG Model  : {args.kg_model}\n\n")

        f.write("ANALYSIS SUMMARY\n" + "=" * 72 + "\n")
        for k, v in analysis.items():
            f.write(f"  {k:<40}: {v}\n")
        f.write("\n")

        f.write("QA RESULTS\n" + "=" * 72 + "\n")
        for r in qa_records:
            f.write(f"\nQ{r['q_num']:04d}  [{r['type']}|{r['level']}]  id={r['_id']}\n")
            f.write(f"  Question   : {r['question']}\n")
            f.write(f"  Gold       : {r['gold_answer']}\n")
            f.write(f"  Predicted  : {r['predicted_answer']}\n")
            ie = r["internal_eval"]
            f.write(f"  EM={r['exact_match']:.0f}  F1={r['f1']:.4f}"
                    f"  Time={r['time_s']}s\n")
            f.write(
                f"  Eval       : P={ie.get('precision', 0.0):.4f}  "
                f"R={ie.get('recall', 0.0):.4f}  "
                f"F1={ie.get('f1', 0.0):.4f}  "
                f"Sub={ie.get('substring_match', 0.0):.4f}  "
                f"R@5={ie.get('evidence_recall_at_5', 0.0):.4f}\n"
            )
            gs = r["graph_stats"]
            f.write(
                f"  DRG        : {gs.get('drg_nodes',0)} nodes  "
                f"{gs.get('drg_edges',0)} edges\n"
            )
            f.write(
                f"  Span       : {gs.get('span_nodes',0)} nodes  "
                f"{gs.get('span_edges',0)} edges\n"
            )
            f.write(
                f"  KG         : {gs.get('kg_nodes',0)} nodes  "
                f"{gs.get('kg_edges',0)} edges  "
                f"{gs.get('kg_communities',0)} communities\n"
            )
            if r["kg_evidence"]:
                f.write(f"  KG evid    : {r['kg_evidence'][0][:120]}\n")
            if r["evidence_spans"]:
                f.write(f"  Top span   : {r['evidence_spans'][0][:120].strip()}...\n")
            f.write("-" * 72 + "\n")

    print(f"Text report saved -> {txt_path}\n")


if __name__ == "__main__":
    main()