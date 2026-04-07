"""
evaluation/graph/hotpot_dataset.py
===================================
Graph-Based QA pipeline evaluation on HotpotQA.

Changes vs. previous version
------------------------------
* All configuration now read from evaluation/config.py (argparse kept as override)
* Per-question log written to  results/graph_eval_log.txt  (NOT the terminal)
* Terminal shows a single unified tqdm progress bar + final summary table only
* Core metrics kept aligned with the project reference summary:
    - exact_match / f1
    - precision / recall
    - substring_match / evidence_recall_at_5
    - em_by_type / f1_by_type
    - avg_reasoning_depth
    - graph size / timing statistics
* Output saved to results/ directory:
    results/graph_eval_log.txt
    results/graph_eval_results.json
    results/graph_eval_results.txt
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

from datetime import datetime
from typing import List, Dict, Tuple, Optional

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
        def __init__(self, total, desc): self.n = 0; self.total = total
        def update(self, n=1):
            self.n += n
            pct = self.n / max(self.total, 1) * 100
            print(f"\rGraph Eval: {self.n}/{self.total} ({pct:.1f}%)", end="", flush=True)
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
# Official HotpotQA normalisation / evaluation helpers
# (kept self-contained so file works standalone too)
# ---------------------------------------------------------------------------
def _normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _get_tokens(s: str) -> List[str]:
    return _normalize_answer(s).split() if s else []


def compute_exact(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    from collections import Counter
    pred_tokens  = _get_tokens(prediction)
    truth_tokens = _get_tokens(ground_truth)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    if not common:
        return 0.0
    num_same  = sum(common.values())
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Convert HotpotQA context → internal "pages" list
# ---------------------------------------------------------------------------
def hotpot_context_to_pages(context: List) -> List[Dict]:
    pages = []
    for page_idx, (title, sentences) in enumerate(context, start=1):
        text = title.strip() + "\n" + " ".join(s.strip() for s in sentences)
        pages.append({"page": page_idx, "text": text})
    return pages


# ---------------------------------------------------------------------------
# KG-augmented evidence extraction helpers  (unchanged from original)
# ---------------------------------------------------------------------------
def _kg_evidence_for_bridge(kg, question: str, sentence_nodes: List[Dict]) -> List[str]:
    evidence: List[str] = []
    if kg is None or kg.graph.number_of_nodes() == 0:
        return evidence
    q_words   = set(_normalize_answer(question).split())
    top_ents  = kg.top_entities(n=20)
    q_entities = [e["entity"] for e in top_ents
                  if any(w in e["entity"] for w in q_words)][:6]
    for i in range(len(q_entities)):
        for j in range(i + 1, len(q_entities)):
            path = kg.shortest_path_evidence(q_entities[i], q_entities[j])
            if path:
                evidence.extend(path[:3])
    return evidence[:10]


def _kg_evidence_for_entity(kg, entity: str, max_hops: int = 2) -> List[str]:
    triples = kg.query_entity(entity, max_hops=max_hops)
    return [f"{t['subject']} {t['relation']} {t['object']}" for t in triples[:8]]


def _kg_boolean_vote(kg, entity1: str, entity2: str, question: str) -> Optional[str]:
    if kg is None or kg.graph.number_of_nodes() == 0:
        return None
    facts1 = _kg_evidence_for_entity(kg, entity1)
    facts2 = _kg_evidence_for_entity(kg, entity2)
    all_facts = " ".join(facts1 + facts2).lower()
    neg_patterns = [r'\bnot\b', r'\bnever\b', r'\bno\b', r'\bneither\b', r"n't\b", r'\bdifferent\b', r'\bvaries\b']
    neg_count = sum(len(re.findall(p, all_facts)) for p in neg_patterns)
    both_kw   = ["both", "same", "also", "either", "share", "similar"]
    e1_present = entity1.lower() in all_facts
    e2_present = entity2.lower() in all_facts
    if neg_count >= 2:
        return "no"
    if any(w in question.lower() for w in both_kw) and e1_present and e2_present and neg_count == 0:
        return "yes"
    return None


# ---------------------------------------------------------------------------
# Main runner — called from evaluation/run_eval.py
# ---------------------------------------------------------------------------
def run_graph_eval() -> Optional[Dict]:
    """
    Run graph pipeline evaluation.
    Returns the analysis dict or None on error.
    """
    # Late import of config so this file can also run standalone
    try:
        from evaluation import config as _cfg
        _NUM   = _cfg.NUM_QUESTIONS
        _SEED  = _cfg.SEED
        _EMBED = _cfg.EMBED_MODEL
        _DRG   = _cfg.DRG_THRESHOLD
        _SPAN  = _cfg.SPAN_THRESHOLD
        _KGM   = _cfg.KG_MODEL
        _KGH   = _cfg.KG_HOPS
        _HPATH = _cfg.get_hotpot_path()
        _RDIR  = _cfg.get_results_dir()
    except ImportError:
        # Standalone fallback — use argparse
        args   = _parse_args()
        _NUM   = args.num
        _SEED  = args.seed
        _EMBED = args.embed_model
        _DRG   = args.drg_threshold
        _SPAN  = args.span_threshold
        _KGM   = args.kg_model
        _KGH   = args.kg_hops
        _HPATH = args.dataset if os.path.isabs(args.dataset) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.dataset)
        _RDIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "results")
        os.makedirs(_RDIR, exist_ok=True)

    print("\n" + "=" * 72)
    print("  GRAPH EVALUATION  (HotpotQA — Graph-Based QA Pipeline)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Lazy-import pipeline components
    # ------------------------------------------------------------------
    # Ensure project root is on the path
    _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

    from parser.drg_nodes import build_nodes
    from parser.drg_graph import DocumentReasoningGraph
    from parser.span_extractor import SpanExtractor
    from parser.span_graph import SpanGraph
    from parser.knowledge_graph import KnowledgeGraph
    from parser.enhanced_reasoner import EnhancedHybridReasoner
    from parser.answer_selector import select_answer
    from parser.evaluator import QAEvaluator

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    if not os.path.exists(_HPATH):
        print(f"[ERROR] Dataset not found: {_HPATH}")
        return None

    print(f"  Loading dataset : {_HPATH}")
    with open(_HPATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # ------------------------------------------------------------------
    # Sample questions (same logic as metrics.sample_questions)
    # ------------------------------------------------------------------
    try:
        from evaluation.metrics import sample_questions
        selected_questions = sample_questions(all_data, _NUM, _SEED)
    except ImportError:
        # Standalone sampling fallback
        selected_questions = _sample_questions_fallback(all_data, _NUM, _SEED)

    num_to_run = len(selected_questions)
    print(f"  Questions       : {num_to_run}  (seed={_SEED})")
    print(f"  Embed model     : {_EMBED}")
    print(f"  DRG threshold   : {_DRG}  |  Span threshold: {_SPAN}")
    print(f"  Results dir     : {_RDIR}")
    print("=" * 72 + "\n")

    # ------------------------------------------------------------------
    # Open log file — all per-question detail goes here
    # ------------------------------------------------------------------
    log_path = os.path.join(_RDIR, "graph_eval_log.txt")
    log_f    = open(log_path, "w", encoding="utf-8")
    log_f.write("GRAPH EVAL LOG  (Graph-Based QA Pipeline on HotpotQA)\n")
    log_f.write(f"Generated  : {datetime.now().isoformat(timespec='seconds')}\n")
    log_f.write(f"Embed      : {_EMBED}  |  KG: {_KGM}\n")
    log_f.write(f"Questions  : {num_to_run}  (seed={_SEED})\n")
    log_f.write("=" * 80 + "\n\n")

    # ------------------------------------------------------------------
    # Per-question evaluation loop
    # ------------------------------------------------------------------
    qa_records:    List[Dict] = []
    sep            = "-" * 80
    pipeline_start = time.time()

    with make_progress_bar(num_to_run, "Graph Eval") as pbar:
        for q_idx, item in enumerate(selected_questions, start=1):
            q_id        = item.get("_id", f"q{q_idx}")
            question    = item["question"]
            gold_answer = item["answer"]
            q_type      = item.get("type", "unknown")
            q_level     = item.get("level", "unknown")
            context     = item["context"]
            # Write question header to log only
            log_f.write(f"\n{sep}\n")
            log_f.write(f"Q{q_idx:04d}/{num_to_run}  [{q_type}|{q_level}]  id={q_id}\n")
            log_f.write(f"Question : {question}\n")
            log_f.write(f"Gold     : {gold_answer}\n")

            q_start = time.time()

            # ── 1. Context → pages ─────────────────────────────────────
            pages = hotpot_context_to_pages(context)
            if not pages:
                log_f.write("  SKIP: empty context\n")
                pbar.update(1)
                continue

            # ── 2. Sentence nodes ──────────────────────────────────────
            sentence_nodes = build_nodes(pages)
            if not sentence_nodes:
                log_f.write("  SKIP: no sentence nodes\n")
                pbar.update(1)
                continue

            sentence_texts = [node["text"] for node in sentence_nodes]

            # ── 3. Knowledge Graph ─────────────────────────────────────
            kg = KnowledgeGraph(model_name=_KGM)
            kg.build_graph(sentence_texts)
            kg_stats = kg.get_stats()
            log_f.write(
                f"  [KG] nodes={kg_stats['nodes']}  edges={kg_stats['edges']}"
                f"  communities={kg_stats['communities']}"
                f"  density={kg_stats['density']:.4f}\n"
            )

            # ── 4. Document Reasoning Graph ────────────────────────────
            drg = DocumentReasoningGraph(model_name=_EMBED)
            drg.add_nodes(sentence_nodes)
            drg.compute_embeddings()
            drg.add_semantic_edges(threshold=_DRG)
            drg.compute_graph_metrics()

            if drg.graph.number_of_nodes() == 0:
                log_f.write("  SKIP: DRG empty\n")
                gc.collect()
                pbar.update(1)
                continue

            # ── 5. Span extraction + Span Graph ───────────────────────
            span_extractor     = SpanExtractor()
            spans              = span_extractor.extract_spans_from_nodes(sentence_nodes)
            span_graph_builder = SpanGraph(model_name=_EMBED)
            span_graph_builder.add_nodes(spans)
            span_graph_builder.compute_embeddings()
            span_graph_builder.add_semantic_edges(threshold=_SPAN)
            span_graph_builder.add_discourse_edges()
            span_graph_builder.add_entity_overlap_edges()
            span_graph_builder.compute_graph_metrics()

            if span_graph_builder.graph.number_of_nodes() == 0:
                log_f.write("  SKIP: Span Graph empty\n")
                gc.collect()
                pbar.update(1)
                continue

            # ── 6. Reasoning ───────────────────────────────────────────
            reasoner    = EnhancedHybridReasoner(
                sentence_graph=drg.graph,
                span_graph=span_graph_builder.graph,
                kg_graph=kg.graph,
                model_name=_EMBED,
            )
            results      = reasoner.enhanced_reasoning(question, k=5)
            final_spans  = results.get("final_spans", [])

            answer         = "No answer found"
            evidence_spans = []
            confidence     = 0.0

            if final_spans:
                answer, evidence_spans, confidence = select_answer(
                    results, span_graph_builder, question,
                    reasoner=reasoner, max_length=220,
                )

            # ── 7. KG-augmented evidence injection ─────────────────────
            kg_evidence: List[str] = []
            if q_type == "bridge":
                kg_evidence = _kg_evidence_for_bridge(kg, question, sentence_nodes)
                if kg_evidence:
                    log_f.write(f"  [KG bridge] {kg_evidence[0][:120]}\n")

            # ── 8. Comparison post-processing (KG-enhanced) ────────────
            if q_type == "comparison":
                try:
                    from parser.comparison_utils import (
                        extract_comparison_entities,
                        classify_comparison_type,
                    )
                    e1, e2       = extract_comparison_entities(question)
                    comp_subtype = classify_comparison_type(question)

                    def _retrieve(entity):
                        hits = []
                        for n in spans:
                            text = n.get("text", "") if isinstance(n, dict) else str(n)
                            if entity.lower() in text.lower():
                                hits.append(text)
                            if len(hits) >= 5: break
                        if len(hits) < 5:
                            for p in pages:
                                if entity.lower() in p.get("text", "").lower():
                                    hits.append(p["text"])
                                if len(hits) >= 5: break
                        hits.extend(_kg_evidence_for_entity(kg, entity))
                        return hits

                    def _get_year(texts):
                        for t in texts:
                            years = re.findall(r'\b(1[89]\d{2}|20\d{2})\b', t)
                            if years: return int(years[0])
                        return None

                    def _clean_entity(ent):
                        noise = r'^(are\s+the|is\s+the|are\s+|is\s+|the\s+|both\s+)'
                        return re.sub(noise, '', ent, flags=re.IGNORECASE).strip()

                    if e1 and e2:
                        e1 = _clean_entity(e1); e2 = _clean_entity(e2)
                        e1_texts = _retrieve(e1); e2_texts = _retrieve(e2)
                        all_evid = " ".join(e1_texts + e2_texts).lower()
                        log_f.write(f"  [comp:{comp_subtype}] e1='{e1}' e2='{e2}'\n")

                        if comp_subtype == "boolean":
                            kg_vote   = _kg_boolean_vote(kg, e1, e2, question)
                            strong_neg = ["neither", "only one", "not both",
                                          "different countr", "different locat", "not the same"]
                            soft_neg   = any(
                                re.search(r'\bnot\s+(?:located|headquartered|based|available'
                                          r'|a\s+\w+|in\s+the)',
                                          " ".join(_retrieve(ent)).lower())
                                for ent in [e1, e2]
                            )
                            both_kw    = ["both", "same", "also", "either", "share"]
                            is_both_q  = any(w in question.lower() for w in both_kw)
                            if kg_vote is not None:
                                answer = kg_vote
                            elif any(n in all_evid for n in strong_neg) or soft_neg:
                                answer = "no"
                            elif is_both_q and e1.lower() in all_evid and e2.lower() in all_evid:
                                answer = "yes"
                            elif e1.lower() in all_evid and e2.lower() in all_evid:
                                answer = "yes"

                        elif answer and answer.lower() not in ("yes", "no", "no answer found"):
                            for entity in [e1, e2]:
                                if answer.lower() in entity.lower() and len(entity) > len(answer):
                                    answer = entity; break

                        if comp_subtype == "comparative":
                            q_lower   = question.lower()
                            is_first  = any(w in q_lower for w in ["first", "earlier", "oldest"])
                            is_recent = any(w in q_lower for w in ["recent", "latest", "newest"])
                            if is_first or is_recent:
                                y1 = _get_year(e1_texts); y2 = _get_year(e2_texts)
                                if y1 and y2:
                                    answer = e1 if (
                                        (is_first and y1 < y2) or (is_recent and y1 > y2)
                                    ) else e2

                        bridge_path = kg.shortest_path_evidence(e1, e2)
                        if bridge_path:
                            kg_evidence.extend(bridge_path[:2])
                except Exception as exc:
                    log_f.write(f"  [comp error] {exc}\n")

            elapsed = time.time() - q_start

            # ── 9. Core metrics ────────────────────────────────────────
            em = compute_exact(answer, gold_answer)
            f1 = compute_f1(answer, gold_answer)

            internal_eval = {
                "exact_match": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "substring_match": 0.0,
            }
            recall_at_5 = 0.0
            if answer != "No answer found":
                internal_eval = QAEvaluator.evaluate(answer, gold_answer)
                if evidence_spans:
                    recall_at_5 = QAEvaluator.evidence_recall_at_k(
                        evidence_spans, gold_answer, k=5)

            reasoning_depth = QAEvaluator.reasoning_depth(
                results.get("traversal_results", []),
                results.get("expansion_results", []),
            )

            # ── Write to log ───────────────────────────────────────────
            conf_label = "Low"
            if confidence >= 0.8:   conf_label = "High"
            elif confidence >= 0.45: conf_label = "Med"

            log_f.write(f"Predicted: {answer}\n")
            log_f.write(
                f"EM={em:.0f}  F1={f1:.4f}  P={internal_eval['precision']:.4f}"
                f"  R={internal_eval['recall']:.4f}  Sub={internal_eval['substring_match']:.0f}"
                f"  Conf={conf_label}({confidence:.2%})"
                f"  Depth={reasoning_depth}  Time={elapsed:.2f}s\n"
            )
            log_f.write(
                f"KG nodes={kg_stats['nodes']} edges={kg_stats['edges']}"
                f" comms={kg_stats['communities']}"
                f" DRG_nodes={drg.graph.number_of_nodes()}"
                f" Span_nodes={span_graph_builder.graph.number_of_nodes()}\n"
            )
            log_f.write(sep + "\n")
            log_f.flush()

            # ── Update progress bar ────────────────────────────────────
            pbar.set_postfix_str(
                f"EM={em:.0f} F1={f1:.2f} Depth={reasoning_depth}"
            )
            pbar.update(1)

            # ── Store record ───────────────────────────────────────────
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
                "precision":       internal_eval["precision"],
                "recall":          internal_eval["recall"],
                "substring_match": internal_eval["substring_match"],
                "confidence":       confidence,
                "time_s":           round(elapsed, 3),
                "reasoning_depth":  reasoning_depth,
                "evidence_count":   len(evidence_spans),
                "evidence_spans":   evidence_spans[:3],
                "kg_evidence":      kg_evidence[:5],
                "evidence_recall_at_5": round(recall_at_5, 4),
                "internal_eval": {
                    "exact_match": round(internal_eval["exact_match"], 4),
                    "precision":   round(internal_eval["precision"], 4),
                    "recall":      round(internal_eval["recall"], 4),
                    "f1":          round(internal_eval["f1"], 4),
                    "substring_match": round(internal_eval["substring_match"], 4),
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
                    "kg_nodes":         kg_stats["nodes"],
                    "kg_edges":         kg_stats["edges"],
                    "kg_communities":   kg_stats["communities"],
                    "kg_density":       kg_stats["density"],
                    "kg_entity_labels": kg_stats["entity_labels"],
                },
            })

            # ── Free memory ────────────────────────────────────────────
            del drg, span_graph_builder, span_extractor, reasoner, results, kg
            gc.collect()
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

    log_f.close()
    total_time = time.time() - pipeline_start

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    answered   = [r for r in qa_records if r["predicted_answer"] != "No answer found"]
    total_q    = len(qa_records)
    levels     = ["easy", "medium", "hard"]
    types      = ["bridge", "comparison"]

    def _avg(recs, key): return sum(r[key] for r in recs) / max(len(recs), 1)

    em_by_level = {lvl: round(_avg([r for r in qa_records if r["level"]==lvl], "exact_match"), 4)
                   for lvl in levels}
    f1_by_level = {lvl: round(_avg([r for r in qa_records if r["level"]==lvl], "f1"), 4)
                   for lvl in levels}
    precision_by_level = {lvl: round(_avg([r for r in qa_records if r["level"]==lvl], "precision"), 4)
                          for lvl in levels}
    recall_by_level = {lvl: round(_avg([r for r in qa_records if r["level"]==lvl], "recall"), 4)
                       for lvl in levels}
    substring_by_level = {lvl: round(_avg([r for r in qa_records if r["level"]==lvl], "substring_match"), 4)
                          for lvl in levels}
    evidence_recall_at_5_by_level = {
        lvl: round(_avg([r for r in qa_records if r["level"] == lvl], "evidence_recall_at_5"), 4)
        for lvl in levels
    }
    questions_by_level = {lvl: len([r for r in qa_records if r["level"] == lvl]) for lvl in levels}
    answered_by_level = {
        lvl: len([r for r in qa_records if r["level"] == lvl and r["predicted_answer"] != "No answer found"])
        for lvl in levels
    }
    bridge_by_level = {lvl: len([r for r in qa_records if r["level"] == lvl and r["type"] == "bridge"]) for lvl in levels}
    comparison_by_level = {lvl: len([r for r in qa_records if r["level"] == lvl and r["type"] == "comparison"]) for lvl in levels}
    em_by_type  = {qt:  round(_avg([r for r in qa_records if r["type"]==qt],   "exact_match"), 4)
                   for qt in types}
    f1_by_type  = {qt:  round(_avg([r for r in qa_records if r["type"]==qt],   "f1"), 4)
                   for qt in types}

    avg_em    = round(_avg(qa_records, "exact_match"), 4)
    avg_f1    = round(_avg(qa_records, "f1"), 4)
    avg_precision = round(_avg(qa_records, "precision"), 4)
    avg_recall = round(_avg(qa_records, "recall"), 4)
    avg_substring = round(_avg(qa_records, "substring_match"), 4)
    avg_depth = round(_avg(answered, "reasoning_depth"), 2) if answered else 0.0
    avg_time  = round(_avg(qa_records, "time_s"), 3)
    avg_eval_r5 = round(_avg(qa_records, "evidence_recall_at_5"), 4)
    avg_eval_r5_by_level = {
        lvl: round(_avg([r for r in qa_records if r["level"] == lvl], "evidence_recall_at_5"), 4)
        for lvl in levels
    }

    drg_nodes_avg = sum(r["graph_stats"].get("drg_nodes", 0) for r in qa_records) / max(total_q, 1)
    drg_edges_avg = sum(r["graph_stats"].get("drg_edges", 0) for r in qa_records) / max(total_q, 1)
    span_nodes_avg = sum(r["graph_stats"].get("span_nodes", 0) for r in qa_records) / max(total_q, 1)
    span_edges_avg = sum(r["graph_stats"].get("span_edges", 0) for r in qa_records) / max(total_q, 1)
    kg_nodes_avg = sum(r["graph_stats"].get("kg_nodes", 0) for r in qa_records) / max(total_q, 1)
    kg_edges_avg = sum(r["graph_stats"].get("kg_edges", 0) for r in qa_records) / max(total_q, 1)

    analysis = {
        "system":                    "GBDQA (Graph Pipeline)",
        "dataset":                   os.path.basename(_HPATH),
        "embed_model":               _EMBED,
        "kg_model":                  _KGM,
        "total_questions":           total_q,
        "answered":                  len(answered),
        "unanswered":                total_q - len(answered),
        "answer_rate_pct":           round(len(answered) / max(total_q, 1) * 100, 1),
        # Core metrics
        "exact_match":               avg_em,
        "precision":                 avg_precision,
        "recall":                    avg_recall,
        "f1":                        avg_f1,
        "substring_match":           avg_substring,
        # Level / type breakdown
        "em_by_level":               em_by_level,
        "f1_by_level":               f1_by_level,
        "precision_by_level":        precision_by_level,
        "recall_by_level":           recall_by_level,
        "substring_by_level":        substring_by_level,
        "questions_by_level":        questions_by_level,
        "answered_by_level":         answered_by_level,
        "bridge_by_level":           bridge_by_level,
        "comparison_by_level":       comparison_by_level,
        "em_by_type":                em_by_type,
        "f1_by_type":                f1_by_type,
        "evidence_recall_at_5":      avg_eval_r5,
        "evidence_recall_at_5_by_level": avg_eval_r5_by_level,
        "avg_reasoning_depth":       avg_depth,
        "drg_avg_nodes":             round(drg_nodes_avg, 1),
        "drg_avg_edges":             round(drg_edges_avg, 1),
        "span_avg_nodes":            round(span_nodes_avg, 1),
        "span_avg_edges":            round(span_edges_avg, 1),
        # Efficiency
        "avg_time_per_question_s":   avg_time,
        "total_pipeline_time_s":     round(total_time, 2),
        "kg_avg_nodes":              round(kg_nodes_avg, 1),
        "kg_avg_edges":              round(kg_edges_avg, 1),
    }

    # ------------------------------------------------------------------
    # Terminal summary table
    # ------------------------------------------------------------------
    _print_graph_summary(analysis, qa_records, levels)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis":     analysis,
        "qa":           qa_records,
    }

    json_path = os.path.join(_RDIR, "graph_eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\n  JSON results → {json_path}")

    txt_path = os.path.join(_RDIR, "graph_eval_results.txt")
    _save_txt_report(txt_path, analysis, qa_records, _EMBED, _KGM)
    print(f"  Text report  → {txt_path}")
    print(f"  Per-Q log    → {log_path}\n")

    return analysis


# ---------------------------------------------------------------------------
# Terminal summary printer
# ---------------------------------------------------------------------------
def _print_graph_summary(analysis: Dict, qa_records: List[Dict], levels: List[str]) -> None:
    W = 72
    print("\n" + "=" * W)
    print("  GRAPH PIPELINE — ANALYSIS SUMMARY")
    print("=" * W)

    rows = [
        ("Questions evaluated",    f"{analysis['total_questions']}"),
        ("Answered",               f"{analysis['answered']}  ({analysis['answer_rate_pct']}%)"),
        ("Exact Match (EM)",       f"{analysis['exact_match']:.4f}  ({analysis['exact_match']*100:.1f}%)"),
        ("F1 Score",               f"{analysis['f1']:.4f}  ({analysis['f1']*100:.1f}%)"),
        ("Precision / Recall",     f"{analysis['precision']:.4f} / {analysis['recall']:.4f}"),
        ("Substring / R@5",        f"{analysis['substring_match']:.4f} / {analysis['evidence_recall_at_5']:.4f}"),
        ("EM bridge / compare",    f"{analysis['em_by_type']['bridge']:.4f} / {analysis['em_by_type']['comparison']:.4f}"),
        ("F1 bridge / compare",    f"{analysis['f1_by_type']['bridge']:.4f} / {analysis['f1_by_type']['comparison']:.4f}"),
        ("Avg reasoning depth",    f"{analysis['avg_reasoning_depth']:.2f}"),
        ("Avg DRG nodes/edges",     f"{analysis['drg_avg_nodes']:.0f} / {analysis['drg_avg_edges']:.0f}"),
        ("Avg Span nodes/edges",    f"{analysis['span_avg_nodes']:.0f} / {analysis['span_avg_edges']:.0f}"),
        ("Avg KG nodes/edges",      f"{analysis['kg_avg_nodes']:.0f} / {analysis['kg_avg_edges']:.0f}"),
        ("Avg time / question",    f"{analysis['avg_time_per_question_s']:.2f}s"),
        ("Total pipeline time",    f"{analysis['total_pipeline_time_s']/60:.1f}min  ({analysis['total_pipeline_time_s']:.0f}s)"),
    ]
    for label, val in rows:
        print(f"  {label:<35}: {val}")
    print("-" * W)

    # Level-wise table
    print(f"\n  {'Level':<10} {'N':>6} {'EM':>10} {'F1':>10} {'Precision / Recall':>22} {'Substring / R@5':>20} {'EM bridge / compare':>24} {'F1 bridge / compare':>24}")
    print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*22} {'-'*20} {'-'*24} {'-'*24}")
    for lvl in levels:
        recs = [r for r in qa_records if r.get("level") == lvl]
        if not recs: continue
        def _a(k): return sum(r[k] for r in recs) / len(recs)
        print(
            f"  {lvl:<10} {len(recs):>6}"
            f" {_a('exact_match')*100:>9.1f}%"
            f" {_a('f1')*100:>9.1f}%"
            f" {(_a('precision')):>9.4f} / {(_a('recall')):<9.4f}"
            f" {(_a('substring_match')):>9.4f} / {analysis['evidence_recall_at_5'] if analysis.get('evidence_recall_at_5') is not None else 0.0:<9.4f}"
            f" {analysis['em_by_type']['bridge']:.4f} / {analysis['em_by_type']['comparison']:.4f}"
            f" {analysis['f1_by_type']['bridge']:.4f} / {analysis['f1_by_type']['comparison']:.4f}"
        )
    print("=" * W)


# ---------------------------------------------------------------------------
# Text report saver
# ---------------------------------------------------------------------------
def _save_txt_report(path: str, analysis: Dict, qa_records: List[Dict],
                     embed_model: str, kg_model: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("GRAPH EVAL RESULTS  (Graph-Based QA Pipeline on HotpotQA)\n")
        f.write(f"Generated : {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Model     : {embed_model}  |  KG: {kg_model}\n\n")
        f.write("ANALYSIS SUMMARY\n" + "=" * 80 + "\n")
        ordered_keys = [
            "system", "dataset", "embed_model", "kg_model", "total_questions", "answered",
            "unanswered", "answer_rate_pct", "exact_match", "precision", "recall", "f1",
            "substring_match", "em_by_level", "f1_by_level", "precision_by_level",
            "recall_by_level", "substring_by_level", "questions_by_level", "answered_by_level",
            "bridge_by_level", "comparison_by_level", "em_by_type", "f1_by_type",
            "evidence_recall_at_5", "evidence_recall_at_5_by_level", "avg_reasoning_depth", "drg_avg_nodes", "drg_avg_edges",
            "span_avg_nodes", "span_avg_edges", "kg_avg_nodes", "kg_avg_edges",
            "avg_time_per_question_s", "total_pipeline_time_s",
        ]
        for k in ordered_keys:
            if k in analysis:
                f.write(f"  {k:<45}: {analysis[k]}\n")


# ---------------------------------------------------------------------------
# Standalone sampling fallback (if metrics.py unavailable)
# ---------------------------------------------------------------------------
def _sample_questions_fallback(all_data, num_questions, seed):
    if num_questions == 0:
        return all_data
    levels = ["easy", "medium", "hard"]
    pools  = {}
    rng    = random.Random(seed)
    for lvl in levels:
        lvl_items = [i for i in all_data if i.get("level") == lvl]
        bridge    = [i for i in lvl_items if i.get("type") == "bridge"]
        comp      = [i for i in lvl_items if i.get("type") == "comparison"]
        rng.shuffle(bridge); rng.shuffle(comp)
        pools[f"{lvl}_bridge"] = bridge
        pools[f"{lvl}_comparison"] = comp
    per_level = num_questions // 3; rem = num_questions % 3
    selected  = []
    for i, lvl in enumerate(levels):
        lc = per_level + (1 if i < rem else 0)
        bc = lc // 2; cc = lc - bc
        selected.extend(pools[f"{lvl}_bridge"][:min(bc, len(pools[f'{lvl}_bridge']))])
        selected.extend(pools[f"{lvl}_comparison"][:min(cc, len(pools[f'{lvl}_comparison']))])
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# CLI parsing for standalone usage
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Graph-Based QA evaluation on HotpotQA")
    p.add_argument("--dataset",        default="hotpot_train_v1.1.json")
    p.add_argument("--num",            type=int, default=300)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--embed-model",    default="BAAI/bge-large-en-v1.5")
    p.add_argument("--drg-threshold",  type=float, default=0.75)
    p.add_argument("--span-threshold", type=float, default=0.70)
    p.add_argument("--kg-model",       default="en_core_web_trf")
    p.add_argument("--kg-hops",        type=int, default=2)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_graph_eval()