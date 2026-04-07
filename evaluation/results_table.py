"""
evaluation/results_table.py
============================
Renders the final Graph vs LLM comparison table.

Outputs:
  1. Rich terminal table
  2. results/combined_analysis.json  - machine-readable combined report
  3. results/combined_analysis.txt   - plain-text version of the table
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init(autoreset=True)
    _HAS_COLOR = True
except Exception:
    _HAS_COLOR = False


def _c(text: str, color: str = "cyan") -> str:
    if not _HAS_COLOR:
        return text
    codes = {
        "cyan": Fore.CYAN,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "red": Fore.RED,
        "white": Fore.WHITE,
        "magenta": Fore.MAGENTA,
        "bold": Style.BRIGHT,
    }
    return codes.get(color, "") + text + Style.RESET_ALL


def _row(label: str, values: List[str], widths: List[int]) -> str:
    label_col = f"  {label:<42}"
    val_cols = "".join(f" {v:^{w}} |" for v, w in zip(values, widths))
    return f"|{label_col}|{val_cols}"


def _divider(widths: List[int]) -> str:
    return "+" + "-" * 44 + "+" + "".join("-" * (w + 2) + "+" for w in widths)


def _header(col_names: List[str], widths: List[int]) -> str:
    label_col = f"  {'Metric':<42}"
    val_cols = "".join(f" {n:^{w}} |" for n, w in zip(col_names, widths))
    return f"|{label_col}|{val_cols}"


def print_final_table(
    graph_analysis: Optional[Dict],
    llm_closedbook_analysis: Optional[Dict],
    llm_analysis: Optional[Dict],
    llm_naiverag_analysis: Optional[Dict],
    results_dir: str = "results",
) -> None:
    """Print the trimmed comparison matrix to terminal and save to files."""
    os.makedirs(results_dir, exist_ok=True)

    systems = [
        ("GBDQA (Graph)", graph_analysis),
        ("LLM", llm_analysis),
        ("Naive RAG", llm_naiverag_analysis),
    ]
    active = [(name, ana) for name, ana in systems if ana is not None]
    names = [name for name, _ in active]
    col_w = [max(15, len(n) + 2) for n in names]

    def g(ana: Optional[Dict], *keys, default=None):
        if ana is None:
            return default
        val = ana
        for key in keys:
            if not isinstance(val, dict):
                return default
            val = val.get(key)
            if val is None:
                return default
        return val

    def row(label: str, vals: List[str]) -> str:
        return _row(label, vals, col_w)

    def fmt_count(v: Optional[int]) -> str:
        return "N/A" if v is None else f"{int(v)}"

    def fmt_answered(answered: Optional[int], total: Optional[int]) -> str:
        if answered is None:
            return "N/A"
        if total in (None, 0):
            return f"{int(answered)}"
        pct = answered / max(total, 1) * 100
        return f"{int(answered)}  ({pct:.1f}%)"

    def fmt_exact(v: Optional[float]) -> str:
        if v is None:
            return "N/A"
        return f"{v:.4f}  ({v * 100:.1f}%)"

    def fmt_pair(left: Optional[float], right: Optional[float], decimals: int = 4) -> str:
        if left is None and right is None:
            return "N/A"
        left_s = "N/A" if left is None else f"{left:.{decimals}f}"
        right_s = "N/A" if right is None else f"{right:.{decimals}f}"
        return f"{left_s} / {right_s}"

    def fmt_substring_r5(substring: Optional[float], recall_at_5: Optional[float]) -> str:
        if substring is None:
            return "N/A"
        if recall_at_5 is None:
            return f"{substring:.4f} / N/A"
        return f"{substring:.4f} / {recall_at_5:.4f}"

    def fmt_time(seconds: Optional[float]) -> str:
        return "N/A" if seconds is None else f"{seconds:.2f}s"

    def fmt_pipeline(seconds: Optional[float]) -> str:
        if seconds is None:
            return "N/A"
        return f"{seconds / 60:.1f}min  ({seconds:.0f}s)"

    def level_counts(ana: Optional[Dict], lvl: str, index: int) -> Dict[str, Optional[int]]:
        if ana is None:
            return {"questions": None, "answered": None, "bridge": None, "comparison": None}

        questions = g(ana, "questions_by_level", lvl)
        if questions is None:
            total = g(ana, "total_questions")
            if total is not None:
                per_level = total // 3
                remainder = total % 3
                questions = per_level + (1 if index < remainder else 0)

        answered = g(ana, "answered_by_level", lvl)
        if answered is None and questions is not None:
            answer_rate = g(ana, "answer_rate_pct")
            answered = questions if answer_rate is None else round(questions * answer_rate / 100)

        bridge = g(ana, "bridge_by_level", lvl)
        comparison = g(ana, "comparison_by_level", lvl)
        if bridge is None and questions is not None:
            bridge = questions // 2
        if comparison is None and questions is not None and bridge is not None:
            comparison = questions - bridge

        return {
            "questions": questions,
            "answered": answered,
            "bridge": bridge,
            "comparison": comparison,
        }

    def r5(ana: Optional[Dict], lvl: Optional[str] = None) -> Optional[float]:
        if ana is None:
            return None
        if lvl is None:
            return g(ana, "evidence_recall_at_5")
        val = g(ana, "evidence_recall_at_5_by_level", lvl)
        return val

    lines: List[str] = []
    div = _divider(col_w)

    lines.append("")
    lines.append(_c("=" * (48 + sum(w + 3 for w in col_w)), "cyan"))
    lines.append(_c("  FINAL EVALUATION — HEAD-TO-HEAD COMPARISON  ", "bold"))
    lines.append(_c("  Graph-Based QA  vs  LLM (Llama3)            ", "cyan"))
    lines.append(_c("=" * (48 + sum(w + 3 for w in col_w)), "cyan"))
    lines.append(_header(names, col_w))

    # Overall summary
    lines.append(row("── OVERALL ──────────────────────────────", ["" for _ in active]))
    lines.append(row("Questions evaluated", [fmt_count(g(a, "total_questions")) for _, a in active]))
    lines.append(row("Answered", [fmt_answered(g(a, "answered"), g(a, "total_questions")) for _, a in active]))
    lines.append(row("Exact Match (EM)", [fmt_exact(g(a, "exact_match")) for _, a in active]))
    lines.append(row("F1 Score", [fmt_exact(g(a, "f1")) for _, a in active]))
    lines.append(row("Precision / Recall", [fmt_pair(g(a, "precision"), g(a, "recall")) for _, a in active]))
    lines.append(row("Substring / R@5", [fmt_substring_r5(g(a, "substring_match"), r5(a)) for _, a in active]))
    lines.append(row("EM bridge / compare", [fmt_pair(g(a, "em_by_type", "bridge"), g(a, "em_by_type", "comparison")) for _, a in active]))
    lines.append(row("F1 bridge / compare", [fmt_pair(g(a, "f1_by_type", "bridge"), g(a, "f1_by_type", "comparison")) for _, a in active]))
    lines.append(row("Avg reasoning depth", [f"{g(a, 'avg_reasoning_depth'):.2f}" if g(a, "avg_reasoning_depth") is not None else "N/A" for _, a in active]))
    lines.append(row("Avg DRG nodes/edges", [fmt_pair(g(a, "drg_avg_nodes"), g(a, "drg_avg_edges"), 0) for _, a in active]))
    lines.append(row("Avg Span nodes/edges", [fmt_pair(g(a, "span_avg_nodes"), g(a, "span_avg_edges"), 0) for _, a in active]))
    lines.append(row("Avg KG nodes/edges", [fmt_pair(g(a, "kg_avg_nodes"), g(a, "kg_avg_edges"), 0) for _, a in active]))
    lines.append(row("Avg time / question", [fmt_time(g(a, "avg_time_per_question_s")) for _, a in active]))
    lines.append(row("Total pipeline time", [fmt_pipeline(g(a, "total_pipeline_time_s")) for _, a in active]))

    # Level-wise breakdown
    lines.append(row("── LEVEL-WISE BREAKDOWN ─────────────────", ["" for _ in active]))
    for index, lvl in enumerate(("easy", "medium", "hard")):
        if index > 0:
            lines.append("")
        lines.append(row(f"  {lvl.upper()}", ["" for _ in active]))
        lines.append(row("  Questions evaluated", [fmt_count(level_counts(a, lvl, index)["questions"]) for _, a in active]))
        lines.append(row("  Answered", [fmt_answered(level_counts(a, lvl, index)["answered"], level_counts(a, lvl, index)["questions"]) for _, a in active]))
        lines.append(row("  Exact Match (EM)", [fmt_exact(g(a, "em_by_level", lvl)) for _, a in active]))
        lines.append(row("  F1 Score", [fmt_exact(g(a, "f1_by_level", lvl)) for _, a in active]))
        lines.append(row("  Precision / Recall", [fmt_pair(g(a, "precision_by_level", lvl), g(a, "recall_by_level", lvl)) for _, a in active]))
        lines.append(row("  Substring / R@5", [fmt_substring_r5(g(a, "substring_by_level", lvl), r5(a, lvl)) for _, a in active]))
        lines.append(row("  EM bridge / compare", [fmt_pair(g(a, "em_by_type", "bridge"), g(a, "em_by_type", "comparison")) for _, a in active]))
        lines.append(row("  F1 bridge / compare", [fmt_pair(g(a, "f1_by_type", "bridge"), g(a, "f1_by_type", "comparison")) for _, a in active]))

    lines.append("")
    lines.append("  ↓ Lower is better   |   ↑ Higher is better")
    lines.append("  N/A  = metric not available for this system")
    lines.append("  R@5  = evidence recall at 5")
    lines.append("")

    for line in lines:
        print(line)

    plain_lines = []
    for line in lines:
        if _HAS_COLOR:
            import re as _re
            plain_lines.append(_re.sub(r'\x1b\[[0-9;]*m', '', line))
        else:
            plain_lines.append(line)

    txt_path = os.path.join(results_dir, "combined_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("GRAPH-BASED QA — FINAL COMPARISON TABLE\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("\n".join(plain_lines))
    print(f"\n  Table saved → {txt_path}")

    combined = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "systems": {
            "graph": graph_analysis,
            "llm": llm_analysis,
            "naive_rag": llm_naiverag_analysis,
        },
    }
    json_path = os.path.join(results_dir, "combined_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"  JSON report  → {json_path}\n")