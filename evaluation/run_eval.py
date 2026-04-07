"""
evaluation/run_eval.py
=======================
Single entry point for the evaluation pipeline.

Usage
-----
    # From project root (with venv active):
    python -m evaluation.run_eval

    # Override mode / sample size / seed from the command line:
    python -m evaluation.run_eval --mode llm --num 6 --seed 42

Configuration
-------------
    CLI arguments override evaluation/config.py for convenience.

    config.RUN_MODE:
        "graph"  → Only run graph pipeline evaluation
        "llm"    → Only run LLM baselines (requires Ollama + llama3)
        "all"    → Run both, then print combined comparison table

    config.NUM_QUESTIONS:
        Number of HotpotQA questions (300 = 100 per difficulty level).
        Set to 0 to use the full dataset.
"""

from __future__ import annotations

import sys
import os
import json
import argparse

# Ensure project root (parent of evaluation/) is on sys.path so that
# `from parser.xxx import yyy` works correctly from any working directory.
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from evaluation import config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Graph-based QA evaluation pipeline")
    parser.add_argument("--mode", choices=["graph", "llm", "all"], default=config.RUN_MODE)
    parser.add_argument("--num", type=int, default=config.NUM_QUESTIONS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.RUN_MODE = args.mode
    config.NUM_QUESTIONS = args.num
    config.SEED = args.seed

    print("\n" + "=" * 72)
    print("  GRAPH-BASED QA  —  EVALUATION PIPELINE")
    print(f"  Mode          : {config.RUN_MODE}")
    print(f"  Questions     : {config.NUM_QUESTIONS}  (seed={config.SEED})")
    print(f"  HotpotQA path : {config.get_hotpot_path()}")
    print(f"  Results dir   : {config.get_results_dir()}")
    print("=" * 72)

    graph_analysis      = None
    llm_analysis        = None
    naive_rag_analysis  = None

    # ------------------------------------------------------------------
    # Graph evaluation
    # ------------------------------------------------------------------
    if config.RUN_MODE in ("graph", "all"):
        from evaluation.graph.hotpot_dataset import run_graph_eval
        graph_analysis = run_graph_eval()
        if graph_analysis is None:
            print("[WARN] Graph evaluation failed or returned no results.")

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------
    if config.RUN_MODE in ("llm", "all"):
        from evaluation.llm.llm_eval import run_llm_eval
        llm_result = run_llm_eval()
        if llm_result is not None:
            llm_analysis = llm_result.get("llm")
        else:
            print("[WARN] LLM evaluation failed (is Ollama running?).")

        from evaluation.llm.naive_rag import run_naive_rag_eval
        rag_result = run_naive_rag_eval()
        if rag_result is not None:
            naive_rag_analysis = rag_result.get("naive_rag")
        else:
            print("[WARN] Naive RAG evaluation failed.")

    # ------------------------------------------------------------------
    # Combined comparison table
    # ------------------------------------------------------------------
    if config.RUN_MODE == "all":
        from evaluation.results_table import print_final_table
        results_dir = config.get_results_dir()
        print_final_table(
            graph_analysis,
            None,
            llm_analysis,
            naive_rag_analysis,
            results_dir=results_dir,
        )

    elif config.RUN_MODE == "graph" and graph_analysis is not None:
        # Even in graph-only mode, show a condensed table with just our system
        from evaluation.results_table import print_final_table
        results_dir = config.get_results_dir()
        print_final_table(
            graph_analysis, None, None, None,
            results_dir=results_dir,
        )

    elif config.RUN_MODE == "llm":
        # LLM-only: print a 3-column table (no graph column)
        # Re-use print_final_table with None for graph
        from evaluation.results_table import print_final_table
        results_dir = config.get_results_dir()
        print_final_table(
            None, None, llm_analysis, naive_rag_analysis,
            results_dir=results_dir,
        )

    print("\n  All done! Results saved to:", config.get_results_dir())
    print("  Files:")
    results_dir = config.get_results_dir()
    for fname in sorted(os.listdir(results_dir)):
        fpath = os.path.join(results_dir, fname)
        size  = os.path.getsize(fpath)
        print(f"    {fname:<40} {size:>10,} bytes")
    print()


if __name__ == "__main__":
    main()
