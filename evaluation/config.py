"""
evaluation/config.py
====================
Central configuration for the Graph-Based QA Evaluation Pipeline.

Edit this file to control what gets evaluated — no CLI arguments needed.
Run:  python -m evaluation.run_eval
"""

import os

# ---------------------------------------------------------------------------
# Core evaluation settings
# ---------------------------------------------------------------------------

# Number of questions to evaluate (split equally across easy/medium/hard).
# Set to 0 to run the entire dataset (very slow).
NUM_QUESTIONS: int = 1     # 100 per difficulty level

# Random seed — identical seed guarantees identical question sets across
# graph and LLM evaluations so comparisons are fair.
SEED: int = 42

# Which evaluations to run:
#   "graph"  → only graph pipeline
#   "llm"    → only LLM baselines
#   "all"    → graph + all LLM baselines, then combined table
RUN_MODE: str = "graph"   # "graph", "llm", or "all"

# ---------------------------------------------------------------------------
# Dataset path
# ---------------------------------------------------------------------------

# Absolute or relative-to-project-root path.
HOTPOT_PATH: str = "hotpot_train_v1.1.json"

# ---------------------------------------------------------------------------
# Graph pipeline settings
# ---------------------------------------------------------------------------

EMBED_MODEL: str      = "BAAI/bge-large-en-v1.5"
DRG_THRESHOLD: float  = 0.75
SPAN_THRESHOLD: float = 0.70
KG_MODEL: str         = "en_core_web_trf"   # spaCy model; "en_core_web_sm" for speed
KG_HOPS: int          = 2

# ---------------------------------------------------------------------------
# LLM / Ollama settings
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str  = "http://localhost:11434"
OLLAMA_MODEL: str     = "llama3"
LLM_MAX_TOKENS: int   = 100        # keep short for extractive-style answers
LLM_TEMPERATURE: float = 0.0       # deterministic
LLM_SLEEP_S: float    = 0.2        # throttle between requests (system stability)
LLM_TIMEOUT_S: int    = 60         # per-request HTTP timeout

# Naive RAG top-k sentences fed to LLM as context
NAIVE_RAG_TOP_K: int  = 5

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------

# Relative to project root (i.e., the directory that contains evaluation/)
RESULTS_DIR: str = "results"

# ---------------------------------------------------------------------------
# Derived / helper paths (do not edit below this line)
# ---------------------------------------------------------------------------

def _project_root() -> str:
    """Return absolute path to project root (parent of evaluation/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_hotpot_path() -> str:
    """Return absolute path to the HotpotQA dataset."""
    p = HOTPOT_PATH
    if not os.path.isabs(p):
        p = os.path.join(_project_root(), p)
    return p


def get_results_dir() -> str:
    """Return absolute path to results directory, creating it if needed."""
    d = RESULTS_DIR
    if not os.path.isabs(d):
        d = os.path.join(_project_root(), d)
    os.makedirs(d, exist_ok=True)
    return d
