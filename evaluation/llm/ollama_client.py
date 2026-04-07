"""
evaluation/llm/ollama_client.py
================================
Wrapper around the local Ollama API for Llama3.

Implements two query modes:
    1. Closed-book  — question only (no context)
    2. LLM baseline — question + full HotpotQA context paragraphs

Also provides a health-check utility.

Performance notes
-----------------
* `LLM_SLEEP_S` pause after each request prevents thermal throttling and
  keeps the Ollama daemon stable across 300-5000 queries.
* `stream=False` avoids streaming complexity and is fine for short answers.
* `temperature=0` gives deterministic outputs for fair evaluation.
"""

from __future__ import annotations

import time
import requests
from typing import List, Optional, Tuple

from evaluation import config


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_ollama_health() -> bool:
    """Return True if Ollama is running and the model is available."""
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags",
                         timeout=5)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        # Accept "llama3" or "llama3:latest" etc.
        return any(config.OLLAMA_MODEL in m for m in models)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def _generate(prompt: str) -> Tuple[str, float]:
    """
    Send `prompt` to Ollama and return (answer_text, latency_ms).

    Raises RuntimeError if the request fails.
    """
    start = time.time()
    try:
        res = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature":    config.LLM_TEMPERATURE,
                    "num_predict":    config.LLM_MAX_TOKENS,
                    "stop":           ["\n\n", "Question:", "Context:"],
                },
            },
            timeout=config.LLM_TIMEOUT_S,
        )
        latency_ms = (time.time() - start) * 1000
        res.raise_for_status()
        answer = res.json().get("response", "").strip()
        # Clean up: take only first line if multi-line answer
        answer = answer.split("\n")[0].strip()
        return answer, round(latency_ms, 1)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama request timed out after {config.LLM_TIMEOUT_S}s")
    except Exception as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def _throttle() -> None:
    """Sleep between requests to keep system stable."""
    time.sleep(config.LLM_SLEEP_S)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_llm_closedbook(question: str) -> Tuple[str, float]:
    """
    Closed-book query: question only, no context.

    Purpose: tests how much the LLM relies on parametric memory (Wikipedia
    training data) rather than document reasoning.  High accuracy here exposes
    memorisation rather than genuine reasoning.

    Prompt design: directive + concise-answer instruction.
    """
    prompt = (
        f"Answer the following question with a short, direct answer "
        f"(1–5 words only). Do not explain.\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    try:
        answer, latency_ms = _generate(prompt)
    except RuntimeError:
        answer, latency_ms = "ERROR", 0.0
    _throttle()
    return answer, latency_ms


def ask_llm_baseline(
    question: str,
    context_paragraphs: List,
) -> Tuple[str, float]:
    """
    LLM-baseline query: question + full HotpotQA context paragraphs.

    This is the primary fair comparison baseline.
    The prompt instructs the model to answer ONLY from the given context,
    keeping answers short and extractive.

    `context_paragraphs` : list of (title, sentences_list) as in HotpotQA.
    """
    ctx_str = _format_context(context_paragraphs)
    prompt = (
        f"Answer the question using ONLY the information in the context below.\n"
        f"Give a short, direct answer (1–5 words). Do not explain.\n\n"
        f"Context:\n{ctx_str}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    try:
        answer, latency_ms = _generate(prompt)
    except RuntimeError:
        answer, latency_ms = "ERROR", 0.0
    _throttle()
    return answer, latency_ms


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_context(context_paragraphs: List) -> str:
    """
    Format HotpotQA context list into a readable string for the prompt.

    Input: list of [title, sentences_list]  or list of plain strings.
    """
    parts = []
    for item in context_paragraphs:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            title, sents = item
            if isinstance(sents, (list, tuple)):
                text = " ".join(str(s) for s in sents)
            else:
                text = str(sents)
            parts.append(f"[{title}]: {text}")
        else:
            parts.append(str(item))
    return "\n".join(parts)
