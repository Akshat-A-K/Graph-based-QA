"""
Model cache utilities to avoid re-loading large transformers multiple times.
"""

import torch
from typing import Dict

_MODEL_CACHE: Dict[str, "SentenceTransformer"] = {}
_QA_PIPELINE_CACHE: Dict[str, object] = {}


def get_sentence_transformer(model_name: str):
    """Return a cached SentenceTransformer instance for a given model name.

    Uses CPU by default so it never competes with the QA pipeline for GPU VRAM.
    The QA pipeline (DeBERTa-v3-large) is the GPU bottleneck; the sentence
    transformer is used for batch pre-encoding and can run efficiently on CPU.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    _MODEL_CACHE[model_name] = model
    return model


def get_qa_pipeline(model_name: str = "deepset/deberta-v3-large-squad2"):
    """Return a cached Transformers extractive-QA pipeline.

    DeBERTa-v3-large uses disentangled attention with a larger capacity model
    (~400M params). Significantly better than base on identity questions
    ("Who is X?"), multi-token answers, and low-resource span extraction.
    Downloads the model on first call; subsequent calls use the in-memory cache.
    Uses GPU when available for faster inference.
    """
    if model_name in _QA_PIPELINE_CACHE:
        return _QA_PIPELINE_CACHE[model_name]

    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading extractive-QA model: {model_name} (device={device})...")
    qa = pipeline("question-answering", model=model_name, device=device)
    _QA_PIPELINE_CACHE[model_name] = qa
    print("Extractive-QA model ready.")
    return qa
