"""
Model cache utilities to avoid re-loading large transformers multiple times.
"""

from typing import Dict

_MODEL_CACHE: Dict[str, "SentenceTransformer"] = {}


def get_sentence_transformer(model_name: str):
    """Return a cached SentenceTransformer instance for a given model name."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model
