"""
Model cache utilities to avoid re-loading large transformers multiple times.
"""

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    print("WARNING: torch not available. CUDA acceleration and some models will be disabled.")

from typing import Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("WARNING: sentence_transformers not available. Semantic features will be disabled.")

_MODEL_CACHE: Dict[str, Any] = {}
_QA_PIPELINE_CACHE: Dict[str, Any] = {}


def get_sentence_transformer(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    try:
        model = SentenceTransformer(model_name, device="cpu")
        _MODEL_CACHE[model_name] = model
        return model
    except Exception as e:
        print(f"Error loading SentenceTransformer: {e}")
        return None


def get_qa_pipeline(model_name: str = "deepset/deberta-v3-large-squad2"):
    if model_name in _QA_PIPELINE_CACHE:
        return _QA_PIPELINE_CACHE[model_name]

    try:
        from transformers import pipeline
        device = 0 if (_TORCH_AVAILABLE and torch.cuda.is_available()) else -1
        print(f"Loading extractive-QA model: {model_name} (device={device})...")
        qa = pipeline("question-answering", model=model_name, device=device)
        _QA_PIPELINE_CACHE[model_name] = qa
        print("Extractive-QA model ready.")
        return qa
    except Exception as e:
        print(f"Error loading QA pipeline: {e}")
        return None
