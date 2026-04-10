import re
from typing import List

# Try to load spaCy model
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (OSError, ImportError, Exception):
    SPACY_AVAILABLE = False
    nlp = None


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy (with regex fallback)."""
    
    try:
        # Try spaCy first (better sentence boundary detection)
        if SPACY_AVAILABLE and nlp is not None:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]
            return sentences
    except Exception:
        pass
    
    # Fallback: Regex-based sentence splitting (Python 3.14 compatible)
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?\u0964\u0965])\s+', text)
    
    # Filter out very short sentences
    sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 3]
    
    return sentences
