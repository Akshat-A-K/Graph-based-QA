import re
import fitz  # PyMuPDF
from typing import List, Dict


def _clean_text(text: str) -> str:
    """Clean PDF text for better sentence splitting and retrieval."""
    if not text:
        return ""

    # Join hyphenated line breaks (e.g., "submis-\nsion" -> "submission")
    text = re.sub(r"-\n(?=\w)", "", text)
    # Normalize newlines to spaces, preserve paragraph breaks
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_pages(pdf_path: str) -> List[Dict]:
    """
    Extract text page by page with metadata.
    Returns list of {page_num, text}
    """

    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = _clean_text(text)

        if not text.strip():
            continue

        pages.append({
            "page": i + 1,
            "text": text
        })

    return pages
