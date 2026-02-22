import re
import fitz  # PyMuPDF
from typing import List, Dict


def _fix_null_encoded(text: str) -> str:
    """Fix UTF-16 BE text runs that PyMuPDF returns as null-interleaved ASCII.

    When a PDF stores a text run in UTF-16 BE encoding, PyMuPDF sometimes
    returns each character preceded by a null byte, e.g.:
        '\\x00e\\x00s\\x00t\\x00i\\x00m\\x00a\\x00t\\x00e\\x00d'
    Stripping the null bytes recovers the original text.  We also turn the
    superscript-1 glyph '¹' back into the rupee sign '₹' when it immediately
    precedes digits (common PDF font-substitution artefact).
    """
    if "\x00" not in text:
        return text
    # Strip runs of (null + printable ASCII) that are clearly null-interleaved
    text = re.sub(
        r"(?:\x00[\x20-\x7e]){3,}",
        lambda m: m.group(0).replace("\x00", ""),
        text,
    )
    # Remove any remaining stray null bytes
    text = text.replace("\x00", "")
    # Superscript-1 used as rupee-sign substitute: '¹634' → '₹634'
    text = re.sub(r"¹(\d)", r"₹\1", text)
    return text


def _fix_spaced_chars(text: str) -> str:
    """Fix PDFs that render text with spaces between every character.

    Some PDFs emit '₹ 6 3 4  c r o r e' instead of '₹634 crore'.
    We detect runs of 4+ single-character tokens separated by single spaces
    and collapse them.  This is pure text normalisation — no domain knowledge.
    """
    def _join(m: "re.Match") -> str:
        parts = m.group(0).split(" ")
        if all(len(p) == 1 for p in parts):
            return "".join(parts)
        return m.group(0)

    text = re.sub(r"\b\w(?: \w){3,}\b", _join, text)
    # Reattach currency / symbol that got separated from its number
    text = re.sub(r"([₹$€£¥])\s+(\d)", r"\1\2", text)
    return text


def _clean_text(text: str) -> str:
    """Clean PDF text for better sentence splitting and retrieval."""
    if not text:
        return ""

    # Fix null-byte interleaved UTF-16 BE artifacts first
    text = _fix_null_encoded(text)
    # Fix spaced-character artifacts before any further normalisation
    text = _fix_spaced_chars(text)
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
