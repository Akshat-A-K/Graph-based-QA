import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import io

try:
    from PIL import Image
    import pytesseract
    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False
try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except Exception:
    _PDFPLUMBER_AVAILABLE = False

from . import config

# Effective flags combining availability + user config
_ENABLE_OCR = config.ENABLE_OCR and _OCR_AVAILABLE
_ENABLE_TABLES = config.ENABLE_TABLES and _PDFPLUMBER_AVAILABLE


def _fix_null_encoded(text: str) -> str:
    """Fix UTF-16 BE text runs that PyMuPDF returns as null-interleaved ASCII.

    When a PDF stores a text run in UTF-16 BE encoding, PyMuPDF sometimes
    returns each character preceded by a null byte, e.g.:
        '\\x00e\\x00s\\x00t\\x00i\\x00m\\x00a\\x00t\\x00e\\x00d'
    Stripping the null bytes recovers the original text.  We also turn the
    superscript-1 glyph '\\u00b9' back into the rupee sign '\\u20b9' when it immediately
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
    # Superscript-1 used as rupee-sign substitute: '\\u00b9634' -> '\\u20b9634'
    text = re.sub(r"\u00b9(\d)", r"\u20b9\1", text)
    return text


def _fix_spaced_chars(text: str) -> str:
    """Fix PDFs that render text with spaces between every character.

    Some PDFs emit '\\u20b9 6 3 4  c r o r e' instead of '\\u20b9634 crore'.
    We detect runs of 4+ single-character tokens separated by single spaces
    and collapse them.  This is pure text normalisation - no domain knowledge.
    """
    def _join(m: "re.Match") -> str:
        parts = m.group(0).split(" ")
        if all(len(p) == 1 for p in parts):
            return "".join(parts)
        return m.group(0)

    text = re.sub(r"\b\w(?: \w){3,}\b", _join, text)
    # Reattach currency / symbol that got separated from its number
    text = re.sub(r"([\u20b9$\u20ac\u00a3\u00a5])\s+(\d)", r"\1\2", text)
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
    pages: List[Dict[str, Any]] = []

    for i, page in enumerate(doc):
        # Use the rich dict output to preserve font sizes and spans
        page_dict = page.get_text("dict")

        blocks_out: List[Dict[str, Any]] = []
        all_spans_sizes = []

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 == text block
                continue

            block_text_lines = []
            max_span_size = 0.0
            bold = False

            for line in block.get("lines", []):
                line_text_parts = []
                for span in line.get("spans", []):
                    s_text = span.get("text", "")
                    s_size = float(span.get("size", 0.0))
                    flags = span.get("flags", 0)
                    # Collect sizes for page-level median
                    all_spans_sizes.append(s_size)
                    max_span_size = max(max_span_size, s_size)
                    # flag 2 often indicates bold; keep heuristic
                    if flags & 2:
                        bold = True

                    line_text_parts.append(s_text)

                line_text = "".join(line_text_parts)
                if line_text.strip():
                    block_text_lines.append(line_text)

            block_text = "\n".join(block_text_lines).strip()
            if not block_text:
                continue

            blocks_out.append({
                "text": block_text,
                "max_font_size": max_span_size,
                "is_bold": bold,
            })

        # Determine median size for heading heuristics
        median_size = 0.0
        if all_spans_sizes:
            try:
                import statistics
                median_size = statistics.median(all_spans_sizes)
            except Exception:
                median_size = float(sum(all_spans_sizes) / len(all_spans_sizes))

        # Mark headings where font size is much larger than median or bold
        for b in blocks_out:
            b["is_heading"] = bool(
                (median_size and b["max_font_size"] >= median_size + 2.0) or b.get("is_bold")
            )

        # Compose plain text for compatibility with older callers
        full_text = "\n\n".join(b["text"] for b in blocks_out)
        full_text = _clean_text(full_text)

        # If page has no text but OCR is available (and enabled), try OCR on the page image
        if not full_text.strip() and _ENABLE_OCR:
            try:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                ocr_text = pytesseract.image_to_string(img)
                full_text = _clean_text(ocr_text)
            except Exception:
                pass

        if not full_text.strip():
            continue

        pages.append({
            "page": i + 1,
            "text": full_text,
            "blocks": blocks_out,
            "median_font_size": median_size,
            "metadata": doc.metadata or {},
        })

    return pages


def extract_document(pdf_path: str, detect_sections: bool = True) -> Dict[str, Any]:
    """Extract document-level structure: metadata, pages, and optional sections.

    Sections are heuristically detected by heading blocks (larger font or bold).
    """
    pages = extract_pages(pdf_path)
    document = {
        "metadata": pages[0].get("metadata", {}) if pages else {},
        "pages": pages,
        "sections": [],
    }

    if detect_sections:
        current_section = {"title": None, "pages": []}
        for p in pages:
            # find heading blocks on page
            headings = [b for b in p.get("blocks", []) if b.get("is_heading")]
            if headings:
                # start new section for first detected heading
                if current_section["title"] or current_section["pages"]:
                    document["sections"].append(current_section)
                    current_section = {"title": None, "pages": []}

                current_section["title"] = headings[0]["text"].split("\n")[0][:200]
                current_section["pages"].append(p)
            else:
                current_section["pages"].append(p)

        if current_section["pages"]:
            document["sections"].append(current_section)

    return document


def _extract_tables_pdfplumber(pdf_path: str) -> List[Dict[str, Any]]:
    """Try to extract tables using pdfplumber when available."""
    tables = []
    if _PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for t in page_tables:
                        # Normalize table rows
                        rows = [[cell for cell in row] for row in t]
                        tables.append({"page": i + 1, "rows": rows})
        except Exception:
            return tables
        return tables

    # Fallback heuristic table extraction when pdfplumber is not available
    # Look for consecutive lines with consistent column counts separated by multiple spaces or pipes
    try:
        pages = extract_pages(pdf_path)
        for p in pages:
            lines = []
            for b in p.get("blocks", []):
                for ln in b["text"].split("\n"):
                    lines.append(ln)

            # Candidate rows: split on 2+ spaces or pipe
            candidate_rows = []
            for ln in lines:
                # normalize multiple spaces to a delimiter
                parts = [c.strip() for c in re.split(r"\s{2,}|\||\t|,", ln) if c.strip()]
                if len(parts) >= 2:
                    candidate_rows.append(parts)

            # Group contiguous rows with same column count
            if candidate_rows:
                current = []
                last_len = len(candidate_rows[0])
                for row in candidate_rows:
                    if len(row) == last_len:
                        current.append(row)
                    else:
                        if len(current) >= 2:
                            tables.append({"page": p.get("page"), "rows": current})
                        current = [row]
                        last_len = len(row)

                if len(current) >= 2:
                    tables.append({"page": p.get("page"), "rows": current})
    except Exception:
        return tables

    return tables


def extract_document_with_tables(pdf_path: str, detect_sections: bool = True, enable_tables: Optional[bool] = None, enable_ocr: Optional[bool] = None) -> Dict[str, Any]:
    """Extended extractor that also attempts table extraction and writes a small summary.

    `enable_tables` and `enable_ocr` override global config when provided.
    """
    doc = extract_document(pdf_path, detect_sections=detect_sections)
    use_tables = _ENABLE_TABLES if enable_tables is None else (enable_tables and _PDFPLUMBER_AVAILABLE)
    # If OCR override requested for inner extract_document, call it again with explicit flag
    if enable_ocr is not None:
        doc = extract_document(pdf_path, detect_sections=detect_sections)

    tables = _extract_tables_pdfplumber(pdf_path) if use_tables else []
    doc["tables"] = tables
    return doc


def save_document_json(document: Dict[str, Any], out_path: str) -> None:
    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
