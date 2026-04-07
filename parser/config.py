"""Runtime configuration for PDF extraction features.

Read options from environment variables to enable OCR/table extraction globally.
"""
import os

# Best-quality embedding default shared across app and batch/Hotpot pipelines.
BEST_RESPONSE_EMBED_MODEL = os.getenv(
    "BEST_RESPONSE_EMBED_MODEL",
    "BAAI/bge-large-en-v1.5"
)

# Pipeline-specific overrides can still be set, but default to the same best model.
UI_EMBED_MODEL = os.getenv("UI_EMBED_MODEL", BEST_RESPONSE_EMBED_MODEL)
BATCH_EMBED_MODEL = os.getenv("BATCH_EMBED_MODEL", BEST_RESPONSE_EMBED_MODEL)
HOTPOT_EMBED_MODEL = os.getenv("HOTPOT_EMBED_MODEL", BEST_RESPONSE_EMBED_MODEL)

# Enable OCR even if pytesseract is available; default: off
ENABLE_OCR = os.getenv("PDF_ENABLE_OCR", "false").lower() in ("1", "true", "yes")

# Enable table extraction (use pdfplumber when available); default: on
ENABLE_TABLES = os.getenv("PDF_ENABLE_TABLES", "true").lower() in ("1", "true", "yes")

# Default output directory for extracted assets
OUTPUT_DIR = os.getenv("PDF_OUTPUT_DIR", ".")
