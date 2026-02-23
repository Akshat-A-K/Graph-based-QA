"""Runtime configuration for PDF extraction features.

Read options from environment variables to enable OCR/table extraction globally.
"""
import os

# Enable OCR even if pytesseract is available; default: off
ENABLE_OCR = os.getenv("PDF_ENABLE_OCR", "false").lower() in ("1", "true", "yes")

# Enable table extraction (use pdfplumber when available); default: on
ENABLE_TABLES = os.getenv("PDF_ENABLE_TABLES", "true").lower() in ("1", "true", "yes")

# Default output directory for extracted assets
OUTPUT_DIR = os.getenv("PDF_OUTPUT_DIR", ".")
