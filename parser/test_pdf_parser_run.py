import os, sys
sys.path.insert(0, os.getcwd())
from parser.pdf_parser import extract_document_with_tables, save_document_json

PDFS = ["A2.pdf", "test.pdf"]

for pdf in PDFS:
    path = os.path.join(os.getcwd(), pdf)
    if not os.path.exists(path):
        print('missing', path)
        continue

    print('\nProcessing', pdf)
    doc = extract_document_with_tables(path)
    out = pdf.replace('.pdf', '.json')
    save_document_json(doc, out)
    print('pages:', len(doc.get('pages', [])), 'sections:', len(doc.get('sections', [])), 'tables:', len(doc.get('tables', [])))
    print('saved ->', out)
