"""
Minimal QA script for memory-constrained environments.
Only builds essential components.
"""
import sys
import gc
from parser.pdf_parser import extract_document_with_tables
from parser.drg_nodes import build_nodes

def simple_answer(question, sentence_nodes):
    """Simple keyword-based answering without heavy models."""
    import re
    from collections import defaultdict
    from rank_bm25 import BM25Okapi
    
    # Extract text from nodes (they are dicts)
    texts = [node["text"] for node in sentence_nodes]
    
    # Tokenize for BM25
    tokenized_texts = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    
    # Query
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get top 3 sentences
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    
    # Return first match or concatenate top matches
    if top_indices:
        best_answer = texts[top_indices[0]]
        # Try to extract answer span from sentence
        words = question.lower().split()
        keywords = [w for w in words if len(w) > 3 and w not in ['what', 'when', 'where', 'which', 'who', 'how', 'does', 'did', 'the', 'is', 'are', 'was', 'were']]
        
        # Look for keyword in answer
        for keyword in keywords:
            if keyword in best_answer.lower():
                # Extract sentence or phrase
                sentences = best_answer.split('.')
                for s in sentences:
                    if keyword in s.lower():
                        return s.strip()
        
        return best_answer
    else:
        return "Unable to find answer"

def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = r"c:\Mtech\Sem_2\INLP\Project\Graph-based-QA\text-conversion-1771764622748.pdf"
    
    print(f"📄 Loading PDF: {pdf_path}")
    doc = extract_document_with_tables(pdf_path)
    pages = doc.get('pages', [])
    sentence_nodes = build_nodes(pages)
    print(f"✅ Loaded {len(sentence_nodes)} sentences")
    
    # Virat Kohli questions
    questions = [
        "Who is Virat Kohli?",
        "When was Virat Kohli born?",
        "Where was Virat Kohli born?",
        "What is Virat Kohli's role in cricket?",
        "Which team does Virat Kohli play for?",
        "What is Virat Kohli's batting style?",
        "How many Test centuries has Virat Kohli scored?",
        "What is Virat Kohli's highest ODI score?",
        "When did Virat Kohli make his international debut?",
        "What records does Virat Kohli hold?",
        "What is Virat Kohli's nickname?",
        "Who did Virat Kohli marry?",
        "What awards has Virat Kohli received?",
        "What is Virat Kohli's jersey number?",
        "Which IPL team does Virat Kohli captain?",
        "What is Virat Kohli's strike rate?",
        "How many runs has Virat Kohli scored in T20s?",
        "What is Virat Kohli's average in ODIs?",
        "When did Virat Kohli become captain of India?",
        "What is Virat Kohli's fitness routine?"
    ]
    
    print(f"\n{'='*80}\n🤔 Answering {len(questions)} questions...\n{'='*80}\n")
    
    for i, question in enumerate(questions, 1):
        try:
            answer = simple_answer(question, sentence_nodes)
            print(f"\nQ{i}: {question}")
            print(f"A: {answer}")
            print("-" * 80)
            gc.collect()  # Clear memory after each question
        except Exception as e:
            print(f"\nQ{i}: {question}")
            print(f"A: Error - {str(e)}")
            print("-" * 80)
    
    print(f"\n{'='*80}\n✅ Completed all questions!\n{'='*80}")

if __name__ == "__main__":
    main()
