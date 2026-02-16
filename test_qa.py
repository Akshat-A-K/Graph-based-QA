"""
Simple QA Test - Test graph building and visualization with sample PDF
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parser.pdf_parser import extract_pages
from parser.drg_nodes import build_nodes
from parser.drg_graph import DocumentReasoningGraph
from parser.span_extractor import SpanExtractor
from parser.span_graph import SpanGraph
from parser.kg_builder import KnowledgeGraphBuilder
from parser.enhanced_reasoner import EnhancedHybridReasoner


def test_qa_system(pdf_path, questions):
    """Test the complete QA pipeline with graph visualization"""
    print("="*70)
    print("QA SYSTEM TEST")
    print("="*70)
    print(f"\nPDF: {pdf_path}")
    
    # 1. Extract text from PDF
    print("\n[1/6] Extracting text from PDF...")
    pages = extract_pages(pdf_path)
    if not pages:
        print("❌ Failed to extract text")
        return
    print(f"✓ Extracted {len(pages)} pages")
    
    # 2. Build sentence nodes
    print("\n[2/6] Building sentence nodes...")
    sentence_nodes = build_nodes(pages)
    print(f"✓ Created {len(sentence_nodes)} sentence nodes")
    
    # 3. Build Document Reasoning Graph (DRG)
    print("\n[3/6] Building Document Reasoning Graph...")
    drg = DocumentReasoningGraph()
    drg.add_nodes(sentence_nodes)
    drg.compute_embeddings()
    drg.add_structural_edges()
    drg.add_semantic_edges(threshold=0.75)
    print(f"✓ DRG: {drg.graph.number_of_nodes()} nodes, {drg.graph.number_of_edges()} edges")
    
    # Export DRG visualization
    os.makedirs('graphs', exist_ok=True)
    drg.export_graph_image('graphs/drg_graph.png')
    print(f"✓ Saved: graphs/drg_graph.png")
    
    # 4. Build Span Graph
    print("\n[4/6] Building Span Graph...")
    span_extractor = SpanExtractor()
    spans = span_extractor.extract_spans_from_nodes(sentence_nodes)
    
    span_graph_builder = SpanGraph()
    span_graph_builder.add_nodes(spans)
    span_graph_builder.compute_embeddings()
    span_graph_builder.add_structural_edges()
    span_graph_builder.add_semantic_edges(threshold=0.7)
    span_graph_builder.add_discourse_edges()
    print(f"✓ Span Graph: {span_graph_builder.graph.number_of_nodes()} nodes, {span_graph_builder.graph.number_of_edges()} edges")
    
    # Export Span Graph visualization
    span_graph_builder.export_graph_image('graphs/span_graph.png')
    print(f"✓ Saved: graphs/span_graph.png")
    
    # 5. Build Knowledge Graph
    print("\n[5/6] Building Knowledge Graph...")
    kg_builder = KnowledgeGraphBuilder()
    kg_data = kg_builder.build_kg(spans)
    kg_graph = kg_data['graph']  # Extract NetworkX graph from result
    print(f"✓ KG: {kg_graph.number_of_nodes()} nodes, {kg_graph.number_of_edges()} edges")
    
    # Export KG visualization
    kg_builder.export_graph_image('graphs/kg_graph.png')
    print(f"✓ Saved: graphs/kg_graph.png")
    
    # 6. Answer questions
    print("\n[6/6] Testing Question Answering...")
    reasoner = EnhancedHybridReasoner(
        sentence_graph=drg.graph,
        span_graph=span_graph_builder.graph,
        knowledge_graph=kg_data  # Pass the complete kg data dict
    )
    
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        results = reasoner.enhanced_reasoning(question, k=3)
        final_spans = results.get('final_spans', [])
        
        if final_spans:
            print(f"   Retrieved {len(final_spans)} spans")
            # Show top answer
            top_span_id = final_spans[0]
            if top_span_id in span_graph_builder.graph.nodes:
                text = span_graph_builder.graph.nodes[top_span_id]['text']
                print(f"   Answer: {text[:150]}...")
        else:
            print("   No answer found")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nGraph visualizations saved to:")
    print("  - graphs/drg_graph.png")
    print("  - graphs/span_graph.png")
    print("  - graphs/kg_graph.png")
    print()


if __name__ == "__main__":
    # Sample test
    sample_pdf = r"C:\Users\kotad\OneDrive\Desktop\INLP\Assignment 1\Assignment-1.pdf"
    sample_questions = [
        "What is the main objective?",
        "What are the key requirements?",
        "What is the evaluation criteria?"
    ]
    
    # Check if PDF path provided as argument
    if len(sys.argv) > 1:
        sample_pdf = sys.argv[1]
    
    if Path(sample_pdf).exists():
        test_qa_system(sample_pdf, sample_questions)
    else:
        print(f"PDF not found: {sample_pdf}")
        print("\nUsage: python test_qa.py <pdf_path>")
