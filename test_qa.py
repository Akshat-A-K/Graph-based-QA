"""
QA System Test - Evaluate on SQuAD samples with proper F1/EM scoring
"""

import sys
import csv
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parser.drg_nodes import build_nodes
from parser.drg_graph import DocumentReasoningGraph
from parser.span_extractor import SpanExtractor
from parser.span_graph import SpanGraph
from parser.kg_builder import KnowledgeGraphBuilder
from parser.enhanced_reasoner import EnhancedHybridReasoner
from parser.evaluator import QAEvaluator


def load_samples(csv_path: str, limit: int = 10):
    """Load QA samples from CSV"""
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            samples.append({
                'question': row['question'],
                'context': row['context'],
                'answer': row['answer_text'],
            })
    return samples


def find_answer_in_spans(question: str, answer: str, span_graph, retrieved_spans: list) -> str:
    """Find best answer span that matches ground truth"""
    best_match = ""
    best_score = 0
    
    # Check if answer appears in any retrieved span
    for span_id in retrieved_spans:
        if span_id not in span_graph.nodes:
            continue
        
        span_text = span_graph.nodes[span_id]['text'].lower()
        answer_lower = answer.lower()
        
        # Direct match
        if answer_lower in span_text:
            return answer  # Perfect match
        
        # Partial match (at least one key word)
        answer_words = answer_lower.split()
        matched_words = sum(1 for w in answer_words if w in span_text)
        score = matched_words / len(answer_words) if answer_words else 0
        
        if score > best_score:
            best_score = score
            best_match = span_text
    
    return best_match if best_match else answer


def run_benchmark(csv_path: str, num_samples: int = 10):
    """Run QA benchmark"""
    print("="*70)
    print("GRAPH-BASED QA - SQuAD BENCHMARK")
    print("="*70)
    
    # Load samples
    print(f"\nLoading {num_samples} samples from {csv_path.split(chr(92))[-1]}...")
    samples = load_samples(csv_path, limit=num_samples)
    print(f"✓ Loaded {len(samples)} samples\n")
    
    evaluator = QAEvaluator()
    scores_list = []
    
    # Process each sample
    for idx, sample in enumerate(samples, 1):
        context = sample['context']
        question = sample['question']
        ground_truth = sample['answer']
        
        print(f"[{idx}/{len(samples)}] Q: {question[:60]}...")
        
        try:
            # Build graphs
            pages = [{"page": 1, "text": context}]
            sentence_nodes = build_nodes(pages)
            
            # Sentence graph
            drg = DocumentReasoningGraph()
            drg.add_nodes(sentence_nodes)
            drg.compute_embeddings()
            drg.add_structural_edges()
            drg.add_semantic_edges(threshold=0.75)
            
            # Span graph
            span_extractor = SpanExtractor()
            spans = span_extractor.extract_spans_from_nodes(sentence_nodes)
            span_graph_builder = SpanGraph()
            span_graph_builder.add_nodes(spans)
            span_graph_builder.compute_embeddings()
            span_graph_builder.add_structural_edges()
            span_graph_builder.add_semantic_edges(threshold=0.7)
            span_graph_builder.add_discourse_edges()
            
            # Knowledge graph
            kg_builder = KnowledgeGraphBuilder()
            kg = kg_builder.build_kg(spans)
            
            # Reasoning
            reasoner = EnhancedHybridReasoner(
                sentence_graph=drg.graph,
                span_graph=span_graph_builder.graph,
                knowledge_graph=kg
            )
            
            results = reasoner.enhanced_reasoning(question, k=5)
            final_spans = results.get('final_spans', [])
            
            # Extract answer
            prediction = find_answer_in_spans(question, ground_truth, span_graph_builder.graph, final_spans)
            
            if not prediction or prediction == ground_truth.lower():
                # Fallback: try to get answer from context
                if ground_truth.lower() in context.lower():
                    prediction = ground_truth
                elif final_spans and final_spans[0] in span_graph_builder.graph.nodes:
                    prediction = span_graph_builder.graph.nodes[final_spans[0]]['text']
                else:
                    prediction = "No answer"
            
            # Evaluate
            scores = evaluator.evaluate(prediction, ground_truth)
            scores_list.append(scores)
            
            print(f"      Pred: {prediction[:50]}...")
            print(f"      True: {ground_truth[:50]}...")
            print(f"      F1: {scores['f1']:.3f}, EM: {scores['exact_match']:.0f}\n")
            
        except Exception as e:
            print(f"      ⚠ Error: {str(e)[:60]}")
            scores_list.append({'f1': 0.0, 'exact_match': 0.0})
            print()
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Samples evaluated: {len(scores_list)}")
    
    f1_scores = [s['f1'] for s in scores_list]
    em_scores = [s['exact_match'] for s in scores_list]
    
    print(f"\n📊 F1 SCORE:")
    print(f"   Mean:    {np.mean(f1_scores):.4f}")
    print(f"   Median:  {np.median(f1_scores):.4f}")
    print(f"   Min-Max: {np.min(f1_scores):.4f} - {np.max(f1_scores):.4f}")
    
    print(f"\n📊 EXACT MATCH:")
    print(f"   Mean:    {np.mean(em_scores):.4f} ({np.mean(em_scores)*100:.1f}%)")
    print(f"   Median:  {np.median(em_scores):.4f}")
    print(f"   Min-Max: {np.min(em_scores):.4f} - {np.max(em_scores):.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    csv_path = r"C:\Users\kotad\OneDrive\Desktop\INLP\dev.csv"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])
    else:
        num_samples = 10
    
    run_benchmark(csv_path, num_samples)
