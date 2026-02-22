import sys
import gc
import torch

from parser.pdf_parser import extract_pages
from parser.drg_nodes import build_nodes
from parser.drg_graph import DocumentReasoningGraph
from parser.span_extractor import SpanExtractor
from parser.span_graph import SpanGraph
from parser.kg_builder import KnowledgeGraphBuilder
from parser.enhanced_reasoner import EnhancedHybridReasoner
from parser.answer_selector import select_answer


def main():
    # Force garbage collection before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = r"c:\Mtech\Sem_2\INLP\Project\Graph-based-QA\A2.pdf"
    
    print(f"📄 Processing PDF: {pdf_path}")
    pages = extract_pages(pdf_path)
    sentence_nodes = build_nodes(pages)

    print("🚀 Building Document Reasoning Graph...")
    drg = DocumentReasoningGraph()
    drg.add_nodes(sentence_nodes)
    drg.compute_embeddings()
    drg.add_structural_edges()
    drg.add_semantic_edges(threshold=0.75)
    
    # Clear memory after DRG
    gc.collect()

    print("📊 Extracting spans...")
    span_extractor = SpanExtractor()
    spans = span_extractor.extract_spans_from_nodes(sentence_nodes)

    print("🕸️ Building Span Graph...")
    span_graph_builder = SpanGraph()
    span_graph_builder.add_nodes(spans)
    span_graph_builder.compute_embeddings()
    span_graph_builder.add_structural_edges()
    span_graph_builder.add_semantic_edges(threshold=0.7)
    span_graph_builder.add_discourse_edges()
    span_graph_builder.add_entity_overlap_edges()
    span_graph_builder.compute_graph_metrics()
    
    # Clear memory after Span Graph
    gc.collect()

    print("🧠 Building Knowledge Graph...")
    kg_builder = KnowledgeGraphBuilder()
    kg_data = kg_builder.build_kg(spans)
    
    # Clear memory after KG
    gc.collect()

    print("🔍 Initializing Reasoner...")
    reasoner = EnhancedHybridReasoner(
        sentence_graph=drg.graph,
        span_graph=span_graph_builder.graph,
        knowledge_graph=kg_data
    )

    if "a2" in pdf_path.lower():
        questions = [
            "What is the deadline for the assignment?",
            "How must the assignment be implemented?",
            "What file type must be submitted?",
            "Which corpus should be used to train word embeddings?",
            "What word embedding methods must be implemented in Section 1?",
            "What analogy formula should be used in Task 2.1?",
            "Which embedding is used for the bias check?",
            "What tagset should be used for POS tagging?",
            "What are the train/validation/test split ratios?",
            "What evaluation metrics must be reported for the POS tagger?",
            "What must be reported about hyperparameters for the embedding methods?",
            "What file name format is required for the submission zip?",
            "What files must be included under the embeddings folder?",
            "What should be included in README.md?",
            "Are Jupyter notebooks allowed for submission?",
            "What should be done about heavy pretrained embedding files in the zip?",
            "Which analogy questions must be answered in the report?",
            "What dataset method should be used to load tagged sentences?",
            "What should the MLP input be for a window size C?",
            "What should be included in the report's error analysis?"
        ]
    else:
        questions = [
            "Who is Virat Kohli?",
            "When was Virat Kohli born?",
            "What role does he play in cricket?",
            "What are his nicknames?",
            "How many international centuries does he have?",
            "Which IPL team does he play for?",
            "Which domestic team does he play for?",
            "What major ICC awards has he won?",
            "What are his notable captaincy records?",
            "Which World Cups or trophies did he win with India?",
            "When did he retire from T20Is?",
            "When did he retire from Test cricket?",
            "Who is his spouse?",
            "What are the names of his children?",
            "Where was he born?",
            "Which school did he attend?",
            "What is his batting position in ODIs?",
            "What is he known for in batting style?",
            "What awards did he receive from the Indian government?",
            "What were his estimated earnings in 2022?"
        ]

    print("Questions and answers:")
    for i, question in enumerate(questions, 1):
        results = reasoner.enhanced_reasoning(question, k=3)
        final_spans = results.get("final_spans", [])
        answer = "No answer found"
        if final_spans:
            answer, _, _, _ = select_answer(
                results,
                span_graph_builder,
                question,
                reasoner=reasoner,
                max_length=220
            )
        print(f"Q{i}: {question}")
        print(f"A{i}: {answer}\n")


if __name__ == "__main__":
    main()
