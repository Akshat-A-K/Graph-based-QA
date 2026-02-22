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
        pdf_path = r"/home/gaurav/inlp_project/OM/Graph-based-QA/test.pdf"
    
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
    elif "test" in pdf_path.lower():
        questions = [
            "What is Abdul Kalam's full name?",                                   # Q1
            "When was Abdul Kalam born?",                                          # Q2
            "When did Abdul Kalam die?",                                           # Q3
            "What was Abdul Kalam's profession?",                                  # Q4
            "From which year to which year did Kalam serve as president of India?",# Q5
            "Where was Abdul Kalam born?",                                         # Q6
            "What subjects did Kalam study?",                                      # Q7
            "Which organisations did Kalam mainly work at as a scientist?",        # Q8
            "What nickname was Kalam given for his missile work?",                 # Q9
            "What role did Kalam play in the Pokhran-II nuclear tests?",           # Q10
            "Which political parties supported Kalam's election as president?",    # Q11
            "What was Kalam popularly referred to as by the public?",              # Q12
            "What is India's highest civilian honour that Kalam received?",        # Q13
            "How did Kalam die and where?",                                        # Q14
            "How old was Kalam when he died?",                                     # Q15
            "Where was Kalam buried and with what honours?",                       # Q16
            "What was Kalam's position among his siblings?",                       # Q17
            "What was the occupation of Kalam's ancestors?",                       # Q18
            "What caused the Kalam family business to fail?",                      # Q19
            "What did Kalam do as a young boy to support his family?",             # Q20
        ]
    else:
        questions = [
            # --- Early life ---
            "What was Virat Kohli's father's profession?",           # Q1
            "Where did Kohli spend his formative years as a child?", # Q2
            "When did Kohli's father die?",                          # Q3
            "What is Kohli's mother's name?",                        # Q4
            "Who are Kohli's siblings?",                             # Q5
            # --- Youth / U-19 career ---
            "How many runs did Kohli score in the 2008 U-19 World Cup?",              # Q6
            "What was Kohli's score in his century at the 2008 U-19 World Cup?",      # Q7
            "What scholarship did Kohli receive in June 2008?",                       # Q8
            "When did Kohli make his first-class debut?",                             # Q9
            # --- International career ---
            "When did Kohli make his international cricket debut?",                   # Q10
            "Where did Kohli score his maiden Test century?",                         # Q11
            "What was Kohli's career-best ODI score?",                                # Q12
            "How many runs did Kohli score in the 2023 ODI World Cup?",               # Q13
            "How many runs did Kohli score in the 2014 ICC World Twenty20?",          # Q14
            "How many runs did Kohli score as Test captain in his first series in Australia?", # Q15
            # --- Captaincy / rankings ---
            "In which year did Kohli become the number one ranked Test batsman?",     # Q16
            "When did Kohli first become number one in ODI batting rankings?",        # Q17
            "In which tournament did Kohli win the Man of the Tournament award in 2016?", # Q18
            # --- IPL ---
            "How much did RCB acquire Kohli for in the 2008 IPL auction?",            # Q19
            "How much was Kohli retained by RCB for ahead of the 2011 IPL season?",  # Q20
            # --- Records ---
            "For how many consecutive years was Kohli named Wisden Leading Cricketer in the World?", # Q21
            "What record did Kohli set at the 2023 ODI World Cup?",                   # Q22
            # --- Personal life ---
            "Who is Kohli's childhood cricket coach?",                                # Q23
            "Who is Kohli's wife?",                                                   # Q24
            "What are Kohli's estimated earnings in 2022?",                           # Q25
        ]

    print("Questions and answers:")
    for i, question in enumerate(questions, 1):
        results = reasoner.enhanced_reasoning(question, k=8)
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

        # --- Context debug (commented out) ---
        # if final_spans:
        #     for rank, span_id in enumerate(final_spans[:5], 1):
        #         if span_id in span_graph_builder.graph.nodes:
        #             node = span_graph_builder.graph.nodes[span_id]
        #             ctx_text = node.get("text", "").strip()
        #             score = results.get("span_scores", {}).get(span_id, 0.0)
        #             print(f"  [{rank}] (score={score:.3f}) {ctx_text}")


if __name__ == "__main__":
    main()
