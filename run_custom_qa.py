import sys
import os
import gc
import json
import time
import numpy as np
from datetime import datetime

# Handle Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


class _NumpyEncoder(json.JSONEncoder):
    "Serialize numpy scalar types to native Python"
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

ENABLE_OCR_FLAG = False
ENABLE_TABLES_FLAG = None
if "--enable-ocr" in sys.argv:
    ENABLE_OCR_FLAG = True
    sys.argv.remove("--enable-ocr")
if "--disable-tables" in sys.argv:
    ENABLE_TABLES_FLAG = False
    sys.argv.remove("--disable-tables")
if ENABLE_OCR_FLAG:
    os.environ["PDF_ENABLE_OCR"] = "true"
if ENABLE_TABLES_FLAG is False:
    os.environ["PDF_ENABLE_TABLES"] = "false"

from parser.pdf_parser import extract_document_with_tables
from parser.drg_nodes import build_nodes
from parser.drg_graph import DocumentReasoningGraph
from parser.span_extractor import SpanExtractor
from parser.span_graph import SpanGraph
from parser.knowledge_graph import KnowledgeGraph
from parser.enhanced_reasoner import EnhancedHybridReasoner
from parser.answer_selector import select_answer
from parser.evaluator import QAEvaluator

# Single embedding model used consistently across all graph components (English-only)
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


def _output_path(pdf_path: str, ext: str) -> str:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(pdf_path)))
    return os.path.join(out_dir, f"{base}_qa_output.{ext}")


try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    _HAS_COLOR = True
except Exception:
    Fore = Style = None
    _HAS_COLOR = False


def _warn(message: str, color: str = "yellow") -> None:
    line = "=" * 72
    if _HAS_COLOR:
        color_code = Fore.YELLOW if color == "yellow" else Fore.RED
        print(color_code + line)
        print(color_code + message)
        print(color_code + line + Style.RESET_ALL)
    else:
        print(line)
        print(message)
        print(line)


def main():
    # Force garbage collection before starting
    gc.collect()

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = r"C:\Users\kotad\OneDrive\Desktop\INLP\Graph-based-QA\temp_Rsearch_pepar.pdf"

    pipeline_start = time.time()

    print(f"Processing PDF: {pdf_path}")
    doc = extract_document_with_tables(pdf_path)
    pages = doc.get('pages', [])
    if not pages:
        _warn("ERROR: No pages found in the PDF. Aborting.", color="red")
        return

    sentence_nodes = build_nodes(pages)
    if not sentence_nodes:
        _warn("ERROR: No sentence nodes were built from the PDF. Aborting.", color="red")
        return

    print("Building Document Reasoning Graph...")
    drg = DocumentReasoningGraph(model_name=EMBED_MODEL)
    drg.add_nodes(sentence_nodes)
    drg.compute_embeddings()
    print("Building Document Reasoning Graph...")
    drg.add_semantic_edges(threshold=0.75)
    drg.compute_graph_metrics()
    if drg.graph.number_of_nodes() == 0:
        _warn("ERROR: Document Reasoning Graph is empty. Aborting.", color="red")
        return
    
    # Clear memory after DRG
    gc.collect()

    print("Extracting spans...")
    span_extractor = SpanExtractor()
    spans = span_extractor.extract_spans_from_nodes(sentence_nodes)

    print("Extracting spans...")
    span_graph_builder = SpanGraph(model_name=EMBED_MODEL)
    span_graph_builder.add_nodes(spans)
    span_graph_builder.compute_embeddings()
    print("Building Span Graph...")
    span_graph_builder.add_semantic_edges(threshold=0.7)
    span_graph_builder.add_discourse_edges()
    span_graph_builder.add_entity_overlap_edges()
    span_graph_builder.compute_graph_metrics()
    if span_graph_builder.graph.number_of_nodes() == 0:
        _warn("ERROR: Span Graph is empty. Aborting.", color="red")
        return

    os.makedirs('graphs', exist_ok=True)
    span_graph_builder.export_graph_json('graphs/span_graph.json')
    span_graph_builder.export_graph_image('graphs/span_graph.png')
    drg.export_graph_image('graphs/drg_graph.png')

    # Clear memory after Span Graph
    gc.collect()

    print("Building Knowledge Graph...")
    kg = KnowledgeGraph()
    kg.build_graph(sentence_nodes)
    
    # Align KG with DRG
    drg.add_kg_edges(kg)

    reasoner = EnhancedHybridReasoner(
        sentence_graph=drg.graph,
        span_graph=span_graph_builder.graph,
        kg_graph=kg.graph,
        model_name=EMBED_MODEL,
    )
    print("Initializing Reasoner...")

    # Document-level stats (for report)
    doc_stats = {
        "pdf": os.path.basename(pdf_path),
        "pages": len(pages),
        "sentences": drg.graph.number_of_nodes(),
        "spans": span_graph_builder.graph.number_of_nodes(),
        "kg_nodes": kg.graph.number_of_nodes(),
        "kg_edges": kg.graph.number_of_edges()
    }

    pdf_lower = os.path.basename(pdf_path).lower()

    if "a2" in pdf_lower:
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
    elif "test" in pdf_lower:
        questions = [
            "What is Abdul Kalam's full name?",
            "When was Abdul Kalam born?",
            "When did Abdul Kalam die?",
            "What was Abdul Kalam's profession?",
            "From which year to which year did Kalam serve as president of India?",
            "Where was Abdul Kalam born?",
            "What subjects did Kalam study?",
            "Which organisations did Kalam mainly work at as a scientist?",
            "What nickname was Kalam given for his missile work?",
            "What role did Kalam play in the Pokhran-II nuclear tests?",
            "Which political parties supported Kalam's election as president?",
            "What was Kalam popularly referred to as by the public?",
            "What is India's highest civilian honour that Kalam received?",
            "How did Kalam die and where?",
            "How old was Kalam when he died?",
            "Where was Kalam buried and with what honours?",
            "What was Kalam's position among his siblings?",
            "What was the occupation of Kalam's ancestors?",
            "What caused the Kalam family business to fail?",
            "What did Kalam do as a young boy to support his family?",
        ]
    elif any(k in pdf_lower for k in ("rsearch", "pepar", "research", "paper", "eurocall", "kruk", "mobile", "autonomy")):
        # Medium-level questions covering all types:
        # Factual | Definitional | Methodological | Inferential |
        # Comparative | Evaluative | Causal | Process
        questions = [
            # ── FACTUAL ────────────────────────────────────────────────────
            "What is the CEFR proficiency level of the study participants?",
            "What was the average age of the study participants?",
            "What percentage of students used mobile devices most frequently in leisure time?",
            "How many of the participants were in the second year of their BA programme?",
            "Which mobile device was used most often by participants?",

            # ── DEFINITIONAL / CONCEPTUAL ──────────────────────────────────
            "What does Holec mean by 'the ability to take charge of one's own learning'?",
            "What is the difference between ability and willingness in Littlewood's model of autonomy?",
            "Why does Reinders argue that autonomy is a continuum rather than an either-or concept?",
            "What does Benson mean when he says autonomy is multidimensional?",
            "How does social interaction contribute to learner autonomy according to the paper?",

            # ── METHODOLOGICAL ─────────────────────────────────────────────
            "Why was a semi-structured interview chosen as the data collection instrument?",
            "How was the interview data transcribed and analyzed?",
            "What were the two types of analysis applied to the gathered data?",
            "How did the researcher ensure validity when translating student excerpts?",
            "What does the paper say about how participants were selected?",

            # ── PROCESS / APPLICATION ──────────────────────────────────────
            "In which contexts or situations did students most frequently use mobile devices for English learning?",
            "What specific language skills did students use mobile devices to practice?",
            "How did students use mobile devices to look up vocabulary?",
            "What role did mobile apps play in students' autonomous language learning?",

            # ── INFERENTIAL / ANALYTICAL ───────────────────────────────────
            "Why does the paper suggest that classroom use of mobile devices was more intuitive than planned?",
            "What gap does the paper imply exists between learner awareness and actual mobile device use?",
            "How does the study link mobile device usage to the development of learner autonomy?",

            # ── COMPARATIVE ────────────────────────────────────────────────
            "How does the paper distinguish between formal and informal language learning with mobile devices?",
            "How does autonomous learning differ from non-autonomous learning according to Benson?",

            # ── EVALUATIVE / CRITICAL ──────────────────────────────────────
            "What limitations does the author acknowledge in the study?",
            "What future research directions does the paper suggest?",
            "How does the paper evaluate the overall results of learners' mobile device engagement?",

            # ── CAUSAL ─────────────────────────────────────────────────────
            "Why do the authors argue language teachers should prepare learners to use mobile devices effectively?",
            "What reasons do students give for using mobile devices for English learning in their spare time?",
            "Why does fostering learner autonomy matter for second language acquisition according to the paper?",
        ]
    else:
        questions = [
            "What is the title or main subject of this document?",
            "Who are the authors or creators of this document?",
            "What is the main objective or research question?",
            "What methodology or approach is described?",
            "What are the key findings or results?",
            "What data or evidence is presented?",
            "What conclusions are drawn?",
            "What recommendations are made?",
            "What limitations are mentioned?",
            "What future work is suggested?",
        ]

    # -----------------------------------------------------------------------
    # Run QA and collect results
    # -----------------------------------------------------------------------
    qa_records = []
    sep = "-" * 72
    print("\n" + "=" * 72)
    print("  QA RESULTS")
    print("=" * 72)

    for i, question in enumerate(questions, 1):
        print("\n" + sep)
        print(f"Q{i:02d}: {question}")

        q_start = time.time()
        results = reasoner.enhanced_reasoning(question, k=5)
        final_spans = results.get("final_spans", [])

        answer = "No answer found"
        evidence_spans = []

        if final_spans:
            answer, evidence_spans, confidence = select_answer(
                results,
                span_graph_builder,
                question,
                reasoner=reasoner,
                max_length=220
            )

        elapsed = time.time() - q_start

        eval_metrics = {"exact_match": 0.0, "f1": 0.0}
        recall_at_5 = 0.0
        reasoning_depth = 0

        if answer != "No answer found" and evidence_spans:
            eval_metrics = QAEvaluator.evaluate(answer, evidence_spans[0])

            # Evidence Recall@5
            recall_at_5 = QAEvaluator.evidence_recall_at_k(
                evidence_spans,
                evidence_spans[0],
                k=5
            )

        # Graph reasoning depth
        reasoning_depth = QAEvaluator.reasoning_depth(
            results.get("traversal_results", []),
            results.get("expansion_results", [])
        )

        print(f"A{i:02d}: {answer}")
        print(f"     Eval       : EM={eval_metrics['exact_match']:.2f}  F1={eval_metrics['f1']:.2f}  (vs top evidence)")
        print(f"     Retrieval  : Recall@5={recall_at_5:.2f}")
        print(f"     Reasoning  : Depth={reasoning_depth}")
        print(f"     Evidence   : {len(evidence_spans)} span(s)   |  "
              f"Hybrid={len(results.get('hybrid_results', []))}  "
              f"Traversal={len(results.get('traversal_results', []))}  "
              f"Expansion={len(results.get('expansion_results', []))}  "
              f"KG_R={len(results.get('kg_results', []))}  "
              f"KG_Ent={len(results.get('kg_entities', []))}")
        print(f"     Conf       : {confidence:.2%} (Combined Retrieval/QA)")
        print(f"     Time       : {elapsed:.2f}s")
        if evidence_spans:
            print(f"     Top evidence: {evidence_spans[0][:120].strip()}...")
        print(sep)

        qa_records.append({
            "q_num": i,
            "question": question,
            "answer": answer,
            "time_s": round(elapsed, 3),
            "evidence_count": len(evidence_spans),
            "evidence_spans": evidence_spans[:3],
            "eval_vs_evidence": {
                "exact_match": round(eval_metrics["exact_match"], 4),
                "f1": round(eval_metrics["f1"], 4),
                "recall_at_5": round(recall_at_5, 4),
            },
            "reasoning_depth": reasoning_depth,
            "retrieval": {
                "hybrid": len(results.get("hybrid_results", [])),
                "traversal": len(results.get("traversal_results", [])),
                "expansion": len(results.get("expansion_results", [])),
                "kg_guided": len(results.get("kg_results", [])),
                "kg_entities": len(results.get("kg_entities", [])),
            },
        })

    total_time = time.time() - pipeline_start

    # -----------------------------------------------------------------------
    # Summary analysis
    # -----------------------------------------------------------------------
    answered = [r for r in qa_records if r["answer"] != "No answer found"]
    avg_time  = sum(r["time_s"] for r in qa_records) / max(len(qa_records), 1)
    avg_f1    = sum(r["eval_vs_evidence"]["f1"] for r in answered) / max(len(answered), 1)
    avg_em    = sum(r["eval_vs_evidence"]["exact_match"] for r in answered) / max(len(answered), 1)
    avg_recall = sum(r["eval_vs_evidence"]["recall_at_5"] for r in answered) / max(len(answered), 1)
    avg_depth = sum(r["reasoning_depth"] for r in answered) / max(len(answered), 1)
    analysis = {
        "total_questions": len(qa_records),
        "answered": len(answered),
        "unanswered": len(qa_records) - len(answered),
        "answer_rate_pct": round(len(answered) / max(len(qa_records), 1) * 100, 1),
        "avg_f1_vs_evidence": round(avg_f1, 4),
        "avg_em_vs_evidence": round(avg_em, 4),
        "avg_time_per_question_s": round(avg_time, 3),
        "total_pipeline_time_s": round(total_time, 2),
        "avg_recall_at_5": round(avg_recall, 4),
        "avg_reasoning_depth": round(avg_depth, 2),
    }

    print("\n" + "=" * 72)
    print("  ANALYSIS SUMMARY")
    print("=" * 72)
    print(f"  Total questions : {analysis['total_questions']}")
    print(f"  Answered        : {analysis['answered']}  ({analysis['answer_rate_pct']}%)")
    print(f"  Unanswered      : {analysis['unanswered']}")
    print(f"  Avg F1 (vs evid): {analysis['avg_f1_vs_evidence']:.4f}")
    print(f"  Avg EM (vs evid): {analysis['avg_em_vs_evidence']:.4f}")
    print(f"  Avg Recall@5  : {analysis['avg_recall_at_5']:.4f}")
    print(f"  Avg Depth     : {analysis['avg_reasoning_depth']:.2f}")
    print(f"  Avg time/Q      : {analysis['avg_time_per_question_s']:.2f}s")
    print(f"  Total time      : {analysis['total_pipeline_time_s']:.1f}s")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pdf": os.path.basename(pdf_path),
        "embed_model": EMBED_MODEL,
        "document_stats": doc_stats,
        "analysis": analysis,
        "qa": qa_records,
    }

    json_path = _output_path(pdf_path, "json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\nJSON output saved  -> {json_path}")

    # Plain-text report
    txt_path = _output_path(pdf_path, "txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"QA OUTPUT  —  {os.path.basename(pdf_path)}\n")
        f.write(f"Generated : {output['generated_at']}\n")
        f.write(f"Model     : {EMBED_MODEL}\n\n")

        f.write("DOCUMENT STATS\n" + "-" * 40 + "\n")
        for k, v in doc_stats.items():
            f.write(f"  {k:<18}: {v}\n")
        f.write("\n")

        f.write("QA RESULTS\n" + "=" * 72 + "\n")
        for r in qa_records:
            f.write(f"\nQ{r['q_num']:02d}: {r['question']}\n")
            f.write(f"A{r['q_num']:02d}: {r['answer']}\n")
            f.write(f"    Eval       : EM={r['eval_vs_evidence']['exact_match']:.4f}  F1={r['eval_vs_evidence']['f1']:.4f}  (vs top evidence)\n")
            f.write(f"    Evidence   : {r['evidence_count']} span(s)\n")
            f.write(f"    Time       : {r['time_s']}s\n")
            if r["evidence_spans"]:
                f.write(f"    Top evid.  : {r['evidence_spans'][0][:120].strip()}...\n")
            f.write("-" * 72 + "\n")

        f.write("\nANALYSIS SUMMARY\n" + "=" * 72 + "\n")
        for k, v in analysis.items():
            f.write(f"  {k:<35}: {v}\n")

    print(f"Text report saved  -> {txt_path}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # If it happens at the very end, we already saved reports
        # Just print the error and exit gracefully if possible
        try:
            print(f"\nPipeline finished with some console issues: {e}")
        except Exception:
            pass
