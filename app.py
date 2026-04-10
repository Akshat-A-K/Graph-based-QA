import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tempfile
import re
from typing import List, Optional

from parser.pdf_parser import extract_document_with_tables
from parser.drg_nodes import build_nodes
from parser.drg_graph import DocumentReasoningGraph
from parser.span_extractor import SpanExtractor
from parser.span_graph import SpanGraph
from parser.knowledge_graph import KnowledgeGraph
from parser.config import HOTPOT_EMBED_MODEL

from parser.enhanced_reasoner import EnhancedHybridReasoner
from parser.advanced_retrieval import normalize_text, tokenize
from parser.answer_selector import select_answer

st.set_page_config(page_title="Graph-Based QA System", page_icon=None, layout="wide", initial_sidebar_state="expanded")

st.title("Graph-Based Document QA System")
st.markdown("""
<div style='background-color: #0e1117; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #4CAF50;'>
    <p style='margin: 0; color: #e0e0e0;'>
        <strong>Multi-level graph reasoning</strong> for faithful document question answering.<br>
        Supports <strong>multilingual queries</strong> including English.
    </p>
</div>
""", unsafe_allow_html=True)

# Session state for caching
if 'graphs' not in st.session_state:
    st.session_state.graphs = None
if 'reasoner' not in st.session_state:
    st.session_state.reasoner = None
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None


def build_graphs_from_pdf(pdf_bytes, pdf_name):
    """Build all graphs from PDF"""
    # Save temp PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    
    try:
        # Extract document (pages + sections + tables)
        doc = extract_document_with_tables(tmp_path)
        pages = doc.get('pages', [])
        
        # Use the same embedding model as Hotpot pipeline for consistent behavior.
        chosen_model = HOTPOT_EMBED_MODEL

        # Build sentence graph
        sentence_nodes = build_nodes(pages)
        drg = DocumentReasoningGraph(model_name=chosen_model)
        drg.add_nodes(sentence_nodes)
        drg.compute_embeddings()
        drg.add_structural_edges()
        drg.add_semantic_edges(threshold=0.75)
        drg.compute_graph_metrics()
        
        # Build span graph
        span_extractor = SpanExtractor()
        spans = span_extractor.extract_spans_from_nodes(sentence_nodes)
        span_graph_builder = SpanGraph()
        # Ensure spans use the same embedding model
        span_graph_builder = SpanGraph(model_name=chosen_model)
        span_graph_builder.add_nodes(spans)
        span_graph_builder.compute_embeddings()
        span_graph_builder.add_structural_edges()
        span_graph_builder.add_semantic_edges(threshold=0.7)
        span_graph_builder.add_discourse_edges()
        span_graph_builder.compute_graph_metrics()
        
        # Build Knowledge Graph from sentences
        sentence_texts = [node['text'] for node in sentence_nodes]
        kg = KnowledgeGraph()
        kg.build_graph(sentence_texts)
        
        # Align KG with DRG
        drg.add_kg_edges(kg)
        
        # Export graphs for visualization
        os.makedirs('graphs', exist_ok=True)
        span_graph_builder.export_graph_json('graphs/span_graph.json')
        span_graph_builder.export_graph_image('graphs/span_graph.png')
        kg.export_json('graphs/knowledge_graph.json')
        kg.export_graphml('graphs/knowledge_graph.graphml')

        drg.export_graph_image('graphs/drg_graph.png')
        
        # Initialize reasoner (use same model for queries and node embeddings)
        reasoner = EnhancedHybridReasoner(
            sentence_graph=drg.graph,
            span_graph=span_graph_builder.graph,
            kg_graph=kg.graph,
            model_name=chosen_model,
            use_cross_encoder=True
        )
        
        return {
            'drg': drg,
            'span_graph': span_graph_builder,
            'kg': kg,
            'reasoner': reasoner,
            'pages': pages,
            'sentence_nodes': sentence_nodes,
            'spans': spans,

        }
    
    finally:
        try:
            os.unlink(tmp_path)
        except PermissionError:
            # On Windows the file handle can remain briefly locked by a backend.
            pass


def extract_answer_text(results, span_graph, query, reasoner=None, max_length=300):
    """Extract readable answer using shared selector logic."""
    return select_answer(
        results=results,
        span_graph=span_graph,
        query=query,
        reasoner=reasoner,
        max_length=max_length
    )


def _kg_evidence_for_entity(kg, entity: str, max_hops: int = 2) -> List[str]:
    triples = kg.query_entity(entity, max_hops=max_hops)
    return [f"{t['subject']} {t['relation']} {t['object']}" for t in triples[:8]]


def _kg_boolean_vote(kg, entity1: str, entity2: str, question: str) -> Optional[str]:
    if kg is None or kg.graph.number_of_nodes() == 0:
        return None

    facts1 = _kg_evidence_for_entity(kg, entity1)
    facts2 = _kg_evidence_for_entity(kg, entity2)
    all_facts = " ".join(facts1 + facts2).lower()

    neg_patterns = [
        r'\bnot\b', r'\bnever\b', r'\bno\b', r'\bneither\b',
        r"n't\b", r'\bdifferent\b', r'\bvaries\b',
    ]
    neg_count = sum(len(re.findall(p, all_facts)) for p in neg_patterns)

    q_lower = question.lower()
    both_kw = ["both", "same", "also", "either", "share", "similar"]

    e1_present = entity1.lower() in all_facts
    e2_present = entity2.lower() in all_facts

    if neg_count >= 2:
        return "no"
    if any(w in q_lower for w in both_kw) and e1_present and e2_present and neg_count == 0:
        return "yes"
    return None


def refine_comparison_answer(question: str, answer: str, graphs: dict) -> str:
    """Mirror comparison refinement from hotpot_dataset for app predictions."""
    try:
        from parser.comparison_utils import (
            extract_comparison_entities,
            classify_comparison_type,
        )

        kg = graphs.get('kg')
        spans = graphs.get('spans', [])
        pages = graphs.get('pages', [])

        e1, e2 = extract_comparison_entities(question)
        comp_subtype = classify_comparison_type(question)

        def _trim_answer_text(text: str) -> str:
            text = (text or "").strip()
            text = re.sub(r'^[\s\("\'\[]+', '', text)
            text = re.sub(r'[\s\)\]\.\,\;\:\!\?"\']+$', '', text)
            return text.strip()

        def _cmp_norm(text: str) -> str:
            return re.sub(r'[^a-z0-9]+', ' ', (text or '').lower()).strip()

        def _entity_match_score(answer_text: str, entity: str) -> int:
            ans = _cmp_norm(answer_text)
            ent = _cmp_norm(entity)
            if not ans or not ent:
                return 0
            score = 0
            if ent in ans or ans in ent:
                score += 5
            score += len(set(ans.split()) & set(ent.split()))
            return score

        def _clean_entity(ent: str) -> str:
            noise = r'^(are\s+the|is\s+the|are\s+|is\s+|the\s+|both\s+)'
            return re.sub(noise, '', ent, flags=re.IGNORECASE).strip()

        def _retrieve(entity: str) -> List[str]:
            hits: List[str] = []
            for n in spans:
                text = n.get("text", "") if isinstance(n, dict) else str(n)
                if entity.lower() in text.lower():
                    hits.append(text)
                if len(hits) >= 5:
                    break
            if len(hits) < 5:
                for p in pages:
                    p_text = p.get("text", "") if isinstance(p, dict) else ""
                    if entity.lower() in p_text.lower():
                        hits.append(p_text)
                    if len(hits) >= 5:
                        break
            if kg is not None:
                hits.extend(_kg_evidence_for_entity(kg, entity))
            return hits

        if e1 and e2:
            e1 = _clean_entity(e1)
            e2 = _clean_entity(e2)
            e1_texts = _retrieve(e1)
            e2_texts = _retrieve(e2)
            all_evid = " ".join(e1_texts + e2_texts).lower()

            if comp_subtype == "boolean":
                kg_vote = _kg_boolean_vote(kg, e1, e2, question) if kg is not None else None
                strong_neg = [
                    "neither", "only one", "not both",
                    "different countr", "different locat", "not the same"
                ]
                soft_neg = any(
                    re.search(
                        r'\bnot\s+(?:located|headquartered|based|available|a\s+\w+|in\s+the)',
                        " ".join(_retrieve(ent)).lower()
                    )
                    for ent in [e1, e2]
                )
                q_lower = question.lower()
                both_kw = ["both", "same", "also", "either", "share"]
                is_both_q = any(w in q_lower for w in both_kw)

                if kg_vote is not None:
                    answer = kg_vote
                elif any(n in all_evid for n in strong_neg) or soft_neg:
                    answer = "no"
                elif is_both_q and e1.lower() in all_evid and e2.lower() in all_evid:
                    answer = "yes"
                elif e1.lower() in all_evid and e2.lower() in all_evid:
                    answer = "yes"

            if comp_subtype == "boolean":
                ans_norm = _cmp_norm(answer)
                if re.search(r'\byes\b', ans_norm):
                    answer = "yes"
                elif re.search(r'\bno\b', ans_norm):
                    answer = "no"
                else:
                    neg_hits = [" not ", " no ", " neither ", " different ", " not both "]
                    answer = "no" if any(h in f" {all_evid} " for h in neg_hits) else "yes"

            if comp_subtype in ("comparative", "select_one"):
                answer = _trim_answer_text(answer)
                s1 = _entity_match_score(answer, e1)
                s2 = _entity_match_score(answer, e2)
                if s1 > s2:
                    answer = e1
                elif s2 > s1:
                    answer = e2
                else:
                    answer = e1 if len(e1_texts) >= len(e2_texts) else e2

        # Fallback strict enforcement for boolean comparison style questions.
        if comp_subtype == "boolean" and str(answer).lower() not in ("yes", "no"):
            ans_norm = _cmp_norm(answer)
            if re.search(r'\byes\b', ans_norm):
                answer = "yes"
            elif re.search(r'\bno\b', ans_norm):
                answer = "no"
            else:
                answer = "no"

        return answer

    except Exception:
        return answer


# Sidebar - PDF Upload
with st.sidebar:
    st.header("Upload Document")
    st.markdown("*Upload a PDF to analyze and ask questions*")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="Upload any PDF document to start asking questions")
    
    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"File: **{uploaded_file.name}** ({file_size_mb:.1f} MB)")
        
        if st.button("Process PDF", use_container_width=True, type="primary"):
            with st.spinner("Building multi-level graphs from document..."):
                st.session_state.graphs = build_graphs_from_pdf(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                st.session_state.pdf_name = uploaded_file.name
            st.success("Document processed successfully.")
            st.balloons()
    
    st.divider()
    
    # Document info
    if st.session_state.graphs:
        st.markdown("---")
        st.markdown("### Document Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            num_sentences = st.session_state.graphs['drg'].graph.number_of_nodes()
            st.metric("Sentences", num_sentences, help="Total sentences in the document")
        with col2:
            num_spans = st.session_state.graphs['span_graph'].graph.number_of_nodes()
            st.metric("Spans", num_spans, help="Fine-grained text spans extracted")
        with col3:
            num_kg_nodes = st.session_state.graphs['kg'].graph.number_of_nodes()
            st.metric("KG Nodes", num_kg_nodes, help="Knowledge graph entities")


# Main content
if st.session_state.graphs:
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        st.header("Ask a Question")
        
        # Main question input
        question = st.text_input(
            "Type your own question:",
            value="",
            placeholder="e.g., What is the deadline?",
            help="Ask in English!",
            key="question_input"
        )
        
        if question:
            with st.spinner("Searching for answer..."):
                results = st.session_state.graphs['reasoner'].enhanced_reasoning(
                    question, 
                    k=5
                )
            
            # Display answer
            st.markdown("---")
            st.subheader("Answer")
            answer, answer_spans, confidence = extract_answer_text(
                results,
                st.session_state.graphs['span_graph'],
                question,
                reasoner=st.session_state.graphs['reasoner']
            )

            # Keep app behavior aligned with hotpot_dataset comparison rules.
            answer = refine_comparison_answer(question, answer, st.session_state.graphs)
            
            # Styled answer box with proper HTML escaping
            answer_clean = answer.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            
            st.markdown(f"""
            <div style='background-color: #1e3a1e; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #4CAF50; margin-bottom: 10px;'>
                <p style='color: #e0e0e0; font-size: 1.1rem; line-height: 1.6; margin: 0;'>{answer_clean}</p>
                <p style='color: #9e9e9e; font-size: 0.85rem; margin-top: 10px; margin-bottom: 0px;'>Combined Confidence: <b>{confidence:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sub-questions if complex
            if results.get("is_aggregated"):
                st.info(f"Question decomposed into {results.get('sub_questions_count')} sub-steps.")
            
            # Show evidence metrics
            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                num_evidence = len(answer_spans)
                st.metric("Evidence Spans", num_evidence, help="Number of supporting spans found")
            with col_conf2:
                kg_ents = len(results.get('kg_entities', []))
                st.metric("KG Entities Used", kg_ents, help="Knowledge Graph entities explicitly activated during reasoning")
            
            # Evidence
            st.markdown("")
            st.subheader("Evidence Spans")
            if len(answer_spans) > 1:
                st.caption(f"Found {len(answer_spans)} supporting evidence spans")
            for i, span_text in enumerate(answer_spans[:5], 1):  # Limit to top 5
                with st.expander(f"Evidence {i}", expanded=(i==1)):
                    st.markdown(f"```\n{span_text}\n```")
            
            # Evidence Chains
            chains = results.get("evidence_chains", [])
            if chains:
                st.markdown("")
                st.subheader("Reasoning Chains")
                for i, chain in enumerate(chains, 1):
                    with st.expander(f"Chain {i}: {chain['type']}", expanded=True):
                        st.write(chain["text"])
                        st.caption(f"Confidence score: {chain['score']:.4f}")
            
            # Breakdown
            st.markdown("")
            with st.expander("Retrieval Breakdown", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Retrieval Methods**")
                    hybrid = len(results.get('hybrid_results', []))
                    traversal = len(results.get('traversal_results', []))
                    expansion = len(results.get('expansion_results', []))
                    kg_guided = len(results.get('kg_results', []))
                    
                    st.metric("Hybrid Retrieval", hybrid, help="BM25 + Semantic + Centrality")
                    st.metric("Graph Traversal", traversal, help="Multi-hop graph navigation")
                    st.metric("Query Expansion", expansion, help="Synonym-based expansion")
                    st.metric("KG-Guided Retrieval", kg_guided, help="Entity-hop knowledge graph retrieval")
                
                with col_b:
                    st.markdown("**Knowledge Graph**")
                    kg_entity_ids = results.get('kg_entities', [])
                    
                    # Get actual entity text from KG
                    if st.session_state.graphs.get('kg'):
                        kg = st.session_state.graphs['kg']
                        # Display top entities from KG (or matched ones if available)
                        # For now, show top 5 entities by degree if no specific IDs provided
                        if kg_entity_ids:
                            # If reasoner provides specific KG entity IDs
                            entity_texts = []
                            for eid in kg_entity_ids[:5]:
                                if eid in kg.graph.nodes:
                                    entity_text = kg.graph.nodes[eid].get('text', eid)
                                    entity_texts.append(f"{entity_text}")
                            if entity_texts:
                                st.caption("**Entities used:**")
                                for et in entity_texts:
                                    st.caption(f"- {et}")
                        else:
                            # Show top entities by degree
                            top_entities = sorted(kg.graph.nodes(data=True), 
                                                  key=lambda x: kg.graph.degree(x[0]), 
                                                  reverse=True)[:3]
                            if top_entities:
                                st.caption("**Top KG Entities:**")
                                for node_id, data in top_entities:
                                    st.caption(f"- {data.get('text', node_id)}")
    
    with col2:
        st.header("Document Stats")
        
        st.metric(
            "Sentences",
            st.session_state.graphs['drg'].graph.number_of_nodes(),
            help="Total sentences in document"
        )
        st.metric(
            "Spans",
            st.session_state.graphs['span_graph'].graph.number_of_nodes(),
            help="Fine-grained text spans"
        )
        st.metric(
            "KG Entities",
            st.session_state.graphs['kg'].graph.number_of_nodes(),
            help="Knowledge graph entities"
        )
        
        st.divider()
        
        st.markdown("### System Features")
        st.markdown("""
        **Graph Visualizations:**
        - `graphs/drg_graph.png`
        - `graphs/span_graph.png`
        - `graphs/knowledge_graph.json` / `.graphml`
        
        **Graph Types:**
        - Sentence-level DRG
        - Span-level graph
        - Knowledge graph (triples)
        
        **Retrieval Methods:**
        - BM25 lexical matching
        - Semantic embeddings
        - Graph centrality
        - Query expansion
        
        **Languages Supported:**
        - English
        """)

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h2 style='color: #4CAF50;'>Upload a PDF document to get started!</h2>
        <p style='color: #888; font-size: 1.1rem;'>Click the upload button in the sidebar to begin</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Multilingual")
        st.markdown("Ask questions in **English**")
    with col2:
        st.markdown("### Evidence-Based")
        st.markdown("Get **faithful answers** with supporting text spans")
    with col3:
        st.markdown("### Zero-Shot")
        st.markdown("**No training required** - works out of the box")
    
    st.markdown("---")
    
    # How it works
    with st.expander("How it works", expanded=True):
        st.markdown("""
        ### Getting Started
        
        1. **Upload PDF**: Click the upload button in the sidebar
        2. **Process**: System builds multi-level graphs from the document
        3. **Ask Questions**: Type your question in natural language
        4. **Get Answer**: System retrieves evidence-based answers
        5. **View Evidence**: See supporting text spans from the document
        
        ### Key Features
        
        - **Language**: English support
        - **Zero-shot**: No training required
        - **Graph-based**: Multi-level reasoning (sentence + span + knowledge)
        - **Faithful**: Evidence-based answers with proper citations
        - **Hybrid retrieval**: BM25, semantic embeddings, graph centrality
        
        ### Example Questions
        
        - *"What is the deadline for submission?"*
        - *"Marks kya hain isme?"* (What are the marks?)
        - *"When should I submit?"*
        - *"Assignment ka weightage kitna hai?"* (What is the assignment weightage?)
        """)
