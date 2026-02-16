"""
Streamlit QA Interface for Graph-Based Document Question Answering
Upload PDF and ask questions with evidence-based retrieval
"""

import streamlit as st
import tempfile
import os

from parser.pdf_parser import extract_pages
from parser.drg_nodes import build_nodes
from parser.drg_graph import DocumentReasoningGraph
from parser.span_extractor import SpanExtractor
from parser.span_graph import SpanGraph
from parser.kg_builder import KnowledgeGraphBuilder
from parser.enhanced_reasoner import EnhancedHybridReasoner


# Page config
st.set_page_config(
    page_title="Graph-Based QA System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Graph-Based Document QA")
st.markdown("""
Multi-level graph reasoning for faithful document question answering.
Upload a PDF and ask questions with evidence-based retrieval.
""")

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
        # Extract pages
        pages = extract_pages(tmp_path)
        
        # Build sentence graph
        sentence_nodes = build_nodes(pages)
        drg = DocumentReasoningGraph()
        drg.add_nodes(sentence_nodes)
        drg.compute_embeddings()
        drg.add_structural_edges()
        drg.add_semantic_edges(threshold=0.75)
        
        # Build span graph
        span_extractor = SpanExtractor()
        spans = span_extractor.extract_spans_from_nodes(sentence_nodes)
        span_graph_builder = SpanGraph()
        span_graph_builder.add_nodes(spans)
        span_graph_builder.compute_embeddings()
        span_graph_builder.add_structural_edges()
        span_graph_builder.add_semantic_edges(threshold=0.7)
        span_graph_builder.add_discourse_edges()
        
        # Build KG
        kg_builder = KnowledgeGraphBuilder()
        kg = kg_builder.build_kg(spans)
        
        # Initialize reasoner
        reasoner = EnhancedHybridReasoner(
            sentence_graph=drg.graph,
            span_graph=span_graph_builder.graph,
            knowledge_graph=kg
        )
        
        return {
            'drg': drg,
            'span_graph': span_graph_builder,
            'reasoner': reasoner,
            'pages': pages,
            'sentence_nodes': sentence_nodes,
            'spans': spans,
            'kg': kg
        }
    
    finally:
        os.unlink(tmp_path)


def extract_answer_text(results, span_graph, max_length=200):
    """Extract readable answer from results"""
    final_spans = results.get('final_spans', [])
    
    if not final_spans:
        return "No answer found", []
    
    # Get text from top spans
    answers = []
    for span_id in final_spans[:3]:
        if span_id in span_graph.graph.nodes:
            text = span_graph.graph.nodes[span_id]['text']
            answers.append(text)
    
    if answers:
        return answers[0][:max_length], answers
    
    return "No answer found", []


# Sidebar - PDF Upload
with st.sidebar:
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"File: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        if st.button("Process PDF", use_container_width=True):
            with st.spinner("Building document graphs..."):
                st.session_state.graphs = build_graphs_from_pdf(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                st.session_state.pdf_name = uploaded_file.name
            st.success("✓ Document processed!")
    
    st.divider()
    
    # Document info
    if st.session_state.graphs:
        col1, col2 = st.columns(2)
        with col1:
            num_sentences = st.session_state.graphs['drg'].graph.number_of_nodes()
            st.metric("Sentences", num_sentences)
        with col2:
            num_spans = st.session_state.graphs['span_graph'].graph.number_of_nodes()
            st.metric("Spans", num_spans)
        
        num_entities = len(st.session_state.graphs['kg']['entities'])
        st.metric("Entities", num_entities)


# Main content
if st.session_state.graphs:
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        st.header("❓ Ask Question")
        question = st.text_input(
            "Enter your question about the document:",
            placeholder="e.g., What is the deadline? When should I submit?"
        )
        
        if question:
            with st.spinner("Searching for answer..."):
                results = st.session_state.graphs['reasoner'].enhanced_reasoning(
                    question, 
                    k=5
                )
            
            # Display answer
            st.subheader("📍 Answer")
            answer, answer_spans = extract_answer_text(
                results,
                st.session_state.graphs['span_graph']
            )
            
            st.success(answer)
            
            # Evidence
            st.subheader("📋 Evidence Spans")
            for i, span_text in enumerate(answer_spans, 1):
                with st.expander(f"Evidence {i}", expanded=(i==1)):
                    st.write(span_text)
            
            # Breakdown
            with st.expander("🔍 Retrieval Breakdown", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Retrieval Methods**")
                    hybrid = len(results.get('hybrid_results', []))
                    traversal = len(results.get('traversal_results', []))
                    expansion = len(results.get('expansion_results', []))
                    
                    st.write(f"• Hybrid: {hybrid} spans")
                    st.write(f"• Traversal: {traversal} spans")
                    st.write(f"• Expansion: {expansion} spans")
                
                with col_b:
                    st.markdown("**Knowledge Graph**")
                    kg_entities = results.get('kg_entities', [])
                    st.write(f"• Entities used: {len(kg_entities)}")
    
    with col2:
        st.header("📊 Statistics")
        
        st.metric(
            "Sentences",
            st.session_state.graphs['drg'].graph.number_of_nodes()
        )
        st.metric(
            "Spans",
            st.session_state.graphs['span_graph'].graph.number_of_nodes()
        )
        st.metric(
            "Entities",
            len(st.session_state.graphs['kg']['entities'])
        )
        
        st.divider()
        
        st.markdown("### 🎯 System Info")
        st.markdown("""
        **Graph Types:**
        - Sentence-level DRG
        - Span-level graph
        - Knowledge graph
        
        **Retrieval:**
        - BM25 lexical matching
        - Semantic embeddings
        - Graph centrality
        
        **Languages:**
        - English ✓
        - 50+ languages ✓
        """)

else:
    st.info("👈 Upload a PDF document to start asking questions!")
    
    # Example
    with st.expander("📖 How it works"):
        st.markdown("""
        1. **Upload PDF**: Click the upload button in the sidebar
        2. **Process**: System builds multi-level graphs from document
        3. **Ask Questions**: Type your question in natural language
        4. **Get Answer**: System retrieves evidence spans from the document
        5. **View Evidence**: See the relevant text supporting the answer
        
        **Key Features:**
        - Multilingual queries (English, Hindi, Hinglish)
        - Zero-shot - no training required
        - Evidence-based answers with span highlighting
        - Graph-based reasoning for faithful results
        """)
