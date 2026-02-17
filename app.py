"""
Streamlit QA Interface for Graph-Based Document Question Answering
Upload PDF and ask questions with evidence-based retrieval
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tempfile

from parser.pdf_parser import extract_pages
from parser.drg_nodes import build_nodes
from parser.drg_graph import DocumentReasoningGraph
from parser.span_extractor import SpanExtractor
from parser.span_graph import SpanGraph
from parser.kg_builder import KnowledgeGraphBuilder
from parser.enhanced_reasoner import EnhancedHybridReasoner
from parser.advanced_retrieval import normalize_text, tokenize


# Page config
st.set_page_config(
    page_title="Graph-Based QA System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Graph-Based Document QA System")
st.markdown("""
<div style='background-color: #0e1117; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #4CAF50;'>
    <p style='margin: 0; color: #e0e0e0;'>
        <strong>Multi-level graph reasoning</strong> for faithful document question answering.<br>
        Supports <strong>multilingual queries</strong> including English, Hindi, and Hinglish.
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
        span_graph_builder.compute_graph_metrics()
        
        # Build KG
        kg_builder = KnowledgeGraphBuilder()
        kg_data = kg_builder.build_kg(spans)
        
        # Export graphs for visualization
        os.makedirs('graphs', exist_ok=True)
        span_graph_builder.export_graph_json('graphs/span_graph.json')
        span_graph_builder.export_graph_image('graphs/span_graph.png')
        kg_builder.export_graph_json('graphs/kg_graph.json')
        kg_builder.export_graph_image('graphs/kg_graph.png')
        drg.export_graph_image('graphs/drg_graph.png')
        
        # Initialize reasoner
        reasoner = EnhancedHybridReasoner(
            sentence_graph=drg.graph,
            span_graph=span_graph_builder.graph,
            knowledge_graph=kg_data,
            model_name="sentence-transformers/LaBSE",
            use_cross_encoder=True
        )
        
        return {
            'drg': drg,
            'span_graph': span_graph_builder,
            'reasoner': reasoner,
            'pages': pages,
            'sentence_nodes': sentence_nodes,
            'spans': spans,
            'kg': kg_data,  # Return the complete kg data
            'kg_builder': kg_builder
        }
    
    finally:
        os.unlink(tmp_path)


def extract_answer_text(results, span_graph, query, reasoner=None, max_length=300):
    """
    Extract readable answer using model ranking scores and sentence context.
    Much more accurate than simple overlap scoring.
    """
    import re
    
    final_spans = results.get('final_spans', [])
    
    if not final_spans:
        return "No answer found", [], 0.0, "Low ✗"
    
    # Get the span scores from results for proper ranking
    span_scores = results.get('span_scores', {})
    
    # Sort spans by their actual model scores (not simple overlap)
    ranked_spans = []
    for span_id in final_spans[:15]:  # Increased from 10 to get better candidates
        if span_id in span_graph.graph.nodes:
            score = span_scores.get(span_id, 0.0)
            text = span_graph.graph.nodes[span_id]['text'].strip()
            sentence_id = span_graph.graph.nodes[span_id].get('sentence_id')
            span_type = span_graph.graph.nodes[span_id].get('span_type', '')
            
            # Boost score for sentence-type spans (more complete)
            if span_type == 'sentence':
                score = score * 1.1
            
            ranked_spans.append((span_id, text, score, sentence_id, span_type))
    
    if not ranked_spans:
        return "No answer found", [], 0.0, "Low ✗"
    
    # Sort by boosted model score (highest first)
    ranked_spans.sort(key=lambda x: x[2], reverse=True)
    
    # Get best answer with sentence context for readability
    best_span_id, best_text, best_score, best_sentence_id, best_span_type = ranked_spans[0]
    
    # Prioritize sentence-type spans for better readability
    if best_span_type == 'sentence' or len(best_text.strip()) >= 60:
        best_answer = best_text
    elif best_sentence_id is not None and best_sentence_id in reasoner.sentence_graph.nodes:
        # Use full sentence for context
        sentence_text = reasoner.sentence_graph.nodes[best_sentence_id]['text']
        if len(sentence_text) < max_length * 1.2:
            best_answer = sentence_text
        else:
            # Sentence too long, keep the span
            best_answer = best_text
    else:
        best_answer = best_text
    
    # Clean up answer
    best_answer = re.sub(r'\s+', ' ', best_answer).strip()
    
    # Collect ONLY highly relevant evidence spans (top 3 with high scores)
    evidence_texts = []
    for _, text, score, _, _ in ranked_spans[:5]:
        # Only show evidence with score >= 60% of best score
        if score >= best_score * 0.6:
            evidence_texts.append(text)
        if len(evidence_texts) >= 3:  # Max 3 evidence spans
            break
    
    # Validate answer against evidence
    if reasoner:
        is_valid, coverage = reasoner.verify_answer_support(best_answer, evidence_texts)
    else:
        is_valid, coverage = True, 0.7
    
    # Truncate if too long
    if len(best_answer) > max_length:
        best_answer = best_answer[:max_length]
        last_period = max(best_answer.rfind('.'), best_answer.rfind('।'))
        if last_period > max_length * 0.6:
            best_answer = best_answer[:last_period + 1]
        else:
            best_answer = best_answer.rsplit(' ', 1)[0] + '...'
    
    # Get confidence from results
    confidence = results.get('confidence', 0.5)
    confidence_label = results.get('confidence_label', 'Medium ◐')
    
    return best_answer, evidence_texts, confidence, confidence_label


# Sidebar - PDF Upload
with st.sidebar:
    st.header("📤 Upload Document")
    st.markdown("*Upload a PDF to analyze and ask questions*")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="Upload any PDF document to start asking questions")
    
    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📄 **{uploaded_file.name}** ({file_size_mb:.1f} MB)")
        
        if st.button("🚀 Process PDF", use_container_width=True, type="primary"):
            with st.spinner("🔨 Building multi-level graphs from document..."):
                st.session_state.graphs = build_graphs_from_pdf(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                st.session_state.pdf_name = uploaded_file.name
            st.success("✅ Document processed successfully!")
            st.balloons()
    
    st.divider()
    
    # Document info
    if st.session_state.graphs:
        st.markdown("---")
        st.markdown("### 📊 Document Statistics")
        col1, col2 = st.columns(2)
        with col1:
            num_sentences = st.session_state.graphs['drg'].graph.number_of_nodes()
            st.metric("📝 Sentences", num_sentences, help="Total sentences in the document")
        with col2:
            num_spans = st.session_state.graphs['span_graph'].graph.number_of_nodes()
            st.metric("🔤 Spans", num_spans, help="Fine-grained text spans extracted")
        
        num_entities = len(st.session_state.graphs['kg']['entities'])
        st.metric("🏷️ Entities", num_entities, help="Named entities identified")


# Main content
if st.session_state.graphs:
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        st.header("❓ Ask a Question")
        
        # Main question input
        question = st.text_input(
            "Type your own question:",
            value="",
            placeholder="e.g., What is the deadline? / Antim tarikh kya hai? / Submission kab tak?",
            help="Ask in English, Hindi, or Hinglish!",
            key="question_input"
        )
        
        if question:
            with st.spinner("🔍 Searching for answer..."):
                results = st.session_state.graphs['reasoner'].enhanced_reasoning(
                    question, 
                    k=5
                )
            
            # Display answer
            st.markdown("---")
            st.subheader("📌 Answer")
            answer, answer_spans, confidence, confidence_label = extract_answer_text(
                results,
                st.session_state.graphs['span_graph'],
                question,
                reasoner=st.session_state.graphs['reasoner']
            )
            
            # Styled answer box with proper HTML escaping
            answer_clean = answer.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            
            st.markdown(f"""
            <div style='background-color: #1e3a1e; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #4CAF50;'>
                <p style='color: #e0e0e0; font-size: 1.1rem; line-height: 1.6; margin: 0;'>{answer_clean}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence & evidence metrics
            col_conf1, col_conf2, col_conf3 = st.columns(3)
            with col_conf1:
                confidence_pct = int(confidence * 100)
                st.metric("Confidence", f"{confidence_pct}%", confidence_label, help="Answer confidence based on evidence quality")
            with col_conf2:
                num_evidence = len(answer_spans)
                st.metric("Evidence Spans", num_evidence, help="Number of supporting spans found")
            with col_conf3:
                kg_ents = len(results.get('kg_entities', []))
                st.metric("KG Entities", kg_ents, help="Knowledge graph entities used")
            
            # Evidence
            st.markdown("")
            st.subheader("📋 Evidence Spans")
            if len(answer_spans) > 1:
                st.caption(f"Found {len(answer_spans)} supporting evidence spans")
            for i, span_text in enumerate(answer_spans[:5], 1):  # Limit to top 5
                with st.expander(f"📄 Evidence {i}", expanded=(i==1)):
                    st.markdown(f"```\n{span_text}\n```")
            
            # Breakdown
            st.markdown("")
            with st.expander("🔬 Retrieval Breakdown", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**🎯 Retrieval Methods**")
                    hybrid = len(results.get('hybrid_results', []))
                    traversal = len(results.get('traversal_results', []))
                    expansion = len(results.get('expansion_results', []))
                    
                    st.metric("Hybrid Retrieval", hybrid, help="BM25 + Semantic + Centrality")
                    st.metric("Graph Traversal", traversal, help="Multi-hop graph navigation")
                    st.metric("Query Expansion", expansion, help="Synonym-based expansion")
                
                with col_b:
                    st.markdown("**🧠 Knowledge Graph**")
                    kg_entity_ids = results.get('kg_entities', [])
                    st.metric("Entities Found", len(kg_entity_ids), help="Named entities used in reasoning")
                    
                    # Get actual entity text from KG
                    if kg_entity_ids and st.session_state.graphs.get('kg'):
                        kg = st.session_state.graphs['kg']
                        entity_texts = []
                        for eid in kg_entity_ids[:5]:
                            for entity in kg['entities']:
                                if entity['entity_id'] == eid:
                                    entity_texts.append(f"{entity['text']} ({entity['entity_type']})")
                                    break
                        if entity_texts:
                            st.caption("**Entities used:**")
                            for et in entity_texts:
                                st.caption(f"• {et}")
    
    with col2:
        st.header("📊 Document Stats")
        
        st.metric(
            "📝 Sentences",
            st.session_state.graphs['drg'].graph.number_of_nodes(),
            help="Total sentences in document"
        )
        st.metric(
            "🔤 Spans",
            st.session_state.graphs['span_graph'].graph.number_of_nodes(),
            help="Fine-grained text spans"
        )
        st.metric(
            "🏷️ Entities",
            len(st.session_state.graphs['kg']['entities']),
            help="Named entities extracted"
        )
        
        st.divider()
        
        st.markdown("### 🎯 System Features")
        st.markdown("""
        **� Graph Visualizations:**
        - `graphs/drg_graph.png`
        - `graphs/span_graph.png`  
        - `graphs/kg_graph.png`
        
        **�🔗 Graph Types:**
        - 📊 Sentence-level DRG
        - 🔤 Span-level graph
        - 🧠 Knowledge graph
        
        **🔍 Retrieval Methods:**
        - 📚 BM25 lexical matching
        - 🧬 Semantic embeddings
        - 🌐 Graph centrality
        - 🔄 Query expansion
        
        **🌍 Languages Supported:**
        - ✅ Hindi / Hinglish / English
        """)

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h2 style='color: #4CAF50;'>👈 Upload a PDF document to get started!</h2>
        <p style='color: #888; font-size: 1.1rem;'>Click the upload button in the sidebar to begin</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🌍 Multilingual")
        st.markdown("Ask questions in **English, Hindi, or Hinglish**")
    with col2:
        st.markdown("### 🎯 Evidence-Based")
        st.markdown("Get **faithful answers** with supporting text spans")
    with col3:
        st.markdown("### ⚡ Zero-Shot")
        st.markdown("**No training required** - works out of the box")
    
    st.markdown("---")
    
    # How it works
    with st.expander("📖 How it works", expanded=True):
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **📤 Upload PDF**: Click the upload button in the sidebar
        2. **🔨 Process**: System builds multi-level graphs from the document
        3. **❓ Ask Questions**: Type your question in natural language
        4. **📌 Get Answer**: System retrieves evidence-based answers
        5. **📋 View Evidence**: See supporting text spans from the document
        
        ### ✨ Key Features
        
        - 🌍 **Multilingual**: English, Hindi, Hinglish support
        - ⚡ **Zero-shot**: No training required
        - 📊 **Graph-based**: Multi-level reasoning (sentence + span + knowledge)
        - 🎯 **Faithful**: Evidence-based answers with proper citations
        - 🔍 **Hybrid retrieval**: BM25, semantic embeddings, graph centrality
        
        ### 💡 Example Questions
        
        - *"What is the deadline for submission?"*
        - *"Marks kya hain isme?"* (What are the marks?)
        - *"When should I submit?"*
        - *"Assignment ka weightage kitna hai?"* (What is the assignment weightage?)
        """)
