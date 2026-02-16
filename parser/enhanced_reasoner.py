"""
Enhanced Hybrid Reasoner with Advanced Retrieval:
- BM25 + Semantic (hybrid retrieval)
- Graph centrality (PageRank, betweenness)
- Cross-encoder re-ranking
- Improved edge weighting
"""

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple, Set

from .advanced_retrieval import (
    BM25Retriever,
    GraphCentrality,
    HybridScorer,
    QueryExpander
)


class EnhancedHybridReasoner:
    """
    Enhanced multi-level reasoning with advanced retrieval techniques.
    Uses BM25, centrality, and cross-encoder re-ranking.
    """
    
    def __init__(
        self, 
        sentence_graph: nx.Graph,
        span_graph: nx.Graph,
        knowledge_graph: Dict,
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        use_cross_encoder=False
    ):
        self.sentence_graph = sentence_graph
        self.span_graph = span_graph
        self.kg = knowledge_graph
        print(f"🌍 Loading multilingual model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✓ Multilingual embeddings ready (supports 50+ languages including Hindi/Hinglish)")
        
        # Advanced components
        self.bm25_spans = None
        self.bm25_sentences = None
        self.span_centrality = None
        self.sentence_centrality = None
        self.query_expander = QueryExpander()
        self.hybrid_scorer = HybridScorer()
        
        # Cross-encoder for re-ranking (optional, slower but more accurate)
        self.use_cross_encoder = use_cross_encoder
        if use_cross_encoder:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize BM25 and centrality
        self._initialize_bm25()
        self._compute_centrality()
        
        print(f"Enhanced hybrid reasoner initialized:")
        print(f"  - Sentence nodes: {sentence_graph.number_of_nodes()}")
        print(f"  - Span nodes: {span_graph.number_of_nodes()}")
        print(f"  - KG entities: {len(knowledge_graph['entities'])}")
        print(f"  - BM25 index: ✓")
        print(f"  - Graph centrality: ✓")
        print(f"  - Cross-encoder: {'✓' if use_cross_encoder else '✗'}")
    
    def _initialize_bm25(self):
        """Build BM25 indices for lexical matching."""
        # Span BM25
        span_texts = [
            self.span_graph.nodes[n]["text"] 
            for n in sorted(self.span_graph.nodes())
        ]
        self.bm25_spans = BM25Retriever()
        self.bm25_spans.fit(span_texts)
        
        # Sentence BM25
        sent_texts = [
            self.sentence_graph.nodes[n]["text"] 
            for n in sorted(self.sentence_graph.nodes())
        ]
        self.bm25_sentences = BM25Retriever()
        self.bm25_sentences.fit(sent_texts)
    
    def _compute_centrality(self):
        """Compute node importance using graph algorithms."""
        print("Computing graph centrality...")
        
        # Span centrality
        self.span_pagerank = GraphCentrality.pagerank(self.span_graph)
        self.span_betweenness = GraphCentrality.betweenness_centrality(self.span_graph)
        
        # Combined centrality (average of multiple metrics)
        self.span_centrality = {}
        for node in self.span_graph.nodes():
            pr = self.span_pagerank.get(node, 0)
            bc = self.span_betweenness.get(node, 0)
            self.span_centrality[node] = (pr + bc) / 2
        
        # Sentence centrality
        self.sent_pagerank = GraphCentrality.pagerank(self.sentence_graph)
        self.sentence_centrality = self.sent_pagerank
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed query using sentence transformer."""
        return self.model.encode([query])[0]
    
    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # ========================================
    # Enhanced Span Retrieval
    # ========================================
    
    def enhanced_span_retrieval(self, query: str, k: int = 10) -> List[int]:
        """
        Hybrid retrieval: BM25 + Semantic + Centrality
        """
        q_emb = self.embed_query(query)
        
        # 1. Semantic scores (embedding similarity)
        semantic_scores = {}
        for span_id in self.span_graph.nodes:
            emb = self.span_graph.nodes[span_id]["embedding"]
            sim = self.cosine(q_emb, emb)
            
            # Type-specific bonuses
            span_type = self.span_graph.nodes[span_id]["span_type"]
            text = self.span_graph.nodes[span_id]["text"].lower()
            
            bonus = 0.0
            if span_type in ["temporal", "deadline"] and "when" in query.lower():
                bonus += 0.15
            if span_type in ["condition", "negation"]:
                bonus += 0.1
            if "deadline" in query.lower() and "deadline" in text:
                bonus += 0.2
            
            semantic_scores[span_id] = sim + bonus
        
        # 2. Lexical scores (BM25)
        lexical_scores = {}
        bm25_results = self.bm25_spans.retrieve(query, k=len(self.span_graph.nodes))
        for doc_id, score in bm25_results:
            span_id = sorted(self.span_graph.nodes())[doc_id]
            lexical_scores[span_id] = score
        
        # 3. Centrality scores
        centrality_scores = self.span_centrality
        
        # 4. Combine all signals
        all_span_ids = list(self.span_graph.nodes())
        sem_scores = [semantic_scores.get(sid, 0) for sid in all_span_ids]
        lex_scores = [lexical_scores.get(sid, 0) for sid in all_span_ids]
        cent_scores = [centrality_scores.get(sid, 0) for sid in all_span_ids]
        
        combined = self.hybrid_scorer.combine_scores(sem_scores, lex_scores, cent_scores)
        
        # Rank by combined score
        ranked = sorted(
            zip(all_span_ids, combined),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [sid for sid, _ in ranked[:k]]
    
    # ========================================
    # Enhanced Graph Traversal
    # ========================================
    
    def enhanced_span_traversal(self, query: str, depth: int = 2, k: int = 10) -> List[int]:
        """
        Graph traversal with centrality-guided expansion.
        """
        # Start with hybrid retrieval seeds
        seeds = self.enhanced_span_retrieval(query, k=5)
        
        visited = set(seeds)
        frontier = list(seeds)
        
        # Expand with centrality-guided selection
        for _ in range(depth):
            new_frontier = []
            
            for span_id in frontier:
                neighbors = list(self.span_graph.neighbors(span_id))
                
                # Sort neighbors by centrality (visit important nodes first)
                neighbors.sort(
                    key=lambda n: self.span_centrality.get(n, 0),
                    reverse=True
                )
                
                for neighbor in neighbors[:5]:  # Top 5 neighbors only
                    if neighbor not in visited:
                        edge_data = self.span_graph.get_edge_data(span_id, neighbor)
                        edge_type = edge_data.get("type", "")
                        
                        # Prioritize discourse edges
                        if edge_type in ["condition", "exception", "temporal", "same_sentence"]:
                            visited.add(neighbor)
                            new_frontier.append(neighbor)
            
            frontier = new_frontier
        
        # Re-score all visited spans
        q_emb = self.embed_query(query)
        scored = []
        
        for span_id in visited:
            # Semantic score
            emb = self.span_graph.nodes[span_id]["embedding"]
            sem_score = self.cosine(q_emb, emb)
            
            # Centrality bonus
            cent_bonus = self.span_centrality.get(span_id, 0) * 0.2
            
            # Discourse bonus
            text = self.span_graph.nodes[span_id]["text"].lower()
            disc_bonus = 0.0
            if any(m in text for m in ["except", "unless", "not", "no"]):
                disc_bonus += 0.1
            
            final_score = sem_score + cent_bonus + disc_bonus
            scored.append((span_id, final_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in scored[:k]]
    
    # ========================================
    # Query Expansion
    # ========================================
    
    def retrieval_with_expansion(self, query: str, k: int = 10) -> List[int]:
        """
        Retrieve using expanded queries and merge results.
        """
        # Generate query variations
        query_variations = self.query_expander.expand(query)
        
        # Retrieve for each variation
        all_results = {}
        for q_var in query_variations:
            results = self.enhanced_span_retrieval(q_var, k=k)
            
            # Accumulate scores
            for rank, span_id in enumerate(results):
                score = 1.0 / (rank + 1)  # Reciprocal rank
                all_results[span_id] = all_results.get(span_id, 0) + score
        
        # Sort by accumulated score
        ranked = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in ranked[:k]]
    
    # ========================================
    # Cross-Encoder Re-ranking
    # ========================================
    
    def rerank_with_cross_encoder(
        self, 
        query: str, 
        candidate_spans: List[int], 
        top_k: int = 10
    ) -> List[int]:
        """
        Re-rank candidates using cross-encoder for better accuracy.
        """
        if not self.use_cross_encoder:
            return candidate_spans[:top_k]
        
        # Prepare query-span pairs
        pairs = []
        for span_id in candidate_spans:
            text = self.span_graph.nodes[span_id]["text"]
            pairs.append([query, text])
        
        # Cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Re-rank
        ranked = sorted(
            zip(candidate_spans, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [sid for sid, _ in ranked[:top_k]]
    
    # ========================================
    # Main Enhanced Reasoning
    # ========================================
    
    def enhanced_reasoning(self, query: str, k: int = 10) -> Dict[str, List]:
        """
        Best reasoning strategy combining all improvements.
        """
        # Step 1: Hybrid retrieval (BM25 + Semantic + Centrality)
        hybrid_results = self.enhanced_span_retrieval(query, k=k*2)
        
        # Step 2: Graph traversal from top seeds
        traversal_results = self.enhanced_span_traversal(query, depth=2, k=k*2)
        
        # Step 3: Query expansion
        expansion_results = self.retrieval_with_expansion(query, k=k*2)
        
        # Step 4: KG-guided (reuse from original)
        kg_entity_ids = self._kg_entity_retrieval(query, k=3)
        kg_span_ids = set()
        for entity_id in kg_entity_ids:
            entity = self.kg["entities"][entity_id]
            kg_span_ids.update(entity["span_ids"])
        
        # Merge all candidates
        all_candidates = set(hybrid_results) | set(traversal_results) | set(expansion_results) | kg_span_ids
        
        # Step 5: Final scoring with all signals
        q_emb = self.embed_query(query)
        final_scores = []
        
        for span_id in all_candidates:
            if span_id not in self.span_graph.nodes:
                continue
            
            # Semantic score
            emb = self.span_graph.nodes[span_id]["embedding"]
            sem_score = self.cosine(q_emb, emb)
            
            # Centrality score
            cent_score = self.span_centrality.get(span_id, 0)
            
            # Presence bonuses
            in_hybrid = 1.0 if span_id in hybrid_results else 0.0
            in_traversal = 1.0 if span_id in traversal_results else 0.0
            in_expansion = 1.0 if span_id in expansion_results else 0.0
            in_kg = 1.0 if span_id in kg_span_ids else 0.0
            
            # Combined score
            final_score = (
                0.4 * sem_score +
                0.2 * cent_score +
                0.1 * in_hybrid +
                0.1 * in_traversal +
                0.1 * in_expansion +
                0.1 * in_kg
            )
            
            final_scores.append((span_id, final_score))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [sid for sid, _ in final_scores[:k*2]]
        
        # Step 6: Optional cross-encoder re-ranking
        if self.use_cross_encoder:
            final_spans = self.rerank_with_cross_encoder(query, top_candidates, top_k=k)
        else:
            final_spans = top_candidates[:k]
        
        return {
            "final_spans": final_spans,
            "hybrid_results": hybrid_results[:5],
            "traversal_results": traversal_results[:5],
            "expansion_results": expansion_results[:5],
            "kg_entities": kg_entity_ids,
            "kg_spans": list(kg_span_ids)[:5]
        }
    
    def _kg_entity_retrieval(self, query: str, k: int = 5) -> List[int]:
        """Retrieve relevant KG entities (from original)."""
        q_emb = self.embed_query(query)
        
        scores = []
        for entity in self.kg["entities"]:
            entity_emb = self.model.encode([entity["text"]])[0]
            sim = self.cosine(q_emb, entity_emb)
            
            # Type-specific boost
            bonus = 0.0
            if "date" in entity["entity_type"].lower():
                if any(w in query.lower() for w in ["when", "deadline", "due"]):
                    bonus += 0.2
            
            scores.append((entity["entity_id"], sim + bonus))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in scores[:k]]
