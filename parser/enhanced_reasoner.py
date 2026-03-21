"""
Enhanced Hybrid Reasoner with Advanced Retrieval:
- BM25 + Semantic (hybrid retrieval)
- Graph centrality (PageRank, betweenness)
- Cross-encoder re-ranking
- Improved edge weighting
- NLTK stopword filtering
- Better relevance scoring
"""

import numpy as np
import networkx as nx
from sentence_transformers import CrossEncoder
from .model_cache import get_sentence_transformer
from typing import List, Dict, Set

# NLTK stopwords
try:
    import nltk
    try:
        from nltk.corpus import stopwords
        STOP_WORDS = set(stopwords.words('english'))
    except LookupError:
        # Download stopwords if not available
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        STOP_WORDS = set(stopwords.words('english'))
except ImportError:
    # Fallback stopwords if NLTK not available
    STOP_WORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}

from .advanced_retrieval import (
    BM25Retriever,
    GraphCentrality,
    HybridScorer,
    normalize_text,
    tokenize
)


class EnhancedHybridReasoner:
    """
    Enhanced multi-level reasoning with advanced retrieval techniques.
    Uses BM25, centrality, and cross-encoder re-ranking.
    """
    
    def __init__(self, sentence_graph: nx.Graph, span_graph: nx.Graph,
                 model_name="sentence-transformers/all-mpnet-base-v2",
                 use_cross_encoder=True):
        self.sentence_graph = sentence_graph
        self.span_graph = span_graph
        print(f"Loading retrieval model: {model_name}")
        self.model = get_sentence_transformer(model_name)

        self.bm25_spans = None
        self.bm25_sentences = None
        self.span_centrality = None
        self.sentence_centrality = None
        self.hybrid_scorer = HybridScorer()

        self.use_cross_encoder = use_cross_encoder
        if use_cross_encoder:
            print("Using cross-encoder for re-ranking")
            try:
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception:
                self.cross_encoder = None
                self.use_cross_encoder = False
                print("Cross-encoder disabled (model load failed).")

        self.stopwords = STOP_WORDS

        self._initialize_bm25()
        self._compute_centrality()

        print("Enhanced hybrid reasoner initialized")
        print(f"  - Sentence nodes: {sentence_graph.number_of_nodes()}")
        print(f"  - Span nodes: {span_graph.number_of_nodes()}")
        print("  - BM25 index: ready")
        print("  - Graph centrality: ready")
        print(f"  - Cross-encoder: {'enabled' if use_cross_encoder else 'disabled'}")

    # KG entity embedding cache removed
    
    def _initialize_bm25(self):
        """Build BM25 indices for lexical matching."""
        # Span BM25
        self.span_node_ids = sorted(self.span_graph.nodes())
        self.span_id_to_doc = {span_id: idx for idx, span_id in enumerate(self.span_node_ids)}
        span_texts = [self.span_graph.nodes[n]["text"] for n in self.span_node_ids]
        self.bm25_spans = BM25Retriever()
        self.bm25_spans.fit(span_texts)
        
        # Sentence BM25
        self.sentence_node_ids = sorted(self.sentence_graph.nodes())
        self.sentence_id_to_doc = {sent_id: idx for idx, sent_id in enumerate(self.sentence_node_ids)}
        sent_texts = [self.sentence_graph.nodes[n]["text"] for n in self.sentence_node_ids]
        self.bm25_sentences = BM25Retriever()
        self.bm25_sentences.fit(sent_texts)

    def _score_sentences(self, query: str) -> Dict[int, float]:
        """Score sentence nodes for query relevance."""
        query_tokens = set(tokenize(query))
        q_emb = self.embed_query(query)

        bm25_scores = {}
        for sent_id in self.sentence_node_ids:
            doc_id = self.sentence_id_to_doc.get(sent_id)
            if doc_id is not None:
                bm25_scores[sent_id] = self.bm25_sentences.score(query, doc_id)

        bm25_vals = list(bm25_scores.values())
        bm25_min = min(bm25_vals) if bm25_vals else 0.0
        bm25_max = max(bm25_vals) if bm25_vals else 1.0

        sentence_scores = {}
        for sent_id in self.sentence_node_ids:
            text = self.sentence_graph.nodes[sent_id]["text"]
            text_tokens = set(tokenize(text))
            overlap = min(1.0, len(query_tokens & text_tokens) / max(1, len(query_tokens)))

            emb = self.sentence_graph.nodes[sent_id].get("embedding")
            if emb is not None:
                sem_score = self.cosine(q_emb, emb)
            else:
                sem_score = 0.0

            raw_bm25 = bm25_scores.get(sent_id, 0.0)
            if bm25_max != bm25_min:
                bm25_score = (raw_bm25 - bm25_min) / (bm25_max - bm25_min)
            else:
                bm25_score = 0.0

            cent_score = self.sentence_centrality.get(sent_id, 0) if self.sentence_centrality else 0.0

            sentence_scores[sent_id] = (
                0.45 * sem_score +
                0.25 * overlap +
                0.2 * bm25_score +
                0.1 * cent_score
            )

        return sentence_scores

    def _semantic_expansions(self, query: str, k: int = 3, tokens_per_span: int = 4) -> List[str]:
        """Build transformer-driven query expansions using top semantic spans."""
        q_emb = self.embed_query(query)
        scored = []
        for span_id in self.span_graph.nodes:
            emb = self.span_graph.nodes[span_id].get("embedding")
            if emb is None:
                continue
            sim = self.cosine(q_emb, emb)
            scored.append((span_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_spans = [sid for sid, _ in scored[:k]]

        query_tokens = set(tokenize(query))
        variations = [query]

        for span_id in top_spans:
            text = self.span_graph.nodes[span_id]["text"]
            tokens = [t for t in tokenize(text) if t not in query_tokens and len(t) > 2]
            if not tokens:
                continue
            expansion = query + " " + " ".join(tokens[:tokens_per_span])
            if normalize_text(expansion) != normalize_text(query):
                variations.append(expansion)

        seen = set()
        unique = []
        for item in variations:
            key = normalize_text(item)
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique
    
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
        query_tokens = set(tokenize(query))

        q_emb = self.embed_query(query)

        # 1. Semantic scores — purely model-driven, no domain-specific bonuses
        semantic_scores = {}
        for span_id in self.span_graph.nodes:
            emb = self.span_graph.nodes[span_id]["embedding"]
            sim = self.cosine(q_emb, emb)

            text = self.span_graph.nodes[span_id]["text"]
            text_tokens = set(tokenize(text))
            sentence_id = self.span_graph.nodes[span_id].get("sentence_id")
            sentence_text = self.sentence_graph.nodes.get(sentence_id, {}).get("text", "")
            sentence_tokens = set(tokenize(sentence_text))

            # Light lexical overlap bonus (model-agnostic, not document-specific)
            overlap = len(query_tokens & text_tokens)
            overlap_bonus = min(0.15, 0.03 * overlap)
            if sentence_tokens & query_tokens:
                overlap_bonus += 0.03

            semantic_scores[span_id] = sim + overlap_bonus
        
        # 2. Lexical scores (BM25)
        lexical_scores = {}
        bm25_results = self.bm25_spans.retrieve(query, k=len(self.span_graph.nodes))
        for doc_id, score in bm25_results:
            span_id = self.span_node_ids[doc_id]
            lexical_scores[span_id] = score
        
        # 3. Centrality scores
        centrality_scores = self.span_centrality
        
        # 4. Combine all signals
        # 4. Combine all signals
        all_span_ids = list(self.span_graph.nodes())

        sem_scores = []
        lex_scores = []
        cent_scores = []

        for sid in all_span_ids:
            sem_scores.append(semantic_scores.get(sid, 0))
            lex_scores.append(lexical_scores.get(sid, 0))
            cent_scores.append(centrality_scores.get(sid, 0))
        
        combined = self.hybrid_scorer.combine_scores(sem_scores, lex_scores, cent_scores)
        
        # Rank by combined score
        ranked = sorted(
            zip(all_span_ids, combined),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [sid for sid, _ in ranked[:k*2]]
    
    # ========================================
    # Enhanced Graph Traversal
    # ========================================
    
    def enhanced_span_traversal(self, query: str, depth: int = 2, k: int = 10) -> List[int]:
        """
        Graph traversal with centrality-guided expansion.
        """
        query_tokens = set(tokenize(query))
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
                text = self.span_graph.nodes[span_id]["text"]
                text_tokens = set(tokenize(text))
                disc_bonus = 0.0
                if any(m in normalize_text(text) for m in ["if", "unless", "provided", "only if", "in case", "when", "except", "excluding", "but not", "other than", "apart from", "not", "no", "never", "cannot", "won't", "shouldn't"]):
                    disc_bonus += 0.1
                overlap_bonus = min(0.2, 0.04 * len(text_tokens & query_tokens))
                
                final_score = sem_score + cent_bonus + disc_bonus + overlap_bonus
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
        # Generate transformer-driven query variations
        query_variations = self._semantic_expansions(query, k=3)
        
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
    # Evidence Diversity
    # ========================================
    
    def mmr_rerank(
        self,
        query: str,
        span_ids: List[int],
        k: int,
        lambda_param: float = 0.6
    ) -> List[int]:
        """Maximal Marginal Relevance reranking.

        Balances relevance to the query against novelty relative to already-
        selected spans.  lambda_param=0.6 gives 60% weight to relevance and
        40% to diversity — tuned empirically for this pipeline.

        MMR(s) = λ·sim(s, query) − (1−λ)·max_{sj ∈ selected} sim(s, sj)
        """
        if not span_ids:
            return []
        if len(span_ids) <= k:
            return span_ids

        q_emb = self.embed_query(query)

        # Pre-fetch embeddings
        embs: Dict[int, np.ndarray] = {}
        relevance: Dict[int, float] = {}
        for sid in span_ids:
            emb = self.span_graph.nodes[sid].get("embedding")
            if emb is None:
                emb = self.model.encode([self.span_graph.nodes[sid]["text"]])[0]
            embs[sid] = emb
            relevance[sid] = self.cosine(q_emb, emb)

        selected: List[int] = []
        remaining = list(span_ids)

        while remaining and len(selected) < k:
            if not selected:
                # First item: pure relevance
                best = max(remaining, key=lambda s: relevance[s])
            else:
                best = max(
                    remaining,
                    key=lambda s: (
                        lambda_param * relevance[s]
                        - (1 - lambda_param) * max(
                            self.cosine(embs[s], embs[sel]) for sel in selected
                        )
                    )
                )
            selected.append(best)
            remaining.remove(best)

        return selected

    def get_diverse_evidence(self, span_ids: List[int], max_spans: int = 5) -> List[int]:
        """
        Select diverse spans using MMR so different document regions are
        represented.  Falls back to section-level dedup when embeddings are
        unavailable.
        """
        if len(span_ids) <= max_spans:
            return span_ids

        # Check whether embeddings are usable
        has_emb = all(
            self.span_graph.nodes[sid].get("embedding") is not None
            for sid in span_ids[:3]
        )
        if has_emb:
            # Use MMR with a stored query proxy — take the centroid of the
            # selected spans as a pseudo-query so we diversify without needing
            # the original query string here.
            centroid = np.mean(
                [self.span_graph.nodes[sid]["embedding"] for sid in span_ids[:max_spans*2]],
                axis=0
            )
            # Treat centroid as query, then run MMR
            selected: List[int] = []
            remaining = list(span_ids)
            while remaining and len(selected) < max_spans:
                if not selected:
                    best = max(remaining, key=lambda s: self.cosine(
                        centroid, self.span_graph.nodes[s]["embedding"]))
                else:
                    best = max(
                        remaining,
                        key=lambda s: (
                            0.6 * self.cosine(centroid, self.span_graph.nodes[s]["embedding"])
                            - 0.4 * max(
                                self.cosine(
                                    self.span_graph.nodes[s]["embedding"],
                                    self.span_graph.nodes[sel]["embedding"]
                                ) for sel in selected
                            )
                        )
                    )
                selected.append(best)
                remaining.remove(best)
            return selected

        # Fallback: section-level dedup
        selected_fb = []
        covered_sections: set = set()
        for span_id in span_ids:
            if len(selected_fb) >= max_spans:
                break
            section = normalize_text(self.span_graph.nodes[span_id].get("section", "unknown"))
            if section not in covered_sections:
                selected_fb.append(span_id)
                covered_sections.add(section)
        for span_id in span_ids:
            if len(selected_fb) >= max_spans:
                break
            if span_id not in selected_fb:
                selected_fb.append(span_id)
        return selected_fb
    
    # ========================================
    # Main Enhanced Reasoning
    # ========================================
    
    def enhanced_reasoning(self, query: str, k: int = 10) -> Dict[str, List]:
        """
        Best reasoning strategy combining all improvements.
        """
        query_tokens = set(tokenize(query))

        sentence_scores = self._score_sentences(query)
        # Step 1: Hybrid retrieval (BM25 + Semantic + Centrality)
        hybrid_results = self.enhanced_span_retrieval(query, k=k*2)
        
        # Step 2: Graph traversal from top seeds
        traversal_results = self.enhanced_span_traversal(query, depth=2, k=k*2)
        
        # Step 3: Query expansion
        expansion_results = self.retrieval_with_expansion(query, k=k*2)
        

        # Merge all candidates
        all_candidates = set(hybrid_results) | set(traversal_results) | set(expansion_results)
        
        # Step 5: Final scoring with all signals
        q_emb = self.embed_query(query)
        final_scores = []
        
        bm25_raw_scores = {}
        for span_id in all_candidates:
            if span_id not in self.span_graph.nodes:
                continue
            doc_id = self.span_id_to_doc.get(span_id)
            if doc_id is not None:
                bm25_raw_scores[span_id] = self.bm25_spans.score(query, doc_id)

        bm25_values = list(bm25_raw_scores.values())
        bm25_min = min(bm25_values) if bm25_values else 0.0
        bm25_max = max(bm25_values) if bm25_values else 1.0

        for span_id in all_candidates:
            if span_id not in self.span_graph.nodes:
                continue

            # Semantic score
            emb = self.span_graph.nodes[span_id]["embedding"]
            sem_score = self.cosine(q_emb, emb)

            # Centrality score
            cent_score = self.span_centrality.get(span_id, 0)

            # Lexical overlap score
            text = self.span_graph.nodes[span_id]["text"]
            text_tokens = set(tokenize(text))
            overlap_score = min(1.0, len(query_tokens & text_tokens) / max(1, len(query_tokens)))

            # Sentence-level relevance bonus
            sentence_id = self.span_graph.nodes[span_id].get("sentence_id")
            sentence_bonus = sentence_scores.get(sentence_id, 0.0)

            # BM25 score (min-max normalized for this candidate set)
            raw_bm25 = bm25_raw_scores.get(span_id, 0.0)
            if bm25_max != bm25_min:
                bm25_score = (raw_bm25 - bm25_min) / (bm25_max - bm25_min)
            else:
                bm25_score = 0.0

            # Multi-source voting: how many retrieval strategies surfaced this span
            in_hybrid    = 1.0 if span_id in hybrid_results    else 0.0
            in_traversal = 1.0 if span_id in traversal_results else 0.0
            in_expansion = 1.0 if span_id in expansion_results else 0.0

            # Combined score — purely model/signal-driven, no domain-specific rules
            final_score = (
                0.35 * sem_score +       # Primary: semantic similarity (LaBSE)
                0.20 * bm25_score +      # Lexical matching (BM25)
                0.12 * overlap_score +   # Token overlap
                0.10 * sentence_bonus +  # Sentence-level relevance
                0.08 * cent_score +      # Graph centrality
                0.07 * in_hybrid +       # Multi-source voting
                0.04 * in_traversal +
                0.03 * in_expansion
            )

            final_scores.append((span_id, final_score))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_candidates = [sid for sid, _ in final_scores]
        top_candidates = ranked_candidates[:k*2]
        
        # Step 6: Optional cross-encoder re-ranking
        if self.use_cross_encoder:
            reranked = self.rerank_with_cross_encoder(query, top_candidates, top_k=k*2)
        else:
            reranked = top_candidates

        # Step 7: MMR reranking — diversify so the answer selector sees spans
        # from different parts of the document, not just the top-scoring region.
        reranked = self.mmr_rerank(query, reranked, k=min(k*2, len(reranked)), lambda_param=0.65)

        def is_relevant(span_id: int) -> bool:
            """Improved relevance check with stopword filtering and semantic threshold"""
            text = self.span_graph.nodes[span_id]["text"]
            text_tokens = set(tokenize(text))
            
            # Remove stopwords for meaningful overlap
            query_content_tokens = query_tokens - self.stopwords
            text_content_tokens = text_tokens - self.stopwords
            
            # Calculate meaningful overlap (excluding stopwords)
            meaningful_overlap = len(query_content_tokens & text_content_tokens)
            final_score_map = {sid: score for sid, score in final_scores}

            # Get semantic score for this span
            span_semantic_score = final_score_map.get(span_id, 0.0)  # O(1)

            for sid, score in final_scores:
                if sid == span_id:
                    span_semantic_score = score
                    break
            
            sentence_id = self.span_graph.nodes[span_id].get("sentence_id")
            sentence_score = sentence_scores.get(sentence_id, 0.0)

            # Minimum semantic threshold — reject very low-scoring spans
            if span_semantic_score < 0.12:
                return False

            # Sufficient lexical overlap → relevant
            if meaningful_overlap >= 2:
                return True

            # Strong semantic score alone is enough
            if span_semantic_score >= 0.40:
                return True

            # One meaningful word + decent score
            if meaningful_overlap >= 1 and span_semantic_score >= 0.30:
                return True

            # Strong sentence-level match
            if sentence_score >= 0.50:
                return True

            # Default: require at least one meaningful overlapping word
            return meaningful_overlap >= 1

        filtered = [sid for sid in reranked if is_relevant(sid)]
        if len(filtered) < k:
            final_spans = (filtered + [sid for sid in reranked if sid not in filtered])[:k]
        else:
            final_spans = filtered[:k]
        
        # Apply diversity filtering to avoid redundant evidence
        diverse_spans = self.get_diverse_evidence(final_spans, max_spans=k)
        
        # Keep a minimal abstain guard using evidence count only.
        if len(diverse_spans) < 1:
            return {
                "final_spans": [],
                "hybrid_results": hybrid_results[:5],
                "traversal_results": traversal_results[:5],
                "expansion_results": expansion_results[:5],
                "kg_entities": [],
                "kg_spans": [],
                "span_scores": {}
            }
        
        # Build span scores mapping for answer extraction
        span_score_map = {sid: score for sid, score in final_scores}
        
        return {
            "final_spans": diverse_spans,
            "hybrid_results": hybrid_results[:5],
            "traversal_results": traversal_results[:5],
            "expansion_results": expansion_results[:5],
            "kg_entities": [],
            "kg_spans": [],
            "span_scores": span_score_map
        }
