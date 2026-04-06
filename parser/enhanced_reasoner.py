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
try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except Exception:
    _CROSS_ENCODER_AVAILABLE = False
from .model_cache import get_sentence_transformer
from typing import List, Dict, Set, Tuple
from collections import defaultdict

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
from .question_processor import QuestionProcessor


class EnhancedHybridReasoner:
    """
    Enhanced multi-level reasoning with advanced retrieval techniques.
    Handles complex queries via decomposition and graph-based multi-hop reasoning.
    """
    
    def __init__(self, sentence_graph: nx.Graph, span_graph: nx.Graph,
                 kg_graph: nx.Graph = None,
                 model_name="sentence-transformers/all-mpnet-base-v2",
                 use_cross_encoder=True):
        self.sentence_graph = sentence_graph
        self.span_graph = span_graph
        self.kg_graph = kg_graph
        print(f"Loading retrieval model: {model_name}")
        self.model = get_sentence_transformer(model_name)
        
        # Initialize question processor
        self.question_processor = QuestionProcessor()

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
        if kg_graph:
            print(f"  - KG nodes: {kg_graph.number_of_nodes()}")
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
        if self.model is None:
            return np.zeros(384)
        return self.model.encode([query])[0]
    
    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    # Retrieve candidate spans using hybrid semantic, lexical, and centrality scoring.
    
    def enhanced_span_retrieval(self, query: str, k: int = 10) -> List[int]:
        """
        Hybrid retrieval: BM25 + Semantic + Centrality
        """
        query_tokens = set(tokenize(query))

        q_emb = self.embed_query(query)

        # 1. Semantic scores - purely model-driven, no domain-specific bonuses
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
    
    # Expand retrieval via graph traversal with centrality-guided neighbors.
    
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
    
    # Re-run retrieval on semantically expanded query variants.
    
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

    # Use KG entities and neighbors to retrieve additional relevant spans.
    
    def kg_guided_retrieval(self, query: str, k: int = 10) -> List[int]:
        """
        Extract entities from query, find 1-hop neighbors in KG, and retrieve spans mentioning them.
        """
        if not self.kg_graph:
            return []
            
        q_norm = normalize_text(query).lower()
        query_entities = [ent for ent in self.kg_graph.nodes() if re.search(rf'\b{re.escape(ent.lower())}\b', q_norm)]
        
        if not query_entities:
            return []
            
        # Find 1-hop neighbors spanning from query entities
        expanded_entities = set(query_entities)
        for ent in query_entities:
            # Directed and undirected neighbors
            for neighbor in self.kg_graph.successors(ent) if self.kg_graph.is_directed() else self.kg_graph.neighbors(ent):
                expanded_entities.add(neighbor)
            if self.kg_graph.is_directed():
                for neighbor in self.kg_graph.predecessors(ent):
                    expanded_entities.add(neighbor)
                
        # Retrieve spans mentioning any of the expanded entities
        candidate_spans = []
        for span_id in self.span_graph.nodes():
            text = self.span_graph.nodes[span_id].get("text", "").lower()
            if any(re.search(rf'\b{re.escape(ent.lower())}\b', text) for ent in expanded_entities):
                candidate_spans.append(span_id)
                
        if not candidate_spans:
            return []
            
        # Score the candidates by semantic similarity to query to rank them
        q_emb = self.embed_query(query)
        scored = []
        for span_id in candidate_spans:
            emb = self.span_graph.nodes[span_id].get("embedding")
            sim = self.cosine(q_emb, emb) if emb is not None else 0.0
            scored.append((span_id, sim))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in scored[:k]]
    
    # Apply optional cross-encoder reranking for higher precision.
    
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
 
    # Keep evidence diverse with MMR and section-aware fallback logic.
    
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
        40% to diversity - tuned empirically for this pipeline.

        MMR(s) = lambda*sim(s, query) - (1-lambda)*max_{sj in selected} sim(s, sj)
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
            # Use MMR with a stored query proxy - take the centroid of the
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
    
    # Run the complete reasoning pipeline, including sub-question handling.
    
    def enhanced_reasoning(self, query: str, k: int = 10) -> Dict[str, List]:
        """
        Best reasoning strategy combining all improvements, including sub-question processing.
        """
        # Step 0: Process question (classify and decompose)
        q_info = self.question_processor.process(query)
        sub_questions = q_info["sub_questions"]
        q_type = q_info["type"]
        is_complex = q_info["is_complex"]

        if is_complex:
            print(f"Decomposing complex question into {len(sub_questions)} sub-questions...")
            all_sub_results = []
            for sub_q in sub_questions:
                print(f"  - Sub-question: {sub_q}")
                sub_res = self._run_reasoning_core(sub_q, k=k)
                all_sub_results.append(sub_res)
            
            # Aggregate sub-question results
            return self._aggregate_reasoning_results(query, all_sub_results, k=k)
        else:
            return self._run_reasoning_core(query, k=k)

    def _aggregate_reasoning_results(self, original_query: str, sub_results: List[Dict], k: int = 10) -> Dict:
        """
        Merge and rerank results from multiple sub-questions.
        """
        combined_spans_set = set()
        all_span_scores = defaultdict(float)
        all_kg_entities = set()
        all_chains = []
        
        # Weighted aggregation (simplified: sum of scores)
        for res in sub_results:
            for span_id in res["final_spans"]:
                combined_spans_set.add(span_id)
                all_span_scores[span_id] += res["span_scores"].get(span_id, 0.0)
            
            all_kg_entities.update(res.get("kg_entities", []))
            all_chains.extend(res.get("evidence_chains", []))

        # Re-rank combined spans
        sorted_spans = sorted(list(combined_spans_set), key=lambda x: all_span_scores[x], reverse=True)
        final_spans = sorted_spans[:k]
        
        # Dedup chains and take best
        all_chains.sort(key=lambda x: x["score"], reverse=True)
        seen_chain_texts = set()
        unique_chains = []
        for c in all_chains:
            if c["text"] not in seen_chain_texts:
                unique_chains.append(c)
                seen_chain_texts.add(c["text"])

        return {
            "final_spans": final_spans,
            "span_scores": dict(all_span_scores),
            "kg_entities": list(all_kg_entities),
            "evidence_chains": unique_chains[:3],
            "is_aggregated": True,
            "sub_questions_count": len(sub_results)
        }

    def _run_reasoning_core(self, query: str, k: int = 10) -> Dict[str, List]:
        """Core reasoning logic extracted from enhanced_reasoning."""
        query_tokens = set(tokenize(query))

        sentence_scores = self._score_sentences(query)
        # Step 1: Hybrid retrieval (BM25 + Semantic + Centrality)
        hybrid_results = self.enhanced_span_retrieval(query, k=k*2)
        
        # Step 2: Graph traversal from top seeds
        traversal_results = self.enhanced_span_traversal(query, depth=2, k=k*2)
        
        # Step 3: Query expansion
        expansion_results = self.retrieval_with_expansion(query, k=k*2)
        
        # Step 4: KG-guided retrieval
        kg_results = self.kg_guided_retrieval(query, k=k*2)

        # Merge all candidates
        all_candidates = set(hybrid_results) | set(traversal_results) | set(expansion_results) | set(kg_results)
        
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
            in_kg        = 1.0 if span_id in kg_results        else 0.0

            # KG bonus: if the span mentions entities found in the KG
            kg_bonus = 0.0
            if self.kg_graph:
                span_text_lower = text.lower()
                for ent in self.kg_graph.nodes():
                    if re.search(rf'\b{re.escape(ent.lower())}\b', span_text_lower):
                        kg_bonus += 0.05
                kg_bonus = min(kg_bonus, 0.15)

            # Combined score - purely model/signal-driven, no domain-specific rules
            final_score = (
                0.28 * sem_score +       # Primary: semantic similarity (LaBSE)
                0.18 * bm25_score +      # Lexical matching (BM25)
                0.12 * overlap_score +   # Token overlap
                0.10 * sentence_bonus +  # Sentence-level relevance
                0.08 * cent_score +      # Graph centrality
                0.06 * in_hybrid +       # Multi-source voting
                0.04 * in_traversal +
                0.03 * in_expansion +
                0.05 * in_kg +           # KG guided retrieval bonus
                0.06 * kg_bonus          # KG entity mention bonus
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

        # Step 7: MMR reranking - diversify
        reranked = self.mmr_rerank(query, reranked, k=min(k*2, len(reranked)), lambda_param=0.65)

        # Step 8: Multi-hop reasoning chains
        evidence_chains = self.build_evidence_chains(query, reranked[:5])

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
            span_semantic_score = final_score_map.get(span_id, 0.0)
            
            sentence_id = self.span_graph.nodes[span_id].get("sentence_id")
            sentence_score = sentence_scores.get(sentence_id, 0.0)

            # Minimum semantic threshold
            if span_semantic_score < 0.12:
                return False

            if meaningful_overlap >= 2:
                return True

            if span_semantic_score >= 0.40:
                return True

            if meaningful_overlap >= 1 and span_semantic_score >= 0.30:
                return True

            if sentence_score >= 0.50:
                return True

            return meaningful_overlap >= 1

        filtered = [sid for sid in reranked if is_relevant(sid)]
        if len(filtered) < k:
            final_spans = (filtered + [sid for sid in reranked if sid not in filtered])[:k]
        else:
            final_spans = filtered[:k]
        
        # Apply diversity filtering
        diverse_spans = self.get_diverse_evidence(final_spans, max_spans=k)
        
        # Add spans from evidence chains to final_spans if they are highly relevant
        for chain in evidence_chains[:2]:
            for node_id in chain["nodes"]:
                if node_id not in diverse_spans and len(diverse_spans) < k + 2:
                    diverse_spans.append(node_id)

        # Build span scores mapping
        span_score_map = {sid: score for sid, score in final_scores}
        
        return {
            "final_spans": diverse_spans,
            "hybrid_results": hybrid_results[:5],
            "traversal_results": traversal_results[:5],
            "expansion_results": expansion_results[:5],
            "kg_results": kg_results[:5],
            "kg_entities": [ent for chain in evidence_chains for ent in chain.get("kg_entities", [])],
            "kg_spans": [sid for chain in evidence_chains for sid in chain.get("nodes", []) if sid in self.span_graph.nodes],
            "evidence_chains": evidence_chains,
            "span_scores": span_score_map
        }

    def build_evidence_chains(self, query: str, top_spans: List[int], max_chains: int = 3) -> List[Dict]:
        """
        Find and score multi-hop reasoning paths between top retrieved spans.
        Uses both DRG and KG if available.
        """
        if not top_spans or len(top_spans) < 2:
            return []

        chains = []
        # Convert span IDs to sentence IDs for DRG traversal
        seed_sentences = []
        for sid in top_spans:
            sent_id = self.span_graph.nodes[sid].get("sentence_id")
            if sent_id is not None and sent_id not in seed_sentences:
                seed_sentences.append(sent_id)

        # 1. DRG-based chains (structural & semantic paths)
        for i in range(min(3, len(seed_sentences))):
            for j in range(i + 1, min(4, len(seed_sentences))):
                start_node = seed_sentences[i]
                end_node = seed_sentences[j]
                
                try:
                    # Find shortest path in the directed DRG
                    path = nx.shortest_path(self.sentence_graph, start_node, end_node, weight='weight')
                    if 1 < len(path) <= 4:  # Reasonable chain length
                        chain_text = [self.sentence_graph.nodes[n]["text"] for n in path]
                        # Score the chain based on node importance and path length
                        score = sum(self.sentence_graph.nodes[n].get("pagerank", 0) for n in path) / len(path)
                        chains.append({
                            "type": "drg_path",
                            "nodes": path,
                            "text": " -> ".join(chain_text),
                            "score": float(score)
                        })
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        # 2. KG-based bridge chains
        if self.kg_graph:
            query_entities = []
            q_norm = normalize_text(query)
            for ent in self.kg_graph.nodes():
                if re.search(rf'\b{re.escape(ent.lower())}\b', q_norm):
                    query_entities.append(ent)
            
            for q_ent in query_entities:
                for s_id in seed_sentences[:3]:
                    s_entities = self.sentence_graph.nodes[s_id].get("kg_entities", [])
                    for s_ent in s_entities:
                        if q_ent == s_ent:
                            continue
                        try:
                            # Use KnowledgeGraph's shortest_path_evidence if available
                            # or just raw NetworkX on self.kg_graph
                            kg_path = nx.shortest_path(self.kg_graph, q_ent, s_ent)
                            if 1 < len(kg_path) <= 3:
                                chains.append({
                                    "type": "kg_bridge",
                                    "kg_entities": kg_path,
                                    "nodes": [s_id],
                                    "text": f"KG Bridge: {' -> '.join(kg_path)}",
                                    "score": 0.5 / len(kg_path)
                                })
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue

        chains.sort(key=lambda x: x["score"], reverse=True)
        return chains[:max_chains]
import re
