"""
Advanced retrieval techniques without GNN:
- BM25 for lexical matching
- Cross-encoder re-ranking
- Graph centrality
- Hybrid scoring
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from collections import Counter
import math


class BM25Retriever:
    """BM25 algorithm for lexical matching (complements embeddings)."""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.docs = []
    
    def fit(self, documents: List[str]):
        """Build BM25 index."""
        self.docs = documents
        self.doc_len = [len(doc.split()) for doc in documents]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        # Calculate document frequencies
        df = {}
        for doc in documents:
            words = set(doc.lower().split())
            for word in words:
                df[word] = df.get(word, 0) + 1
        
        # Calculate IDF
        num_docs = len(documents)
        for word, freq in df.items():
            self.idf[word] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)
    
    def score(self, query: str, doc_id: int) -> float:
        """Calculate BM25 score for a query-document pair."""
        score = 0.0
        doc = self.docs[doc_id]
        doc_words = doc.lower().split()
        doc_len = self.doc_len[doc_id]
        
        query_words = query.lower().split()
        word_freqs = Counter(doc_words)
        
        for word in query_words:
            if word not in word_freqs:
                continue
            
            freq = word_freqs[word]
            idf = self.idf.get(word, 0)
            
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Retrieve top-k documents by BM25 score."""
        scores = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class GraphCentrality:
    """Calculate node importance using graph algorithms."""
    
    @staticmethod
    def pagerank(graph: nx.Graph, alpha=0.85) -> Dict[int, float]:
        """PageRank for node importance."""
        return nx.pagerank(graph, alpha=alpha)
    
    @staticmethod
    def betweenness_centrality(graph: nx.Graph) -> Dict[int, float]:
        """Betweenness centrality (nodes on shortest paths)."""
        return nx.betweenness_centrality(graph)
    
    @staticmethod
    def degree_centrality(graph: nx.Graph) -> Dict[int, float]:
        """Degree centrality (number of connections)."""
        return nx.degree_centrality(graph)
    
    @staticmethod
    def closeness_centrality(graph: nx.Graph) -> Dict[int, float]:
        """Closeness centrality (average distance to all nodes)."""
        try:
            return nx.closeness_centrality(graph)
        except:
            # For disconnected graphs
            return {node: 0.0 for node in graph.nodes()}
    
    @staticmethod
    def eigenvector_centrality(graph: nx.Graph, max_iter=100) -> Dict[int, float]:
        """Eigenvector centrality (importance based on neighbors)."""
        try:
            return nx.eigenvector_centrality(graph, max_iter=max_iter)
        except:
            # Fallback to PageRank if convergence fails
            return GraphCentrality.pagerank(graph)


class HybridScorer:
    """Combine multiple signals for better ranking."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize with signal weights.
        
        Args:
            weights: Dict like {'semantic': 0.4, 'lexical': 0.3, 'centrality': 0.3}
        """
        self.weights = weights or {
            'semantic': 0.5,
            'lexical': 0.3,
            'centrality': 0.2
        }
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalization."""
        if not scores or max(scores) == min(scores):
            return [0.0] * len(scores)
        
        min_score = min(scores)
        max_score = max(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def combine_scores(
        self,
        semantic_scores: List[float],
        lexical_scores: List[float],
        centrality_scores: List[float]
    ) -> List[float]:
        """Weighted combination of scores."""
        # Normalize each signal
        sem_norm = self.normalize_scores(semantic_scores)
        lex_norm = self.normalize_scores(lexical_scores)
        cent_norm = self.normalize_scores(centrality_scores)
        
        # Weighted sum
        combined = []
        for sem, lex, cent in zip(sem_norm, lex_norm, cent_norm):
            score = (
                self.weights['semantic'] * sem +
                self.weights['lexical'] * lex +
                self.weights['centrality'] * cent
            )
            combined.append(score)
        
        return combined


class QueryExpander:
    """Expand queries with synonyms and related terms."""
    
    def __init__(self):
        # Domain-specific synonym dictionary
        self.synonyms = {
            'deadline': ['due date', 'submission date', 'last date', 'final date'],
            'extension': ['extra time', 'additional time', 'more time'],
            'submit': ['upload', 'turn in', 'hand in', 'provide'],
            'required': ['mandatory', 'must', 'necessary', 'needed'],
            'allowed': ['permitted', 'acceptable', 'can', 'may'],
            'prohibited': ['not allowed', 'forbidden', 'cannot', 'banned'],
        }
    
    def expand(self, query: str) -> List[str]:
        """Generate query variations."""
        variations = [query]
        query_lower = query.lower()
        
        for word, syns in self.synonyms.items():
            if word in query_lower:
                for syn in syns:
                    expanded = query_lower.replace(word, syn)
                    variations.append(expanded)
        
        return variations
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Simple keyword extraction (remove common words)
        stopwords = {'the', 'is', 'are', 'was', 'a', 'an', 'what', 'when', 'where', 
                     'who', 'how', 'can', 'will', 'be', 'for', 'to', 'of', 'in'}
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords


class EdgeWeighting:
    """Improved edge weight calculation."""
    
    @staticmethod
    def semantic_weight(similarity: float, threshold: float = 0.5) -> float:
        """
        Convert similarity to edge weight with smooth decay.
        Only strong similarities get high weights.
        """
        if similarity < threshold:
            return 0.0
        
        # Exponential scaling for high similarities
        normalized = (similarity - threshold) / (1.0 - threshold)
        return normalized ** 2
    
    @staticmethod
    def structural_weight(node1_attrs: Dict, node2_attrs: Dict) -> float:
        """
        Calculate structural similarity between nodes.
        Same sentence > same section > same page.
        """
        weight = 0.0
        
        # Same sentence
        if node1_attrs.get('sentence_id') == node2_attrs.get('sentence_id'):
            weight += 1.0
        
        # Same section
        elif node1_attrs.get('section') == node2_attrs.get('section'):
            weight += 0.5
        
        # Same page
        elif node1_attrs.get('page') == node2_attrs.get('page'):
            weight += 0.3
        
        # Adjacent (for spans)
        if 'span_id' in node1_attrs:
            dist = abs(node1_attrs['span_id'] - node2_attrs['span_id'])
            if dist == 1:
                weight += 0.8
            elif dist == 2:
                weight += 0.4
        
        return min(weight, 1.0)
    
    @staticmethod
    def discourse_weight(node1_text: str, node2_text: str) -> float:
        """
        Higher weight for discourse-related spans.
        """
        text1_lower = node1_text.lower()
        text2_lower = node2_text.lower()
        
        discourse_markers = {
            'condition': ['if', 'unless', 'provided', 'only if'],
            'exception': ['except', 'excluding', 'but not', 'other than'],
            'temporal': ['before', 'after', 'until', 'by', 'deadline'],
            'causal': ['because', 'since', 'therefore', 'thus', 'hence'],
        }
        
        weight = 0.0
        
        for marker_type, markers in discourse_markers.items():
            has_marker_1 = any(m in text1_lower for m in markers)
            has_marker_2 = any(m in text2_lower for m in markers)
            
            if has_marker_1 or has_marker_2:
                weight += 0.3
            
            if has_marker_1 and has_marker_2:
                weight += 0.4
        
        return min(weight, 1.0)
