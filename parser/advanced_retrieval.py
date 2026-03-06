"""
Advanced retrieval techniques without GNN:
- BM25 for lexical matching
- Cross-encoder re-ranking
- Graph centrality
- Hybrid scoring
"""

import math
import re
import unicodedata
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from collections import Counter


def normalize_text(text: str) -> str:
    """Normalize text for retrieval across scripts and punctuation."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u0964", ".").replace("\u0965", ".")
    normalized = normalized.casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text using Unicode-aware word boundaries."""
    normalized = normalize_text(text)
    return re.findall(r"\w+", normalized, flags=re.UNICODE)


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
        tokenized = [tokenize(doc) for doc in documents]
        self.doc_len = [len(tokens) for tokens in tokenized]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        # Calculate document frequencies
        df = {}
        for tokens in tokenized:
            words = set(tokens)
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
        doc_words = tokenize(doc)
        doc_len = self.doc_len[doc_id]
        
        query_words = tokenize(query)
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
        pass
    
    def expand(self, query: str) -> List[str]:
        """Generate query variations."""
        return [query]
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        words = tokenize(query)
        keywords = [w for w in words if len(w) > 2]
        
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
            "condition": ["if", "unless", "provided", "only if", "in case", "when"],
            "exception": ["except", "excluding", "but not", "other than", "apart from"],
            "temporal": ["before", "after", "until", "by", "deadline", "during", "while"],
            "negation": ["not", "no", "never", "cannot", "won't", "shouldn't"],
            "requirement": ["must", "shall", "required", "need to", "have to", "should"],
            "causation": ["because", "since", "therefore", "thus", "hence", "so"],
            "contrast": ["but", "however", "although", "while", "whereas", "yet"],
            "addition": ["and", "also", "moreover", "furthermore", "additionally"],
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
    