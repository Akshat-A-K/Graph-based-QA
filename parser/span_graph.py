"""
Span Graph Builder: Create graph structure over fine-grained text spans.
Combines structural, semantic, and discourse relations.
"""

import networkx as nx
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict


class SpanGraph:
    """Build and manage span-level document graph."""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print("Loading embedding model for span graph...")
        self.model = SentenceTransformer(model_name)
        self.graph = nx.Graph()
    
    def add_nodes(self, spans: List[Dict]):
        """Add span nodes to graph."""
        for span in spans:
            self.graph.add_node(
                span["span_id"],
                text=span["text"],
                span_type=span["span_type"],
                page=span["page"],
                section=span["section"],
                sentence_id=span["sentence_id"]
            )
    
    def compute_embeddings(self):
        """Compute embeddings for all spans."""
        print("Computing span embeddings...")
        texts = [self.graph.nodes[n]["text"] for n in self.graph.nodes]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for i, node_id in enumerate(self.graph.nodes):
            self.graph.nodes[node_id]["embedding"] = embeddings[i]
    
    def add_structural_edges(self):
        """Add edges based on document structure."""
        print("Adding structural edges to span graph...")
        
        nodes = list(self.graph.nodes)
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                d1 = self.graph.nodes[n1]
                d2 = self.graph.nodes[n2]
                
                # Same sentence
                if d1["sentence_id"] == d2["sentence_id"]:
                    self.graph.add_edge(n1, n2, type="same_sentence", weight=1.0)
                
                # Same section
                elif d1["section"] == d2["section"]:
                    self.graph.add_edge(n1, n2, type="same_section", weight=0.5)
                
                # Same page
                elif d1["page"] == d2["page"]:
                    self.graph.add_edge(n1, n2, type="same_page", weight=0.3)
                
                # Adjacent spans (within 2 positions)
                if abs(n1 - n2) <= 2:
                    self.graph.add_edge(n1, n2, type="adjacent", weight=0.8)
    
    def add_semantic_edges(self, threshold=0.7):
        """Add edges based on semantic similarity."""
        print("Adding semantic edges to span graph...")
        
        node_ids = list(self.graph.nodes)
        embeddings = [self.graph.nodes[n]["embedding"] for n in node_ids]
        
        sim_matrix = cosine_similarity(embeddings)
        
        for i in tqdm(range(len(node_ids))):
            for j in range(i + 1, len(node_ids)):
                sim = sim_matrix[i][j]
                
                if sim >= threshold:
                    # Don't duplicate if already connected
                    if not self.graph.has_edge(node_ids[i], node_ids[j]):
                        self.graph.add_edge(
                            node_ids[i],
                            node_ids[j],
                            type="semantic",
                            weight=float(sim)
                        )
    
    def add_discourse_edges(self):
        """Add discourse relation edges (condition, exception, temporal)."""
        print("Adding discourse edges...")
        
        # Discourse markers
        condition_markers = ["if", "unless", "provided", "only if"]
        exception_markers = ["except", "excluding", "but not", "other than"]
        temporal_markers = ["before", "after", "until", "by", "deadline"]
        negation_markers = ["not", "no", "never", "cannot"]
        
        nodes = list(self.graph.nodes)
        
        for node_id in nodes:
            text = self.graph.nodes[node_id]["text"].lower()
            
            # Find spans with discourse markers
            has_condition = any(marker in text for marker in condition_markers)
            has_exception = any(marker in text for marker in exception_markers)
            has_temporal = any(marker in text for marker in temporal_markers)
            has_negation = any(marker in text for marker in negation_markers)
            
            # Mark node properties
            self.graph.nodes[node_id]["is_condition"] = has_condition
            self.graph.nodes[node_id]["is_exception"] = has_exception
            self.graph.nodes[node_id]["is_temporal"] = has_temporal
            self.graph.nodes[node_id]["is_negation"] = has_negation
            
            # Connect condition/exception spans to nearby spans
            if has_condition or has_exception:
                sentence_id = self.graph.nodes[node_id]["sentence_id"]
                
                # Find other spans in same sentence
                for other_id in nodes:
                    if other_id != node_id:
                        if self.graph.nodes[other_id]["sentence_id"] == sentence_id:
                            edge_type = "condition" if has_condition else "exception"
                            self.graph.add_edge(
                                node_id, 
                                other_id, 
                                type=edge_type,
                                weight=0.9
                            )
    
    def build_graph(self, spans: List[Dict]):
        """Build complete span graph."""
        self.add_nodes(spans)
        self.compute_embeddings()
        self.add_structural_edges()
        self.add_semantic_edges()
        self.add_discourse_edges()
        
        print("Span graph built.")
        print("Nodes:", self.graph.number_of_nodes())
        print("Edges:", self.graph.number_of_edges())
        
        return self.graph


def build_span_graph(spans: List[Dict]) -> nx.Graph:
    """
    Main entry point: build span graph from spans.
    
    Args:
        spans: List of span nodes from span_extractor.build_span_nodes()
    
    Returns:
        NetworkX graph with span-level nodes and edges
    """
    sg = SpanGraph()
    return sg.build_graph(spans)
