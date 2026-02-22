"""
Enhanced Span Graph Builder
Advanced discourse relations, coreference, semantic clustering, and visualization export
"""

import json
import networkx as nx
import numpy as np
from tqdm import tqdm
from .model_cache import get_sentence_transformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class SpanGraph:
    """Build and manage enhanced span-level document graph."""

    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"):
        print("🔗 Initializing Enhanced Span Graph Builder...")
        self.model = get_sentence_transformer(model_name)
        self.graph = nx.DiGraph()  # Directed for better reasoning
        
        # Enhanced discourse markers
        self.discourse_markers = {
            "condition": ["if", "unless", "provided", "only if", "in case", "when"],
            "exception": ["except", "excluding", "but not", "other than", "apart from"],
            "temporal": ["before", "after", "until", "by", "deadline", "during", "while"],
            "negation": ["not", "no", "never", "cannot", "won't", "shouldn't"],
            "requirement": ["must", "shall", "required", "need to", "have to", "should"],
            "causation": ["because", "since", "therefore", "thus", "hence", "so"],
            "contrast": ["but", "however", "although", "while", "whereas", "yet"],
            "addition": ["and", "also", "moreover", "furthermore", "additionally"],
        }
    
    def add_nodes(self, spans: List[Dict]):
        """Add span nodes to graph."""
        for span in spans:
            self.graph.add_node(
                span["span_id"],
                text=span["text"],
                span_type=span["span_type"],
                page=span["page"],
                section=span["section"],
                sentence_id=span["sentence_id"],
                entities=span.get("entities", [])
            )
    
    def compute_embeddings(self):
        """Compute embeddings and importance scores for all spans."""
        print("📊 Computing span embeddings and importance...")
        texts = [self.graph.nodes[n]["text"] for n in self.graph.nodes]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Compute span importance based on length and position
        for i, node_id in enumerate(self.graph.nodes):
            self.graph.nodes[node_id]["embedding"] = embeddings[i]
            # Importance = word count (longer spans more important)
            self.graph.nodes[node_id]["importance"] = len(texts[i].split())
            self.graph.nodes[node_id]["length"] = len(texts[i])
    
    def add_structural_edges(self):
        """Add enhanced structural edges based on document structure."""
        print("🔗 Adding structural edges...")
        
        nodes = list(self.graph.nodes)
        
        for i in range(len(nodes)):
            n1 = nodes[i]
            d1 = self.graph.nodes[n1]
            
            for j in range(i + 1, len(nodes)):
                n2 = nodes[j]
                d2 = self.graph.nodes[n2]
                
                # Same sentence (bidirectional, high weight)
                if d1["sentence_id"] == d2["sentence_id"]:
                    self.graph.add_edge(n1, n2, type="same_sentence", weight=1.0)
                    self.graph.add_edge(n2, n1, type="same_sentence", weight=1.0)
                
                # Sequential spans (directed)
                elif n2 - n1 == 1:
                    self.graph.add_edge(n1, n2, type="sequential", weight=0.9)
                    self.graph.add_edge(n2, n1, type="backward", weight=0.7)
                
                # Same section (bidirectional)
                elif d1["section"] == d2["section"] and d1["section"] != "GLOBAL":
                    self.graph.add_edge(n1, n2, type="same_section", weight=0.5)
                    self.graph.add_edge(n2, n1, type="same_section", weight=0.5)
                
                # Same page
                elif d1["page"] == d2["page"]:
                    self.graph.add_edge(n1, n2, type="same_page", weight=0.3)
                    self.graph.add_edge(n2, n1, type="same_page", weight=0.3)
                
                # Proximity (within 5 spans)
                if 1 < abs(n1 - n2) <= 5:
                    proximity_weight = 1.0 / abs(n1 - n2)
                    self.graph.add_edge(n1, n2, type="proximity", weight=proximity_weight)
                    self.graph.add_edge(n2, n1, type="proximity", weight=proximity_weight)
    
    def add_semantic_edges(self, threshold=0.7):
        """Add semantic similarity edges with adaptive thresholding."""
        print("🧠 Adding semantic edges...")
        
        node_ids = list(self.graph.nodes)
        embeddings = [self.graph.nodes[n]["embedding"] for n in node_ids]
        
        sim_matrix = cosine_similarity(embeddings)
        
        # Adaptive threshold based on distribution
        all_sims = [sim_matrix[i][j] for i in range(len(node_ids)) for j in range(i + 1, len(node_ids))]
        if all_sims:
            dynamic_threshold = max(threshold, np.percentile(all_sims, 80))
        else:
            dynamic_threshold = threshold
        
        edge_count = 0
        for i in tqdm(range(len(node_ids)), desc="Semantic edges"):
            for j in range(i + 1, len(node_ids)):
                sim = sim_matrix[i][j]
                
                if sim >= dynamic_threshold:
                    # Don't duplicate if already connected
                    if not self.graph.has_edge(node_ids[i], node_ids[j]):
                        self.graph.add_edge(
                            node_ids[i],
                            node_ids[j],
                            type="semantic",
                            weight=float(sim)
                        )
                        self.graph.add_edge(
                            node_ids[j],
                            node_ids[i],
                            type="semantic",
                            weight=float(sim)
                        )
                        edge_count += 1
        
        print(f"  ✓ Added {edge_count} semantic edge pairs")
    
    def add_discourse_edges(self):
        """Add enhanced discourse relation edges."""
        print("💬 Adding discourse edges...")
        
        nodes = list(self.graph.nodes)
        edge_count = 0
        
        for node_id in nodes:
            text = self.graph.nodes[node_id]["text"].lower()
            
            # Detect discourse types
            discourse_types = []
            for discourse_type, markers in self.discourse_markers.items():
                if any(marker in text for marker in markers):
                    discourse_types.append(discourse_type)
                    self.graph.nodes[node_id][f"is_{discourse_type}"] = True
            
            # Store all discourse types for this span
            self.graph.nodes[node_id]["discourse_types"] = discourse_types
            
            # Connect discourse spans to relevant nearby spans
            if discourse_types:
                sentence_id = self.graph.nodes[node_id]["sentence_id"]
                
                # Find spans in same or nearby sentences
                for other_id in nodes:
                    if other_id != node_id:
                        other_sent = self.graph.nodes[other_id]["sentence_id"]
                        
                        # Same sentence or adjacent sentences
                        if abs(other_sent - sentence_id) <= 1:
                            for dtype in discourse_types:
                                self.graph.add_edge(
                                    node_id,
                                    other_id,
                                    type=dtype,
                                    weight=0.85
                                )
                                edge_count += 1
        
        print(f"  ✓ Added {edge_count} discourse edges")

    def add_entity_overlap_edges(self):
        """Connect spans that share extracted entities."""
        print("🔗 Adding entity overlap edges...")
        entity_map = defaultdict(list)

        for node_id in self.graph.nodes:
            entities = self.graph.nodes[node_id].get("entities", [])
            for ent in entities:
                key = str(ent).strip().lower()
                if key:
                    entity_map[key].append(node_id)

        edge_count = 0
        for ent, node_list in entity_map.items():
            if len(node_list) < 2:
                continue
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    n1, n2 = node_list[i], node_list[j]
                    self.graph.add_edge(n1, n2, type="entity", weight=0.8, entity=ent)
                    self.graph.add_edge(n2, n1, type="entity", weight=0.8, entity=ent)
                    edge_count += 1

        print(f"  ✓ Added {edge_count} entity edge pairs")
    
    def compute_graph_metrics(self):
        """Compute centrality and importance metrics for spans."""
        print("📈 Computing graph metrics...")
        
        # PageRank for global importance
        try:
            pagerank = nx.pagerank(self.graph, weight='weight')
            for node_id in self.graph.nodes:
                self.graph.nodes[node_id]['pagerank'] = pagerank.get(node_id, 0)
        except:
            pass
        
        # Degree centrality
        for node_id in self.graph.nodes:
            self.graph.nodes[node_id]['degree'] = self.graph.degree(node_id)
    
    def export_graph_json(self, output_path: str):
        """Export span graph to JSON for visualization."""
        data = {
            "nodes": [
                {
                    "id": n,
                    "text": self.graph.nodes[n].get('text', '')[:100],  # Truncate for viz
                    "type": self.graph.nodes[n].get('span_type', ''),
                    "page": self.graph.nodes[n].get('page', 0),
                    "importance": self.graph.nodes[n].get('importance', 0),
                    "discourse": self.graph.nodes[n].get('discourse_types', []),
                    "pagerank": self.graph.nodes[n].get('pagerank', 0),
                }
                for n in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "type": self.graph.edges[u, v].get('type', 'unknown'),
                    "weight": self.graph.edges[u, v].get('weight', 0.5),
                }
                for u, v in self.graph.edges()
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Span graph exported to {output_path}")
    
    def export_graph_graphml(self, output_path: str):
        """Export span graph to GraphML for Gephi/Cytoscape."""
        # Create a copy without numpy arrays and lists for GraphML export
        export_graph = self.graph.copy()
        for node in export_graph.nodes():
            # Remove or convert non-GraphML compatible attributes
            if 'embedding' in export_graph.nodes[node]:
                del export_graph.nodes[node]['embedding']
            if 'discourse_types' in export_graph.nodes[node]:
                # Convert list to string
                discourse_list = export_graph.nodes[node]['discourse_types']
                export_graph.nodes[node]['discourse_types'] = ','.join(discourse_list) if discourse_list else ''
        
        nx.write_graphml(export_graph, output_path)
        print(f"✅ Span graph exported to {output_path}")
    
    def export_graph_image(self, output_path: str, figsize=(14, 12)):
        """Export span graph as image visualization."""
        plt.figure(figsize=figsize)
        
        # Use spring layout with fewer iterations for speed
        pos = nx.spring_layout(self.graph, k=1.5, iterations=20)
        
        # Color nodes by discourse type
        node_colors = []
        for n in self.graph.nodes():
            discourse = self.graph.nodes[n].get('discourse_types', [])
            if 'requirement' in discourse:
                node_colors.append('red')
            elif 'condition' in discourse:
                node_colors.append('orange')
            elif 'temporal' in discourse:
                node_colors.append('green')
            else:
                node_colors.append('lightblue')
        
        # Draw nodes
        node_sizes = [self.graph.nodes[n].get('importance', 5) * 50 for n in self.graph.nodes()]
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.7)
        
        # Draw edges (no arrows for large graphs)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.15, arrows=False, width=0.3)
        
        # Draw labels
        labels = {n: f"{n}" for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=7)
        
        plt.title("Span Graph (Discourse-Aware)", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Span graph image saved to {output_path}")
    
    def build_graph(self, spans: List[Dict]):
        """Build complete enhanced span graph."""
        self.add_nodes(spans)
        self.compute_embeddings()
        self.add_structural_edges()
        self.add_semantic_edges()
        self.add_discourse_edges()
        self.add_entity_overlap_edges()
        self.compute_graph_metrics()
        
        print("\n✅ Enhanced Span Graph built successfully!")
        print(f"   🔤 Nodes: {self.graph.number_of_nodes()}")
        print(f"   🔗 Edges: {self.graph.number_of_edges()}")
        
        # Edge type breakdown
        edge_types = Counter([data.get('type', 'unknown') for _, _, data in self.graph.edges(data=True)])
        print("\n   Edge types:")
        for edge_type, count in edge_types.most_common():
            print(f"      • {edge_type}: {count}")
        
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
