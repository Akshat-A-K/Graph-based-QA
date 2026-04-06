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
        print("Initializing Enhanced Span Graph Builder...")
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

    def add_entity_nodes(self):
        """Create entity nodes and connect spans to them."""
        entity_to_node = {}

        for node_id in list(self.graph.nodes):
            node_data = self.graph.nodes[node_id]
            if node_data.get("span_type") == "entity_node":
                continue

            entities = node_data.get("entities", [])
            for ent in entities:
                ent = str(ent).lower().strip()

                # remove invalid entities
                if not ent or len(ent) < 2:
                    continue

                if ent not in entity_to_node:
                    ent_id = f"entity::{ent}"
                    self.graph.add_node(
                        ent_id,
                        text=ent,
                        span_type="entity_node",
                        page=-1,
                        section="GLOBAL",
                        sentence_id=-1,
                        entities=[]
                    )
                    entity_to_node[ent] = ent_id

                ent_id = entity_to_node[ent]

                # span -> entity
                if not self.graph.has_edge(node_id, ent_id):
                    self.graph.add_edge(
                        node_id,
                        ent_id,
                        type="mentions",
                        weight=1.0
                    )

                # entity -> span
                if not self.graph.has_edge(ent_id, node_id):
                    self.graph.add_edge(
                        ent_id,
                        node_id,
                        type="mentioned_in",
                        weight=1.0
                    )
    def add_entity_entity_edges(self):
        """
        Connect entity nodes that appear in the same span.
        This creates entity-entity relationships (GraphRAG style).
        """

        print("Adding entity -> entity relations...")

        edge_count = 0

        # iterate over span nodes only
        for node_id, data in self.graph.nodes(data=True):

            if data.get("span_type") == "entity_node":
                continue

            entities = data.get("entities", [])

            if len(entities) < 2:
                continue

            # limit entity connections per span to at most 2 entities
            max_relations = min(2, len(entities))

            for i in range(max_relations):
                for j in range(i + 1, max_relations):
                    e1 = f"entity::{entities[i].lower().strip()}"
                    e2 = f"entity::{entities[j].lower().strip()}"

                    if self.graph.has_node(e1) and self.graph.has_node(e2):
                        if not self.graph.has_edge(e1, e2):
                            self.graph.add_edge(e1, e2, type="entity_relation", weight=0.9)
                            edge_count += 1

                        if not self.graph.has_edge(e2, e1):
                            self.graph.add_edge(e2, e1, type="entity_relation", weight=0.9)

        print(f"  Added {edge_count} entity-entity edges")
    def compute_embeddings(self):
        """Compute embeddings and importance scores for all spans."""
        print("Computing span embeddings and importance...")
        
        if self.model is None:
            print("WARNING: Span model is None. Using zero embeddings.")
            for node_id in self.graph.nodes:
                self.graph.nodes[node_id]["embedding"] = np.zeros(384)
                self.graph.nodes[node_id]["importance"] = 1.0
                self.graph.nodes[node_id]["length"] = len(self.graph.nodes[node_id].get("text", ""))
            return

        texts = [self.graph.nodes[n]["text"] for n in self.graph.nodes]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Compute span importance based on length and position
        for i, node_id in enumerate(self.graph.nodes):
            self.graph.nodes[node_id]["embedding"] = embeddings[i]
            # Importance = word count (longer spans more important)
            self.graph.nodes[node_id]["importance"] = len(texts[i].split())
            self.graph.nodes[node_id]["length"] = len(texts[i])
    
    def add_structural_edges(self):
        """Add strong structural edges only (for QA reasoning)."""

        print("Adding structural edges...")

        nodes = [
            n for n in self.graph.nodes
            if self.graph.nodes[n].get("span_type") != "entity_node"
        ]

        edge_count = 0

        for i in range(len(nodes)):
            n1 = nodes[i]
            d1 = self.graph.nodes[n1]

            for j in range(i + 1, len(nodes)):
                n2 = nodes[j]
                d2 = self.graph.nodes[n2]

                # same sentence (very strong)
                if d1["sentence_id"] == d2["sentence_id"]:
                    self.graph.add_edge(n1, n2, type="same_sentence", weight=1.0)
                    self.graph.add_edge(n2, n1, type="same_sentence", weight=1.0)
                    edge_count += 2

                # sequential spans (sentence order)
                elif isinstance(n1, int) and isinstance(n2, int) and (n2 - n1 == 1):
                    self.graph.add_edge(n1, n2, type="sequential", weight=0.9)
                    self.graph.add_edge(n2, n1, type="backward", weight=0.7)
                    edge_count += 2

        print(f"  Added {edge_count} structural edges")
    
    def add_semantic_edges(self, threshold=0.75, max_neighbors=3):
        """Add semantic similarity edges with adaptive thresholding and neighbor cap.

        Keeps only the top `max_neighbors` neighbors per node (above threshold)
        to avoid near-complete connectivity.
        """
        print("Adding semantic edges...")

        node_ids = list(self.graph.nodes)
        embeddings = [self.graph.nodes[n]["embedding"] for n in node_ids]

        sim_matrix = cosine_similarity(embeddings)

        # Adaptive threshold based on distribution (higher percentile reduces noise)
        all_sims = [sim_matrix[i][j] for i in range(len(node_ids)) for j in range(i + 1, len(node_ids))]
        if all_sims:
            dynamic_threshold = max(threshold, np.percentile(all_sims, 80))
        else:
            dynamic_threshold = threshold

        edge_count = 0
        # For each node, select top neighbors above dynamic_threshold up to max_neighbors
        for i in tqdm(range(len(node_ids)), desc="Semantic edges"):
            sims = []
            for j in range(len(node_ids)):
                if i == j:
                    continue
                sims.append((j, float(sim_matrix[i][j])))

            # Filter by threshold then sort
            filtered = [p for p in sims if p[1] >= dynamic_threshold]
            filtered.sort(key=lambda x: x[1], reverse=True)
            # Cap neighbors
            topk = filtered[:max_neighbors]

            for j, sim in topk:
                src = node_ids[i]
                tgt = node_ids[j]
                if not self.graph.has_edge(src, tgt):
                    self.graph.add_edge(src, tgt, type="semantic", weight=sim)
                    edge_count += 1
                # also add reverse if missing (keeps reasoning symmetric)
                if not self.graph.has_edge(tgt, src):
                    self.graph.add_edge(tgt, src, type="semantic", weight=sim)

        print(f"  Added {edge_count} semantic edges")
    
    def add_discourse_edges(self):
        """Add enhanced discourse relation edges."""
        print("Adding discourse edges...")
        
        nodes = [n for n in self.graph.nodes if self.graph.nodes[n].get("span_type") != "entity_node"]
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
        
        print(f"  Added {edge_count} discourse edges")

    def add_entity_overlap_edges(self):
        """Connect spans that share extracted entities."""
        print("Adding entity overlap edges...")
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

        print(f"  Added {edge_count} entity edge pairs")
    
    def compute_graph_metrics(self):
        """Compute centrality and importance metrics for spans."""
        print("Computing graph metrics...")
        
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
        
        print(f"Span graph exported to {output_path}")
    
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
            if 'entities' in export_graph.nodes[node]:
                entity_list = export_graph.nodes[node]['entities']
                export_graph.nodes[node]['entities'] = ','.join(str(e) for e in entity_list) if entity_list else ''
        
        nx.write_graphml(export_graph, output_path)
        print(f"Span graph exported to {output_path}")
    
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
        print(f"Span graph image saved to {output_path}")
    
    def build_graph(self, spans: List[Dict]):
        """Build complete enhanced span graph."""
        self.add_nodes(spans)
        self.add_entity_nodes()
        self.add_entity_entity_edges() 
        self.compute_embeddings()
        self.add_structural_edges()
        self.add_semantic_edges()
        self.add_discourse_edges()
        self.add_entity_overlap_edges()
        self.compute_graph_metrics()
        
        print("\nEnhanced Span Graph built successfully.")
        print(f"   Nodes: {self.graph.number_of_nodes()}")
        print(f"   Edges: {self.graph.number_of_edges()}")
        
        # Edge type breakdown
        edge_types = Counter([data.get('type', 'unknown') for _, _, data in self.graph.edges(data=True)])
        print("\n   Edge types:")
        for edge_type, count in edge_types.most_common():
            print(f"      - {edge_type}: {count}")
        
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
