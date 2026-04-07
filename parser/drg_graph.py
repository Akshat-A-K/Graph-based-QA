import networkx as nx
import numpy as np
from tqdm import tqdm
from .model_cache import get_sentence_transformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import re
import matplotlib.pyplot as plt
from .config import BATCH_EMBED_MODEL
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class DocumentReasoningGraph:
    def __init__(
        self,
        # Use an English-focused embedding model by default
        model_name=BATCH_EMBED_MODEL,
        enable_model_ner: bool = False
    ):
        print("Loading DRG model...")
        self.model = get_sentence_transformer(model_name)
        self.graph = nx.DiGraph()  # Use directed graph for better reasoning
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.entity_map = defaultdict(list)  # Maps entities to node IDs
        self.ner_pipeline = None
        self.use_ner = enable_model_ner

    def _get_ner_pipeline(self):
        if not self.use_ner or self.ner_pipeline is not None:
            return self.ner_pipeline

        try:
            from .model_cache import get_ner_pipeline
            # Use an English NER model
            self.ner_pipeline = get_ner_pipeline("dslim/bert-base-NER")
        except Exception:
            self.ner_pipeline = None
            self.use_ner = False

        return self.ner_pipeline

    # Add sentence nodes with basic metadata.
    def add_nodes(self, nodes):
        for node in nodes:
            self.graph.add_node(
                node["node_id"],
                text=node["text"],
                page=node["page"],
                section=node["section"],
                sent_index=node["sent_index"]
            )

    # Compute embeddings and lightweight node importance scores.
    def compute_embeddings(self):
        print("Computing embeddings and importance scores...")

        if self.model is None:
            print("WARNING: DRG model is None. Using zero embeddings.")
            for node_id in self.graph.nodes:
                self.graph.nodes[node_id]["embedding"] = np.zeros(384)
                self.graph.nodes[node_id]["importance"] = 1.0
                self.graph.nodes[node_id]["length"] = 0
            return

        texts = [self.graph.nodes[n]["text"] for n in self.graph.nodes]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Compute TF-IDF for importance scoring
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            tfidf_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
        except Exception:
            # Fallback if TF-IDF fails
            tfidf_scores = np.ones(len(texts))

        for i, node_id in enumerate(self.graph.nodes):
            self.graph.nodes[node_id]["embedding"] = embeddings[i]
            self.graph.nodes[node_id]["importance"] = float(tfidf_scores[i])
            self.graph.nodes[node_id]["length"] = len(texts[i].split())
            
        # Extract entities for linking
        self._extract_entities()

    # Add structural edges such as adjacency, section, and proximity links.
    def add_structural_edges(self):
        print("Adding enhanced structural edges...")

        nodes = list(self.graph.nodes)

        for i in range(len(nodes)):
            n1 = nodes[i]
            d1 = self.graph.nodes[n1]
            
            for j in range(i + 1, len(nodes)):
                n2 = nodes[j]
                d2 = self.graph.nodes[n2]

                # Adjacent sentences (directed - forward flow)
                if (d1["page"] == d2["page"] and 
                    d2["sent_index"] - d1["sent_index"] == 1):
                    self.graph.add_edge(n1, n2, type="adjacent", weight=1.0)
                    # Backward edge for context
                    self.graph.add_edge(n2, n1, type="context", weight=0.8)
                
                # Same page (undirected)
                elif d1["page"] == d2["page"]:
                    self.graph.add_edge(n1, n2, type="page", weight=0.5)
                    self.graph.add_edge(n2, n1, type="page", weight=0.5)

                # Same section (undirected)
                if d1["section"] == d2["section"] and d1["section"] != "GLOBAL":
                    self.graph.add_edge(n1, n2, type="section", weight=0.6)
                    self.graph.add_edge(n2, n1, type="section", weight=0.6)
                
                # Near neighbors (within 3 sentences)
                if (d1["page"] == d2["page"] and 
                    1 < abs(d1["sent_index"] - d2["sent_index"]) <= 3):
                    proximity_weight = 1.0 / abs(d1["sent_index"] - d2["sent_index"]) 
                    self.graph.add_edge(n1, n2, type="proximity", weight=proximity_weight)
                    self.graph.add_edge(n2, n1, type="proximity", weight=proximity_weight)

    # Add semantic edges and then connect entity-overlap nodes.
    def add_semantic_edges(self, threshold=0.75):
        print("Adding semantic and entity-based edges...")

        node_ids = list(self.graph.nodes)
        embeddings = [self.graph.nodes[n]["embedding"] for n in node_ids]

        sim_matrix = cosine_similarity(embeddings)
        
        # Use dynamic threshold based on distribution
        all_sims = []
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                all_sims.append(sim_matrix[i][j])
        
        # Adaptive threshold: top percentile
        if all_sims:
            dynamic_threshold = max(threshold, np.percentile(all_sims, 85))
        else:
            dynamic_threshold = threshold

        edge_count = 0
        for i in tqdm(range(len(node_ids)), desc="Semantic edges"):
            for j in range(i + 1, len(node_ids)):
                sim = sim_matrix[i][j]

                if sim >= dynamic_threshold:
                    # Bidirectional semantic edges
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
        
        print(f"Added {edge_count} semantic edge pairs")
        
        # Add entity-based coreference edges
        self._add_entity_edges()

    # Extract named entities and key terms for coreference-style linking.
    def _extract_entities(self):
        """Extract named entities and key terms from sentences"""
        print("Extracting entities...")
        
        # Simple pattern-based entity extraction (fallback)
        patterns = [
            (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', 'PERSON/ORG'),  # Proper nouns
            (r'\b\d{4}\b', 'YEAR'),  # Years
            (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'DATE'),  # Dates
            (r'\b\d+%\b', 'PERCENTAGE'),  # Percentages
            (r'\b\d+\s*(?:marks?|points?|credits?)\b', 'SCORE'),  # Scores
        ]
        
        for node_id in self.graph.nodes:
            text = self.graph.nodes[node_id]['text']
            entities = []

            if self.use_ner and self._get_ner_pipeline() is not None:
                try:
                    ner_results = self.ner_pipeline(text)
                except Exception:
                    ner_results = []

                for ent in ner_results:
                    ent_text = str(ent.get("word", "")).strip()
                    ent_label = str(ent.get("entity_group", "ENTITY"))
                    if not ent_text:
                        continue
                    entity_key = f"{ent_label}:{ent_text.lower()}"
                    entities.append(entity_key)
                    self.entity_map[entity_key].append(node_id)

            if not entities:
                for pattern, entity_type in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entity_key = f"{entity_type}:{match.lower()}"
                        entities.append(entity_key)
                        self.entity_map[entity_key].append(node_id)
            
            self.graph.nodes[node_id]['entities'] = entities
    
    # Connect nodes that mention the same entities.
    def _add_entity_edges(self):
        """Connect sentences that mention the same entities"""
        print("Adding entity coreference edges...")
        
        edge_count = 0
        for entity, node_list in self.entity_map.items():
            if len(node_list) > 1:
                # Connect all nodes mentioning this entity
                for i in range(len(node_list)):
                    for j in range(i + 1, len(node_list)):
                        n1, n2 = node_list[i], node_list[j]
                        # Add bidirectional entity edges
                        self.graph.add_edge(n1, n2, type="entity", weight=0.85, entity=entity)
                        self.graph.add_edge(n2, n1, type="entity", weight=0.85, entity=entity)
                        edge_count += 1
        
        print(f"Added {edge_count} entity coreference edge pairs")
    
    # Align DRG nodes with KG entities and add cross-graph links.
    def add_kg_edges(self, kg_graph_obj):
        """
        Align KG entities with DRG nodes and add cross-graph edges.
        kg_graph_obj is an instance of KnowledgeGraph.
        """
        print("Adding KG-DRG cross-linking edges...")
        if kg_graph_obj is None or kg_graph_obj.graph.number_of_nodes() == 0:
            return

        edge_count = 0
        kg_entities = list(kg_graph_obj.graph.nodes())
        
        # Build mapping for faster lookup
        entity_to_nodes = defaultdict(list)
        for node_id in self.graph.nodes:
            text = self.graph.nodes[node_id]['text'].lower()
            for ent in kg_entities:
                # Use word-boundary regex for precise matching
                if re.search(rf'\b{re.escape(ent.lower())}\b', text):
                    entity_to_nodes[ent].append(node_id)
                    # Store KG entity in DRG node metadata
                    if 'kg_entities' not in self.graph.nodes[node_id]:
                        self.graph.nodes[node_id]['kg_entities'] = []
                    self.graph.nodes[node_id]['kg_entities'].append(ent)

        # Add edges between nodes that share KG entities
        for ent, node_list in entity_to_nodes.items():
            if len(node_list) > 1:
                for i in range(len(node_list)):
                    for j in range(i + 1, len(node_list)):
                        n1, n2 = node_list[i], node_list[j]
                        self.graph.add_edge(n1, n2, type="kg_overlap", weight=0.9, entity=ent)
                        self.graph.add_edge(n2, n1, type="kg_overlap", weight=0.9, entity=ent)
                        edge_count += 1
        
        print(f"Added {edge_count} KG-DRG cross-linking edges")
    
    # Compute centrality values used during retrieval and reasoning.
    def compute_graph_metrics(self):
        """Compute node centrality and importance metrics"""
        print("Computing graph centrality metrics...")
        
        # PageRank for importance
        try:
            pagerank = nx.pagerank(self.graph, weight='weight')
            for node_id in self.graph.nodes:
                self.graph.nodes[node_id]['pagerank'] = pagerank.get(node_id, 0)
        except Exception:
            pass
        
        # Degree centrality
        for node_id in self.graph.nodes:
            self.graph.nodes[node_id]['degree'] = self.graph.degree(node_id)
    
    def export_graph_image(self, output_path: str, figsize=(12, 10)):
        """Export DRG graph as image visualization."""
        plt.figure(figsize=figsize)
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw nodes with importance-based sizing
        node_sizes = [self.graph.nodes[n].get('importance', 1) * 300 for n in self.graph.nodes()]
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes, 
                               node_color='lightblue', alpha=0.7)
        
        # Draw edges with type-based colors (no arrows for large graphs)
        edge_colors = []
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            if edge_type == 'semantic':
                edge_colors.append('red')
            elif edge_type == 'entity':
                edge_colors.append('green')
            elif edge_type == 'adjacent':
                edge_colors.append('blue')
            else:
                edge_colors.append('gray')
        
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, 
                               alpha=0.3, arrows=False)
        
        # Draw labels (truncated text)
        labels = {n: f"{n}\n{self.graph.nodes[n]['text'][:20]}..." 
                  for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=6)
        
        plt.title("Document Reasoning Graph (DRG)", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"DRG image saved to {output_path}")
