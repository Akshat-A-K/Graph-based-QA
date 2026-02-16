import networkx as nx
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentReasoningGraph:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.graph = nx.Graph()

    # -------------------------
    # STEP 1: add nodes
    # -------------------------
    def add_nodes(self, nodes):
        for node in nodes:
            self.graph.add_node(
                node["node_id"],
                text=node["text"],
                page=node["page"],
                section=node["section"],
                sent_index=node["sent_index"]
            )

    # -------------------------
    # STEP 2: embeddings
    # -------------------------
    def compute_embeddings(self):
        print("Computing embeddings...")

        texts = [self.graph.nodes[n]["text"] for n in self.graph.nodes]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        for i, node_id in enumerate(self.graph.nodes):
            self.graph.nodes[node_id]["embedding"] = embeddings[i]

    # -------------------------
    # STEP 3: structural edges
    # -------------------------
    def add_structural_edges(self):
        print("Adding structural edges...")

        nodes = list(self.graph.nodes)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1 = nodes[i]
                n2 = nodes[j]

                d1 = self.graph.nodes[n1]
                d2 = self.graph.nodes[n2]

                # same page
                if d1["page"] == d2["page"]:
                    self.graph.add_edge(n1, n2, type="page")

                # same section
                if d1["section"] == d2["section"]:
                    self.graph.add_edge(n1, n2, type="section")

                # adjacent sentences
                if (
                    d1["page"] == d2["page"]
                    and abs(d1["sent_index"] - d2["sent_index"]) == 1
                ):
                    self.graph.add_edge(n1, n2, type="adjacent")

    # -------------------------
    # STEP 4: semantic edges
    # -------------------------
    def add_semantic_edges(self, threshold=0.75):
        print("Adding semantic edges...")

        node_ids = list(self.graph.nodes)
        embeddings = [self.graph.nodes[n]["embedding"] for n in node_ids]

        sim_matrix = cosine_similarity(embeddings)

        for i in tqdm(range(len(node_ids))):
            for j in range(i + 1, len(node_ids)):
                sim = sim_matrix[i][j]

                if sim >= threshold:
                    self.graph.add_edge(
                        node_ids[i],
                        node_ids[j],
                        type="semantic",
                        weight=float(sim)
                    )

    # -------------------------
    # BUILD FULL GRAPH
    # -------------------------
    def build_graph(self, nodes):
        self.add_nodes(nodes)
        self.compute_embeddings()
        self.add_structural_edges()
        self.add_semantic_edges()

        print("Graph built.")
        print("Nodes:", self.graph.number_of_nodes())
        print("Edges:", self.graph.number_of_edges())

        return self.graph
