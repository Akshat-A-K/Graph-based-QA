"""
knowledge_graph.py
==================
Enhanced Knowledge Graph with:
  - Richer triple extraction (NER linking, noun-chunk fallback, compound verbs,
    passive voice, prepositional chains, copula "be" relations)
  - Entity typing via spaCy NER labels
  - Relation normalisation & deduplication
  - Graph analytics (PageRank, betweenness, community detection)
  - Retrieval API: query_entity(), shortest_path_evidence(), entity_bridge()
  - JSON / GraphML export (unchanged interface)
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

# Optional imports with graceful fallback when packages are unavailable.
try:
    import spacy
    _SPACY_OK = True
except ImportError:
    _SPACY_OK = False

try:
    from networkx.algorithms.community import greedy_modularity_communities
    _COMMUNITY_OK = True
except ImportError:
    _COMMUNITY_OK = False


# Text normalization helpers used for robust node and relation matching.
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "this", "that", "these", "those",
    "it", "its", "they", "their", "he", "she", "his", "her",
    "we", "our", "you", "your", "i", "my",
}

def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip leading determiners."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    tokens = text.split()
    while tokens and tokens[0] in _STOP_WORDS:
        tokens = tokens[1:]
    return " ".join(tokens)


def _span_text(token) -> str:
    """Return the subtree text for a token (compact noun phrase)."""
    # Walk left/right compounds & modifiers only (avoids pulling whole clauses)
    left = [t for t in token.lefts
            if t.dep_ in ("compound", "amod", "nmod", "nummod", "det")]
    right = [t for t in token.rights
             if t.dep_ in ("compound", "amod", "nmod", "nummod")]
    parts = left + [token] + right
    return " ".join(t.text for t in parts)


# Build, analyze, and query a directed knowledge graph from document text.
class KnowledgeGraph:
    """
    Directed knowledge graph built from natural-language sentences.

    Node attributes
    ---------------
    text        : str   - normalised surface form
    label       : str   - NER label or "entity" / "noun_chunk"
    frequency   : int   - how many times this entity was seen
    pagerank    : float - PageRank score (computed after build)
    community   : int   - community id (computed after build)

    Edge attributes
    ---------------
    relation    : str   - normalised relation phrase
    type        : str   - "relation" | "is_a" | "part_of" | "coref"
    weight      : float - edge weight (incremented on duplicate triples)
    """

    # Initialize graph state and optional spaCy pipeline.
    def __init__(self, model_name: str = "en_core_web_trf"):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._entity_freq: Dict[str, int] = defaultdict(int)
        self._ner_label: Dict[str, str] = {}       # normalised text -> NER label
        self.nlp = None

        if not _SPACY_OK:
            print("spaCy not installed - KnowledgeGraph extraction disabled.")
            return

        for name in (model_name, "en_core_web_lg", "en_core_web_sm"):
            try:
                self.nlp = spacy.load(name)
                print(f"[KG] Loaded spaCy model: {name}")
                break
            except Exception:
                continue

        if self.nlp is None:
            print("[KG] No spaCy model found - extraction disabled.")

    # Extract relation triples from sentence-level linguistic structure.
    def _extract_triples(self, text: str) -> List[Tuple[str, str, str, str]]:
        """
        Return list of (subject, relation, object, triple_type) tuples.

        triple_type in {"relation", "is_a", "copula"}
        """
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        triples: List[Tuple[str, str, str, str]] = []

        # Keep sentence-local NER references for entity labeling.
        ner_spans = {ent.start: ent for ent in doc.ents}

        def _best_text(token) -> str:
            """Use NER span text if available, else build from subtree."""
            # check if token starts a named entity
            for ent in doc.ents:
                if token.i >= ent.start and token.i < ent.end:
                    return ent.text
            return _span_text(token)

        def _record_ner(token):
            """Store NER label for the node we're about to create."""
            for ent in doc.ents:
                if token.i >= ent.start and token.i < ent.end:
                    norm = _normalise(ent.text)
                    if norm:
                        self._ner_label[norm] = ent.label_
                    return

        for sent in doc.sents:
            for token in sent:

                # Handle verbal predicates and verb-centered relations.
                if token.pos_ == "VERB":
                    subjects: List[str] = []
                    objects: List[str]  = []

                    # Build full verb phrase (aux + verb + particle)
                    aux_parts = [
                        c.text for c in token.children
                        if c.dep_ in ("aux", "auxpass", "neg", "prt", "part")
                    ]
                    verb_phrase = " ".join(aux_parts + [token.lemma_])

                    is_passive = any(
                        c.dep_ == "auxpass" for c in token.children
                    )

                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            subjects.append(_best_text(child))
                            _record_ner(child)
                        elif child.dep_ in ("dobj", "attr"):
                            objects.append(_best_text(child))
                            _record_ner(child)
                        elif child.dep_ == "prep":
                            prep_word = child.text
                            for gc in child.children:
                                if gc.dep_ == "pobj":
                                    objects.append(_best_text(gc))
                                    verb_phrase_full = f"{verb_phrase} {prep_word}"
                                    for s in subjects:
                                        o_text = _best_text(gc)
                                        if s and o_text:
                                            triples.append(
                                                (s, verb_phrase_full,
                                                 o_text, "relation")
                                            )
                                    _record_ner(gc)

                    # Passive -> flip for readability (obj is_passive subj)
                    for s in subjects:
                        for o in objects:
                            if s and o:
                                if is_passive:
                                    triples.append(
                                        (o, f"is {verb_phrase} by", s, "relation")
                                    )
                                else:
                                    triples.append(
                                        (s, verb_phrase, o, "relation")
                                    )

                # Map copula patterns to simple "is" relations.
                elif token.pos_ == "AUX" and token.lemma_ == "be":
                    subjects: List[str] = []
                    predicates: List[str] = []
                    for child in token.children:
                        if child.dep_ in ("nsubj",):
                            subjects.append(_best_text(child))
                            _record_ner(child)
                        elif child.dep_ in ("attr", "acomp"):
                            predicates.append(_best_text(child))
                            _record_ner(child)
                    for s in subjects:
                        for p in predicates:
                            if s and p:
                                triples.append((s, "is", p, "is_a"))

        # Add fallback links when no verb-centered triples were extracted.
        # When a sentence has no verb triples, connect consecutive named
        # entities with a generic "related_to" edge so the graph stays connected.
        if not triples:
            ents = [ent for ent in doc.ents if len(ent.text.split()) <= 5]
            for i in range(len(ents) - 1):
                s = ents[i].text
                o = ents[i + 1].text
                triples.append((s, "related_to", o, "relation"))
                self._ner_label[_normalise(s)] = ents[i].label_
                self._ner_label[_normalise(o)] = ents[i + 1].label_

        return triples

    # Build the graph from sentence text and extracted triples.
    def build_graph(self, sentences: List[str]) -> None:
        """Build the knowledge graph from a list of sentences."""
        print("[KG] Building Knowledge Graph...")
        edge_count = 0

        for item in sentences:
            # Handle list of strings or list of node dictionaries
            if isinstance(item, dict):
                sent_text = item.get("text", "")
            else:
                sent_text = str(item)
            
            if not sent_text:
                continue

            for subj_raw, rel_raw, obj_raw, triple_type in self._extract_triples(sent_text):
                subj = _normalise(subj_raw)
                obj  = _normalise(obj_raw)
                rel  = rel_raw.strip().lower()

                if not subj or not obj or len(subj) < 2 or len(obj) < 2:
                    continue
                if subj == obj:
                    continue

                # Node frequency
                self._entity_freq[subj] += 1
                self._entity_freq[obj]  += 1

                # Add / update nodes
                for node_id in (subj, obj):
                    if not self.graph.has_node(node_id):
                        self.graph.add_node(
                            node_id,
                            text=node_id,
                            label=self._ner_label.get(node_id, "entity"),
                            frequency=1,
                            pagerank=0.0,
                            community=-1,
                        )
                    else:
                        self.graph.nodes[node_id]["frequency"] = \
                            self._entity_freq[node_id]
                        # Update label if we now have a proper NER tag
                        if (self.graph.nodes[node_id]["label"] == "entity"
                                and node_id in self._ner_label):
                            self.graph.nodes[node_id]["label"] = \
                                self._ner_label[node_id]

                # Add / update edge (weighted)
                if self.graph.has_edge(subj, obj):
                    self.graph[subj][obj]["weight"] += 1.0
                else:
                    self.graph.add_edge(
                        subj, obj,
                        relation=rel,
                        type=triple_type,
                        weight=1.0,
                    )
                    edge_count += 1

        self._compute_analytics()
        print(
            f"[KG] Built: {self.graph.number_of_nodes()} nodes, "
            f"{edge_count} unique edges"
        )

    # Compute graph analytics used for ranking and diagnostics.
    def _compute_analytics(self) -> None:
        """Compute PageRank, betweenness centrality, and communities."""
        if self.graph.number_of_nodes() == 0:
            return

        # PageRank
        try:
            pr = nx.pagerank(self.graph, alpha=0.85, max_iter=200)
            for node, score in pr.items():
                if self.graph.has_node(node):
                    self.graph.nodes[node]["pagerank"] = round(score, 6)
        except Exception:
            pass

        # Community detection on undirected version
        if _COMMUNITY_OK:
            try:
                undirected = self.graph.to_undirected()
                communities = list(greedy_modularity_communities(undirected))
                for comm_id, comm_nodes in enumerate(communities):
                    for node in comm_nodes:
                        if self.graph.has_node(node):
                            self.graph.nodes[node]["community"] = comm_id
            except Exception:
                pass

    # Query and path APIs used during QA reasoning.
    def query_entity(
        self, entity: str, max_hops: int = 2
    ) -> List[Dict]:
        """
        Return all triples reachable within *max_hops* from *entity*.

        Returns list of dicts:
            {"subject": str, "relation": str, "object": str,
             "type": str, "weight": float}
        """
        norm = _normalise(entity)
        if not self.graph.has_node(norm):
            # fuzzy fallback: partial match
            candidates = [
                n for n in self.graph.nodes
                if norm in n or n in norm
            ]
            if not candidates:
                return []
            norm = max(candidates,
                       key=lambda n: self.graph.nodes[n].get("pagerank", 0))

        visited: Set[str] = set()
        frontier = {norm}
        results: List[Dict] = []

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for u, v, data in self.graph.out_edges(node, data=True):
                    results.append({
                        "subject":  u,
                        "relation": data.get("relation", ""),
                        "object":   v,
                        "type":     data.get("type", "relation"),
                        "weight":   data.get("weight", 1.0),
                    })
                    next_frontier.add(v)
                for u, v, data in self.graph.in_edges(node, data=True):
                    results.append({
                        "subject":  u,
                        "relation": data.get("relation", ""),
                        "object":   v,
                        "type":     data.get("type", "relation"),
                        "weight":   data.get("weight", 1.0),
                    })
                    next_frontier.add(u)
            frontier = next_frontier - visited

        # Deduplicate
        seen: Set[Tuple] = set()
        unique: List[Dict] = []
        for r in results:
            key = (r["subject"], r["relation"], r["object"])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        # Sort by weight descending
        unique.sort(key=lambda x: x["weight"], reverse=True)
        return unique

    def shortest_path_evidence(
        self, entity1: str, entity2: str
    ) -> Optional[List[str]]:
        """
        Return the sequence of relation labels on the shortest undirected
        path between *entity1* and *entity2*, or None if unreachable.

        Useful for multi-hop bridge questions.
        """
        n1 = _normalise(entity1)
        n2 = _normalise(entity2)

        # fuzzy fallback
        def _resolve(name: str) -> Optional[str]:
            if self.graph.has_node(name):
                return name
            candidates = [n for n in self.graph.nodes
                          if name in n or n in name]
            if not candidates:
                return None
            return max(candidates,
                       key=lambda n: self.graph.nodes[n].get("pagerank", 0))

        n1 = _resolve(n1)
        n2 = _resolve(n2)
        if n1 is None or n2 is None:
            return None

        try:
            undirected = self.graph.to_undirected()
            path = nx.shortest_path(undirected, n1, n2)
            evidence: List[str] = []
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                rel = (
                    self.graph[a][b].get("relation", "")
                    if self.graph.has_edge(a, b)
                    else self.graph[b][a].get("relation", "?")
                )
                evidence.append(f"{a} -[{rel}]-> {b}")
            return evidence
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def entity_bridge(
        self, entity1: str, entity2: str
    ) -> List[str]:
        """
        Return intermediate entities that connect *entity1* to *entity2*
        (bridge nodes on the shortest path).
        """
        path_evidence = self.shortest_path_evidence(entity1, entity2)
        if path_evidence is None or len(path_evidence) <= 1:
            return []
        # Extract intermediate node names (between -> markers)
        nodes = []
        for step in path_evidence[:-1]:          # exclude last hop
            # step looks like "node_a -[rel]-> node_b"
            parts = re.split(r"\s*-\[.*?\]->\s*", step)
            if len(parts) == 2:
                nodes.append(parts[1].strip())   # the target of this hop
        return nodes

    def top_entities(self, n: int = 10) -> List[Dict]:
        """Return the top-n entities by PageRank score."""
        nodes = [
            {
                "entity":    d.get("text", nid),
                "label":     d.get("label", "entity"),
                "pagerank":  d.get("pagerank", 0.0),
                "frequency": d.get("frequency", 1),
                "community": d.get("community", -1),
            }
            for nid, d in self.graph.nodes(data=True)
        ]
        nodes.sort(key=lambda x: x["pagerank"], reverse=True)
        return nodes[:n]

    def entities_by_label(self, label: str) -> List[str]:
        """Return all entity texts with a given NER label (e.g. 'PERSON')."""
        return [
            d.get("text", nid)
            for nid, d in self.graph.nodes(data=True)
            if d.get("label", "") == label
        ]

    def get_stats(self) -> Dict:
        """Return a dict of graph statistics."""
        label_counts: Dict[str, int] = defaultdict(int)
        for _, d in self.graph.nodes(data=True):
            label_counts[d.get("label", "entity")] += 1

        communities = set(
            d.get("community", -1)
            for _, d in self.graph.nodes(data=True)
            if d.get("community", -1) >= 0
        )

        return {
            "nodes":       self.graph.number_of_nodes(),
            "edges":       self.graph.number_of_edges(),
            "communities": len(communities),
            "entity_labels": dict(label_counts),
            "density":     round(nx.density(self.graph), 6),
            "is_connected": nx.is_weakly_connected(self.graph)
                            if self.graph.number_of_nodes() > 0 else False,
        }

    # Export graph structure and metadata for inspection and tooling.
    def export_json(self, filepath: str) -> None:
        """Export the knowledge graph to JSON."""
        data = {
            "nodes": [
                {
                    "id":        nid,
                    "text":      d.get("text", nid),
                    "label":     d.get("label", "entity"),
                    "pagerank":  d.get("pagerank", 0.0),
                    "frequency": d.get("frequency", 1),
                    "community": d.get("community", -1),
                }
                for nid, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source":   u,
                    "target":   v,
                    "relation": d.get("relation", ""),
                    "type":     d.get("type", "relation"),
                    "weight":   d.get("weight", 1.0),
                }
                for u, v, d in self.graph.edges(data=True)
            ],
            "stats": self.get_stats(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[KG] Exported to {filepath}")

    def export_graphml(self, filepath: str) -> None:
        """Export the knowledge graph to GraphML."""
        nx.write_graphml(self.graph, filepath)
        print(f"[KG] Exported GraphML to {filepath}")