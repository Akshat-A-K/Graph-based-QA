"""
Enhanced Knowledge Graph Builder
Advanced entity recognition, relation extraction, and graph visualization export
"""

import re
import json
import networkx as nx
from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    entity_id: int
    text: str
    entity_type: str  # DATE, TIME, PERSON, ORG, REQUIREMENT, CONSTRAINT
    span_ids: List[int]  # source spans


@dataclass
class Relation:
    """Represents a relation between entities."""
    relation_id: int
    source_entity_id: int
    target_entity_id: int
    relation_type: str  # TEMPORAL, CONDITIONAL, CAUSAL, RESTRICTION
    confidence: float


class KnowledgeGraphBuilder:
    """Build a knowledge graph from document spans."""
    
    def __init__(self):
        print("🧠 Initializing Enhanced Knowledge Graph Builder...")

        self.ner_pipeline = None
        self.rebel_pipeline = None
        self.use_ner = True
        self.use_rebel = True

        try:
            from transformers import pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model="Davlan/bert-base-multilingual-cased-ner-hrl",
                aggregation_strategy="simple"
            )
        except Exception:
            self.ner_pipeline = None
            self.use_ner = False

        try:
            from transformers import pipeline
            self.rebel_pipeline = pipeline(
                "text2text-generation",
                model="Babelscape/rebel-large"
            )
        except Exception:
            self.rebel_pipeline = None
            self.use_rebel = False
        
        # Enhanced entity extraction patterns with better coverage
        self.entity_patterns = {
            "DATE": [
                r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}',
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',  # ISO format
            ],
            "TIME": [
                r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|AM|PM|hours?)?\b',
                r'\b(?:23:59|11:59|00:00|noon|midnight)\b',
            ],
            "PERSON": [
                r'\b(?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.|Sir|Madam)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b(?=\s+(?:said|wrote|mentioned|stated|proposed|suggested))',
            ],
            "ORG": [
                r'\b(?:University|Department|Institute|Ministry|Committee|Board|Company|Corporation|Agency)\s+(?:of\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            ],
            "SCORE": [  # NEW: Detect marks, grades, scores
                r'\b\d+\s*(?:marks?|points?|credits?|grade|percentage|%)\b',
                r'\b(?:grade|marks?)\s*:?\s*\d+',
            ],
            "PERCENTAGE": [
                r'\b\d+(?:\.\d+)?\s*%',
                r'\b\d+(?:\.\d+)?\s*percent',
            ],
            "NUMBER": [  # NEW: Generic numbers
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            ],
            "REQUIREMENT": [
                r'\b(?:must|shall|required to|mandatory to|obligatory to|need to|have to)\s+[a-z\s]+?(?=\.)',
            ],
            "CONSTRAINT": [
                r'\b(?:not allowed|cannot|must not|shall not|prohibited|forbidden|banned)\s+[a-z\s]+?(?=\.)',
                r'\b(?:except|unless|only if|provided that|on condition)\s+[a-z\s]+',
            ],
            "KEYWORD": [  # NEW: Domain-specific keywords
                r'\b(?:assignment|submission|deadline|task|project|exam|test|quiz|lecture|tutorial)\b',
            ],
        }
        
        # Enhanced relation patterns
        self.relation_patterns = {
            "TEMPORAL": [
                r'\b(?:before|after|until|by|deadline|due date|on|during|while|when)\b',
            ],
            "CONDITIONAL": [
                r'\b(?:if|unless|provided that|only if|in case|assuming|given that)\b',
            ],
            "CAUSAL": [
                r'\b(?:because|since|as|therefore|thus|hence|consequently|results in|leads to|causes)\b',
            ],
            "RESTRICTION": [
                r'\b(?:except|excluding|not including|but not|other than|without)\b',
            ],
            "REQUIREMENT": [
                r'\b(?:must|shall|required|need to|have to|should|ought to)\b',
            ],
            "COMPOSITION": [  # NEW: Part-of relations
                r'\b(?:consists of|contains|includes|comprises|made up of)\b',
            ],
            "EQUIVALENCE": [  # NEW: Is-a relations
                r'\b(?:is|are|equals|means|refers to|defined as)\b',
            ],
        }
        
        # Graph for visualization
        self.nx_graph = nx.DiGraph()
    
    def extract_entities(self, spans: List[Dict]) -> List[Entity]:
        """Extract entities from spans."""
        entities = []
        entity_id = 0
        seen_entities = set()  # avoid duplicates

        if self.use_ner and self.ner_pipeline is not None:
            for span in spans:
                text = span["text"]
                span_id = span["span_id"]
                try:
                    ner_results = self.ner_pipeline(text)
                except Exception:
                    ner_results = []

                for ent in ner_results:
                    ent_text = str(ent.get("word", "")).strip()
                    ent_label = str(ent.get("entity_group", "ENTITY"))
                    if not ent_text:
                        continue
                    entity_key = (ent_text.lower(), ent_label)
                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        entities.append(Entity(
                            entity_id=entity_id,
                            text=ent_text,
                            entity_type=ent_label,
                            span_ids=[span_id]
                        ))
                        entity_id += 1
                    else:
                        for entity in entities:
                            if entity.text.lower() == ent_text.lower() and entity.entity_type == ent_label:
                                if span_id not in entity.span_ids:
                                    entity.span_ids.append(span_id)
                                break

            if entities:
                return entities
        
        for span in spans:
            text = span["text"]
            span_id = span["span_id"]
            
            # Try each entity type pattern
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entity_text = match.group(0).strip()
                        
                        # Normalize for deduplication
                        entity_key = (entity_text.lower(), entity_type)
                        
                        if entity_key not in seen_entities:
                            seen_entities.add(entity_key)
                            entities.append(Entity(
                                entity_id=entity_id,
                                text=entity_text,
                                entity_type=entity_type,
                                span_ids=[span_id]
                            ))
                            entity_id += 1
                        else:
                            # Update existing entity with new span reference
                            for entity in entities:
                                if entity.text.lower() == entity_text.lower() and entity.entity_type == entity_type:
                                    if span_id not in entity.span_ids:
                                        entity.span_ids.append(span_id)
                                    break
        
        return entities

    def _parse_rebel_output(self, text: str) -> List[Dict]:
        """Parse REBEL model output into triplets."""
        triplets = []
        if not text:
            return triplets

        parts = text.split("<triplet>")
        for part in parts:
            part = part.strip()
            if not part:
                continue

            subj = ""
            obj = ""
            rel = ""
            tokens = part.split("<subj>")
            if len(tokens) < 2:
                continue
            after_subj = tokens[1]
            subj = after_subj.split("<obj>")[0].strip()
            after_obj = after_subj.split("<obj>")[-1]
            obj = after_obj.split("<rel>")[0].strip()
            rel = after_obj.split("<rel>")[-1].strip()

            if subj and obj and rel:
                triplets.append({"subject": subj, "object": obj, "relation": rel})

        return triplets

    def _extract_relations_model(self, entities: List[Entity], spans: List[Dict]) -> List[Relation]:
        """Extract relations using REBEL model output."""
        relations = []
        relation_id = 0

        entity_by_key = { (e.text.lower(), e.entity_type): e for e in entities }
        entity_by_text = { e.text.lower(): e for e in entities }

        for span in spans:
            text = span["text"]
            try:
                outputs = self.rebel_pipeline(text, max_length=256, truncation=True)
            except Exception:
                outputs = []

            if not outputs:
                continue

            generated = outputs[0].get("generated_text", "")
            triplets = self._parse_rebel_output(generated)
            if not triplets:
                continue

            for triplet in triplets:
                subj_text = triplet["subject"].strip()
                obj_text = triplet["object"].strip()
                rel_text = triplet["relation"].strip()

                if not subj_text or not obj_text or not rel_text:
                    continue

                subj_key = subj_text.lower()
                obj_key = obj_text.lower()

                subj_entity = entity_by_text.get(subj_key)
                if subj_entity is None:
                    subj_entity = Entity(
                        entity_id=len(entities),
                        text=subj_text,
                        entity_type="ENTITY",
                        span_ids=[span["span_id"]]
                    )
                    entities.append(subj_entity)
                    entity_by_text[subj_key] = subj_entity

                obj_entity = entity_by_text.get(obj_key)
                if obj_entity is None:
                    obj_entity = Entity(
                        entity_id=len(entities),
                        text=obj_text,
                        entity_type="ENTITY",
                        span_ids=[span["span_id"]]
                    )
                    entities.append(obj_entity)
                    entity_by_text[obj_key] = obj_entity

                relations.append(Relation(
                    relation_id=relation_id,
                    source_entity_id=subj_entity.entity_id,
                    target_entity_id=obj_entity.entity_id,
                    relation_type=rel_text.upper().replace(" ", "_"),
                    confidence=0.7
                ))
                relation_id += 1

        return relations
    
    def extract_relations(
        self, 
        entities: List[Entity], 
        spans: List[Dict]
    ) -> List[Relation]:
        """Extract relations between entities based on co-occurrence and patterns."""
        if self.use_rebel and self.rebel_pipeline is not None:
            relations = self._extract_relations_model(entities, spans)
            if relations:
                return relations

        relations = []
        relation_id = 0
        
        # Build span_id -> entity_ids mapping
        span_to_entities = defaultdict(list)
        for entity in entities:
            for span_id in entity.span_ids:
                span_to_entities[span_id].append(entity.entity_id)
        
        # Find relations within spans
        for span in spans:
            span_id = span["span_id"]
            text = span["text"]
            
            entity_ids = span_to_entities.get(span_id, [])
            
            # If multiple entities in same span, check for relations
            if len(entity_ids) >= 2:
                # Check which relation type applies
                relation_type = self._detect_relation_type(text)
                
                if relation_type:
                    # Create relations between entity pairs
                    for i in range(len(entity_ids) - 1):
                        for j in range(i + 1, len(entity_ids)):
                            relations.append(Relation(
                                relation_id=relation_id,
                                source_entity_id=entity_ids[i],
                                target_entity_id=entity_ids[j],
                                relation_type=relation_type,
                                confidence=0.8
                            ))
                            relation_id += 1
        
        # Additional heuristic: temporal relations between dates and requirements
        for entity in entities:
            if entity.entity_type == "REQUIREMENT":
                # Find nearest DATE entity
                for other_entity in entities:
                    if other_entity.entity_type == "DATE":
                        # Check if they share spans or are in nearby spans
                        shared = set(entity.span_ids) & set(other_entity.span_ids)
                        if shared or self._spans_are_close(entity.span_ids, other_entity.span_ids):
                            relations.append(Relation(
                                relation_id=relation_id,
                                source_entity_id=entity.entity_id,
                                target_entity_id=other_entity.entity_id,
                                relation_type="TEMPORAL",
                                confidence=0.7
                            ))
                            relation_id += 1
        
        return relations
    
    def _detect_relation_type(self, text: str) -> str:
        """Detect relation type from text using patterns."""
        text_lower = text.lower()
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return relation_type
        
        return None
    
    def _spans_are_close(self, span_ids_1: List[int], span_ids_2: List[int]) -> bool:
        """Check if two sets of span IDs are close (within 3 spans)."""
        for s1 in span_ids_1:
            for s2 in span_ids_2:
                if abs(s1 - s2) <= 3:
                    return True
        return False
    
    def build_kg(self, spans: List[Dict]) -> Dict:
        """
        Build enhanced knowledge graph from spans.
        
        Returns:
            Dictionary with 'entities', 'relations', and 'graph' keys
        """
        print("🔨 Building knowledge graph...")
        
        entities = self.extract_entities(spans)
        relations = self.extract_relations(entities, spans)
        
        # Build NetworkX graph for visualization
        self._build_nx_graph(entities, relations)
        
        print(f"✅ KG built: {len(entities)} entities, {len(relations)} relations")
        
        return {
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "text": e.text,
                    "entity_type": e.entity_type,
                    "span_ids": e.span_ids
                }
                for e in entities
            ],
            "relations": [
                {
                    "relation_id": r.relation_id,
                    "source": r.source_entity_id,
                    "target": r.target_entity_id,
                    "type": r.relation_type,
                    "confidence": r.confidence
                }
                for r in relations
            ],
            "graph": self.nx_graph  # Add NetworkX graph
        }
    
    def _build_nx_graph(self, entities: List[Entity], relations: List[Relation]):
        """Build NetworkX graph from entities and relations."""
        # Add entity nodes
        for entity in entities:
            self.nx_graph.add_node(
                entity.entity_id,
                label=entity.text,
                type=entity.entity_type,
                span_ids=entity.span_ids
            )
        
        # Add relation edges
        for relation in relations:
            self.nx_graph.add_edge(
                relation.source_entity_id,
                relation.target_entity_id,
                relation=relation.relation_type,
                confidence=relation.confidence
            )
    
    def export_graph_json(self, output_path: str):
        """Export knowledge graph to JSON for visualization."""
        data = {
            "nodes": [
                {"id": n, **self.nx_graph.nodes[n]}
                for n in self.nx_graph.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **self.nx_graph.edges[u, v]
                }
                for u, v in self.nx_graph.edges()
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ KG exported to {output_path}")
    
    def export_graph_graphml(self, output_path: str):
        """Export knowledge graph to GraphML for Gephi/Cytoscape."""
        # Create a copy without lists for GraphML export
        export_graph = self.nx_graph.copy()
        for node in export_graph.nodes():
            if 'span_ids' in export_graph.nodes[node]:
                # Convert list to string
                span_ids = export_graph.nodes[node]['span_ids']
                export_graph.nodes[node]['span_ids'] = ','.join(map(str, span_ids)) if span_ids else ''
        
        nx.write_graphml(export_graph, output_path)
        print(f"✅ KG exported to {output_path}")
    
    def export_graph_image(self, output_path: str, figsize=(12, 10)):
        """Export knowledge graph as image visualization."""
        if self.nx_graph.number_of_nodes() == 0:
            print("⚠️  No entities to visualize")
            return
        
        plt.figure(figsize=figsize)
        
        # Use hierarchical layout
        pos = nx.spring_layout(self.nx_graph, k=2, iterations=50)
        
        # Color nodes by entity type
        entity_type_colors = {
            'DATE': 'lightcoral',
            'TIME': 'lightgreen',
            'PERSON': 'lightblue',
            'ORG': 'lightyellow',
            'SCORE': 'gold',
            'PERCENTAGE': 'orange',
            'NUMBER': 'lightgray',
            'KEYWORD': 'pink',
            'REQUIREMENT': 'red',
            'CONSTRAINT': 'darkred',
        }
        
        node_colors = [entity_type_colors.get(self.nx_graph.nodes[n].get('type', ''), 'white') 
                       for n in self.nx_graph.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(self.nx_graph, pos, node_color=node_colors,
                               node_size=800, alpha=0.8)
        
        # Draw edges with relation labels (no arrows for faster rendering)
        nx.draw_networkx_edges(self.nx_graph, pos, alpha=0.4, arrows=False,
                               width=2)
        
        # Draw labels
        labels = {n: self.nx_graph.nodes[n].get('label', str(n))[:15] 
                  for n in self.nx_graph.nodes()}
        nx.draw_networkx_labels(self.nx_graph, pos, labels, font_size=8)
        
        # Draw edge labels
        edge_labels = {(u, v): self.nx_graph.edges[u, v].get('relation', '')[:10]
                       for u, v in self.nx_graph.edges()}
        nx.draw_networkx_edge_labels(self.nx_graph, pos, edge_labels, font_size=6)
        
        plt.title("Knowledge Graph (Entities & Relations)", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ KG image saved to {output_path}")


def build_knowledge_graph(spans: List[Dict]) -> Dict:
    """
    Main entry point: build KG from spans.
    
    Args:
        spans: List of span nodes from span_extractor.build_span_nodes()
    
    Returns:
        Knowledge graph with entities and relations
    """
    builder = KnowledgeGraphBuilder()
    return builder.build_kg(spans)
