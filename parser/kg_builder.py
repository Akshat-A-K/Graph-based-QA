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
    
    def extract_relations(
        self, 
        entities: List[Entity], 
        spans: List[Dict]
    ) -> List[Relation]:
        """Extract relations between entities based on co-occurrence and patterns."""
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
