"""
Span-level text extraction for fine-grained document reasoning.
Extracts meaningful text spans (phrases, clauses) beyond sentence boundaries.
"""

import re
import spacy
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class Span:
    """Represents a text span with metadata."""
    span_id: int
    text: str
    start_char: int
    end_char: int
    span_type: str  # clause, phrase, named_entity, keyword
    page: int
    section: str
    sentence_id: int
    entities: List[str] = field(default_factory=list)


class SpanExtractor:
    """Extract fine-grained spans from sentences."""
    
    def __init__(self, ner_model: str = "Jean-Baptiste/roberta-large-ner-english"):
        self.ner_model = ner_model
        self.ner_pipeline = None
        self.use_ner = True

        try:
            from .model_cache import get_ner_pipeline
            self.ner_pipeline = get_ner_pipeline(self.ner_model)
        except Exception:
            self.ner_pipeline = None
            self.use_ner = False

        # initialize spaCy dependency parser (may raise if model missing)
        try:
            self.nlp = spacy.load("en_core_web_trf")
            print("Loaded spaCy model for dependency parsing.")
        except Exception:
            self.nlp = None

        # Patterns for clause boundaries (more conservative to avoid over-splitting)
        self.clause_markers = [
            r';\s+',
            r':\s+(?=[A-Z])',  # Only split on colon if followed by capital
        ]
        
    
    def extract_spans_from_sentence(
        self, 
        sentence: str, 
        sentence_id: int,
        page: int,
        section: str,
        start_span_id: int = 0
    ) -> List[Span]:
        """Extract multiple spans from a single sentence with improved quality."""
        spans = []
        span_id = start_span_id
        
        # 1. ALWAYS include the full sentence as a span (improves readability)
        spans.append(Span(
            span_id=span_id,
            text=sentence.strip(),
            start_char=0,
            end_char=len(sentence),
            span_type="sentence",
            page=page,
            section=section,
            sentence_id=sentence_id,
            entities=[]
        ))
        span_id += 1
        
        # 2. Extract entities using model-based NER (preferred)
        extracted_positions = set()
        ner_entities = []

        if self.use_ner and self.ner_pipeline is not None:
            try:
                ner_results = self.ner_pipeline(sentence)
            except Exception:
                ner_results = []

            for ent in ner_results:
                start = int(ent.get("start", 0))
                end = int(ent.get("end", 0))
                label = str(ent.get("entity_group", "ENTITY"))
                text = sentence[start:end].strip()

                # Skip punctuation or meaningless spans
                if (
                    not text
                    or end <= start
                    or len(text) < 2
                    or text in {".", ",", ";", ":"}
                ):
                    continue

                # Avoid overlaps with existing extracted spans
                overlap = False
                for s, e in extracted_positions:
                    if not (end <= s or start >= e):
                        overlap = True
                        break

                if overlap:
                    continue

                spans.append(Span(
                    span_id=span_id,
                    text=text,
                    start_char=start,
                    end_char=end,
                    span_type=f"ner:{label}",
                    page=page,
                    section=section,
                    sentence_id=sentence_id,
                    entities=[text.lower()]
                ))
                extracted_positions.add((start, end))
                ner_entities.append(text.lower().strip())
                span_id += 1

        # 3. Extract dependency-based spans via spaCy (noun chunks and verb clauses)
        if self.nlp is not None:
            try:
                dep_spans, span_id = self._extract_dependency_spans(
                    sentence,
                    sentence_id,
                    page,
                    section,
                    span_id
                )
                # Avoid overlaps with previously extracted positions
                for sspan in dep_spans:
                    overlap = False
                    for s, e in extracted_positions:
                        if not (sspan.start_char >= e or sspan.end_char <= s):
                            overlap = True
                            break
                    if not overlap:
                        spans.append(sspan)
                        extracted_positions.add((sspan.start_char, sspan.end_char))
            except Exception:
                # Fallback: do nothing if spaCy fails
                pass
        
        # 4. Only split into clauses if sentence is very long (>150 chars)
        if len(sentence) > 150:
            clauses = self._split_into_clauses(sentence)
            offset = 0
            for clause in clauses:
                if len(clause.strip()) > 30:  # Higher minimum for clause quality
                    # Check if not already covered by important patterns
                    overlap = False
                    for start, end in extracted_positions:
                        if not (offset + len(clause) <= start or offset >= end):
                            overlap = True
                            break
                    
                    if not overlap:
                        spans.append(Span(
                            span_id=span_id,
                            text=clause.strip(),
                            start_char=offset,
                            end_char=offset + len(clause),
                            span_type="clause",
                            page=page,
                            section=section,
                            sentence_id=sentence_id,
                            entities=[]
                        ))
                        span_id += 1
                offset += len(clause)
        
        # 5. Attach sentence-level entity list to the full sentence span
        if ner_entities:
            spans[0].entities = sorted(set(ner_entities))

        return spans
    
    def _split_into_clauses(self, sentence: str) -> List[str]:
        """Split sentence into clauses based on markers."""
        # Start with the full sentence
        clauses = [sentence]
        
        # Split on each marker pattern
        for pattern in self.clause_markers:
            new_clauses = []
            for clause in clauses:
                parts = re.split(pattern, clause)
                new_clauses.extend(parts)
            clauses = new_clauses
        
        return [c for c in clauses if c.strip()]


    def _extract_dependency_spans(self, sentence, sentence_id, page, section, span_id):
        spans = []
        if self.nlp is None:
            return spans, span_id

        doc = self.nlp(sentence)

        # ── noun phrases ──
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if not text:
                continue
            spans.append(Span(
                span_id=span_id,
                text=text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                span_type="noun_phrase",
                page=page, section=section,
                sentence_id=sentence_id, entities=[]
            ))
            span_id += 1

        # ── CARDINAL + head noun (moved OUTSIDE noun_chunks loop) ──
        for token in doc:
            if token.pos_ == "NUM" or token.ent_type_ == "CARDINAL":
                head = token.head
                if head.pos_ in ("NOUN", "PROPN"):
                    subtree_tokens = sorted(
                        [t for t in head.subtree
                        if t.pos_ in ("NUM", "NOUN", "ADV", "ADJ", "PROPN")
                        and t.dep_ in ("nummod", "advmod", "amod", "compound",
                                        "ROOT", "nsubj", "dobj")],
                        key=lambda t: t.i
                    )
                    if subtree_tokens:
                        start_idx = subtree_tokens[0].idx
                        end_idx   = subtree_tokens[-1].idx + len(subtree_tokens[-1].text)
                    else:
                        start_idx = min(token.idx, head.idx)
                        end_idx   = max(token.idx + len(token.text),
                                        head.idx   + len(head.text))

                    text = sentence[start_idx:end_idx].strip()
                    if len(text) > 2 and any(c.isdigit() for c in text):
                        spans.append(Span(
                            span_id=span_id,
                            text=text,
                            start_char=start_idx,
                            end_char=end_idx,
                            span_type="noun_phrase",
                            page=page, section=section,
                            sentence_id=sentence_id, entities=[]
                        ))
                        span_id += 1

        # ── verb clauses (moved OUTSIDE noun_chunks loop) ──
        for token in doc:
            if token.pos_ == "VERB":
                subtree = list(token.subtree)
                if not subtree:
                    continue
                start = subtree[0].idx
                end   = subtree[-1].idx + len(subtree[-1].text)
                if start < 0 or end <= start:
                    continue
                clause = sentence[start:end].strip()
                if len(clause.split()) > 2:
                    spans.append(Span(
                        span_id=span_id,
                        text=clause,
                        start_char=start, end_char=end,
                        span_type="verb_clause",
                        page=page, section=section,
                        sentence_id=sentence_id, entities=[]
                    ))
                    span_id += 1

        return spans, span_id   # ← now correctly OUTSIDE all loops
    
    # removed regex-based phrase classifier; dependency-based extraction used instead
    
    def extract_spans_from_nodes(self, sentence_nodes: List[Dict]) -> List[Dict]:
        """Convert sentence nodes to span nodes."""
        all_spans = []
        seen_spans = set()
        span_id = 0
        
        for sent_node in sentence_nodes:
            spans = self.extract_spans_from_sentence(
                sentence=sent_node["text"],
                sentence_id=sent_node["node_id"],
                page=sent_node["page"],
                section=sent_node["section"],
                start_span_id=span_id
            )
            
            for span in spans:
                key = (span.text.lower().strip(), span.sentence_id)
                if key not in seen_spans:
                    all_spans.append({
                        "span_id": span.span_id,
                        "text": span.text,
                        "span_type": span.span_type,
                        "page": span.page,
                        "section": span.section,
                        "sentence_id": span.sentence_id,
                        "start_char": span.start_char,
                        "end_char": span.end_char,
                        "entities": span.entities
                    })
                    seen_spans.add(key)
                    span_id += 1
        
        return all_spans


def build_span_nodes(sentence_nodes: List[Dict]) -> List[Dict]:
    """
    Main entry point: convert sentence nodes to span nodes.
    
    Args:
        sentence_nodes: List of sentence-level nodes from drg_nodes.build_nodes()
    
    Returns:
        List of span-level nodes with finer granularity
    """
    extractor = SpanExtractor()
    return extractor.extract_spans_from_nodes(sentence_nodes)
