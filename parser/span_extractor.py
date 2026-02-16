"""
Span-level text extraction for fine-grained document reasoning.
Extracts meaningful text spans (phrases, clauses) beyond sentence boundaries.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


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


class SpanExtractor:
    """Extract fine-grained spans from sentences."""
    
    def __init__(self):
        # Patterns for clause boundaries
        self.clause_markers = [
            r',\s+(?:if|unless|except|when|where|while|although|because|since)',
            r';\s+',
            r'\s+(?:but|and|or)\s+',
            r':\s+',
        ]
        
        # Patterns for important phrases
        self.important_patterns = [
            r'\b(?:deadline|due date|submission|extension)\b[^.;]*',
            r'\b(?:submission format|file structure|zip file|compressed|compression)\b[^.;]*',
            r'\b(?:not|no|never|cannot|must not|shall not)\b[^.;]*',
            r'\b(?:if|unless|except|only if|provided that)\b[^.;]*',
            r'\b(?:must|shall|should|required|mandatory)\b[^.;]*',
            r'\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
            r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?',
        ]
    
    def extract_spans_from_sentence(
        self, 
        sentence: str, 
        sentence_id: int,
        page: int,
        section: str,
        start_span_id: int = 0
    ) -> List[Span]:
        """Extract multiple spans from a single sentence."""
        spans = []
        span_id = start_span_id
        
        # 1. Extract clauses by splitting on markers
        clauses = self._split_into_clauses(sentence)
        
        offset = 0
        for clause in clauses:
            if len(clause.strip()) > 10:  # minimum clause length
                spans.append(Span(
                    span_id=span_id,
                    text=clause.strip(),
                    start_char=offset,
                    end_char=offset + len(clause),
                    span_type="clause",
                    page=page,
                    section=section,
                    sentence_id=sentence_id
                ))
                span_id += 1
            offset += len(clause)
        
        # 2. Extract important phrases (dates, conditions, negations)
        for pattern in self.important_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                phrase = match.group(0).strip()
                if len(phrase) > 5:
                    # Determine phrase type
                    phrase_type = self._classify_phrase(phrase)
                    
                    spans.append(Span(
                        span_id=span_id,
                        text=phrase,
                        start_char=match.start(),
                        end_char=match.end(),
                        span_type=phrase_type,
                        page=page,
                        section=section,
                        sentence_id=sentence_id
                    ))
                    span_id += 1
        
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
    
    def _classify_phrase(self, phrase: str) -> str:
        """Classify phrase type based on content."""
        phrase_lower = phrase.lower()
        
        if re.search(r'\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)', phrase_lower):
            return "temporal"
        
        if re.search(r'\b(?:if|unless|except|only if|provided that)\b', phrase_lower):
            return "condition"
        
        if re.search(r'\b(?:not|no|never|cannot|must not|shall not)\b', phrase_lower):
            return "negation"
        
        if re.search(r'\b(?:must|shall|should|required|mandatory|obligatory)\b', phrase_lower):
            return "requirement"
        
        if re.search(r'\b(?:deadline|due|submission)\b', phrase_lower):
            return "deadline"
        
        return "phrase"
    
    def extract_spans_from_nodes(self, sentence_nodes: List[Dict]) -> List[Dict]:
        """Convert sentence nodes to span nodes."""
        all_spans = []
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
                all_spans.append({
                    "span_id": span.span_id,
                    "text": span.text,
                    "span_type": span.span_type,
                    "page": span.page,
                    "section": span.section,
                    "sentence_id": span.sentence_id,
                    "start_char": span.start_char,
                    "end_char": span.end_char
                })
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
