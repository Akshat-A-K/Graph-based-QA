"""Span-level text extraction"""

import re
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
    
    def __init__(self, ner_model: str = "Davlan/bert-base-multilingual-cased-ner-hrl"):
        self.ner_model = ner_model
        self.ner_pipeline = None
        self.use_ner = True

        try:
            from transformers import pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model,
                aggregation_strategy="simple"
            )
        except Exception:
            self.ner_pipeline = None
            self.use_ner = False

        # Patterns for clause boundaries (more conservative to avoid over-splitting)
        self.clause_markers = [
                r';\s+(?=[A-Z])',            # semicolon followed by capital
                r':\s+(?=[A-Z])',            # colon followed by capital
                r'—\s+',                     # em-dash (strong clause break)
                r'-\s+',                     # en-dash
                r'\)\s+(?=[A-Z])',          # closing parenthesis followed by capital
                r',\s+(?:(?:but|however|although|while|whereas|so|then|therefore)\b)',
        ]
        
        # Enhanced patterns for important phrases with better coverage
        self.important_patterns = [
            # Deadlines and dates
            r'\b(?:deadline|due date|last date|final date|submission date|closing date)[^.;]*?(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|23:59|11:59)',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}[^.;]{0,50}',
            r'\b(?:submission|submit|deadline|due)[^.;]{0,80}(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2})',
            
            # Submission formats and file structure
            r'\b(?:submission format|file format|file structure|directory structure|folder structure|compressed|zip file|archive)[^.;]*',
            r'\b(?:format|structure|organization)(?:\s+of|\s+for)?\s+(?:submission|files?|assignment|project)[^.;]{0,100}',
            
            # Constraints and negations (keep them together with context)
            r'\b(?:not allowed|cannot|must not|shall not|prohibited|forbidden|banned|restriction)[^.;]{0,80}',
            r'\b(?:except|excluding|does not include|but not|other than|apart from|with the exception)[^.;]{0,100}',
            
            # Conditional statements (keep full condition)
            r'\b(?:if|unless|provided that|only if|in case|on condition that)[^.;]{0,120}',
            
            # Requirements (with context)
            r'\b(?:must|shall|required to|mandatory|need to|have to|should|obligatory)[^.;]{0,100}',
            
            # Times
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|AM|PM|hours?)?[^.;]{0,30}',
            r'\b(?:23:59|11:59|00:00|noon|midnight)[^.;]{0,30}',
            
            # Scores and marks
            r'\b\d+\s*(?:marks?|points?|credits?|grade|percentage|%)[^.;]{0,50}',
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

                if not text or end <= start:
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
                ner_entities.append(text.lower())
                span_id += 1

        # 3. Extract important phrases (regex fallback only)
        if not extracted_positions:
            extracted_positions = set()
            for pattern in self.important_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    phrase = match.group(0).strip()
                    if len(phrase) > 10:  # Minimum phrase length for quality
                        # Determine phrase type
                        phrase_type = self._classify_phrase(phrase)
                        
                        # Check for overlap with existing spans to avoid duplicates
                        overlap = False
                        for start, end in extracted_positions:
                            if not (match.end() <= start or match.start() >= end):
                                overlap = True
                                break
                        
                        if not overlap:
                            spans.append(Span(
                                span_id=span_id,
                                text=phrase,
                                start_char=match.start(),
                                end_char=match.end(),
                                span_type=phrase_type,
                                page=page,
                                section=section,
                                sentence_id=sentence_id,
                                entities=[]
                            ))
                            extracted_positions.add((match.start(), match.end()))
                            span_id += 1
        
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
                    "end_char": span.end_char,
                    "entities": span.entities
                })
                span_id += 1
        
        return all_spans

