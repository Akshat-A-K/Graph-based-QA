import re
from typing import Tuple, Optional

BOOLEAN_PATTERNS = [
    r'^are\s+both\b',
    r'^do\s+both\b',
    r'^did\s+both\b',
    r'^is\s+.{0,60}\s+or\s+.{0,60}\b',
    r'^are\s+both\s+',
    r'^are\s+the\s+',           # "Are the X and Y both..."
    r'^are\s+[a-z]',            # "Are both..." / "Are Catasetum..."
    r'^did\s+',                 # "Did both X and Y..."
    r'^do\s+',                  # "Do both X and Y..."
    r'^is\s+',                  # "Is X or Y..."
    r'\bboth\s+located\b',
    r'\bboth\s+headquartered\b',
    r'\bboth\s+available\b',
    r'\bboth\s+located\b',
    r'\bboth\s+a\s+',
    r'\bboth\s+\w+\?$',         # ends with "both singers?" etc
]

COMPARATIVE_PATTERNS = [
    r'\bmore\b', r'\bless\b', r'\blarger\b', r'\bsmaller\b',
    r'\bfirst\b', r'\bearlier\b', r'\brecent\b', r'\bolder\b',
    r'\bwider\b', r'\bbigger\b', r'\bdiverse\b',
    # NOTE: removed \bstill in\b - that's select_one not comparative
]

COMMON_PROPERTY_PATTERNS = [
    r'\bin common\b',
    r'\bmutual\b',
    r'\bwhat type of\b',
    r'\bwhat kind of\b',
    r'\bwhat genre\b',
    r'\bboth what\b',
    r'\bshared\b',
    r'\bwhich occupation\b',
    r'\bwhat did both\b',
    r'\bwhat do both\b',
    r'^[A-Z].{0,80},\s+are\s+(?:a\s+)?(?:genus|type|kind|form|example)',
    r'are\s+what\s+(?:type|kind|sort|form|genre)',
    r'have\s+which\s+\w+',
    r'share\s+which\s+\w+',
    r'shared\s+.{0,30}\s+and\s+what',
]


def classify_comparison_type(question: str) -> str:
    q = question.lower().strip()

    # Difference
    if re.search(r'\bdifferent\b|\bhow is\b', q):
        return "difference"

    # Common property - check before boolean
    if any(re.search(p, q, re.IGNORECASE) for p in COMMON_PROPERTY_PATTERNS):
        return "common_property"

    # Boolean
    if any(re.search(p, q) for p in BOOLEAN_PATTERNS):
        return "boolean"

    # Comparative ranking
    if any(re.search(p, q) for p in COMPARATIVE_PATTERNS):
        return "comparative"

    return "select_one"


def _strip_question_prefix(question: str) -> str:
    """Remove leading question words before entity extraction."""
    return re.sub(
        r'^(?:Are\s+|Is\s+|Do\s+|Did\s+|Were\s+|Was\s+)'
        r'(?:both\s+|either\s+)?(?:the\s+)?',
        '', question, flags=re.IGNORECASE
    ).strip()


def extract_comparison_entities(question: str) -> Tuple[Optional[str], Optional[str]]:

    # 1. "In between X and Y" / "Between X and Y"
    between_match = re.search(
        r'(?:in\s+)?between\s+(.{2,80}?)\s+and\s+(.{2,80})',
        question, re.IGNORECASE
    )
    if between_match:
        return between_match.group(1).strip(), between_match.group(2).strip()

    # 2. After comma: "X or Y" - most reliable pattern
    comma_split = question.split(',', 1)
    if len(comma_split) == 2:
        entity_part = comma_split[1].strip().rstrip('?')
        or_match = re.search(
            r'^(.{2,80}?)\s+or\s+(.{2,80})$',
            entity_part
        )
        if or_match:
            return or_match.group(1).strip(), or_match.group(2).strip()

    # 3. Semicolon separator: "...; X or Y"
    semi_match = re.search(
        r';\s*(.{2,80}?)\s+or\s+(.{2,80})',
        question
    )
    if semi_match:
        return semi_match.group(1).strip(), semi_match.group(2).strip()

    # 4. "both X and Y" - only when "both" precedes entities (not at end)
    # "Are both X and Y..." or "both X and Y share..."
    and_match = re.search(
        r'\bboth\s+(?:the\s+)?(.{2,80}?)\s+and\s+(?:the\s+)?(.{2,80?})'
        r'(?=\s+(?:are|were|have|do|did|share|is|a\b|an\b|\?))',
        question, re.IGNORECASE
    )
    if and_match:
        return and_match.group(1).strip(), and_match.group(2).strip()

    # 5. Entities BEFORE comma: "X and Y, are/were/have..."
    # Strip question prefix first to avoid capturing "Are the" as part of entity
    stripped = _strip_question_prefix(question)
    pre_comma = re.match(
        r'^(.{2,80}?)\s+and\s+(?:the\s+)?(.{2,80}?)\s*,',
        stripped, re.IGNORECASE
    )
    if pre_comma:
        return pre_comma.group(1).strip(), pre_comma.group(2).strip()

    # 6. "Are the X and the Y both..." - boolean with "the" prefixes
    # Extract around "and" between question start and "both"
    are_and_both = re.search(
        r'(?:Are|Is|Were|Was)\s+(?:the\s+)?(.{2,80}?)\s+and\s+'
        r'(?:the\s+)?(.{2,80?})\s+both\b',
        question, re.IGNORECASE
    )
    if are_and_both:
        return are_and_both.group(1).strip(), are_and_both.group(2).strip()

    # 7. "X and Y" before verb (general)
    and_match2 = re.search(
        r'(?:the\s+)?(.{2,60}?)\s+and\s+(?:the\s+)?(.{2,60}?)'
        r'(?=\s*,|\s+both\b|\s+are\b|\s+have\b|\s+share\b'
        r'|\s+were\b|\s+be\b|\s+is\b)',
        question, re.IGNORECASE
    )
    if and_match2:
        return and_match2.group(1).strip(), and_match2.group(2).strip()

    # 8. "X or Y" anywhere (last resort - most greedy)
    or_match2 = re.search(
        r'(?:the\s+)?(.{2,60}?)\s+or\s+(?:the\s+)?(.{2,60})',
        question
    )
    if or_match2:
        return or_match2.group(1).strip(), or_match2.group(2).strip()

    return None, None