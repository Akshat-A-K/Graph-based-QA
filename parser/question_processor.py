import re
from typing import List, Dict, Tuple
try:
    from transformers import pipeline
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

from .advanced_retrieval import normalize_text

class QuestionProcessor:
    """
    Handles question type classification and sub-question generation.
    Uses transformer models for high-quality decomposition.
    """
    
    def __init__(self, classifier_model="facebook/bart-large-mnli", 
                 generator_model="google/flan-t5-base"):
        self.classification_pipeline = None
        self.generation_pipeline = None
        
        if _TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading question classifier: {classifier_model}")
                self.classification_pipeline = pipeline("zero-shot-classification", model=classifier_model)
                print(f"Loading sub-question generator: {generator_model}")
                self.generation_pipeline = pipeline("text2text-generation", model=generator_model)
            except Exception as e:
                print(f"Error loading transformer models: {e}. Falling back to rule-based.")
        
        self.question_types = [
            "factual", "definitional", "methodological", 
            "comparative", "causal", "process", "temporal", "bridge"
        ]

    def classify_type(self, question: str) -> str:
        """Classify the question into one of the predefined types."""
        if self.classification_pipeline:
            try:
                result = self.classification_pipeline(
                    question, 
                    candidate_labels=self.question_types
                )
                return result["labels"][0]
            except Exception:
                pass
        
        # Rule-based fallback
        q_lower = question.lower()
        if any(w in q_lower for w in ["compare", "difference", "versus", "vs"]):
            return "comparative"
        if any(w in q_lower for w in ["why", "because", "reason"]):
            return "causal"
        if any(w in q_lower for w in ["when", "date", "year", "deadline"]):
            return "temporal"
        if any(w in q_lower for w in ["how", "process", "step", "method"]):
            return "process"
        if any(w in q_lower for w in ["what is", "define", "meaning"]):
            return "definitional"
        
        return "factual"

    def generate_sub_questions(self, question: str, q_type: str) -> List[str]:
        """Generate sub-questions for complex or multi-hop queries."""
        if q_type not in ["comparative", "bridge", "causal", "process"]:
            return [question]
            
        if self.generation_pipeline:
            try:
                prompt = f"Decompose this complex question into 2-3 simpler sub-questions: {question}"
                results = self.generation_pipeline(prompt, max_length=128, num_return_sequences=1)
                generated_text = results[0]["generated_text"]
                # Split by common delimiters
                sub_qs = re.split(r'\?|\n|;|(?<=\d\.)', generated_text)
                sub_qs = [q.strip() + "?" for q in sub_qs if len(q.strip()) > 10]
                if sub_qs:
                    return sub_qs
            except Exception:
                pass
                
        # Rule-based fallback for decomposition (basic)
        if " and " in question and "?" in question:
            parts = question.replace("?", "").split(" and ")
            if len(parts) == 2:
                return [parts[0].strip() + "?", parts[1].strip() + "?"]
        
        return [question]

    def process(self, question: str) -> Dict:
        """Full processing pipeline: classify and decompose."""
        q_type = self.classify_type(question)
        sub_questions = self.generate_sub_questions(question, q_type)
        
        return {
            "original_question": question,
            "type": q_type,
            "sub_questions": sub_questions,
            "is_complex": len(sub_questions) > 1
        }
