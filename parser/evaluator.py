"""
Simple QA Evaluation Metrics
"""

import re
from typing import Dict


class QAEvaluator:
    """Evaluate QA predictions with standard metrics"""
    
    @staticmethod
    def normalize_text(s: str) -> str:
        """Normalize text for comparison"""
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)  # Remove articles
        s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
        s = ' '.join(s.split())  # Remove extra whitespace
        return s
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> float:
        """Binary exact match score"""
        pred_norm = QAEvaluator.normalize_text(prediction)
        gt_norm = QAEvaluator.normalize_text(ground_truth)
        return float(pred_norm == gt_norm)
    
    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """Token-level F1 score"""
        pred_tokens = QAEvaluator.normalize_text(prediction).split()
        gt_tokens = QAEvaluator.normalize_text(ground_truth).split()
        
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        common = set(pred_tokens) & set(gt_tokens)
        num_same = len(common)
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    @staticmethod
    def evaluate(prediction: str, ground_truth: str) -> Dict[str, float]:
        """Get all metrics for a prediction"""
        return {
            'exact_match': QAEvaluator.exact_match(prediction, ground_truth),
            'f1': QAEvaluator.f1_score(prediction, ground_truth)
        }
