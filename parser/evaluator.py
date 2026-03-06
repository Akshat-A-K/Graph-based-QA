"""
QA Evaluation Metrics for Graph-based QA Systems
"""

import re
import string
from typing import Dict, List
from collections import Counter


class QAEvaluator:
    """Evaluate QA predictions using standard metrics"""

    # -------------------------------------------------------
    # TEXT NORMALIZATION
    # -------------------------------------------------------

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for fair comparison"""

        if not text:
            return ""

        text = text.lower()

        text = ''.join(
            ch for ch in text
            if ch not in set(string.punctuation)
        )

        text = re.sub(r'\b(a|an|the)\b', ' ', text)

        text = " ".join(text.split())

        return text

    # -------------------------------------------------------
    # TOKENIZATION
    # -------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize normalized text"""
        return QAEvaluator.normalize_text(text).split()

    # -------------------------------------------------------
    # EXACT MATCH
    # -------------------------------------------------------

    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> float:
        pred = QAEvaluator.normalize_text(prediction)
        gt = QAEvaluator.normalize_text(ground_truth)

        return float(pred == gt)

    # -------------------------------------------------------
    # PRECISION / RECALL / F1
    # -------------------------------------------------------

    @staticmethod
    def precision_recall_f1(prediction: str, ground_truth: str) -> Dict[str, float]:

        pred_tokens = QAEvaluator.tokenize(prediction)
        gt_tokens = QAEvaluator.tokenize(ground_truth)

        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            # Relaxed matching fallback: try substring or numeric equivalence
            if QAEvaluator.substring_match(prediction, ground_truth):
                # partial credit for substring matches
                precision = 0.6
                recall = 0.6
                f1 = 2 * precision * recall / (precision + recall)
                return {"precision": precision, "recall": recall, "f1": f1}

            # Numeric comparison: treat numerically-equal answers as correct
            try:
                import re
                nums_pred = re.findall(r"[-+]?[0-9]*\.?[0-9]+", prediction)
                nums_gt = re.findall(r"[-+]?[0-9]*\.?[0-9]+", ground_truth)
                if nums_pred and nums_gt:
                    # compare first numeric tokens
                    p = float(nums_pred[0])
                    g = float(nums_gt[0])
                    if abs(p - g) <= max(1e-6, 0.01 * max(abs(g), 1.0)):
                        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
            except Exception:
                pass

            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)

        f1 = 2 * precision * recall / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # -------------------------------------------------------
    # SUBSTRING MATCH
    # -------------------------------------------------------

    @staticmethod
    def substring_match(prediction: str, ground_truth: str) -> float:

        pred = QAEvaluator.normalize_text(prediction)
        gt = QAEvaluator.normalize_text(ground_truth)

        if pred in gt or gt in pred:
            return 1.0

        return 0.0

    # -------------------------------------------------------
    # EVIDENCE RECALL@K
    # -------------------------------------------------------

    @staticmethod
    def evidence_recall_at_k(
        evidence_spans: List[str],
        ground_truth: str,
        k: int = 5
    ) -> float:

        if not evidence_spans:
            return 0.0

        gt_norm = QAEvaluator.normalize_text(ground_truth)

        for span in evidence_spans[:k]:
            span_norm = QAEvaluator.normalize_text(span)

            if gt_norm in span_norm or span_norm in gt_norm:
                return 1.0

        return 0.0

    # -------------------------------------------------------
    # GRAPH REASONING DEPTH
    # -------------------------------------------------------

    @staticmethod
    def reasoning_depth(traversal_results=None, expansion_results=None):
        """Compute reasoning depth (number of unique nodes visited).

        Accepts either a single `results` dict (with keys
        `traversal_results` and `expansion_results`) or two separate lists
        passed as `traversal_results` and `expansion_results`.
        """
        nodes = set()

        # If a single dict was passed in as the first argument
        if isinstance(traversal_results, dict) and expansion_results is None:
            results = traversal_results
            for r in results.get("traversal_results", []):
                if isinstance(r, dict) and "node" in r:
                    nodes.add(r["node"])

            for r in results.get("expansion_results", []):
                if isinstance(r, dict) and "node" in r:
                    nodes.add(r["node"])
            return len(nodes)

        # Otherwise treat the two args as iterable lists (may be None)
        for r in traversal_results or []:
            try:
                nodes.add(r["node"])
            except Exception:
                # support simple values
                nodes.add(r)

        for r in expansion_results or []:
            try:
                nodes.add(r["node"])
            except Exception:
                nodes.add(r)

        return len(nodes)

    # -------------------------------------------------------
    # FULL EVALUATION
    # -------------------------------------------------------

    @staticmethod
    def evaluate(
        prediction: str,
        ground_truth: str
    ) -> Dict[str, float]:

        prf = QAEvaluator.precision_recall_f1(prediction, ground_truth)

        return {
            "exact_match": QAEvaluator.exact_match(prediction, ground_truth),
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "substring_match": QAEvaluator.substring_match(prediction, ground_truth)
        }

    # -------------------------------------------------------
    # MULTIPLE GROUND TRUTH SUPPORT
    # -------------------------------------------------------

    @staticmethod
    def evaluate_multiple(
        prediction: str,
        ground_truths: List[str]
    ) -> Dict[str, float]:

        scores = [
            QAEvaluator.evaluate(prediction, gt)
            for gt in ground_truths
        ]

        best_scores = {
            metric: max(score[metric] for score in scores)
            for metric in scores[0]
        }

        return best_scores