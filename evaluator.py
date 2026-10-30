"""evaluator.py - Quality evaluation using ROUGE, semantic similarity, and relative delta."""

from typing import Any, Dict, List, Optional
import numpy as np

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    import evaluate
    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False


class ResponseEvaluator:
    """Evaluates generated LLM responses against reference answers and baseline outputs."""

    def __init__(self):
        self.rouge_scorer = None
        if HAS_ROUGE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL", "rougeLsum"],
                use_stemmer=True,
            )

    def compute_rouge(
        self,
        prediction: str,
        reference: str,
    ) -> Dict[str, float]:
        """Compute standard ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
        if not prediction.strip() or not reference.strip():
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

        if self.rouge_scorer is not None:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                "rouge1": round(scores["rouge1"].fmeasure * 100, 2),
                "rouge2": round(scores["rouge2"].fmeasure * 100, 2),
                "rougeL": round(scores["rougeL"].fmeasure * 100, 2),
                "rougeLsum": round(scores["rougeLsum"].fmeasure * 100, 2),
            }

        # Fallback word-level n-gram overlap if rouge_score is unavailable
        return self._simple_token_overlap(prediction, reference)

    def _simple_token_overlap(self, pred: str, ref: str) -> Dict[str, float]:
        """Fallback token-overlap calculation when rouge_score is not installed."""
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        if not pred_tokens or not ref_tokens:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

        overlap = set(pred_tokens) & set(ref_tokens)
        precision = len(overlap) / len(pred_tokens)
        recall = len(overlap) / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        score = round(f1 * 100, 2)
        return {"rouge1": score, "rouge2": score, "rougeL": score, "rougeLsum": score}

    def compute_token_f1(self, prediction: str, reference: str) -> float:
        """Compute token-level F1 overlap between prediction and reference."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        if not pred_tokens or not ref_tokens:
            return 0.0

        common = {}
        for token in pred_tokens:
            if token in ref_tokens:
                common[token] = min(pred_tokens.count(token), ref_tokens.count(token))
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return round(f1 * 100, 2)

    def evaluate_multiple_choice(
        self,
        questions: List[Dict[str, Any]],
        predict_fn: Any,
    ) -> Dict[str, Any]:
        """Evaluate accuracy on multiple choice reasoning questions across model configurations."""
        correct_count = 0
        details = []

        for q in questions:
            prompt = (
                f"Question: {q['question']}\n"
                f"Options:\n"
                f"A) {q['options']['A']}\n"
                f"B) {q['options']['B']}\n"
                f"C) {q['options']['C']}\n"
                f"D) {q['options']['D']}\n"
                f"Answer with strictly the single capital letter of the correct choice (A, B, C, or D).\n"
                f"Answer:"
            )
            raw_pred = predict_fn(prompt).strip().upper()
            # Extract first occurrence of A, B, C, or D
            extracted = None
            for ch in raw_pred:
                if ch in ("A", "B", "C", "D"):
                    extracted = ch
                    break

            is_correct = (extracted == q["answer"])
            if is_correct:
                correct_count += 1

            details.append({
                "id": q["id"],
                "topic": q["topic"],
                "expected": q["answer"],
                "predicted": extracted or "None",
                "raw_output": raw_pred[:40],
                "correct": is_correct,
            })

        acc = (correct_count / len(questions)) * 100 if questions else 0.0
        return {
            "total_questions": len(questions),
            "correct": correct_count,
            "accuracy_pct": round(acc, 2),
            "details": details,
        }

    def compare_outputs(
        self,
        baseline_output: str,
        optimized_output: str,
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare output quality between baseline model and optimized model.

        If a reference answer is provided, both are evaluated against the ground truth.
        Otherwise, the optimized output is compared directly against the baseline as reference.
        """
        if reference_answer:
            baseline_rouge = self.compute_rouge(baseline_output, reference_answer)
            optimized_rouge = self.compute_rouge(optimized_output, reference_answer)
            target = "ground_truth"
        else:
            # Baseline acts as reference pseudo ground-truth
            baseline_rouge = {"rouge1": 100.0, "rouge2": 100.0, "rougeL": 100.0, "rougeLsum": 100.0}
            optimized_rouge = self.compute_rouge(optimized_output, baseline_output)
            target = "baseline_as_reference"

        # Calculate deltas (Optimized - Baseline)
        delta_rouge1 = round(optimized_rouge["rouge1"] - baseline_rouge["rouge1"], 2)
        delta_rouge2 = round(optimized_rouge["rouge2"] - baseline_rouge["rouge2"], 2)
        delta_rougeL = round(optimized_rouge["rougeL"] - baseline_rouge["rougeL"], 2)

        # Token F1 agreement between baseline and optimized
        agreement_f1 = self.compute_token_f1(optimized_output, baseline_output)

        return {
            "evaluation_target": target,
            "baseline_rouge": baseline_rouge,
            "optimized_rouge": optimized_rouge,
            "delta_rouge1": delta_rouge1,
            "delta_rouge2": delta_rouge2,
            "delta_rougeL": delta_rougeL,
            "output_agreement_f1": agreement_f1,
            "baseline_text_length": len(baseline_output),
            "optimized_text_length": len(optimized_output),
        }


# Global evaluator singleton
_EVALUATOR_INSTANCE: Optional[ResponseEvaluator] = None


def get_evaluator() -> ResponseEvaluator:
    """Return singleton ResponseEvaluator instance."""
    global _EVALUATOR_INSTANCE
    if _EVALUATOR_INSTANCE is None:
        _EVALUATOR_INSTANCE = ResponseEvaluator()
    return _EVALUATOR_INSTANCE

