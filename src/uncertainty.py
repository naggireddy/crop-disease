"""
OOD / Uncertainty Rejection Gate
Enforces safety boundaries on predictions using confidence thresholding and
predictive Shannon entropy to prevent forced hallucinated diagnoses.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from config import config

class UncertaintySafetyGate:
    """
    Evaluates prediction confidence and entropy against safety thresholds.
    Configurable boundary centered around 80% default target.
    """
    def __init__(
        self,
        confidence_threshold: float = config.OOD_CONFIDENCE_THRESHOLD,
        max_entropy_threshold: float = config.MAX_ENTROPY_THRESHOLD
    ):
        self.confidence_threshold = confidence_threshold
        self.max_entropy_threshold = max_entropy_threshold

    @staticmethod
    def compute_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
        """Computes Shannon entropy: H(p) = -sum(p * log(p))."""
        probs = np.clip(probabilities, eps, 1.0)
        probs = probs / np.sum(probs)
        return float(-np.sum(probs * np.log(probs)))

    def evaluate(
        self,
        probabilities: np.ndarray,
        class_labels: List[str],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluates prediction probabilities.
        Returns accepted diagnosis or uncertain rejection with advice.
        """
        probs = np.asarray(probabilities).flatten()
        top_indices = np.argsort(probs)[::-1][:top_k]

        best_idx = int(top_indices[0])
        best_label = class_labels[best_idx] if best_idx < len(class_labels) else "unknown"
        best_confidence = float(probs[best_idx])
        entropy = self.compute_entropy(probs)

        # Build top-k candidate list
        candidates = []
        for idx in top_indices:
            label = class_labels[int(idx)] if int(idx) < len(class_labels) else "unknown"
            candidates.append({
                "label": label,
                "confidence": round(float(probs[int(idx)]), 4)
            })

        # Uncertainty Decision
        is_confident = best_confidence >= self.confidence_threshold
        is_low_entropy = entropy <= self.max_entropy_threshold
        accepted = is_confident and is_low_entropy

        if accepted:
            crop, disease = best_label.split("__", 1) if "__" in best_label else (best_label, "unknown")
            return {
                "prediction": best_label,
                "crop": crop,
                "disease": disease,
                "confidence": round(best_confidence, 4),
                "entropy": round(entropy, 4),
                "status": "accepted",
                "uncertain": False,
                "candidates": candidates,
                "message": f"Diagnosis accepted with {best_confidence*100:.1f}% confidence."
            }
        else:
            reason = []
            if not is_confident:
                reason.append(f"Confidence ({best_confidence*100:.1f}%) is below safety threshold ({self.confidence_threshold*100:.1f}%).")
            if not is_low_entropy:
                reason.append(f"Prediction entropy ({entropy:.2f}) indicates ambiguous symptom presentation.")

            return {
                "prediction": None,
                "candidate_prediction": best_label,
                "confidence": round(best_confidence, 4),
                "entropy": round(entropy, 4),
                "status": "uncertain",
                "uncertain": True,
                "candidates": candidates,
                "reason": " ".join(reason),
                "message": "Diagnosis uncertain. Please retake photo with clearer lighting and a single leaf filling the frame, or request expert agronomist review."
            }

