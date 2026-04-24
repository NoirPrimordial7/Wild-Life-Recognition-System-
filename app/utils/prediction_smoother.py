from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from .prediction_service import PredictionResult


@dataclass(slots=True)
class SmootherDecision:
    display_result: PredictionResult | None
    accepted_new_result: bool
    reason: str


class PredictionSmoother:
    def __init__(self, window_size: int = 5, minimum_votes: int = 2) -> None:
        self.window_size = max(2, window_size)
        self.minimum_votes = max(2, minimum_votes)
        self._recent_labels: deque[str] = deque(maxlen=self.window_size)
        self._stable_result: PredictionResult | None = None

    @property
    def stable_result(self) -> PredictionResult | None:
        return self._stable_result

    def reset(self) -> None:
        self._recent_labels.clear()
        self._stable_result = None

    def observe(self, result: PredictionResult, threshold: float, enabled: bool = True) -> SmootherDecision:
        if result.confidence < threshold:
            if self._stable_result is None:
                return SmootherDecision(None, False, f"Confidence below threshold ({result.confidence:.1%}).")
            return SmootherDecision(
                self._stable_result,
                False,
                f"Confidence below threshold ({result.confidence:.1%}). Keeping last stable result.",
            )

        if not enabled:
            changed = self._stable_result is None or self._stable_result.predicted_label != result.predicted_label
            self._stable_result = result
            self._recent_labels.clear()
            self._recent_labels.append(result.predicted_label)
            return SmootherDecision(self._stable_result, changed, "Prediction updated without smoothing.")

        self._recent_labels.append(result.predicted_label)

        if self._stable_result is None:
            self._stable_result = result
            return SmootherDecision(self._stable_result, True, "Initial stable prediction accepted.")

        if result.predicted_label == self._stable_result.predicted_label:
            self._stable_result = result
            return SmootherDecision(self._stable_result, False, "Stable prediction refreshed.")

        label_votes = Counter(self._recent_labels)[result.predicted_label]
        if label_votes >= self.minimum_votes:
            self._stable_result = result
            self._recent_labels.clear()
            self._recent_labels.append(result.predicted_label)
            return SmootherDecision(
                self._stable_result,
                True,
                f"Prediction changed after {label_votes} matching samples.",
            )

        return SmootherDecision(
            self._stable_result,
            False,
            f"Holding previous stable prediction while sampling new label ({label_votes} vote(s)).",
        )
