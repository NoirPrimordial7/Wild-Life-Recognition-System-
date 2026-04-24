from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

try:
    import cv2
except ModuleNotFoundError as exc:
    raise RuntimeError("OpenCV is not installed. Run `pip install -r requirements.txt`.") from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise RuntimeError("NumPy is not installed. Run `pip install -r requirements.txt`.") from exc

from .animal_info_loader import get_animal_info, load_animal_info
from .class_names import load_class_names
from .model_loader import load_model


@dataclass(slots=True)
class PredictionCandidate:
    index: int
    label: str
    confidence: float


@dataclass(slots=True)
class PredictionResult:
    predicted_index: int
    predicted_label: str
    confidence: float
    animal_info: dict[str, Any] | None
    top_5_predictions: list[PredictionCandidate]


class PredictionService:
    def __init__(self) -> None:
        self.model = load_model()
        self.output_count = int(self.model.output_shape[-1])
        self.class_names = load_class_names(expected_count=self.output_count)
        self.animal_info_data = load_animal_info()
        self.class_mismatch_warning: str | None = None

        if len(self.class_names) != self.output_count:
            self.class_mismatch_warning = (
                f"Model outputs {self.output_count} classes but class_names.json contains "
                f"{len(self.class_names)} labels."
            )
            warnings.warn(self.class_mismatch_warning, stacklevel=2)

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Received an empty frame for prediction.")

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224), interpolation=cv2.INTER_AREA)
        normalized_frame = resized_frame.astype("float32") / 255.0
        return np.expand_dims(normalized_frame, axis=0)

    def predict_frame(self, frame_bgr: np.ndarray) -> PredictionResult:
        prepared_frame = self.preprocess_frame(frame_bgr)
        predictions = self.model.predict(prepared_frame, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_label = self.class_names[predicted_index] if predicted_index < len(self.class_names) else f"Class {predicted_index}"
        confidence = float(predictions[predicted_index])
        animal_info = get_animal_info(predicted_label, self.animal_info_data)

        top_indices = np.argsort(predictions)[::-1][:5]
        top_5_predictions = [
            PredictionCandidate(
                index=int(index),
                label=self.class_names[int(index)] if int(index) < len(self.class_names) else f"Class {int(index)}",
                confidence=float(predictions[int(index)]),
            )
            for index in top_indices
        ]

        return PredictionResult(
            predicted_index=predicted_index,
            predicted_label=predicted_label,
            confidence=confidence,
            animal_info=animal_info,
            top_5_predictions=top_5_predictions,
        )

    def predict_video_frame(self, frame_bgr: np.ndarray) -> PredictionResult:
        return self.predict_frame(frame_bgr)

    def predict_image_path(self, image_path: str | Path) -> PredictionResult:
        resolved_path = Path(image_path)
        frame = cv2.imread(str(resolved_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image file: {resolved_path}")
        return self.predict_frame(frame)
