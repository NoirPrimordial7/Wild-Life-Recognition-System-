from __future__ import annotations

from utils.runtime_bootstrap import maybe_relaunch_with_project_runtime

maybe_relaunch_with_project_runtime()

import argparse
import time
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit("OpenCV is not installed. Run `pip install -r requirements.txt`.") from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit("NumPy is not installed. Run `pip install -r requirements.txt`.") from exc

from utils.animal_info_loader import format_animal_info, get_animal_info, load_animal_info
from utils.class_names import load_class_names
from utils.model_loader import get_project_root, load_model

FRAME_SIZE = (224, 224)
WINDOW_NAME = "Wildlife Detector - Press Q to quit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wildlife classification on webcam or video.")
    parser.add_argument(
        "--source",
        default="webcam",
        help='Use "webcam" for the default camera or provide a video file path.',
    )
    return parser.parse_args()


def resolve_source(source: str) -> int | str:
    normalized = source.strip()
    if normalized.lower() == "webcam":
        return 0
    if normalized.isdigit():
        return int(normalized)

    source_path = Path(normalized)
    if not source_path.is_absolute():
        source_path = (get_project_root() / source_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Video source not found: {source_path}")
    return str(source_path)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    normalized_frame = resized_frame.astype("float32") / 255.0
    return np.expand_dims(normalized_frame, axis=0)


def annotate_frame(frame: np.ndarray, label: str, confidence: float, fps: float) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (470, 105), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    lines = [
        f"Animal: {label}",
        f"Confidence: {confidence:.2%}",
        f"FPS: {fps:.1f}",
    ]

    for index, line in enumerate(lines):
        y_position = 38 + (index * 24)
        cv2.putText(
            frame,
            line,
            (24, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def print_prediction_change(label: str, confidence: float, animal_info_data: dict) -> None:
    print(f"\nPrediction changed: {label} ({confidence:.2%})", flush=True)
    animal_info = get_animal_info(label, animal_info_data)
    print(format_animal_info(animal_info), flush=True)
    print("-" * 60, flush=True)


def main() -> None:
    args = parse_args()
    model = load_model()
    output_count = int(model.output_shape[-1])
    class_names = load_class_names(expected_count=output_count)
    animal_info_data = load_animal_info()

    source = resolve_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    smoothed_fps = 0.0
    last_label: str | None = None

    try:
        while True:
            loop_started = time.perf_counter()
            success, frame = capture.read()
            if not success:
                print("End of stream reached.", flush=True)
                break

            predictions = model.predict(preprocess_frame(frame), verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index])
            predicted_label = (
                class_names[predicted_index]
                if predicted_index < len(class_names)
                else f"Class {predicted_index}"
            )

            if predicted_label != last_label:
                print_prediction_change(predicted_label, confidence, animal_info_data)
                last_label = predicted_label

            instant_fps = 1.0 / max(time.perf_counter() - loop_started, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else (0.85 * smoothed_fps) + (0.15 * instant_fps)

            annotated_frame = annotate_frame(frame.copy(), predicted_label, confidence, smoothed_fps)
            cv2.imshow(WINDOW_NAME, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
