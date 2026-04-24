from __future__ import annotations

from utils.runtime_bootstrap import maybe_relaunch_with_project_runtime

maybe_relaunch_with_project_runtime()

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit("NumPy is not installed. Run `pip install -r requirements.txt`.") from exc

from utils.class_names import load_class_names
from utils.model_loader import load_model


def main() -> None:
    model = load_model()
    class_names = load_class_names(expected_count=int(model.output_shape[-1]))

    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Class names ({len(class_names)}):")
    print(class_names)

    dummy_input = np.random.random((1, 224, 224, 3)).astype("float32")
    predictions = model.predict(dummy_input, verbose=0)[0]
    top_index = int(np.argmax(predictions))
    top_confidence = float(predictions[top_index])

    print(f"Top predicted index: {top_index}")
    print(f"Top confidence: {top_confidence:.6f}")


if __name__ == "__main__":
    main()
