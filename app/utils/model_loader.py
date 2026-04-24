from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "animal_classification_model_final.h5"


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_default_model_path() -> Path:
    return MODEL_PATH


def _tensorflow_install_message() -> str:
    return (
        "TensorFlow is not available for the current Python environment.\n"
        "Install the project dependencies inside a Python 3.10 virtual environment:\n"
        "  python -m venv venv\n"
        "  venv\\Scripts\\activate\n"
        "  pip install -r requirements.txt\n"
        "If your default `python` is not Python 3.10, use `py -3.10 -m venv venv` on Windows."
    )


def import_tensorflow() -> Any:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    try:
        import tensorflow as tf  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(_tensorflow_install_message()) from exc
    except Exception as exc:
        raise RuntimeError(f"TensorFlow import failed: {exc}\n\n{_tensorflow_install_message()}") from exc
    return tf


def load_model(model_path: str | Path | None = None) -> Any:
    resolved_model_path = Path(model_path) if model_path else MODEL_PATH
    if not resolved_model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {resolved_model_path}")

    tf = import_tensorflow()
    try:
        return tf.keras.models.load_model(resolved_model_path, compile=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {resolved_model_path}: {exc}") from exc
