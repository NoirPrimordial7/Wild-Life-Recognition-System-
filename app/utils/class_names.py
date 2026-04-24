from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

from .model_loader import get_project_root

CLASS_NAMES_PATH = get_project_root() / "data" / "class_names.json"
LEGACY_CLASS_SOURCES = [
    get_project_root() / "old_scripts" / "main7.py",
    get_project_root() / "old_scripts" / "main6.py",
]


def _extract_class_names_from_python(path: Path) -> list[str] | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "Classnames" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    return None


def _build_class_names_from_sources() -> list[str]:
    for source_path in LEGACY_CLASS_SOURCES:
        if not source_path.is_file():
            continue
        names = _extract_class_names_from_python(source_path)
        if names:
            return names

    animal_info_path = get_project_root() / "data" / "animal_info.json"
    if animal_info_path.is_file():
        payload = json.loads(animal_info_path.read_text(encoding="utf-8"))
        names = [str(animal.get("name", "")).strip() for animal in payload.get("animals", []) if animal.get("name")]
        if names:
            return names

    raise FileNotFoundError("Could not derive class names from the available project files.")


def create_class_names_file(path: str | Path | None = None) -> Path:
    output_path = Path(path) if path else CLASS_NAMES_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_names = _build_class_names_from_sources()
    payload = {"class_names": class_names}
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_class_names(class_names_path: str | Path | None = None, expected_count: int | None = None) -> list[str]:
    path = Path(class_names_path) if class_names_path else CLASS_NAMES_PATH
    if not path.is_file():
        create_class_names_file(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        class_names = payload.get("class_names", [])
    elif isinstance(payload, list):
        class_names = payload
    else:
        raise ValueError(f"Unsupported class names format in {path}")

    normalized_names = [str(name).strip() for name in class_names if str(name).strip()]
    if expected_count is not None and len(normalized_names) != expected_count:
        warnings.warn(
            f"Model output count ({expected_count}) does not match class count ({len(normalized_names)}).",
            stacklevel=2,
        )

    return normalized_names
