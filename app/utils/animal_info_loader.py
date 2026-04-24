from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .model_loader import get_project_root

ANIMAL_INFO_PATH = get_project_root() / "data" / "animal_info.json"


def normalize_animal_name(name: str) -> str:
    normalized = name.replace("_", " ").replace("-", " ").strip().lower()
    return re.sub(r"\s+", " ", normalized)


def load_animal_info(path: str | Path | None = None) -> dict[str, Any]:
    info_path = Path(path) if path else ANIMAL_INFO_PATH
    if not info_path.is_file():
        raise FileNotFoundError(f"Animal info file not found: {info_path}")

    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Animal info file is not valid JSON: {info_path}") from exc

    animals = payload.get("animals")
    if not isinstance(animals, list):
        raise ValueError(f"Animal info file must contain an 'animals' list: {info_path}")

    return payload


def get_animal_info(predicted_animal: str, animal_info_data: dict[str, Any]) -> dict[str, Any] | None:
    target_name = normalize_animal_name(predicted_animal)
    for animal in animal_info_data.get("animals", []):
        candidate_name = normalize_animal_name(str(animal.get("name", "")))
        if candidate_name == target_name:
            return animal
    return None


def format_animal_info(animal: dict[str, Any] | None) -> str:
    if not animal:
        return "No animal information found."

    animal_name = animal.get("name", "Unknown animal")
    details = animal.get("details", {})
    if not isinstance(details, dict) or not details:
        return f"{animal_name}: No extra details available."

    lines = [f"{animal_name}:"]
    for key, value in details.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)
