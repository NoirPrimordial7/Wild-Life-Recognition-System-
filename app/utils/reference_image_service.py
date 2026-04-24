from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw

from .model_loader import get_project_root

SUPPORTED_REFERENCE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DOWNLOAD_REPORT_FILENAME = "download_report.json"


def normalize_class_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


class ReferenceImageService:
    def __init__(self, reference_dir: str | Path | None = None) -> None:
        self.reference_dir = Path(reference_dir) if reference_dir else get_project_root() / "assets" / "animal_reference_images"
        self.reference_dir.mkdir(parents=True, exist_ok=True)

    def build_output_path(self, class_name: str, extension: str) -> Path:
        normalized_extension = extension.lower()
        if normalized_extension not in SUPPORTED_REFERENCE_EXTENSIONS:
            normalized_extension = ".jpg"
        return self.reference_dir / f"{normalize_class_name(class_name)}{normalized_extension}"

    def get_download_report_path(self) -> Path:
        return self.reference_dir / DOWNLOAD_REPORT_FILENAME

    def get_reference_image_path(self, class_name: str) -> Path | None:
        normalized_name = normalize_class_name(class_name)
        candidate_stems = {
            normalized_name,
            class_name.strip().lower(),
            class_name.strip().replace(" ", "_").lower(),
            class_name.strip().replace(" ", "-").lower(),
        }

        for extension in SUPPORTED_REFERENCE_EXTENSIONS:
            for stem in candidate_stems:
                candidate = self.reference_dir / f"{stem}{extension}"
                if candidate.is_file():
                    return candidate

        for candidate in self.reference_dir.iterdir():
            if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_REFERENCE_EXTENSIONS:
                continue
            if normalize_class_name(candidate.stem) == normalized_name:
                return candidate
        return None

    def load_reference_image(self, class_name: str, size: tuple[int, int]) -> Image.Image:
        image_path = self.get_reference_image_path(class_name)
        if image_path and image_path.is_file():
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
            image.thumbnail(size, Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
            canvas = Image.new("RGB", size, "#101c17")
            paste_x = (canvas.width - image.width) // 2
            paste_y = (canvas.height - image.height) // 2
            canvas.paste(image, (paste_x, paste_y))
            return canvas

        return self.create_placeholder_image(class_name, size)

    def create_placeholder_image(self, class_name: str, size: tuple[int, int]) -> Image.Image:
        width, height = size
        image = Image.new("RGB", size, "#0f1b17")
        draw = ImageDraw.Draw(image)

        draw.rounded_rectangle((12, 12, width - 12, height - 12), radius=24, outline="#274338", width=2, fill="#162621")
        draw.text((24, 34), "Reference Image", fill="#d7e4d8")
        draw.text((24, 74), class_name or "Unknown animal", fill="#bb9457")
        draw.text((24, 116), "No reference image yet.", fill="#9fb2a3")
        draw.text((24, 148), "Run image downloader or", fill="#9fb2a3")
        draw.text((24, 176), "add a file to assets/animal_reference_images/", fill="#9fb2a3")
        return image
