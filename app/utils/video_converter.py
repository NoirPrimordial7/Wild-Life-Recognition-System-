from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess

try:
    import cv2
except ModuleNotFoundError as exc:
    raise RuntimeError("OpenCV is not installed. Run `pip install -r requirements.txt`.") from exc

try:
    import imageio_ffmpeg
except ModuleNotFoundError as exc:
    raise RuntimeError("imageio-ffmpeg is not installed. Run `pip install -r requirements.txt`.") from exc

from .model_loader import get_project_root

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".3gp", ".webm"}


@dataclass(slots=True)
class VideoConversionResult:
    success: bool
    output_path: Path | None
    message: str
    converted: bool


def is_video_readable_by_opencv(input_path: str | Path) -> bool:
    video_path = Path(input_path)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return False

        for _ in range(3):
            success, frame = capture.read()
            if success and frame is not None and frame.size > 0:
                return True
        return False
    finally:
        capture.release()


def _build_output_path(input_path: Path, output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{input_path.stem}_converted_{timestamp}.mp4"


def _run_ffmpeg_command(command: list[str]) -> tuple[bool, str]:
    completed_process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = "\n".join(part for part in [completed_process.stdout.strip(), completed_process.stderr.strip()] if part)
    return completed_process.returncode == 0, combined_output


def convert_video_for_opencv(input_path: str | Path, output_dir: str | Path | None = None) -> VideoConversionResult:
    source_path = Path(input_path).expanduser().resolve()
    if not source_path.is_file():
        return VideoConversionResult(False, None, f"Video file not found: {source_path}", False)

    if source_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        return VideoConversionResult(
            False,
            None,
            f"Unsupported video format: {source_path.suffix}. Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}",
            False,
        )

    if is_video_readable_by_opencv(source_path):
        return VideoConversionResult(
            True,
            source_path,
            "Video is already readable by OpenCV. Conversion skipped.",
            False,
        )

    target_directory = Path(output_dir) if output_dir else get_project_root() / "assets" / "videos" / "converted"
    target_directory.mkdir(parents=True, exist_ok=True)
    output_path = _build_output_path(source_path, target_directory)

    ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
    primary_command = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(source_path),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    primary_success, primary_output = _run_ffmpeg_command(primary_command)

    if not primary_success:
        fallback_command = [
            ffmpeg_executable,
            "-y",
            "-i",
            str(source_path),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-r",
            "30",
            "-c:v",
            "mpeg4",
            "-q:v",
            "4",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        fallback_success, fallback_output = _run_ffmpeg_command(fallback_command)
        if not fallback_success:
            message = fallback_output or primary_output or "FFmpeg conversion failed with an unknown error."
            return VideoConversionResult(False, None, message, False)

    if not is_video_readable_by_opencv(output_path):
        return VideoConversionResult(
            False,
            None,
            f"Video was converted but OpenCV still could not read it: {output_path}",
            False,
        )

    return VideoConversionResult(
        True,
        output_path,
        f"Converted video saved to: {output_path}",
        True,
    )
