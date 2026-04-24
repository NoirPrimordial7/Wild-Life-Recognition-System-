from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class DetectionRecord:
    detected_at: str
    source_type: str
    source_name: str
    label: str
    confidence: float
    frame_number: int | None = None
    video_timestamp: str | None = None
    playback_speed: str | None = None
    ai_interval: str | None = None
    top_5_predictions: list[str] | None = None


def format_video_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class HistoryService:
    def __init__(self, max_items: int = 80) -> None:
        self.max_items = max_items
        self._records: list[DetectionRecord] = []
        self._last_event_key: tuple[str, str] | None = None

    def reset_session(self) -> None:
        self._last_event_key = None

    def clear(self) -> None:
        self._records.clear()
        self._last_event_key = None

    def get_records(self) -> list[DetectionRecord]:
        return list(self._records)

    def add_detection(
        self,
        *,
        source_type: str,
        source_name: str,
        label: str,
        confidence: float,
        frame_number: int | None = None,
        video_timestamp: str | None = None,
        playback_speed: str | None = None,
        ai_interval: str | None = None,
        top_5_predictions: list[str] | None = None,
        add_all: bool = False,
    ) -> bool:
        event_key = (source_type, label)
        if not add_all and self._last_event_key == event_key:
            return False

        record = DetectionRecord(
            detected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source_type=source_type,
            source_name=source_name,
            label=label,
            confidence=confidence,
            frame_number=frame_number,
            video_timestamp=video_timestamp,
            playback_speed=playback_speed,
            ai_interval=ai_interval,
            top_5_predictions=top_5_predictions,
        )
        self._records.insert(0, record)
        self._records = self._records[: self.max_items]
        self._last_event_key = event_key
        return True

    def to_report_text(self) -> str:
        if not self._records:
            return "No detections recorded."

        lines: list[str] = []
        for record in self._records:
            line = (
                f"[{record.detected_at}] {record.source_type} | {record.label} | "
                f"{record.confidence:.2%} | source={record.source_name}"
            )
            if record.frame_number is not None:
                line += f" | frame={record.frame_number}"
            if record.video_timestamp is not None:
                line += f" | time={record.video_timestamp}"
            if record.playback_speed is not None:
                line += f" | speed={record.playback_speed}"
            if record.ai_interval is not None:
                line += f" | interval={record.ai_interval}"
            if record.top_5_predictions:
                line += " | top5=" + "; ".join(record.top_5_predictions)
            lines.append(line)
        return "\n".join(lines)
