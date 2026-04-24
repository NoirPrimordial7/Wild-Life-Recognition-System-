from __future__ import annotations

APP_BG = "#06100d"
SIDEBAR_BG = "#0b1512"
CARD_BG = "#12211d"
CARD_ALT_BG = "#172923"
CARD_BORDER = "#1e3a31"
PREVIEW_BG = "#0a1311"
ACCENT = "#c69a5b"
ACCENT_HOVER = "#d2aa72"
SUCCESS = "#4caf84"
WARNING = "#f0a35e"
ERROR = "#e76f51"
TEXT_PRIMARY = "#f4f6f1"
TEXT_MUTED = "#91a59a"

WINDOW_DEFAULT_SIZE = "1400x850"
WINDOW_MIN_WIDTH = 1100
WINDOW_MIN_HEIGHT = 700
SIDEBAR_WIDTH = 250
RIGHT_PANEL_WIDTH = 390
PREVIEW_ASPECT_RATIO = 16 / 9

IMAGE_FILE_TYPES = [("Images", "*.jpg *.jpeg *.png *.webp *.bmp")]
VIDEO_FILE_TYPES = [("Videos", "*.mp4 *.mov *.mkv *.avi *.3gp *.webm")]

VIDEO_INTERVAL_FRAME_OPTIONS = {
    "Every 8 frames": 8,
    "Every 16 frames": 16,
    "Every 24 frames": 24,
    "Every 30 frames": 30,
    "Every 60 frames": 60,
    "Every 120 frames": 120,
}
VIDEO_INTERVAL_SECONDS_OPTIONS = {
    "Every 0.5 sec": 0.5,
    "Every 1 sec": 1.0,
    "Every 2 sec": 2.0,
    "Every 5 sec": 5.0,
}
VIDEO_INTERVAL_OPTIONS = list(VIDEO_INTERVAL_FRAME_OPTIONS.keys()) + list(VIDEO_INTERVAL_SECONDS_OPTIONS.keys())
PLAYBACK_SPEED_OPTIONS = ["0.25x", "0.5x", "0.75x", "1.0x", "1.25x", "1.5x", "2.0x"]


def apply_dashboard_theme(ctk_module) -> None:
    ctk_module.set_appearance_mode("dark")
    ctk_module.set_default_color_theme("blue")
