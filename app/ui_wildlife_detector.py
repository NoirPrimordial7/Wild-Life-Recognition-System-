from __future__ import annotations

from utils.runtime_bootstrap import maybe_relaunch_with_project_runtime

maybe_relaunch_with_project_runtime()

import os
import subprocess
import threading
import time
from pathlib import Path
import sys
from tkinter import TclError

try:
    from tkinter import filedialog, messagebox
except Exception as exc:
    raise SystemExit(
        "Tkinter could not be imported. Run `python app/check_ui_environment.py` and repair or reinstall "
        "Python 3.10 with Tcl/Tk support if the check fails."
    ) from exc

try:
    import customtkinter as ctk
except ModuleNotFoundError as exc:
    raise SystemExit(
        "CustomTkinter is not installed. Run `pip install -r requirements.txt`, then try "
        "`python app/ui_wildlife_detector.py` again."
    ) from exc
except Exception as exc:
    raise SystemExit(
        "CustomTkinter could not be imported cleanly. Run `python app/check_ui_environment.py` to diagnose "
        "the UI environment, then reinstall dependencies if needed."
    ) from exc

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit("OpenCV is not installed. Run `pip install -r requirements.txt`.") from exc

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError as exc:
    raise SystemExit("Pillow is not installed. Run `pip install -r requirements.txt`.") from exc

from utils.animal_info_loader import format_animal_info
from utils.history_service import DetectionRecord, HistoryService, format_video_timestamp
from utils.model_loader import get_project_root
from utils.prediction_service import PredictionResult, PredictionService
from utils.prediction_smoother import PredictionSmoother, SmootherDecision
from utils.reference_image_service import ReferenceImageService
from utils.video_converter import VideoConversionResult, convert_video_for_opencv, is_video_readable_by_opencv
from ui.theme import (
    ACCENT,
    ACCENT_HOVER,
    APP_BG,
    CARD_ALT_BG,
    CARD_BG,
    CARD_BORDER,
    ERROR,
    IMAGE_FILE_TYPES,
    PLAYBACK_SPEED_OPTIONS,
    PREVIEW_ASPECT_RATIO,
    PREVIEW_BG,
    RIGHT_PANEL_WIDTH,
    SIDEBAR_BG,
    SIDEBAR_WIDTH,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    VIDEO_FILE_TYPES,
    VIDEO_INTERVAL_FRAME_OPTIONS,
    VIDEO_INTERVAL_OPTIONS,
    VIDEO_INTERVAL_SECONDS_OPTIONS,
    WARNING,
    WINDOW_DEFAULT_SIZE,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    apply_dashboard_theme,
)

apply_dashboard_theme(ctk)

RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


class WildlifeDetectorApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Wildlife Detection System")
        self.geometry(WINDOW_DEFAULT_SIZE)
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.configure(fg_color=APP_BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.project_root = get_project_root()
        self.assets_dir = self.project_root / "assets"
        self.reports_dir = self.assets_dir / "reports"
        self.converted_videos_dir = self.assets_dir / "videos" / "converted"
        self.reference_images_dir = self.assets_dir / "animal_reference_images"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.converted_videos_dir.mkdir(parents=True, exist_ok=True)
        self.reference_images_dir.mkdir(parents=True, exist_ok=True)

        self.prediction_service: PredictionService | None = None
        self.service_error: str | None = None
        self.reference_image_service = ReferenceImageService(self.reference_images_dir)
        self.prediction_smoother = PredictionSmoother(window_size=5, minimum_votes=2)
        self.history_service = HistoryService(max_items=120)

        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.video_seek_lock = threading.Lock()
        self.pending_seek_frame: int | None = None
        self.video_prediction_lock = threading.Lock()
        self.video_prediction_running = False

        self.current_source_type = "idle"
        self.player_state = "idle"
        self.current_source_name = "Idle"
        self.current_source_path: str | None = None
        self.current_video_path: Path | None = None
        self.current_frame_number = 0
        self.current_video_seconds = 0.0
        self.current_fps = 0.0
        self.current_video_fps = 0.0
        self.current_video_total_frames = 0
        self.current_video_total_duration_seconds: float | None = None
        self.latest_raw_result: PredictionResult | None = None
        self.latest_display_result: PredictionResult | None = None
        self.preview_image_ref: ctk.CTkImage | None = None
        self.reference_image_ref: ctk.CTkImage | None = None
        self.reference_download_thread: threading.Thread | None = None
        self.timeline_update_internal = False
        self.muted = True
        self.session_id = 0

        self.confidence_threshold_var = ctk.DoubleVar(value=50.0)
        self.webcam_interval_var = ctk.DoubleVar(value=0.9)
        self.video_interval_mode_var = ctk.StringVar(value="Frames")
        self.video_interval_value_var = ctk.StringVar(value="Every 24 frames")
        self.playback_speed_var = ctk.StringVar(value="1.0x")
        self.smoothing_enabled_var = ctk.BooleanVar(value=True)
        self.record_all_history_var = ctk.BooleanVar(value=False)
        self.confidence_threshold = 0.50
        self.webcam_interval_seconds = 0.90
        self.video_interval_mode = "Frames"
        self.video_interval_selection = "Every 24 frames"
        self.playback_speed_label = "1.0x"
        self.playback_speed_multiplier = 1.0
        self.smoothing_enabled = True

        self._build_layout()
        self._on_threshold_changed(self.confidence_threshold_var.get())
        self._on_webcam_interval_changed(self.webcam_interval_var.get())
        self._on_video_interval_mode_changed(self.video_interval_mode_var.get())
        self._on_playback_speed_changed(self.playback_speed_var.get())
        self._reset_panels()
        self._set_status_message("Preparing model and metadata...")
        self._set_preview_note("Load a webcam, image, or video source to begin.")
        self._update_playback_buttons("idle")
        self.after(150, self._load_prediction_service_async)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_center_panel()
        self._build_right_panel()

    def _build_sidebar(self) -> None:
        self.sidebar = ctk.CTkFrame(self, width=SIDEBAR_WIDTH, corner_radius=0, fg_color=SIDEBAR_BG)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(8, weight=1)

        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=22, pady=(26, 18))

        brand_title = ctk.CTkLabel(
            brand_frame,
            text="Wildlife Intelligence",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        brand_title.pack(anchor="w")

        brand_subtitle = ctk.CTkLabel(
            brand_frame,
            text="Professional desktop dashboard for live wildlife classification and review.",
            wraplength=195,
            justify="left",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_MUTED,
        )
        brand_subtitle.pack(anchor="w", pady=(8, 0))

        button_specs = [
            ("Live Webcam", self.start_webcam, ACCENT, ACCENT_HOVER),
            ("Upload Image", self.select_image, CARD_ALT_BG, "#233a31"),
            ("Upload Video", self.select_video, CARD_ALT_BG, "#233a31"),
            ("Convert Mobile Video", self.convert_mobile_video, CARD_ALT_BG, "#233a31"),
            ("Open Assets Folder", self.open_assets_folder, CARD_ALT_BG, "#233a31"),
        ]
        for text, command, color, hover in button_specs:
            button = ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
                height=46,
                corner_radius=16,
                fg_color=color,
                hover_color=hover,
                text_color=TEXT_PRIMARY,
                anchor="w",
                font=ctk.CTkFont(size=14, weight="bold"),
            )
            button.pack(fill="x", padx=18, pady=6)

        status_card = ctk.CTkFrame(self.sidebar, fg_color=CARD_BG, corner_radius=20, border_width=1, border_color=CARD_BORDER)
        status_card.pack(fill="x", padx=18, pady=(26, 10))

        self.model_status_label = ctk.CTkLabel(
            status_card,
            text="Model status: Loading...",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=WARNING,
        )
        self.model_status_label.pack(anchor="w", padx=16, pady=(16, 6))

        self.model_status_detail = ctk.CTkLabel(
            status_card,
            text="Loading model, class names, and animal info.",
            wraplength=190,
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.model_status_detail.pack(anchor="w", padx=16, pady=(0, 16))

        session_card = ctk.CTkFrame(self.sidebar, fg_color=CARD_BG, corner_radius=20, border_width=1, border_color=CARD_BORDER)
        session_card.pack(fill="x", padx=18, pady=(8, 22))

        session_title = ctk.CTkLabel(
            session_card,
            text="Active Session",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        session_title.pack(anchor="w", padx=16, pady=(16, 8))

        self.session_source_label = ctk.CTkLabel(
            session_card,
            text="Source: Idle",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        self.session_source_label.pack(fill="x", padx=16)

        self.session_mode_label = ctk.CTkLabel(
            session_card,
            text="Playback: Stopped",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        self.session_mode_label.pack(fill="x", padx=16, pady=(6, 16))

    def _build_center_panel(self) -> None:
        self.center_panel = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.center_panel.grid(row=0, column=1, sticky="nsew", padx=(24, 18), pady=24)
        self.center_panel.grid_rowconfigure(1, weight=1)
        self.center_panel.grid_columnconfigure(0, weight=1)

        header_frame = ctk.CTkFrame(self.center_panel, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=0)

        hero_title = ctk.CTkLabel(
            header_frame,
            text="Wildlife Monitoring Console",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        hero_title.grid(row=0, column=0, sticky="w")

        hero_subtitle = ctk.CTkLabel(
            header_frame,
            text="Playback uploaded videos smoothly, sample frames periodically, stabilize predictions, and export a detection timeline.",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_MUTED,
        )
        hero_subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        chip_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        chip_frame.grid(row=0, column=1, rowspan=2, sticky="e")
        self.model_chip = self._make_chip(chip_frame, "Model: Loading", WARNING)
        self.source_chip = self._make_chip(chip_frame, "Source: Idle", TEXT_MUTED)
        self.ai_chip = self._make_chip(chip_frame, "AI: Stopped", TEXT_MUTED)
        for index, chip in enumerate((self.model_chip, self.source_chip, self.ai_chip)):
            chip.grid(row=0, column=index, padx=(8, 0), sticky="e")

        self.preview_card = ctk.CTkFrame(
            self.center_panel,
            fg_color=CARD_BG,
            corner_radius=28,
            border_width=1,
            border_color=CARD_BORDER,
        )
        self.preview_card.grid(row=1, column=0, sticky="nsew")
        self.preview_card.grid_rowconfigure(1, weight=1)
        self.preview_card.grid_columnconfigure(0, weight=1)

        preview_header = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        preview_header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 10))
        preview_header.grid_columnconfigure(0, weight=1)

        preview_title = ctk.CTkLabel(
            preview_header,
            text="Live Preview",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        preview_title.grid(row=0, column=0, sticky="w")

        self.preview_note_label = ctk.CTkLabel(
            preview_header,
            text="Idle",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.preview_note_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.preview_label = ctk.CTkLabel(
            self.preview_card,
            text="",
            fg_color=PREVIEW_BG,
            corner_radius=22,
            width=820,
            height=560,
            text_color=TEXT_MUTED,
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 14))

        controls_frame = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        controls_frame.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 10))
        controls_frame.grid_columnconfigure(tuple(range(9)), weight=1)

        self.play_button = self._build_control_button(controls_frame, "Start / Play", self.play_current_video)
        self.pause_button = self._build_control_button(controls_frame, "Pause", self.pause_current_video)
        self.resume_button = self._build_control_button(controls_frame, "Resume", self.resume_current_video)
        self.stop_button = self._build_control_button(controls_frame, "Stop", self.stop_processing)
        self.restart_button = self._build_control_button(controls_frame, "Restart", self.restart_current_video)
        self.seek_back_button = self._build_control_button(controls_frame, "-5s", lambda: self.seek_video_by_seconds(-5))
        self.seek_forward_button = self._build_control_button(controls_frame, "+5s", lambda: self.seek_video_by_seconds(5))
        self.prev_frame_button = self._build_control_button(controls_frame, "Prev Frame", lambda: self.step_video_frame(-1))
        self.next_frame_button = self._build_control_button(controls_frame, "Next Frame", lambda: self.step_video_frame(1))

        for index, button in enumerate(
            [
                self.play_button,
                self.pause_button,
                self.resume_button,
                self.stop_button,
                self.restart_button,
                self.seek_back_button,
                self.seek_forward_button,
                self.prev_frame_button,
                self.next_frame_button,
            ]
        ):
            button.grid(row=0, column=index, padx=4, sticky="ew")

        secondary_controls = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        secondary_controls.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 10))
        secondary_controls.grid_columnconfigure(1, weight=1)
        secondary_controls.grid_columnconfigure(3, weight=2)

        self.speed_down_button = self._build_control_button(secondary_controls, "Speed -", self.decrease_playback_speed)
        self.speed_down_button.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.playback_speed_menu = ctk.CTkOptionMenu(
            secondary_controls,
            variable=self.playback_speed_var,
            values=PLAYBACK_SPEED_OPTIONS,
            fg_color=CARD_ALT_BG,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            command=self._on_playback_speed_changed,
        )
        self.playback_speed_menu.grid(row=0, column=1, padx=6, sticky="ew")

        self.speed_up_button = self._build_control_button(secondary_controls, "Speed +", self.increase_playback_speed)
        self.speed_up_button.grid(row=0, column=2, padx=6, sticky="ew")

        self.video_interval_value_menu = ctk.CTkOptionMenu(
            secondary_controls,
            variable=self.video_interval_value_var,
            values=VIDEO_INTERVAL_OPTIONS,
            fg_color=CARD_ALT_BG,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            command=self._on_video_interval_value_changed,
        )
        self.video_interval_value_menu.grid(row=0, column=3, padx=6, sticky="ew")

        self.mute_button = self._build_control_button(secondary_controls, "Muted", self.toggle_mute_placeholder)
        self.mute_button.grid(row=0, column=4, padx=6, sticky="ew")

        for offset, label in enumerate(["8F", "16F", "24F", "60F"], start=5):
            button = self._build_control_button(secondary_controls, label, lambda value=label: self.set_quick_frame_interval(value))
            button.grid(row=0, column=offset, padx=(6 if offset > 5 else 6, 0), sticky="ew")

        timeline_frame = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        timeline_frame.grid(row=4, column=0, sticky="ew", padx=24, pady=(0, 24))
        timeline_frame.grid_columnconfigure(0, weight=1)

        self.timeline_slider = ctk.CTkSlider(
            timeline_frame,
            from_=0,
            to=1,
            number_of_steps=1,
            progress_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            command=self._on_timeline_seek,
        )
        self.timeline_slider.grid(row=0, column=0, sticky="ew")
        self.timeline_slider.set(0)

        self.timeline_label = ctk.CTkLabel(
            timeline_frame,
            text="--:-- / --:-- | Frame 0 / 0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MUTED,
        )
        self.timeline_label.grid(row=1, column=0, sticky="w", pady=(8, 0))

    def _build_right_panel(self) -> None:
        self.right_panel = ctk.CTkScrollableFrame(self, width=RIGHT_PANEL_WIDTH, fg_color="transparent", corner_radius=0)
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(0, 24), pady=24)
        self.right_panel.grid_columnconfigure(0, weight=1)

        self.result_card = self._make_card(self.right_panel, "Current Result")
        self.result_card.grid(row=0, column=0, sticky="ew", pady=(0, 14))

        self.result_name_label = ctk.CTkLabel(
            self.result_card,
            text="Waiting for prediction",
            wraplength=330,
            justify="left",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        self.result_name_label.pack(anchor="w", padx=18)

        self.result_confidence_label = ctk.CTkLabel(
            self.result_card,
            text="Confidence: --",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        self.result_confidence_label.pack(anchor="w", padx=18, pady=(6, 2))

        self.result_note_label = ctk.CTkLabel(
            self.result_card,
            text="Threshold and smoothing status will appear here.",
            wraplength=330,
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.result_note_label.pack(anchor="w", padx=18, pady=(0, 14))

        self.top_prediction_rows: list[tuple[ctk.CTkLabel, ctk.CTkProgressBar, ctk.CTkLabel]] = []
        for _ in range(5):
            row = ctk.CTkFrame(self.result_card, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=4)
            row.grid_columnconfigure(0, weight=1)
            row.grid_columnconfigure(1, weight=0)

            label = ctk.CTkLabel(row, text="--", anchor="w", text_color=TEXT_PRIMARY, font=ctk.CTkFont(size=13, weight="bold"))
            label.grid(row=0, column=0, sticky="w")
            percent = ctk.CTkLabel(row, text="0%", text_color=TEXT_MUTED, font=ctk.CTkFont(size=12))
            percent.grid(row=0, column=1, sticky="e")

            progress = ctk.CTkProgressBar(row, height=12, corner_radius=7, progress_color=ACCENT, fg_color="#223830")
            progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
            progress.set(0)

            self.top_prediction_rows.append((label, progress, percent))

        self.reference_card = self._make_card(self.right_panel, "Reference Image")
        self.reference_card.grid(row=1, column=0, sticky="ew", pady=(0, 14))

        self.reference_image_label = ctk.CTkLabel(
            self.reference_card,
            text="",
            fg_color=PREVIEW_BG,
            corner_radius=18,
            width=330,
            height=240,
        )
        self.reference_image_label.pack(fill="both", padx=18, pady=(0, 12))

        self.reference_caption_label = ctk.CTkLabel(
            self.reference_card,
            text="No reference image yet. Run image downloader.",
            wraplength=330,
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.reference_caption_label.pack(anchor="w", padx=18, pady=(0, 16))

        self.info_card = self._make_card(self.right_panel, "Animal Details")
        self.info_card.grid(row=2, column=0, sticky="ew", pady=(0, 14))

        self.info_textbox = ctk.CTkTextbox(
            self.info_card,
            height=170,
            corner_radius=16,
            fg_color=CARD_ALT_BG,
            text_color=TEXT_PRIMARY,
            border_spacing=12,
            wrap="word",
        )
        self.info_textbox.pack(fill="both", expand=True, padx=18, pady=(0, 16))

        self.status_card = self._make_card(self.right_panel, "Stream Status")
        self.status_card.grid(row=3, column=0, sticky="ew", pady=(0, 14))

        self.status_message_label = ctk.CTkLabel(
            self.status_card,
            text="Ready.",
            wraplength=330,
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.status_message_label.pack(anchor="w", padx=18, pady=(0, 12))

        self.source_value_label = self._build_status_value(self.status_card, "Source: Idle")
        self.fps_value_label = self._build_status_value(self.status_card, "Processing FPS: 0.0")
        self.video_fps_value_label = self._build_status_value(self.status_card, "Video FPS: --")
        self.playback_speed_value_label = self._build_status_value(self.status_card, "Playback Speed: 1.0x")
        self.frame_value_label = self._build_status_value(self.status_card, "Frame: 0")
        self.timestamp_value_label = self._build_status_value(self.status_card, "Time: --:--")
        self.interval_value_label = self._build_status_value(self.status_card, "Sampling: Every 24 frames")

        self.settings_card = self._make_card(self.right_panel, "Controls & Settings")
        self.settings_card.grid(row=4, column=0, sticky="ew", pady=(0, 14))

        self.threshold_label = ctk.CTkLabel(
            self.settings_card,
            text="Confidence Threshold: 50%",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        self.threshold_label.pack(anchor="w", padx=18)

        self.threshold_slider = ctk.CTkSlider(
            self.settings_card,
            from_=20,
            to=95,
            number_of_steps=75,
            variable=self.confidence_threshold_var,
            progress_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            command=self._on_threshold_changed,
        )
        self.threshold_slider.pack(fill="x", padx=18, pady=(8, 14))

        self.webcam_interval_label = ctk.CTkLabel(
            self.settings_card,
            text="Webcam Prediction Interval: 0.90 s",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        self.webcam_interval_label.pack(anchor="w", padx=18)

        self.webcam_interval_slider = ctk.CTkSlider(
            self.settings_card,
            from_=0.5,
            to=2.0,
            number_of_steps=30,
            variable=self.webcam_interval_var,
            progress_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            command=self._on_webcam_interval_changed,
        )
        self.webcam_interval_slider.pack(fill="x", padx=18, pady=(8, 14))

        mode_row = ctk.CTkFrame(self.settings_card, fg_color="transparent")
        mode_row.pack(fill="x", padx=18, pady=(0, 10))

        mode_label = ctk.CTkLabel(
            mode_row,
            text="Identify animal every",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        mode_label.pack(anchor="w")

        self.video_interval_mode_control = ctk.CTkSegmentedButton(
            mode_row,
            values=["Frames", "Seconds"],
            variable=self.video_interval_mode_var,
            command=self._on_video_interval_mode_changed,
            fg_color=CARD_ALT_BG,
            selected_color=ACCENT,
            selected_hover_color=ACCENT_HOVER,
        )
        self.video_interval_mode_control.pack(fill="x", pady=(8, 0))

        self.settings_video_interval_value_menu = ctk.CTkOptionMenu(
            self.settings_card,
            variable=self.video_interval_value_var,
            values=VIDEO_INTERVAL_OPTIONS,
            fg_color=CARD_ALT_BG,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            command=self._on_video_interval_value_changed,
        )
        self.settings_video_interval_value_menu.pack(fill="x", padx=18, pady=(0, 14))

        self.playback_speed_label_widget = ctk.CTkLabel(
            self.settings_card,
            text="Playback Speed (video only)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        self.playback_speed_label_widget.pack(anchor="w", padx=18)

        self.settings_playback_speed_menu = ctk.CTkOptionMenu(
            self.settings_card,
            variable=self.playback_speed_var,
            values=PLAYBACK_SPEED_OPTIONS,
            fg_color=CARD_ALT_BG,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            command=self._on_playback_speed_changed,
        )
        self.settings_playback_speed_menu.pack(fill="x", padx=18, pady=(8, 14))

        self.smoothing_switch = ctk.CTkSwitch(
            self.settings_card,
            text="Enable prediction smoothing",
            variable=self.smoothing_enabled_var,
            command=self._on_smoothing_changed,
            progress_color=ACCENT,
            button_color=TEXT_PRIMARY,
            button_hover_color=TEXT_PRIMARY,
            text_color=TEXT_PRIMARY,
        )
        self.smoothing_switch.pack(anchor="w", padx=18, pady=6)

        self.history_switch = ctk.CTkSwitch(
            self.settings_card,
            text="Record every sampled detection",
            variable=self.record_all_history_var,
            progress_color=ACCENT,
            button_color=TEXT_PRIMARY,
            button_hover_color=TEXT_PRIMARY,
            text_color=TEXT_PRIMARY,
        )
        self.history_switch.pack(anchor="w", padx=18, pady=6)

        settings_button_row = ctk.CTkFrame(self.settings_card, fg_color="transparent")
        settings_button_row.pack(fill="x", padx=18, pady=(12, 16))
        settings_button_row.grid_columnconfigure((0, 1), weight=1)

        save_button = ctk.CTkButton(
            settings_button_row,
            text="Save Report",
            command=self.save_result_report,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#20150b",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=14,
            height=40,
        )
        save_button.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        clear_button = ctk.CTkButton(
            settings_button_row,
            text="Clear History",
            command=self.clear_history,
            fg_color=CARD_ALT_BG,
            hover_color="#233a31",
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=14,
            height=40,
        )
        clear_button.grid(row=0, column=1, padx=(6, 0), sticky="ew")

        self.download_images_button = ctk.CTkButton(
            self.settings_card,
            text="Download Animal Images",
            command=self.download_animal_images,
            fg_color=CARD_ALT_BG,
            hover_color="#233a31",
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=14,
            height=40,
        )
        self.download_images_button.pack(fill="x", padx=18, pady=(0, 16))

        self.history_card = self._make_card(self.right_panel, "Detection Timeline")
        self.history_card.grid(row=5, column=0, sticky="ew", pady=(0, 8))

        self.history_list_frame = ctk.CTkScrollableFrame(self.history_card, height=260, fg_color=CARD_ALT_BG, corner_radius=16)
        self.history_list_frame.pack(fill="both", expand=True, padx=18, pady=(0, 18))

    def _make_card(self, parent, title: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=24, border_width=1, border_color=CARD_BORDER)
        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        title_label.pack(anchor="w", padx=18, pady=(18, 12))
        return card

    def _build_status_value(self, parent, text: str) -> ctk.CTkLabel:
        label = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        label.pack(fill="x", padx=18, pady=3)
        return label

    def _build_control_button(self, parent, text: str, command) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            height=34,
            corner_radius=10,
            fg_color=CARD_ALT_BG,
            hover_color="#233a31",
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=12, weight="bold"),
        )

    def _make_chip(self, parent, text: str, color: str) -> ctk.CTkLabel:
        return ctk.CTkLabel(
            parent,
            text=text,
            fg_color=CARD_ALT_BG,
            text_color=color,
            corner_radius=12,
            padx=10,
            pady=5,
            font=ctk.CTkFont(size=12, weight="bold"),
        )

    def _update_status_chips(self) -> None:
        model_text = "Model: Loaded" if self.prediction_service is not None else "Model: Not Loaded"
        model_color = SUCCESS if self.prediction_service is not None else WARNING
        source_label = self.current_source_type.title() if self.current_source_type != "idle" else "Idle"

        if self.player_state == "playing" or self.player_state == "webcam_running":
            ai_text, ai_color = "AI: Running", SUCCESS
        elif self.player_state == "paused":
            ai_text, ai_color = "AI: Paused", WARNING
        else:
            ai_text, ai_color = "AI: Stopped", TEXT_MUTED

        self.model_chip.configure(text=model_text, text_color=model_color)
        self.source_chip.configure(text=f"Source: {source_label}", text_color=TEXT_PRIMARY)
        self.ai_chip.configure(text=ai_text, text_color=ai_color)

    def _load_prediction_service_async(self) -> None:
        threading.Thread(target=self._load_prediction_service_worker, daemon=True).start()

    def _load_prediction_service_worker(self) -> None:
        try:
            service = PredictionService()
        except Exception as exc:
            self.after(0, lambda: self._handle_service_error(str(exc)))
            return
        self.after(0, lambda: self._handle_service_loaded(service))

    def _handle_service_loaded(self, service: PredictionService) -> None:
        self.prediction_service = service
        self.service_error = None
        self.model_status_label.configure(text="Model status: Loaded", text_color=SUCCESS)

        detail = "Model, class names, and animal details are ready."
        if service.class_mismatch_warning:
            detail = service.class_mismatch_warning
            self.model_status_label.configure(text="Model status: Warning", text_color=WARNING)
        self.model_status_detail.configure(text=detail)
        self._set_status_message("Model loaded successfully. Choose a source from the sidebar.")
        self._update_status_chips()

    def _handle_service_error(self, error_message: str) -> None:
        self.prediction_service = None
        self.service_error = error_message
        self.model_status_label.configure(text="Model status: Not Loaded", text_color=ERROR)
        self.model_status_detail.configure(text=error_message)
        self._set_status_message("The model could not be loaded.")
        self._update_status_chips()
        messagebox.showerror("Model Load Error", error_message)

    def _ensure_service_ready(self) -> bool:
        if self.prediction_service is not None:
            return True
        if self.service_error:
            messagebox.showerror("Model Not Ready", self.service_error)
            return False
        messagebox.showinfo("Model Loading", "The model is still loading. Please wait a moment and try again.")
        return False

    def _prepare_new_session(self, source_type: str, source_name: str, source_path: str | None = None) -> None:
        self.stop_processing(update_status=False)
        self.stop_event.clear()
        self.pause_event.clear()
        self.session_id += 1
        self.current_source_type = source_type
        self.player_state = source_type if source_type in {"idle", "image"} else f"{source_type}_loaded"
        self.current_source_name = source_name
        self.current_source_path = source_path
        self.current_frame_number = 0
        self.current_video_seconds = 0.0
        self.current_fps = 0.0
        self.current_video_fps = 0.0
        self.current_video_total_frames = 0
        self.current_video_total_duration_seconds = None
        with self.video_seek_lock:
            self.pending_seek_frame = None
        with self.video_prediction_lock:
            self.video_prediction_running = False
        self.latest_raw_result = None
        self.latest_display_result = None
        self.prediction_smoother.reset()
        self.history_service.reset_session()
        self.session_source_label.configure(text=f"Source: {source_name}")
        self._update_status_chips()

    def start_webcam(self) -> None:
        if not self._ensure_service_ready():
            return

        self.current_video_path = None
        self._prepare_new_session("webcam", "Webcam", "camera://0")
        self.player_state = "webcam_running"
        self._set_preview_note("Opening webcam feed...")
        self._set_status_message("Opening webcam feed...")
        self._update_playback_buttons("webcam_playing")

        self.worker_thread = threading.Thread(target=self._webcam_loop, daemon=True)
        self.worker_thread.start()

    def _webcam_loop(self) -> None:
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            self.after(0, lambda: self._handle_stream_error("Camera Error", "Could not open webcam (index 0)."))
            return

        last_prediction_clock = 0.0
        smoothed_fps = 0.0

        try:
            while not self.stop_event.is_set():
                loop_started = time.perf_counter()
                success, frame = capture.read()
                if not success:
                    self.after(0, lambda: self._set_status_message("Webcam stream stopped unexpectedly."))
                    break

                self.current_frame_number += 1
                now = time.perf_counter()
                sampled_result: PredictionResult | None = None
                decision: SmootherDecision | None = None

                if self.prediction_service and (last_prediction_clock == 0.0 or now - last_prediction_clock >= self._get_webcam_interval()):
                    sampled_result = self.prediction_service.predict_video_frame(frame)
                    decision = self.prediction_smoother.observe(
                        sampled_result,
                        threshold=self._get_confidence_threshold(),
                        enabled=self.smoothing_enabled,
                    )
                    last_prediction_clock = now

                instant_fps = 1.0 / max(time.perf_counter() - loop_started, 1e-6)
                smoothed_fps = instant_fps if smoothed_fps == 0.0 else (0.85 * smoothed_fps) + (0.15 * instant_fps)

                panel_result = self._choose_panel_result(sampled_result, decision)
                preview_frame = self._compose_preview_frame(
                    frame.copy(),
                    panel_result,
                    line_two=f"Confidence: {panel_result.confidence:.1%}" if panel_result else "Sampling webcam feed...",
                    meta=f"Source: Webcam  |  FPS: {smoothed_fps:.1f}",
                )
                status_note = decision.reason if decision else f"Sampling webcam every {self._get_webcam_interval():.2f} seconds."

                self.after(
                    0,
                    lambda pf=preview_frame, sampled=sampled_result, decision=decision, panel=panel_result, fps=smoothed_fps, note=status_note: self._handle_stream_update(
                        preview_frame=pf,
                        sampled_result=sampled,
                        decision=decision,
                        panel_result=panel,
                        fps=fps,
                        frame_number=self.current_frame_number,
                        video_seconds=None,
                        status_note=note,
                    ),
                )
        except Exception as exc:
            self.after(0, lambda: self._handle_stream_error("Webcam Error", str(exc)))
        finally:
            capture.release()

    def select_image(self) -> None:
        if not self._ensure_service_ready():
            return

        image_path = filedialog.askopenfilename(title="Select an image", filetypes=IMAGE_FILE_TYPES)
        if not image_path:
            return

        self.current_video_path = None
        self._prepare_new_session("image", f"Image: {Path(image_path).name}", image_path)
        self.player_state = "image_loaded"
        self._set_preview_note("Loading image and running one-time prediction...")
        self._set_status_message("Processing image...")
        self._update_playback_buttons("image")

        self.worker_thread = threading.Thread(target=self._image_prediction_worker, args=(Path(image_path),), daemon=True)
        self.worker_thread.start()

    def _image_prediction_worker(self, image_path: Path) -> None:
        try:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise FileNotFoundError(f"Could not read image file: {image_path}")
            sampled_result = self.prediction_service.predict_image_path(image_path) if self.prediction_service else None
            panel_result = sampled_result
            preview_frame = self._compose_preview_frame(
                frame.copy(),
                panel_result,
                line_two=f"Confidence: {panel_result.confidence:.1%}" if panel_result else "Prediction unavailable",
                meta="Single image classification",
            )
            self.after(
                0,
                lambda pf=preview_frame, result=sampled_result: self._handle_stream_update(
                    preview_frame=pf,
                    sampled_result=result,
                    decision=None,
                    panel_result=panel_result,
                    fps=0.0,
                    frame_number=1,
                    video_seconds=None,
                    status_note="Image classified successfully.",
                    force_history=True,
                ),
            )
        except Exception as exc:
            self.after(0, lambda: self._handle_stream_error("Image Error", str(exc)))

    def select_video(self) -> None:
        if not self._ensure_service_ready():
            return

        video_path = filedialog.askopenfilename(title="Select a video", filetypes=VIDEO_FILE_TYPES)
        if not video_path:
            return

        selected_path = Path(video_path)
        if not is_video_readable_by_opencv(selected_path):
            should_convert = messagebox.askyesno(
                "Convert Video",
                "OpenCV could not read this video directly. Convert Mobile Video may fix this file.\n\nDo you want to convert it to a compatible MP4 now?",
            )
            if should_convert:
                self._prepare_new_session("video", f"Video: {selected_path.name}", str(selected_path))
                self.current_video_path = selected_path
                self._set_preview_note("Converting selected video...")
                self._set_status_message("Converting selected video...")
                self._update_playback_buttons("video_loading")
                self.worker_thread = threading.Thread(target=self._conversion_worker, args=(selected_path, True), daemon=True)
                self.worker_thread.start()
            return

        self._load_video_for_manual_playback(selected_path)

    def _load_video_for_manual_playback(self, video_path: Path) -> None:
        if not self._ensure_service_ready():
            return

        self._prepare_new_session("video", f"Video: {video_path.name}", str(video_path))
        self.current_video_path = video_path
        self.player_state = "video_loaded"

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            messagebox.showerror(
                "Video Error",
                f"Could not open video:\n{video_path}\n\nTry Convert Mobile Video if this came from a phone.",
            )
            self._update_playback_buttons("idle")
            return

        try:
            capture_fps = capture.get(cv2.CAP_PROP_FPS)
            self.current_video_fps = capture_fps if capture_fps and capture_fps > 0 else 24.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.current_video_total_frames = total_frames if total_frames > 0 else 0
            self.current_video_total_duration_seconds = (
                self.current_video_total_frames / self.current_video_fps
                if self.current_video_total_frames > 0 and self.current_video_fps > 0
                else None
            )
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError("Could not read the first video frame.")

            self.current_frame_number = 1
            self.current_video_seconds = 0.0
            self._configure_timeline()
            preview_frame = self._compose_preview_frame(
                frame.copy(),
                None,
                line_two="Video loaded. Press Start to analyze.",
                meta=f"{self._format_time_progress(0.0)}  |  Frame {self._format_frame_progress(1)}",
                status_line="Video loaded. Press Start to analyze.",
            )
            self._render_preview_frame(preview_frame)
            self._update_status_panel("Video loaded. Press Start to analyze.", 0.0, 1, 0.0)
            self._set_preview_note(f"Loaded {video_path.name}. Press Start / Play to begin AI analysis.")
            self._update_playback_buttons("video_loaded")
        except Exception as exc:
            messagebox.showerror(
                "Video Error",
                f"Could not read video:\n{video_path}\n\n{exc}\n\nTry Convert Mobile Video if this came from a phone.",
            )
            self._set_status_message("Video could not be read.")
            self._update_playback_buttons("idle")
        finally:
            capture.release()

    def _start_video_playback(self, video_path: Path, start_frame: int | None = None) -> None:
        if not self._ensure_service_ready():
            return

        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_processing(update_status=False)
            self.stop_event.clear()

        if self.current_source_type != "video":
            self._prepare_new_session("video", f"Video: {video_path.name}", str(video_path))
        else:
            self.stop_event.clear()
            self.pause_event.clear()
        self.current_video_path = video_path
        self.player_state = "playing"
        if start_frame is not None:
            self.current_frame_number = max(0, start_frame)
        self._set_preview_note(f"Playing {video_path.name}")
        self._set_status_message(f"Playing {video_path.name}")
        self._update_playback_buttons("video_playing")

        self.worker_thread = threading.Thread(target=self._video_loop, args=(video_path, self.current_frame_number), daemon=True)
        self.worker_thread.start()

    def _video_loop(self, video_path: Path, start_frame: int = 0) -> None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            self.after(0, lambda: self._handle_stream_error("Video Error", f"Could not open video: {video_path}"))
            return

        capture_fps = capture.get(cv2.CAP_PROP_FPS)
        playback_fps = capture_fps if capture_fps and capture_fps > 0 else 24.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.current_video_fps = playback_fps
        self.current_video_total_frames = total_frames if total_frames > 0 else 0
        self.current_video_total_duration_seconds = (
            self.current_video_total_frames / playback_fps if self.current_video_total_frames > 0 and playback_fps > 0 else None
        )
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(start_frame, max(self.current_video_total_frames - 1, 0)))
        base_frame_delay = 1.0 / max(min(playback_fps, 60.0), 1.0)
        last_sampled_frame = -10_000
        last_sampled_seconds = -10_000.0
        smoothed_loop_fps = 0.0

        try:
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    self.after(0, lambda: self._update_playback_buttons("video_paused"))
                    time.sleep(0.1)
                    continue

                with self.video_seek_lock:
                    seek_frame = self.pending_seek_frame
                    self.pending_seek_frame = None
                if seek_frame is not None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, seek_frame))
                    last_sampled_frame = -10_000
                    last_sampled_seconds = -10_000.0

                self.after(0, lambda: self._update_playback_buttons("video_playing"))
                loop_started = time.perf_counter()
                success, frame = capture.read()
                if not success:
                    self.player_state = "stopped"
                    self.after(0, lambda: self._set_status_message(f"Finished playing {video_path.name}."))
                    self.after(0, lambda: self._set_preview_note(f"Finished playing {video_path.name}."))
                    self.after(0, lambda: self._update_playback_buttons("video_stopped"))
                    break

                decoded_frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                self.current_frame_number = decoded_frame_number if decoded_frame_number > 0 else self.current_frame_number + 1
                capture_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
                video_seconds = capture_msec / 1000.0 if capture_msec and capture_msec > 0 else 0.0
                if video_seconds <= 0 and playback_fps > 0:
                    video_seconds = max(0.0, (self.current_frame_number - 1) / playback_fps)
                self.current_video_seconds = video_seconds

                sampled_result: PredictionResult | None = None
                decision: SmootherDecision | None = None
                if self.prediction_service and self._should_sample_video(self.current_frame_number, video_seconds, last_sampled_frame, last_sampled_seconds):
                    if self._start_video_prediction(frame.copy(), self.current_frame_number, video_seconds):
                        last_sampled_frame = self.current_frame_number
                        last_sampled_seconds = video_seconds
                    status_note = (
                        f"AI check queued. Next check {self._describe_next_ai_check(self.current_frame_number, video_seconds)}."
                    )
                else:
                    status_note = f"Next AI check {self._describe_next_ai_check(self.current_frame_number, video_seconds)}."

                instant_fps = 1.0 / max(time.perf_counter() - loop_started, 1e-6)
                smoothed_loop_fps = instant_fps if smoothed_loop_fps == 0.0 else (0.85 * smoothed_loop_fps) + (0.15 * instant_fps)

                panel_result = self._choose_panel_result(sampled_result, decision)
                speed_label = self._get_playback_speed_label()
                preview_frame = self._compose_preview_frame(
                    frame.copy(),
                    panel_result,
                    line_two=f"Confidence: {panel_result.confidence:.1%}" if panel_result else "Playing video...",
                    meta=f"{speed_label}  |  AI: {self._describe_video_sampling_rule()}",
                )

                self.after(
                    0,
                    lambda pf=preview_frame, sampled=sampled_result, decision=decision, panel=panel_result, fps=smoothed_loop_fps, seconds=video_seconds, note=status_note: self._handle_stream_update(
                        preview_frame=pf,
                        sampled_result=sampled,
                        decision=decision,
                        panel_result=panel,
                        fps=fps,
                        frame_number=self.current_frame_number,
                        video_seconds=seconds,
                        status_note=note,
                    ),
                )

                adjusted_frame_delay = base_frame_delay / self._get_playback_speed_multiplier()
                sleep_time = adjusted_frame_delay - (time.perf_counter() - loop_started)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as exc:
            self.after(0, lambda: self._handle_stream_error("Video Error", str(exc)))
        finally:
            capture.release()

    def play_current_video(self) -> None:
        if self.current_video_path is None:
            messagebox.showinfo("No Video Loaded", "Load a video first to use playback controls.")
            return
        if self.current_source_type != "video":
            messagebox.showinfo("Video Required", "Upload a video before using video playback controls.")
            return
        if self.player_state == "paused":
            self.resume_current_video()
            return
        self._start_video_playback(self.current_video_path)

    def pause_current_video(self) -> None:
        if self.current_source_type != "video" or self.player_state != "playing":
            return
        self.pause_event.set()
        self.player_state = "paused"
        self._set_preview_note("Video paused.")
        self._set_status_message("Video paused.")
        self._render_video_frame_at(self.current_frame_number, overlay_text="Paused")
        self._update_playback_buttons("video_paused")

    def resume_current_video(self) -> None:
        if self.current_source_type != "video" or self.player_state != "paused":
            return
        self.pause_event.clear()
        self.player_state = "playing"
        self._set_preview_note("Video resumed.")
        self._set_status_message("Video resumed.")
        self._update_playback_buttons("video_playing")

    def restart_current_video(self) -> None:
        if self.current_video_path is None:
            messagebox.showinfo("No Video Loaded", "Load a video first to restart playback.")
            return
        self.stop_processing(update_status=False)
        self._render_video_frame_at(0, overlay_text="Video restarted. Press Start to analyze.")
        self.player_state = "video_loaded"
        self._set_preview_note("Restarted to the first frame. Press Start / Play to begin.")
        self._set_status_message("Restarted to first frame.")
        self._update_playback_buttons("video_loaded")

    def seek_video_by_seconds(self, seconds_delta: float) -> None:
        if self.current_video_path is None or self.current_source_type != "video":
            return
        fps = self.current_video_fps if self.current_video_fps > 0 else 24.0
        frame_delta = int(round(seconds_delta * fps))
        self._seek_video_to_frame(self.current_frame_number + frame_delta)

    def step_video_frame(self, frame_delta: int) -> None:
        if self.current_video_path is None or self.current_source_type != "video":
            return
        was_playing = self.player_state == "playing"
        if was_playing:
            self.pause_current_video()
        self._seek_video_to_frame(self.current_frame_number + frame_delta)

    def _seek_video_to_frame(self, frame_number: int) -> None:
        if self.current_video_path is None:
            return
        target_frame = max(0, frame_number)
        if self.current_video_total_frames > 0:
            target_frame = min(target_frame, self.current_video_total_frames - 1)

        if self.player_state == "playing":
            with self.video_seek_lock:
                self.pending_seek_frame = target_frame
            self.current_frame_number = target_frame
            self._update_timeline(target_frame, self._seconds_for_frame(target_frame))
            return

        self._render_video_frame_at(target_frame)

    def _render_video_frame_at(self, frame_number: int, overlay_text: str | None = None) -> None:
        if self.current_video_path is None:
            return

        capture = cv2.VideoCapture(str(self.current_video_path))
        if not capture.isOpened():
            messagebox.showerror("Video Error", f"Could not open video:\n{self.current_video_path}")
            return

        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError("Could not read frame at the selected position.")
            decoded_frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or frame_number)
            video_seconds = self._seconds_for_frame(decoded_frame_number)
            self.current_frame_number = decoded_frame_number
            self.current_video_seconds = video_seconds
            panel_result = self.latest_display_result or self.latest_raw_result
            preview_frame = self._compose_preview_frame(
                frame.copy(),
                panel_result,
                line_two="Seeking preview. Press Start to analyze.",
                meta=f"{self._format_time_progress(video_seconds)}  |  Frame {self._format_frame_progress(decoded_frame_number)}",
                status_line=overlay_text,
            )
            self._render_preview_frame(preview_frame)
            self._update_status_panel("Seeked video preview.", self.current_fps, decoded_frame_number, video_seconds)
            self._update_timeline(decoded_frame_number, video_seconds)
            state_map = {"video_loaded": "video_loaded", "paused": "video_paused", "stopped": "video_stopped"}
            self._update_playback_buttons(state_map.get(self.player_state, "video_loaded"))
        except Exception as exc:
            messagebox.showerror("Seek Error", str(exc))
        finally:
            capture.release()

    def toggle_mute_placeholder(self) -> None:
        self.muted = not self.muted
        self.mute_button.configure(text="Muted" if self.muted else "Audio N/A")
        self._set_status_message("Audio playback is not implemented; video analysis is visual-only.")

    def decrease_playback_speed(self) -> None:
        index = PLAYBACK_SPEED_OPTIONS.index(self._get_playback_speed_label()) if self._get_playback_speed_label() in PLAYBACK_SPEED_OPTIONS else 3
        self._set_playback_speed_by_index(max(0, index - 1))

    def increase_playback_speed(self) -> None:
        index = PLAYBACK_SPEED_OPTIONS.index(self._get_playback_speed_label()) if self._get_playback_speed_label() in PLAYBACK_SPEED_OPTIONS else 3
        self._set_playback_speed_by_index(min(len(PLAYBACK_SPEED_OPTIONS) - 1, index + 1))

    def _set_playback_speed_by_index(self, index: int) -> None:
        self.playback_speed_var.set(PLAYBACK_SPEED_OPTIONS[index])
        self._on_playback_speed_changed(PLAYBACK_SPEED_OPTIONS[index])

    def set_quick_frame_interval(self, label: str) -> None:
        interval_map = {
            "8F": "Every 8 frames",
            "16F": "Every 16 frames",
            "24F": "Every 24 frames",
            "60F": "Every 60 frames",
        }
        selected = interval_map[label]
        self.video_interval_value_var.set(selected)
        self._on_video_interval_value_changed(selected)

    def stop_processing(self, update_status: bool = True) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self.session_id += 1
        if self.worker_thread and self.worker_thread.is_alive() and threading.current_thread() is not self.worker_thread:
            self.worker_thread.join(timeout=1.5)
        self.worker_thread = None

        if update_status:
            self._set_preview_note("Processing stopped.")
            self._set_status_message("Processing stopped.")

        if self.current_source_type == "video" and self.current_video_path is not None:
            self.player_state = "stopped"
            if update_status:
                self._render_video_frame_at(self.current_frame_number, overlay_text="Stopped")
            self._update_playback_buttons("video_stopped")
            self.session_mode_label.configure(text="Playback: Stopped")
        elif self.current_source_type == "webcam":
            self.player_state = "idle"
            self._update_playback_buttons("idle")
            self.session_mode_label.configure(text="Playback: Stopped")
        else:
            self.player_state = "idle" if self.current_source_type == "idle" else self.player_state
            self._update_playback_buttons("idle")

    def convert_mobile_video(self) -> None:
        video_path = filedialog.askopenfilename(title="Select a mobile video", filetypes=VIDEO_FILE_TYPES)
        if not video_path:
            return

        selected_path = Path(video_path)
        self.current_video_path = selected_path
        self._prepare_new_session("video", f"Video: {selected_path.name}", str(selected_path))
        self._set_preview_note("Converting mobile video...")
        self._set_status_message("Converting mobile video...")
        self._update_playback_buttons("video_loading")
        self.worker_thread = threading.Thread(target=self._conversion_worker, args=(selected_path, False), daemon=True)
        self.worker_thread.start()

    def _conversion_worker(self, input_path: Path, auto_run_after_conversion: bool) -> None:
        try:
            conversion_result = convert_video_for_opencv(input_path, self.converted_videos_dir)
        except Exception as exc:
            self.after(0, lambda: self._handle_stream_error("Conversion Error", str(exc)))
            return
        self.after(0, lambda: self._handle_conversion_result(conversion_result, auto_run_after_conversion))

    def _handle_conversion_result(self, result: VideoConversionResult, auto_run_after_conversion: bool) -> None:
        if not result.success or result.output_path is None:
            messagebox.showerror("Conversion Error", result.message)
            self._set_status_message("Video conversion failed.")
            self._set_preview_note("Video conversion failed.")
            self._update_playback_buttons("idle")
            return

        self._set_status_message(result.message)
        self._set_preview_note(result.message)
        self.current_video_path = result.output_path
        messagebox.showinfo("Video Conversion", result.message)

        if auto_run_after_conversion:
            self._load_video_for_manual_playback(result.output_path)
            return

        should_load = messagebox.askyesno("Load Converted Video", "Do you want to load the converted video in the player now?")
        if should_load:
            self._load_video_for_manual_playback(result.output_path)
        else:
            self._update_playback_buttons("video_stopped")

    def open_assets_folder(self) -> None:
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(self.assets_dir))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.assets_dir)])
            else:
                subprocess.Popen(["xdg-open", str(self.assets_dir)])
        except Exception as exc:
            messagebox.showerror("Open Folder Error", str(exc))

    def download_animal_images(self) -> None:
        if self.reference_download_thread and self.reference_download_thread.is_alive():
            messagebox.showinfo("Downloader Running", "Animal reference image download is already running.")
            return

        self.download_images_button.configure(state="disabled")
        self._set_status_message("Downloading reference images from public sources...")
        self.reference_download_thread = threading.Thread(target=self._download_animal_images_worker, daemon=True)
        self.reference_download_thread.start()

    def _download_animal_images_worker(self) -> None:
        command = [sys.executable, str(self.project_root / "app" / "download_animal_reference_images.py")]
        try:
            result = subprocess.run(
                command,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            self.after(0, lambda: self._finish_animal_image_download(False, "", str(exc)))
            return

        self.after(0, lambda: self._finish_animal_image_download(result.returncode == 0, result.stdout, result.stderr))

    def _finish_animal_image_download(self, success: bool, stdout: str, stderr: str) -> None:
        self.download_images_button.configure(state="normal")
        details = "\n".join((stdout or stderr).strip().splitlines()[-8:]).strip()
        if success:
            self._set_status_message("Animal image download completed.")
            if self.latest_display_result is not None:
                self._update_reference_image(self.latest_display_result.predicted_label)
            elif self.latest_raw_result is not None:
                self._update_reference_image(self.latest_raw_result.predicted_label)
            messagebox.showinfo(
                "Animal Images",
                details or "Animal images downloaded or skipped. Check assets/animal_reference_images/ for results.",
            )
            return

        self._set_status_message("Animal image download failed.")
        messagebox.showerror(
            "Animal Images",
            details or "The image downloader failed. Run `python app/download_animal_reference_images.py` in a terminal for details.",
        )

    def clear_history(self) -> None:
        self.history_service.clear()
        self._render_history_view()
        self._set_status_message("Detection history cleared.")

    def save_result_report(self) -> None:
        panel_result = self.latest_display_result or self.latest_raw_result
        if panel_result is None:
            messagebox.showinfo("No Result", "Run a prediction first, then save the report.")
            return

        report_path = self.reports_dir / f"wildlife_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        report_lines = [
            "Wildlife Intelligence Report",
            "===========================",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Source type: {self.current_source_type}",
            f"Source path: {self.current_source_path or 'N/A'}",
            f"Detection interval: {self._describe_current_interval_for_report()}",
            f"Confidence threshold: {self._get_confidence_threshold():.0%}",
            f"Smoothing enabled: {self.smoothing_enabled_var.get()}",
            f"Playback speed: {self._get_playback_speed_label() if self.current_source_type == 'video' else 'Real-time'}",
            "",
            f"Current result: {panel_result.predicted_label}",
            f"Confidence: {panel_result.confidence:.2%}",
            "",
            "Top 5 predictions:",
        ]
        for candidate in panel_result.top_5_predictions:
            report_lines.append(f"- {candidate.label}: {candidate.confidence:.2%}")

        report_lines.extend(
            [
                "",
                "Animal information:",
                format_animal_info(panel_result.animal_info),
                "",
                "Detection timeline:",
                self.history_service.to_report_text(),
            ]
        )

        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        messagebox.showinfo("Report Saved", f"Saved report to:\n{report_path}")
        self._set_status_message(f"Saved report: {report_path.name}")

    def _start_video_prediction(self, frame, frame_number: int, video_seconds: float) -> bool:
        with self.video_prediction_lock:
            if self.video_prediction_running:
                return False
            self.video_prediction_running = True

        threading.Thread(
            target=self._video_prediction_worker,
            args=(
                frame,
                frame_number,
                video_seconds,
                self._get_playback_speed_label(),
                self._describe_video_sampling_rule(),
                self._get_confidence_threshold(),
                self.smoothing_enabled,
                self.session_id,
            ),
            daemon=True,
        ).start()
        return True

    def _video_prediction_worker(
        self,
        frame,
        frame_number: int,
        video_seconds: float,
        playback_speed: str,
        ai_interval: str,
        confidence_threshold: float,
        smoothing_enabled: bool,
        session_id: int,
    ) -> None:
        try:
            sampled_result = self.prediction_service.predict_video_frame(frame) if self.prediction_service else None
            decision = (
                self.prediction_smoother.observe(
                    sampled_result,
                    threshold=confidence_threshold,
                    enabled=smoothing_enabled,
                )
                if sampled_result is not None
                else None
            )
        except Exception as exc:
            self.after(0, lambda: self._set_status_message(f"Video prediction failed: {exc}"))
            sampled_result = None
            decision = None
        finally:
            with self.video_prediction_lock:
                self.video_prediction_running = False

        if sampled_result is not None:
            self.after(
                0,
                lambda result=sampled_result, decision=decision, frame_number=frame_number, seconds=video_seconds, speed=playback_speed, interval=ai_interval, session_id=session_id: self._handle_video_prediction_result(
                    result,
                    decision,
                    frame_number,
                    seconds,
                    speed,
                    interval,
                    session_id,
                ),
            )

    def _handle_video_prediction_result(
        self,
        sampled_result: PredictionResult,
        decision: SmootherDecision | None,
        frame_number: int,
        video_seconds: float,
        playback_speed: str,
        ai_interval: str,
        session_id: int,
    ) -> None:
        if self.current_source_type != "video" or session_id != self.session_id:
            return

        self.latest_raw_result = sampled_result
        panel_result = self._choose_panel_result(sampled_result, decision)
        if decision and decision.display_result is not None:
            self.latest_display_result = decision.display_result

        status_note = decision.reason if decision else f"AI checked {ai_interval}."
        self._update_result_panel(panel_result, sampled_result, status_note)
        self._update_status_panel(status_note, self.current_fps, self.current_frame_number, self.current_video_seconds)

        should_add = self.record_all_history_var.get() or (decision.accepted_new_result if decision else True)
        if panel_result is not None and should_add:
            self._record_detection(
                panel_result,
                frame_number,
                video_seconds,
                add_all=self.record_all_history_var.get(),
                playback_speed=playback_speed,
                ai_interval=ai_interval,
            )

    def _handle_stream_update(
        self,
        *,
        preview_frame,
        sampled_result: PredictionResult | None,
        decision: SmootherDecision | None,
        panel_result: PredictionResult | None,
        fps: float,
        frame_number: int,
        video_seconds: float | None,
        status_note: str,
        force_history: bool = False,
    ) -> None:
        self.current_fps = fps
        self.current_frame_number = frame_number
        self.current_video_seconds = video_seconds or 0.0

        if preview_frame is not None:
            self._render_preview_frame(preview_frame)

        if sampled_result is not None:
            self.latest_raw_result = sampled_result
        if decision and decision.display_result is not None:
            self.latest_display_result = decision.display_result
        elif panel_result is not None and self.current_source_type == "image":
            self.latest_display_result = panel_result

        self._update_result_panel(panel_result, sampled_result, status_note)
        self._update_status_panel(status_note, fps, frame_number, video_seconds)
        if self.current_source_type == "video":
            self._update_timeline(frame_number, video_seconds)

        if force_history and panel_result is not None:
            self._record_detection(panel_result, frame_number, video_seconds, add_all=True)
        elif sampled_result is not None and panel_result is not None:
            should_add = self.record_all_history_var.get() or (decision.accepted_new_result if decision else True)
            if should_add:
                self._record_detection(panel_result, frame_number, video_seconds, add_all=self.record_all_history_var.get())

    def _record_detection(
        self,
        result: PredictionResult,
        frame_number: int,
        video_seconds: float | None,
        add_all: bool,
        playback_speed: str | None = None,
        ai_interval: str | None = None,
    ) -> None:
        self.history_service.add_detection(
            source_type=self.current_source_type,
            source_name=self.current_source_name,
            label=result.predicted_label,
            confidence=result.confidence,
            frame_number=frame_number if self.current_source_type == "video" else None,
            video_timestamp=format_video_timestamp(video_seconds) if self.current_source_type == "video" else None,
            playback_speed=playback_speed if self.current_source_type == "video" else None,
            ai_interval=ai_interval if self.current_source_type == "video" else None,
            top_5_predictions=[
                f"{candidate.label}: {candidate.confidence:.2%}" for candidate in result.top_5_predictions
            ]
            if self.current_source_type == "video"
            else None,
            add_all=add_all,
        )
        self._render_history_view()

    def _render_history_view(self) -> None:
        for child in self.history_list_frame.winfo_children():
            child.destroy()

        records = self.history_service.get_records()
        if not records:
            empty_label = ctk.CTkLabel(
                self.history_list_frame,
                text="No detections recorded yet.",
                text_color=TEXT_MUTED,
                font=ctk.CTkFont(size=13),
            )
            empty_label.pack(anchor="w", padx=10, pady=10)
            return

        for record in records:
            self._render_history_record(record)

    def _render_history_record(self, record: DetectionRecord) -> None:
        entry = ctk.CTkFrame(self.history_list_frame, fg_color="#1b2b25", corner_radius=14, border_width=1, border_color="#223931")
        entry.pack(fill="x", padx=8, pady=6)

        title = ctk.CTkLabel(
            entry,
            text=f"{record.label}  |  {record.confidence:.1%}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        title.pack(anchor="w", padx=12, pady=(10, 2))

        details = f"{record.detected_at}  |  {record.source_type}"
        if record.video_timestamp:
            details += f"  |  time {record.video_timestamp}"
        if record.frame_number is not None:
            details += f"  |  frame {record.frame_number}"
        if record.playback_speed:
            details += f"  |  {record.playback_speed}"
        if record.ai_interval:
            details += f"  |  {record.ai_interval}"
        subtitle = ctk.CTkLabel(
            entry,
            text=details,
            wraplength=300,
            justify="left",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        )
        subtitle.pack(anchor="w", padx=12, pady=(0, 10))

    def _update_result_panel(
        self,
        panel_result: PredictionResult | None,
        sampled_result: PredictionResult | None,
        status_note: str,
    ) -> None:
        if panel_result is None:
            self.result_name_label.configure(text="Sampling...", text_color=TEXT_PRIMARY)
            self.result_confidence_label.configure(text="Confidence: --")
            self.result_note_label.configure(text=status_note)
            self._set_reference_image_placeholder("No stable prediction yet")
            return

        visible_result = panel_result
        threshold = self._get_confidence_threshold()
        if visible_result.confidence < threshold:
            title = "Uncertain prediction"
            detail = f"Best guess: {visible_result.predicted_label} | {visible_result.confidence:.1%}"
            self.result_name_label.configure(text_color=WARNING)
        else:
            title = visible_result.predicted_label
            detail = f"Confidence: {visible_result.confidence:.1%}"
            self.result_name_label.configure(text_color=TEXT_PRIMARY)

        self.result_name_label.configure(text=title)
        self.result_confidence_label.configure(text=detail)
        self.result_note_label.configure(text=status_note)

        for row, candidate in zip(self.top_prediction_rows, visible_result.top_5_predictions):
            label_widget, progress_widget, percent_widget = row
            label_widget.configure(text=candidate.label)
            progress_widget.set(candidate.confidence)
            percent_widget.configure(text=f"{candidate.confidence:.1%}")
        for row in self.top_prediction_rows[len(visible_result.top_5_predictions):]:
            label_widget, progress_widget, percent_widget = row
            label_widget.configure(text="--")
            progress_widget.set(0)
            percent_widget.configure(text="0%")

        self.info_textbox.configure(state="normal")
        self.info_textbox.delete("1.0", "end")
        self.info_textbox.insert("1.0", format_animal_info(visible_result.animal_info))
        self.info_textbox.configure(state="disabled")

        self._update_reference_image(visible_result.predicted_label)

    def _update_reference_image(self, label: str) -> None:
        image = self.reference_image_service.load_reference_image(label, (330, 240))
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(330, 240))
        self.reference_image_ref = ctk_image
        self.reference_image_label.configure(image=ctk_image, text="")
        if self.reference_image_service.get_reference_image_path(label):
            self.reference_caption_label.configure(text=f"Reference image for {label}")
        else:
            self.reference_caption_label.configure(text="No reference image yet. Run image downloader.")

    def _set_reference_image_placeholder(self, title: str) -> None:
        image = self.reference_image_service.create_placeholder_image(title, (330, 240))
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(330, 240))
        self.reference_image_ref = ctk_image
        self.reference_image_label.configure(image=ctk_image, text="")
        self.reference_caption_label.configure(text="No reference image yet. Run image downloader.")

    def _update_status_panel(self, status_note: str, fps: float, frame_number: int, video_seconds: float | None) -> None:
        self.current_fps = fps
        self.source_value_label.configure(text=f"Source: {self.current_source_name}")
        self.fps_value_label.configure(text=f"Processing FPS: {fps:.1f}")
        if self.current_source_type == "video":
            self.video_fps_value_label.configure(text=f"Video FPS: {self.current_video_fps:.2f}")
            self.playback_speed_value_label.configure(text=f"Playback Speed: {self._get_playback_speed_label()}")
            self.frame_value_label.configure(text=f"Frame: {self._format_frame_progress(frame_number)}")
            self.timestamp_value_label.configure(text=f"Time: {self._format_time_progress(video_seconds)}")
        elif self.current_source_type == "webcam":
            self.video_fps_value_label.configure(text="Video FPS: Webcam live")
            self.playback_speed_value_label.configure(text="Playback Speed: Real-time webcam")
            self.frame_value_label.configure(text=f"Frame: {frame_number}")
            self.timestamp_value_label.configure(text="Time: Live")
        else:
            self.video_fps_value_label.configure(text="Video FPS: N/A")
            self.playback_speed_value_label.configure(text="Playback Speed: N/A")
            self.frame_value_label.configure(text=f"Frame: {frame_number}")
            self.timestamp_value_label.configure(text=f"Time: {format_video_timestamp(video_seconds)}")
        self.interval_value_label.configure(text=f"Sampling: {self._describe_current_interval_for_status()}")
        self._set_status_message(status_note)

    def _configure_timeline(self) -> None:
        max_frame = max(self.current_video_total_frames - 1, 1)
        self.timeline_slider.configure(from_=0, to=max_frame, number_of_steps=max_frame)
        self._update_timeline(self.current_frame_number, self.current_video_seconds)

    def _update_timeline(self, frame_number: int, video_seconds: float | None) -> None:
        if not hasattr(self, "timeline_slider"):
            return
        self.timeline_update_internal = True
        try:
            max_frame = max(self.current_video_total_frames - 1, 1)
            self.timeline_slider.set(max(0, min(frame_number, max_frame)))
        finally:
            self.timeline_update_internal = False

        self.timeline_label.configure(
            text=f"{self._format_time_progress(video_seconds)} | Frame {self._format_frame_progress(frame_number)}"
        )

    def _on_timeline_seek(self, value: float) -> None:
        if self.timeline_update_internal or self.current_source_type != "video" or self.current_video_path is None:
            return
        self._seek_video_to_frame(int(round(float(value))))

    def _seconds_for_frame(self, frame_number: int) -> float:
        fps = self.current_video_fps if self.current_video_fps > 0 else 24.0
        return max(0.0, (frame_number - 1) / fps)

    def _render_preview_frame(self, frame_bgr) -> None:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        preview_image = Image.fromarray(rgb_frame)
        canvas_size = self._get_preview_canvas_size()
        preview_image.thumbnail(canvas_size, RESAMPLE)

        canvas = Image.new("RGB", canvas_size, PREVIEW_BG)
        paste_x = (canvas.width - preview_image.width) // 2
        paste_y = (canvas.height - preview_image.height) // 2
        canvas.paste(preview_image, (paste_x, paste_y))

        ctk_image = ctk.CTkImage(light_image=canvas, dark_image=canvas, size=canvas_size)
        self.preview_image_ref = ctk_image
        self.preview_label.configure(image=ctk_image, text="")

    def _render_preview_placeholder(self, message: str) -> None:
        canvas_size = self._get_preview_canvas_size()
        width, height = canvas_size
        canvas = Image.new("RGB", canvas_size, PREVIEW_BG)
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((18, 18, width - 18, height - 18), radius=24, outline="#1f352d", width=2, fill="#080d0c")
        center_y = max(80, height // 2 - 42)
        draw.text((46, center_y), message, fill=TEXT_PRIMARY)
        draw.text((46, center_y + 38), "Video analysis starts only after Start / Play.", fill=TEXT_MUTED)
        ctk_image = ctk.CTkImage(light_image=canvas, dark_image=canvas, size=canvas_size)
        self.preview_image_ref = ctk_image
        self.preview_label.configure(image=ctk_image, text="")

    def _get_preview_canvas_size(self) -> tuple[int, int]:
        width = max(640, self.preview_label.winfo_width() - 12)
        height = max(360, self.preview_label.winfo_height() - 12)
        aspect_height = int(width / PREVIEW_ASPECT_RATIO)
        if aspect_height <= height:
            height = aspect_height
        else:
            width = int(height * PREVIEW_ASPECT_RATIO)
        return width, height

    def _compose_preview_frame(
        self,
        frame_bgr,
        result: PredictionResult | None,
        line_two: str,
        meta: str,
        status_line: str | None = None,
    ):
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (18, 18), (620, 124), (7, 11, 9), -1)
        frame_bgr = cv2.addWeighted(overlay, 0.38, frame_bgr, 0.62, 0)

        if status_line:
            title = status_line
        elif result is None:
            title = "Scanning..."
        elif result.confidence < self._get_confidence_threshold():
            title = f"Uncertain: {result.predicted_label}"
        else:
            title = result.predicted_label

        cv2.putText(frame_bgr, title, (34, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (246, 243, 234), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, line_two, (34, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 224, 220), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, meta, (34, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (185, 199, 190), 1, cv2.LINE_AA)
        return frame_bgr

    def _choose_panel_result(
        self,
        sampled_result: PredictionResult | None,
        decision: SmootherDecision | None,
    ) -> PredictionResult | None:
        if decision and decision.display_result is not None:
            return decision.display_result
        if sampled_result is not None:
            return sampled_result
        if self.latest_display_result is not None:
            return self.latest_display_result
        return self.latest_raw_result

    def _should_sample_video(
        self,
        frame_number: int,
        video_seconds: float,
        last_sampled_frame: int,
        last_sampled_seconds: float,
    ) -> bool:
        if self.video_interval_mode == "Seconds":
            interval_seconds = VIDEO_INTERVAL_SECONDS_OPTIONS.get(self.video_interval_selection, 1.0)
            return last_sampled_seconds < 0 or video_seconds - last_sampled_seconds >= interval_seconds

        interval_frames = VIDEO_INTERVAL_FRAME_OPTIONS.get(self.video_interval_selection, 24)
        return frame_number == 1 or frame_number % interval_frames == 0

    def _describe_video_sampling_rule(self) -> str:
        return self.video_interval_selection

    def _describe_next_ai_check(self, frame_number: int, video_seconds: float) -> str:
        if self.video_interval_mode == "Seconds":
            interval_seconds = VIDEO_INTERVAL_SECONDS_OPTIONS.get(self.video_interval_selection, 1.0)
            next_second = video_seconds + interval_seconds
            return f"near {format_video_timestamp(next_second)}"

        interval_frames = VIDEO_INTERVAL_FRAME_OPTIONS.get(self.video_interval_selection, 24)
        remainder = frame_number % interval_frames
        frames_until_next = interval_frames if remainder == 0 else interval_frames - remainder
        next_frame = frame_number + frames_until_next
        if self.current_video_total_frames > 0:
            next_frame = min(next_frame, self.current_video_total_frames)
        return f"at frame {next_frame}"

    def _describe_current_interval_for_status(self) -> str:
        if self.current_source_type == "webcam":
            return f"Webcam every {self._get_webcam_interval():.2f} sec"
        if self.current_source_type == "video":
            return self._describe_video_sampling_rule()
        return "Single prediction"

    def _describe_current_interval_for_report(self) -> str:
        if self.current_source_type == "video":
            return f"Video sampling: {self._describe_video_sampling_rule()}"
        if self.current_source_type == "webcam":
            return f"Webcam sampling: every {self._get_webcam_interval():.2f} seconds"
        return "Single image prediction"

    def _get_confidence_threshold(self) -> float:
        return self.confidence_threshold

    def _get_webcam_interval(self) -> float:
        return self.webcam_interval_seconds

    def _get_playback_speed_label(self) -> str:
        return self.playback_speed_label

    def _get_playback_speed_multiplier(self) -> float:
        return self.playback_speed_multiplier

    def _format_frame_progress(self, frame_number: int) -> str:
        if self.current_video_total_frames > 0:
            return f"{frame_number} / {self.current_video_total_frames}"
        return str(frame_number)

    def _format_time_progress(self, video_seconds: float | None) -> str:
        current = format_video_timestamp(video_seconds)
        if self.current_video_total_duration_seconds is not None:
            total = format_video_timestamp(self.current_video_total_duration_seconds)
            return f"{current} / {total}"
        return current

    def _on_threshold_changed(self, value: float) -> None:
        self.confidence_threshold = float(value) / 100.0
        self.threshold_label.configure(text=f"Confidence Threshold: {value:.0f}%")

    def _on_webcam_interval_changed(self, value: float) -> None:
        self.webcam_interval_seconds = round(float(value), 2)
        self.webcam_interval_label.configure(text=f"Webcam Prediction Interval: {self.webcam_interval_seconds:.2f} s")

    def _on_smoothing_changed(self) -> None:
        self.smoothing_enabled = bool(self.smoothing_enabled_var.get())

    def _on_video_interval_mode_changed(self, value: str) -> None:
        self.video_interval_mode = value
        if value == "Seconds":
            if self.video_interval_value_var.get() not in VIDEO_INTERVAL_SECONDS_OPTIONS:
                self.video_interval_value_var.set("Every 1 sec")
        else:
            if self.video_interval_value_var.get() not in VIDEO_INTERVAL_FRAME_OPTIONS:
                self.video_interval_value_var.set("Every 24 frames")
        self._on_video_interval_value_changed(self.video_interval_value_var.get())

    def _on_video_interval_value_changed(self, value: str) -> None:
        self.video_interval_selection = value
        self.video_interval_mode = "Seconds" if value in VIDEO_INTERVAL_SECONDS_OPTIONS else "Frames"
        self.video_interval_mode_var.set(self.video_interval_mode)
        if hasattr(self, "interval_value_label"):
            self.interval_value_label.configure(text=f"Sampling: {self._describe_current_interval_for_status()}")

    def _on_playback_speed_changed(self, value: str) -> None:
        self.playback_speed_label = value
        try:
            self.playback_speed_multiplier = max(float(value.lower().replace("x", "")), 0.25)
        except ValueError:
            self.playback_speed_label = "1.0x"
            self.playback_speed_multiplier = 1.0

        if hasattr(self, "playback_speed_value_label"):
            self.playback_speed_value_label.configure(text=f"Playback Speed: {self.playback_speed_label}")
        if self.current_source_type == "video":
            if self.pause_event.is_set():
                self._update_playback_buttons("video_paused")
            elif self.worker_thread and self.worker_thread.is_alive():
                self._update_playback_buttons("video_playing")

    def _set_status_message(self, message: str) -> None:
        self.status_message_label.configure(text=message)

    def _set_preview_note(self, message: str) -> None:
        self.preview_note_label.configure(text=message)

    def _handle_stream_error(self, title: str, message: str) -> None:
        self.stop_processing(update_status=False)
        self._set_status_message(message)
        self._set_preview_note(message)
        messagebox.showerror(title, message)

    def _update_playback_buttons(self, state: str) -> None:
        states = {
            "idle": {"play": "disabled", "pause": "disabled", "resume": "disabled", "stop": "disabled", "restart": "disabled", "seek": "disabled", "speed": "normal", "label": "Playback: Stopped"},
            "image": {"play": "disabled", "pause": "disabled", "resume": "disabled", "stop": "disabled", "restart": "disabled", "seek": "disabled", "speed": "normal", "label": "Playback: Image mode"},
            "webcam_playing": {"play": "disabled", "pause": "disabled", "resume": "disabled", "stop": "normal", "restart": "disabled", "seek": "disabled", "speed": "disabled", "label": "Playback: Webcam live"},
            "video_loading": {"play": "disabled", "pause": "disabled", "resume": "disabled", "stop": "disabled", "restart": "disabled", "seek": "disabled", "speed": "normal", "label": "Playback: Preparing video"},
            "video_loaded": {"play": "normal", "pause": "disabled", "resume": "disabled", "stop": "disabled", "restart": "normal", "seek": "normal", "speed": "normal", "label": f"Playback: Video loaded @ {self._get_playback_speed_label()}"},
            "video_playing": {"play": "disabled", "pause": "normal", "resume": "disabled", "stop": "normal", "restart": "normal", "seek": "normal", "speed": "normal", "label": f"Playback: Playing video @ {self._get_playback_speed_label()}"},
            "video_paused": {"play": "normal", "pause": "disabled", "resume": "normal", "stop": "normal", "restart": "normal", "seek": "normal", "speed": "normal", "label": f"Playback: Paused @ {self._get_playback_speed_label()}"},
            "video_stopped": {"play": "normal" if self.current_video_path else "disabled", "pause": "disabled", "resume": "disabled", "stop": "disabled", "restart": "normal" if self.current_video_path else "disabled", "seek": "normal" if self.current_video_path else "disabled", "speed": "normal", "label": "Playback: Stopped"},
        }
        config = states.get(state, states["idle"])

        self.play_button.configure(state=config["play"])
        self.pause_button.configure(state=config["pause"])
        self.resume_button.configure(state=config["resume"])
        self.stop_button.configure(state=config["stop"])
        self.restart_button.configure(state=config["restart"])
        for button in (self.seek_back_button, self.seek_forward_button, self.prev_frame_button, self.next_frame_button):
            button.configure(state=config["seek"])
        self.speed_down_button.configure(state=config["speed"])
        self.speed_up_button.configure(state=config["speed"])
        self.playback_speed_menu.configure(state=config["speed"])
        self.session_mode_label.configure(text=config["label"])
        self._update_status_chips()

    def _reset_panels(self) -> None:
        self.result_name_label.configure(text="Waiting for prediction", text_color=TEXT_PRIMARY)
        self.result_confidence_label.configure(text="Confidence: --")
        self.result_note_label.configure(text="Adjust the controls to tune sampling and stability.")

        for label_widget, progress_widget, percent_widget in self.top_prediction_rows:
            label_widget.configure(text="--")
            progress_widget.set(0)
            percent_widget.configure(text="0%")

        self.info_textbox.configure(state="normal")
        self.info_textbox.delete("1.0", "end")
        self.info_textbox.insert("1.0", "Animal details will appear here once a prediction is available.")
        self.info_textbox.configure(state="disabled")

        self._set_reference_image_placeholder("No prediction yet")
        self._render_preview_placeholder("Upload an image/video or start webcam detection.")
        self._update_timeline(0, None)
        self._render_history_view()

    def _on_close(self) -> None:
        self.stop_processing(update_status=False)
        self.destroy()


def main() -> None:
    try:
        app = WildlifeDetectorApp()
    except TclError as exc:
        raise SystemExit(
            "Tkinter could not start. Run `python app/check_ui_environment.py` for a full diagnostic, then "
            "repair or reinstall Python 3.10 with Tcl/Tk support before running "
            "`python app/ui_wildlife_detector.py` again."
        ) from exc
    app.mainloop()


if __name__ == "__main__":
    main()
