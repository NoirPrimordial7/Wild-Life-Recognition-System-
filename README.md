# Wildlife Detection System

## Project Overview
This project uses the existing trained `animal_classification_model_final.h5` model to classify the main animal visible in an image or video frame. It shows prediction confidence, top predictions, animal information from `animal_info.json`, reference images when available, and a detection history that can be exported as a report.

This is image classification, not bounding-box object detection. The app predicts the most likely animal for the whole frame and does not draw detection boxes.

## Features
- Webcam detection with periodic AI checks
- Image upload and one-time classification
- Video upload with a VLC-style media player
- Uploaded videos do not auto-play; the first frame is shown until `Start / Play`
- Playback controls: `Start / Play`, `Pause`, `Resume`, `Stop`, `Restart`, `-5s`, `+5s`, `Prev Frame`, `Next Frame`
- Playback speed control: `0.25x`, `0.5x`, `0.75x`, `1.0x`, `1.25x`, `1.5x`, `2.0x`
- AI interval control by frames or time
- Quick AI interval buttons: `8F`, `16F`, `24F`, `60F`
- Mobile video conversion for common phone formats
- Animal reference image cards
- Detection history and report export
- Cross-platform setup scripts for Windows, macOS, and Linux

## Project Structure
```text
Wildlife detection system/
|-- app/
|   |-- check_ui_environment.py
|   |-- download_animal_reference_images.py
|   |-- live_wildlife_detector.py
|   |-- setup_project.py
|   |-- test_model_load.py
|   |-- ui_wildlife_detector.py
|   |-- ui/
|   |   |-- __init__.py
|   |   |-- animal_info_panel.py
|   |   |-- history_panel.py
|   |   |-- media_controls.py
|   |   |-- preview_panel.py
|   |   |-- report_panel.py
|   |   |-- result_panel.py
|   |   |-- settings_panel.py
|   |   |-- sidebar.py
|   |   `-- theme.py
|   `-- utils/
|       |-- animal_info_loader.py
|       |-- class_names.py
|       |-- history_service.py
|       |-- model_loader.py
|       |-- prediction_service.py
|       |-- prediction_smoother.py
|       |-- reference_image_service.py
|       |-- runtime_bootstrap.py
|       `-- video_converter.py
|-- assets/
|-- data/
|   |-- animal_info.json
|   `-- class_names.json
|-- models/
|   `-- animal_classification_model_final.h5
|-- install_from_github.py
|-- setup_unix.sh
|-- setup_windows.bat
|-- setup_windows.ps1
`-- requirements.txt
```

## Setup On Current Machine
Use Python 3.10. Python 3.14 is not recommended for TensorFlow compatibility.

```powershell
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app/check_ui_environment.py
python app/test_model_load.py
python app/ui_wildlife_detector.py
```

Tkinter/Tcl must work for the desktop UI. If `app/check_ui_environment.py` fails, repair or reinstall Python 3.10 and make sure Tcl/Tk and IDLE are enabled.

## Setup On New Windows PC
Double-click:

```bat
setup_windows.bat
```

Or run:

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

The Windows setup checks Git, checks `py -3.10`, creates `venv`, installs dependencies, runs the UI environment check, and runs the model load test.

## Setup On Mac/Linux
```bash
chmod +x setup_unix.sh
./setup_unix.sh
```

macOS Python 3.10 install option:

```bash
brew install python@3.10
```

Linux Python/Tk install option:

```bash
sudo apt install python3.10 python3.10-venv python3-tk
```

## Install From GitHub
Run this from any folder where you want the project cloned:

```powershell
python install_from_github.py
```

Repository:

```text
https://github.com/NoirPrimordial7/Wild-life-detection-system.git
```

## Run Commands
Desktop UI:

```powershell
python app/ui_wildlife_detector.py
```

Direct webcam script:

```powershell
python app/live_wildlife_detector.py --source webcam
```

Model test:

```powershell
python app/test_model_load.py
```

UI environment diagnostic:

```powershell
python app/check_ui_environment.py
```

## Video AI Detection
Uploaded videos load manually. After selecting a video, the app shows the first frame and waits for `Start / Play`.

Frame intervals classify when the decoded frame reaches the selected interval:

```text
Every 8 frames
Every 16 frames
Every 24 frames
Every 30 frames
Every 60 frames
Every 120 frames
```

Time intervals classify by the video timestamp:

```text
Every 0.5 sec
Every 1 sec
Every 2 sec
Every 5 sec
```

Webcam detection is different: it stays real-time and uses its own timer interval, so playback speed does not apply.

## Reference Images
Download public educational/demo reference images where available:

```powershell
python app/download_animal_reference_images.py
python app/download_animal_reference_images.py --force
python app/download_animal_reference_images.py --limit 10
python app/download_animal_reference_images.py --only "Tiger"
```

Images are saved in `assets/animal_reference_images/`, with a report at `assets/animal_reference_images/download_report.json`.

## GitHub Notes
- Run `git status` before pushing.
- Do not force push.
- If your branch is behind GitHub, use a backup branch and `git pull --rebase origin main`.
- Stop and resolve conflicts carefully if Git reports merge conflicts.
