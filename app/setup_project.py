from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IS_WINDOWS = os.name == "nt"
VENV_DIR = PROJECT_ROOT / "venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
MODEL_PATH = PROJECT_ROOT / "models" / "animal_classification_model_final.h5"
CLASS_NAMES_PATH = PROJECT_ROOT / "data" / "class_names.json"
ANIMAL_INFO_PATH = PROJECT_ROOT / "data" / "animal_info.json"


def print_status(level: str, name: str, details: str) -> None:
    print(f"[{level}] {name}: {details}")


def recreate_instructions() -> str:
    if IS_WINDOWS:
        return (
            "rmdir /s /q venv\n"
            "py -3.10 -m venv venv\n"
            "venv\\Scripts\\activate\n"
            "pip install -r requirements.txt"
        )
    return (
        "rm -rf venv\n"
        "python3.10 -m venv venv\n"
        "source venv/bin/activate\n"
        "pip install -r requirements.txt"
    )


def run_command(name: str, command: list[str]) -> bool:
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or result.stderr).strip()
    details = output.splitlines()[-1] if output else "Command completed."
    if result.returncode == 0:
        print_status("PASS", name, details)
        return True
    print_status("FAIL", name, details)
    return False


def check_python_version() -> bool:
    version = sys.version_info
    if (version.major, version.minor) == (3, 10):
        print_status("PASS", "Python 3.10", f"Using Python {version.major}.{version.minor}.{version.micro}")
        return True

    print_status(
        "WARN",
        "Python 3.10",
        (
            f"Running Python {version.major}.{version.minor}.{version.micro}. "
            "TensorFlow and Tkinter are most reliable with Python 3.10 for this project."
        ),
    )
    return False


def check_venv_health() -> bool:
    if VENV_DIR.exists() and not VENV_PYTHON.exists():
        print_status("FAIL", "Virtual environment", "venv exists but its Python executable is missing.")
        print("Recreate the virtual environment with:")
        print(recreate_instructions())
        return False

    if VENV_PYTHON.exists():
        print_status("PASS", "Virtual environment", f"Found reusable venv at {VENV_DIR}")
        return True

    print_status("WARN", "Virtual environment", "No venv folder detected yet. Run a setup script to create it.")
    return False


def check_import(module_name: str, label: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print_status("FAIL", label, str(exc))
        return False

    version = getattr(module, "__version__", None)
    detail = f"Imported successfully from {getattr(module, '__file__', 'built-in module')}"
    if version:
        detail += f" (version {version})"
    print_status("PASS", label, detail)
    return True


def check_pip() -> bool:
    return run_command("pip", [sys.executable, "-m", "pip", "--version"])


def check_required_files() -> bool:
    required_paths = [
        MODEL_PATH,
        CLASS_NAMES_PATH,
        ANIMAL_INFO_PATH,
    ]
    overall_ok = True
    for path in required_paths:
        if path.exists():
            print_status("PASS", "Required file", str(path.relative_to(PROJECT_ROOT)))
        else:
            print_status("FAIL", "Required file", f"Missing {path.relative_to(PROJECT_ROOT)}")
            overall_ok = False
    return overall_ok


def check_tkinter_directly() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
    except Exception as exc:
        print_status("FAIL", "Tkinter/Tcl", str(exc))
        print("Fix: repair/reinstall Python 3.10 and make sure Tcl/Tk and IDLE are installed.")
        return False

    print_status("PASS", "Tkinter/Tcl", "Tk root created and destroyed successfully")
    return True


def check_class_names_count() -> bool:
    if not CLASS_NAMES_PATH.exists():
        print_status("FAIL", "class_names count", "data/class_names.json is missing")
        return False

    try:
        class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print_status("FAIL", "class_names count", str(exc))
        return False

    if not isinstance(class_names, list):
        print_status("FAIL", "class_names count", "class_names.json must contain a JSON list")
        return False

    if len(class_names) <= 0:
        print_status("FAIL", "class_names count", "No class names found")
        return False

    print_status("PASS", "class_names count", f"{len(class_names)} labels found")
    return True


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.splitlines()[0]}")
    print(f"Platform: {platform.platform()}")
    print()

    overall_success = True

    check_python_version()
    if not check_venv_health():
        overall_success = False

    if not check_pip():
        overall_success = False

    if not check_import("tensorflow", "TensorFlow import"):
        overall_success = False
    if not check_import("cv2", "OpenCV import"):
        overall_success = False
    if not check_import("customtkinter", "CustomTkinter import"):
        overall_success = False

    if not check_tkinter_directly():
        overall_success = False

    if not run_command("Tkinter/Tcl environment", [sys.executable, "app/check_ui_environment.py"]):
        overall_success = False

    if not check_required_files():
        overall_success = False
    if not check_class_names_count():
        overall_success = False

    if not run_command("Model test", [sys.executable, "app/test_model_load.py"]):
        overall_success = False

    print()
    print("Optional: download public animal reference images with:")
    print("  python app/download_animal_reference_images.py")
    print("Final UI command:")
    print("  python app/ui_wildlife_detector.py")

    if overall_success:
        print()
        print("Setup verification PASSED.")
        return 0

    print()
    print("Setup verification completed with failures. Review the messages above before running the UI.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
