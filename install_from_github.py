from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/NoirPrimordial7/Wild-life-detection-system.git"
DEFAULT_FOLDER = "Wild-life-detection-system"
IS_WINDOWS = os.name == "nt"


def run(command: list[str], cwd: Path | None = None, allow_fail: bool = False) -> int:
    print(f"> {' '.join(command)}")
    result = subprocess.run(command, cwd=str(cwd) if cwd else None, check=False)
    if result.returncode != 0 and not allow_fail:
        raise SystemExit(result.returncode)
    return result.returncode


def find_python() -> str:
    if IS_WINDOWS:
        try:
            subprocess.run(["py", "-3.10", "-c", "import sys"], check=True, capture_output=True)
            return "py -3.10"
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise SystemExit("Install Python 3.10 from python.org and enable Tcl/Tk and IDLE.")

    python310 = shutil.which("python3.10")
    if python310:
        return python310

    python3 = shutil.which("python3")
    if python3:
        print("WARNING: python3.10 was not found. Falling back to python3.")
        return python3

    raise SystemExit("Python 3 is not installed.")


def python_command(python_spec: str, *args: str) -> list[str]:
    if python_spec == "py -3.10":
        return ["py", "-3.10", *args]
    return [python_spec, *args]


def venv_python(project_root: Path) -> Path:
    return project_root / ("venv/Scripts/python.exe" if IS_WINDOWS else "venv/bin/python")


def ensure_git() -> None:
    if not shutil.which("git"):
        raise SystemExit("Git is not installed. Install Git and run this installer again.")


def clone_or_update(target: Path) -> Path:
    if target.exists():
        if (target / ".git").is_dir():
            print(f"Existing clone found: {target}")
            run(["git", "pull", "--ff-only"], cwd=target)
            return target

        raise SystemExit(
            f"Target folder already exists and is not a Git clone:\n{target}\n"
            "Choose another --target folder or move the existing folder manually."
        )

    run(["git", "clone", REPO_URL, str(target)])
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Wildlife Detection System from GitHub.")
    parser.add_argument("--target", type=Path, default=Path.cwd() / DEFAULT_FOLDER, help="Folder to clone or update.")
    args = parser.parse_args()

    ensure_git()
    python_spec = find_python()
    project_root = clone_or_update(args.target.resolve())
    project_python = venv_python(project_root)

    if not project_python.exists():
        run(python_command(python_spec, "-m", "venv", "venv"), cwd=project_root)
    else:
        print(f"Reusing existing virtual environment: {project_root / 'venv'}")

    run([str(project_python), "-m", "pip", "install", "--upgrade", "pip"], cwd=project_root)
    run([str(project_python), "-m", "pip", "install", "-r", "requirements.txt"], cwd=project_root)
    run([str(project_python), "app/check_ui_environment.py"], cwd=project_root, allow_fail=True)
    run([str(project_python), "app/test_model_load.py"], cwd=project_root)

    print()
    print("Installation complete.")
    print(f"Project folder: {project_root}")
    print("Run the UI with:")
    if IS_WINDOWS:
        print(r"  .\venv\Scripts\python.exe app\ui_wildlife_detector.py")
    else:
        print("  ./venv/bin/python app/ui_wildlife_detector.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
