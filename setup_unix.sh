#!/usr/bin/env bash
set -euo pipefail

# macOS: brew install python@3.10
# Linux: sudo apt install python3.10 python3.10-venv python3-tk

REPO_URL="https://github.com/NoirPrimordial7/Wild-life-detection-system.git"
REPO_FOLDER_NAME="Wild-life-detection-system"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_ROOT"

step() {
  echo
  echo "== $1 =="
}

step "Checking Git"
if ! command -v git >/dev/null 2>&1; then
  echo "Git is not installed. Install Git and run this setup again."
  exit 1
fi
git --version

step "Checking Python"
if command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="python3.10"
  echo "Using python3.10"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
  echo "python3.10 was not found. Falling back to python3."
  echo "TensorFlow is most reliable with Python 3.10 for this project."
else
  echo "Python 3 is not installed."
  exit 1
fi

if [[ ! -f "$PROJECT_ROOT/app/ui_wildlife_detector.py" ]]; then
  TARGET_ROOT="$SCRIPT_ROOT/$REPO_FOLDER_NAME"
  if [[ -d "$TARGET_ROOT/.git" ]]; then
    step "Updating existing clone"
    cd "$TARGET_ROOT"
    git pull --ff-only
  else
    step "Cloning project"
    cd "$SCRIPT_ROOT"
    git clone "$REPO_URL" "$REPO_FOLDER_NAME"
    cd "$TARGET_ROOT"
  fi
  PROJECT_ROOT="$(pwd)"
else
  step "Using existing project folder"
  cd "$PROJECT_ROOT"
  if [[ -d ".git" ]]; then
    if [[ -z "$(git status --porcelain)" ]]; then
      git pull --ff-only
    else
      git status --short
      echo "Existing local changes are preserved. Pull manually after committing or stashing local work."
    fi
  fi
fi

step "Creating virtual environment"
if [[ ! -x "$PROJECT_ROOT/venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv venv
else
  echo "Reusing existing venv."
fi

VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

step "Installing dependencies"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.txt

step "Running verification"
if ! "$VENV_PYTHON" app/check_ui_environment.py; then
  echo "Tkinter/Tcl check failed. Install Tk support for your Python version and rerun this setup."
fi
"$VENV_PYTHON" app/test_model_load.py

step "Ready"
echo "Project folder: $PROJECT_ROOT"
echo "Run the UI with:"
echo "  ./venv/bin/python app/ui_wildlife_detector.py"
