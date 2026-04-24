#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "Wildlife Detection System macOS setup"
echo "Project root: $PROJECT_ROOT"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
  PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  PYTHON_OK="$($PYTHON_BIN -c 'import sys; print(int(sys.version_info >= (3, 10)))')"
else
  PYTHON_OK="0"
fi

if [[ "$PYTHON_OK" != "1" ]]; then
  echo
  echo "Python 3.10+ was not found."
  echo "Install Python 3.10+ with Homebrew:"
  echo "  brew install python@3.10"
  echo
  echo "If you do not use Homebrew, install Python from python.org and use python3."
  exit 1
fi

echo "Using $PYTHON_BIN $PYTHON_VERSION"

if [[ ! -d "venv" ]]; then
  echo "Creating virtual environment..."
  "$PYTHON_BIN" -m venv venv
else
  echo "Reusing existing virtual environment."
fi

source venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
if ! pip install -r requirements.txt; then
  echo
  echo "Dependency installation failed."
  echo "If TensorFlow fails on macOS, try:"
  echo "  brew install libomp"
  echo "  pip install tensorflow"
  exit 1
fi

echo "Running UI environment check..."
python app/check_ui_environment.py

echo "Running model load test..."
python app/test_model_load.py

echo
echo "Setup completed successfully."
echo "Launch the app with:"
echo "  python app/main.py"
