@echo off
setlocal enabledelayedexpansion

:: 1. Check Git
where git >nul 2>nul
if errorlevel 1 (
    echo ❌ Git not found. Please install Git from https://git-scm.com/downloads
    pause
    exit /b
)

:: 2. Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.10+ from https://www.python.org/downloads/windows/
    pause
    exit /b
)

:: 3. Clone repo
echo 📦 Cloning project...
git clone https://github.com/NoirPrimordial7/Wild-life-detection-system.git
cd Wild-life-detection-system

:: 4. Create virtual environment
echo 🛠️ Creating virtual environment...
python -m venv venv

:: 5. Activate it
call venv\Scripts\activate

:: 6. Upgrade pip
echo 🔁 Upgrading pip...
python -m pip install --upgrade pip

:: 7. Install dependencies
echo 📦 Installing dependencies...
pip install tensorflow pillow numpy opencv-python

:: Optional (for Tkinter on some setups – most already have it)
:: python -m pip install tk

:: 8. Run app
echo 🚀 Running the Jungle Wildlife Detector...
python main7.py

pause
