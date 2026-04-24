from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BOOTSTRAP_FLAG = "WILDLIFE_RUNTIME_BOOTSTRAPPED"


def maybe_relaunch_with_project_runtime() -> None:
    if os.environ.get(BOOTSTRAP_FLAG) == "1":
        return
    if os.name != "nt":
        return

    project_root = Path(__file__).resolve().parents[2]
    preferred_python = project_root / "venv" / "Scripts" / "python.exe"

    try:
        current_python = Path(sys.executable).resolve()
    except OSError:
        current_python = Path(sys.executable)

    if preferred_python.is_file():
        try:
            if current_python == preferred_python.resolve():
                return
        except OSError:
            pass

        environment = os.environ.copy()
        environment[BOOTSTRAP_FLAG] = "1"
        completed_process = subprocess.run(
            [str(preferred_python), *sys.argv],
            env=environment,
            check=False,
        )
        raise SystemExit(completed_process.returncode)
