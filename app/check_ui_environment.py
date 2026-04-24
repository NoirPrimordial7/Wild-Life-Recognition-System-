from __future__ import annotations

from utils.runtime_bootstrap import maybe_relaunch_with_project_runtime

maybe_relaunch_with_project_runtime()

import platform
import sys


WINDOWS_TK_FIX_STEPS = """Windows fix steps:
A) Repair/reinstall Python 3.10 from python.org
B) During installation, make sure Tcl/Tk and IDLE are installed
C) Recreate venv after Python repair:
   rmdir /s /q venv
   py -3.10 -m venv venv
   venv\\Scripts\\activate
   pip install -r requirements.txt
"""


def print_result(name: str, success: bool, details: str) -> None:
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {name}: {details}")


def main() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.splitlines()[0]}")
    print(f"Platform: {platform.platform()}")
    print()

    overall_success = True
    tkinter_module = None

    try:
        import tkinter as tk

        tkinter_module = tk
        print_result("Import tkinter", True, f"Imported successfully from {getattr(tk, '__file__', 'built-in module')}")
    except Exception as exc:
        print_result("Import tkinter", False, str(exc))
        print()
        print(WINDOWS_TK_FIX_STEPS)
        overall_success = False

    if tkinter_module is not None:
        try:
            root = tkinter_module.Tk()
            root.withdraw()
            root.update_idletasks()
            root.destroy()
            print_result("Create Tk root window", True, "Tk root created and destroyed successfully")
        except Exception as exc:
            print_result("Create Tk root window", False, str(exc))
            print()
            print(WINDOWS_TK_FIX_STEPS)
            overall_success = False

    try:
        import customtkinter as ctk

        version = getattr(ctk, "__version__", "unknown")
        print_result("Import customtkinter", True, f"Imported successfully (version {version})")
    except Exception as exc:
        print_result("Import customtkinter", False, str(exc))
        print()
        print("Install or reinstall project dependencies with:")
        print("  pip install -r requirements.txt")
        overall_success = False

    print()
    if overall_success:
        print("UI environment check PASSED.")
        return 0

    print("UI environment check FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
