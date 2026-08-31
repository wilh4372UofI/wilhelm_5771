"""Standalone environment check for CS 4771/5771.

Run inside your activated course virtual environment:

    python check_environment.py

Exits 0 if the environment is ready, 1 otherwise.
"""

import importlib
import platform
import sys

REQUIRED_PACKAGES = ["numpy", "pandas", "matplotlib", "sklearn", "pytest"]
MIN_PYTHON = (3, 11)


def check_package(name: str) -> tuple[bool, str]:
    """Import a package and report whether it loaded, plus its version."""
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        return False, str(exc)
    return True, getattr(module, "__version__", "unknown")


def main() -> int:
    python_ok = sys.version_info[:2] >= MIN_PYTHON
    print(f"Python     : {platform.python_version()} "
          f"({'OK' if python_ok else 'TOO OLD - need 3.11+'})")
    print(f"Executable : {sys.executable}")
    print(f"Platform   : {platform.platform()}")
    print("Packages:")

    all_ok = True
    for name in REQUIRED_PACKAGES:
        installed, detail = check_package(name)
        all_ok = all_ok and installed
        status = f"OK   {detail}" if installed else f"MISSING ({detail})"
        print(f"  {name:<12} {status}")

    if python_ok and all_ok:
        print("\nEnvironment OK.")
        return 0
    print("\nEnvironment NOT ready - see the lines above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())