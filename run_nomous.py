#!/usr/bin/env python3
"""Nomous one-command launcher.

This script wraps :mod:`scripts.start` so new users can run a single command to
bootstrap the environment, install dependencies, and start the application. It
provides a friendlier error message if Python modules required by the launcher
are missing from the host interpreter.
"""

from __future__ import annotations

import importlib
import sys


def main() -> int:
    """Import and execute ``scripts.start.main`` with defensive error handling."""

    try:
        start_module = importlib.import_module("scripts.start")
    except ModuleNotFoundError as exc:  # pragma: no cover - unlikely in repo
        missing = exc.name or "unknown"
        sys.stderr.write(
            "Nomous launcher could not import required module "
            f"'{missing}'.\n"
        )
        sys.stderr.write(
            "Ensure you are running the launcher from the repository root and "
            "that the file structure is intact.\n"
        )
        return 1

    if not hasattr(start_module, "main"):
        sys.stderr.write(
            "The Nomous start module is missing a 'main' entry point.\n"
        )
        return 1

    return int(start_module.main())


if __name__ == "__main__":
    sys.exit(main())
