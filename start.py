#!/usr/bin/env python3
"""Compatibility shim for legacy ``python start.py`` entry point.

Historically the Nomous launcher lived at ``scripts/start.py``. Some users still
try to invoke ``python start.py`` from the repository root, which previously
resulted in a confusing ``FileNotFoundError``. This thin wrapper simply proxies
to :mod:`run_nomous` so the legacy command continues to work.
"""

from __future__ import annotations

import sys

from run_nomous import main as run_nomous_main


if __name__ == "__main__":
    sys.exit(run_nomous_main())
