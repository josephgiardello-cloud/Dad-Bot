#!/usr/bin/env python3
"""Thin entrypoint for launching the Dad Bot FastAPI service.

This wrapper keeps the public startup path explicit while reusing the
canonical app_runtime launch flow.
"""

from __future__ import annotations

import sys

from launch import main as run_launch_main


def main() -> int:
    return run_launch_main(["--api", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
