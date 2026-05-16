#!/usr/bin/env python3
"""Thin entrypoint for launching the Dad Bot FastAPI service.

This wrapper keeps the public startup path explicit while reusing the
canonical app_runtime launch flow.
"""

from __future__ import annotations

import sys

import launch


def main() -> int:
    return int(launch.main(["--api", *sys.argv[1:]]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
