#!/usr/bin/env python3
"""Thin entrypoint for launching the Dad Bot FastAPI service.

This wrapper keeps the public startup path explicit while reusing the
canonical app_runtime launch flow.
"""

from __future__ import annotations

import sys

from dadbot.app_runtime import main as run_app_main
from Dad import DadBot


def main() -> int:
    return run_app_main(["--serve-api", *sys.argv[1:]], dadbot_cls=DadBot, script_path=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
