#!/usr/bin/env python3
"""Thin launcher entrypoint routed through canonical app runtime startup."""

from __future__ import annotations

import sys

from dadbot.app_runtime import main as app_main
from Dad import DadBot


def main() -> int:
    return app_main(["--web", *sys.argv[1:]], dadbot_cls=DadBot, script_path=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
