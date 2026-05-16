#!/usr/bin/env python3
"""Canonical launcher entrypoint for Dad Bot."""

from __future__ import annotations

import sys

from dadbot.app_runtime import main as app_main
from dadbot.core.dadbot import DadBot


def _normalize_launch_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    for arg in argv:
        if arg in {"--ui", "--web"}:
            continue
        if arg == "--api":
            normalized.append("--serve-api")
            continue
        normalized.append(arg)
    return normalized


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    return app_main(_normalize_launch_args(args), dadbot_cls=DadBot, script_path=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
