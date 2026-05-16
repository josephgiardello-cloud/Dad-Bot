#!/usr/bin/env python3
"""Canonical launcher entrypoint for Dad Bot."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--ui", action="store_true", default=True)
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--api", action="store_true")
    parsed, passthrough = parser.parse_known_args(args)

    launch_args = list(passthrough)
    if parsed.api:
        launch_args = [arg for arg in launch_args if arg != "--cli"]
        launch_args.append("--serve-api")
    elif parsed.cli:
        launch_args = [arg for arg in launch_args if arg != "--serve-api"]
        launch_args.append("--cli")

    return app_main(_normalize_launch_args(launch_args), dadbot_cls=DadBot, script_path=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
