"""Package entrypoint for the Streamlit UI surface."""

from __future__ import annotations

from dad_streamlit import main as _streamlit_main


def run() -> int:
    return int(_streamlit_main() or 0)


def main() -> int:
    return run()


__all__ = ["main", "run"]