from __future__ import annotations

from dadbot.app_runtime import main as run_app_main
from dadbot.core.dadbot import DadBot


def main() -> int:
    return int(run_app_main(dadbot_cls=DadBot, script_path=__file__))


if __name__ == "__main__":
    raise SystemExit(main())
