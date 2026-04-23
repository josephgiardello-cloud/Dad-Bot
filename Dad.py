from dadbot.app_runtime import main as run_app_main
from dadbot.core.dadbot import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(run_app_main(dadbot_cls=DadBot, script_path=__file__))
