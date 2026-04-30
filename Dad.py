from dadbot.app_runtime import ensure_streamlit_app_file, main as run_app_main  # noqa: F401
from dadbot.background import BackgroundTaskManager  # noqa: F401 - re-exported for tests
from dadbot.core.dadbot import *  # noqa: F403
from dadbot_system import DadServiceClient  # noqa: F401 - re-exported for tests

if __name__ == "__main__":
    raise SystemExit(run_app_main(dadbot_cls=DadBot, script_path=__file__))
