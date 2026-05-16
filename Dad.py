import sys

from dadbot.app_runtime import ensure_streamlit_app_file  # noqa: F401
from dadbot.background import BackgroundTaskManager  # noqa: F401 - re-exported for tests
from dadbot.core.dadbot import *  # noqa: F403
from dadbot_system import DadServiceClient  # noqa: F401 - re-exported for tests

from launch import main as run_launch_main


def main() -> int:
    return run_launch_main(sys.argv[1:])

if __name__ == "__main__":
    raise SystemExit(main())
