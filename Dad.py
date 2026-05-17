import sys

import launch
from dadbot.app_runtime import ensure_streamlit_app_file  # noqa: F401 - re-exported for tests
from dadbot.background import BackgroundTaskManager  # noqa: F401 - re-exported for tests
from dadbot.core.dadbot import *  # noqa: F403
from dadbot_system import DadServiceClient  # noqa: F401 - re-exported for tests


def main() -> int:
    return int(launch.main(sys.argv[1:]) or 0)

if __name__ == "__main__":
    raise SystemExit(main())
