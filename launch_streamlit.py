try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    load_dotenv = None

import dad_streamlit  # noqa: F401 - importing runs the Streamlit entrypoint
