"""Dad Bot Streamlit app entrypoint and routing helpers.

This module provides a stable package entrypoint while `dad_streamlit.py`
remains the runtime script used by `dadbot.app_runtime`.
"""

from dad_streamlit import main

__all__ = ["main"]
