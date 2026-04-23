#!/usr/bin/env python3
"""
Quick launcher — double-click friendly.
Run: python launch.py
"""

import subprocess
import sys
import time
import webbrowser


def main():
    print("🧔 Starting DadBot...")
    try:
        proc = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "dad_streamlit.py",
            "--server.headless=false",
            "--server.port=8501",
        ])

        # Give Streamlit a moment to start before opening browser
        time.sleep(4)
        webbrowser.open("http://localhost:8501")

        print("✅ DadBot is running at http://localhost:8501")
        print("   Browser should open automatically.")
        print("   Press Ctrl+C to stop.\n")
        proc.wait()
    except KeyboardInterrupt:
        print("\n👋 DadBot stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
