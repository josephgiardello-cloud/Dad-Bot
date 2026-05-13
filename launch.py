#!/usr/bin/env python3
"""
Quick launcher — double-click friendly.
Run: python launch.py

Starts the DadBot web UI and opens it in your default browser.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def launch_streamlit():
    """Launch DadBot via Streamlit."""
    print("🧔 Starting DadBot...\n")
    
    # Verify Ollama is available
    try:
        subprocess.check_output(
            ["ollama", "--version"],
            stderr=subprocess.DEVNULL,
            text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Ollama not found or not running")
        print("\n   Make sure Ollama is installed and running:")
        print("   1. Download from https://ollama.com")
        print("   2. Start Ollama service")
        print("   3. Run: ollama pull llama3.2")
        print("\n   Then try again: python launch.py")
        sys.exit(1)
    
    try:
        # Start Streamlit
        print("   Opening browser to http://localhost:8501...\n")
        time.sleep(0.5)
        
        # Try to open browser (non-blocking)
        webbrowser.open("http://localhost:8501", new=2)
        
        # Start Streamlit and wait for it
        subprocess.check_call([
            sys.executable, "-m", "streamlit", "run",
            "dad_streamlit.py",
            "--logger.level=info",
            "--client.showErrorDetails=true"
        ])
    except KeyboardInterrupt:
        print("\n👋 DadBot stopped.")
        sys.exit(0)
    except FileNotFoundError:
        print("❌ Streamlit not found. Try reinstalling:")
        print("   python install.py")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error (code {e.returncode}):")
        print("\n   Troubleshooting:")
        print("   • Check Ollama is running: ollama serve")
        print("   • Port 8501 in use? Kill it or use: streamlit run dad_streamlit.py --server.port 8502")
        print("   • Full reinstall: python install.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    launch_streamlit()
