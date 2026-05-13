#!/usr/bin/env python3
"""
DadBot One-Command Installer + Launcher
Run: python install.py
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REQUIRED_MODELS = ["llama3.2", "nomic-embed-text"]
OPTIONAL_MODELS = ["flux"]  # large image model — skip if slow


def check_python():
    if sys.version_info < (3, 13):
        print(f"❌ Python 3.13+ required (you have {sys.version_info.major}.{sys.version_info.minor})")
        print("   Download from: https://www.python.org/downloads/")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")


def install_dependencies():
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[voice,service]"])
    except subprocess.CalledProcessError as e:
        print("❌ Dependency installation failed.")
        print("   Error:", str(e))
        print("\n   Troubleshooting:")
        print("   1. Ensure you're using a fresh Python virtual environment")
        print("   2. Check that pip is up-to-date: python -m pip install --upgrade pip")
        print("   3. On Windows, you may need Visual Studio Build Tools for C++ support")
        print("   4. Run again with: python install.py")
        sys.exit(1)
    print("✅ Dependencies installed")


def setup_static_and_profile():
    Path("static").mkdir(exist_ok=True)
    Path("session_logs").mkdir(exist_ok=True)
    print("✅ Static & session_logs folders ready")

    profile = Path("dad_profile.json")
    if not profile.exists():
        template = Path("dad_profile.template.json")
        if template.exists():
            shutil.copy(template, profile)
            print("✅ Created dad_profile.json from template")
        else:
            profile.write_text(
                json.dumps(
                    {
                        "name": "Dad",
                        "voice": {"tts_backend": "pyttsx3", "piper_model_path": ""},
                        "avatar": {},
                        "ical_feed_url": "",
                    },
                    indent=2,
                )
            )
            print("✅ Created minimal dad_profile.json")

    memory = Path("dad_memory.json")
    if not memory.exists():
        memory.write_text("{}")
        print("✅ Created empty dad_memory.json")


def _run_ollama(*args):
    try:
        return subprocess.check_output(["ollama", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None


def check_ollama():
    print("\n🔍 Checking Ollama...")
    version = _run_ollama("--version")
    if version is None:
        print("⚠️  Ollama not found or not running.")
        print("\n   📥 Install Ollama:")
        print("      • macOS/Linux: https://ollama.com/download")
        print("      • Windows: https://ollama.com/download/windows")
        print("\n   After installation, run:")
        print("      ollama pull llama3.2")
        print("      ollama pull nomic-embed-text")
        print("\n   Then run this installer again:")
        print("      python install.py")
        return False
    print(f"✅ Ollama {version.strip()}")
    return True


def pull_models():
    print("\n📥 Pulling required models...")
    print("   (This may take several minutes on first run)\n")
    for model in REQUIRED_MODELS:
        print(f"   • {model}...", end=" ", flush=True)
        try:
            subprocess.check_call(
                ["ollama", "pull", model],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("✅")
        except Exception as e:
            print(f"❌\n      Error: {e}")
            print("      Try manually: ollama pull " + model)
            print("      Then re-run: python install.py")

    print("\n   Optional models:")
    for model in OPTIONAL_MODELS:
        print(f"   • {model} (large, ~5GB)...", end=" ", flush=True)
        try:
            subprocess.check_call(
                ["ollama", "pull", model],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("✅")
        except Exception:
            print("⏭️  Skipped (optional, not required for chat)")


def launch():
    print("\n🚀 Launching DadBot...")
    print("   Opening browser to http://localhost:8501")
    time.sleep(1)
    try:
        # Use streamlit command for better subprocess management
        subprocess.check_call([
            sys.executable, "-m", "streamlit", "run", 
            "dad_streamlit.py", 
            "--logger.level=info"
        ])
    except subprocess.CalledProcessError as e:
        print(f"❌ Launch failed with error code {e.returncode}")
        print("\n   Troubleshooting:")
        print("   • Check that Ollama is running: ollama serve")
        print("   • Port 8501 already in use? Check: lsof -i :8501")
        print("   • Try launching manually: streamlit run dad_streamlit.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to launch: {e}")
        print("\n   Try manually: python launch.py")


def main():
    print("🧔 DadBot Installer & Launcher\n" + "=" * 35)
    check_python()
    install_dependencies()
    setup_static_and_profile()
    if check_ollama():
        pull_models()
        launch()
    else:
        print("\nAfter starting Ollama, run this script again or just run:")
        print("   python launch.py")


if __name__ == "__main__":
    main()
