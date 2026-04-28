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
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")


def install_dependencies():
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[voice,service]"])
    except subprocess.CalledProcessError:
        print("❌ pip install failed — check the error above and try again.")
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
            profile.write_text(json.dumps({
                "name": "Dad",
                "voice": {"tts_backend": "pyttsx3", "piper_model_path": ""},
                "avatar": {},
                "ical_feed_url": "",
            }, indent=2))
            print("✅ Created minimal dad_profile.json")

    memory = Path("dad_memory.json")
    if not memory.exists():
        memory.write_text("{}")
        print("✅ Created empty dad_memory.json")


def _run_ollama(*args):
    try:
        return subprocess.check_output(["ollama", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def check_ollama():
    print("\n🔍 Checking Ollama...")
    if _run_ollama("--version") is None:
        print("⚠️  Ollama not detected or not running.")
        print("   Please install Ollama from https://ollama.com and run it.")
        print("   Then pull recommended models: ollama pull llama3.2")
        return False
    print(f"✅ Ollama detected: {_run_ollama('--version')}")
    return True


def pull_models():
    print("\n📥 Pulling required models (may take a few minutes on first run)...")
    for model in REQUIRED_MODELS:
        print(f"   Pulling {model}...", end=" ", flush=True)
        try:
            subprocess.check_call(
                ["ollama", "pull", model],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print("✅")
        except Exception:
            print("⚠️  Failed — will retry on next launch.")

    for model in OPTIONAL_MODELS:
        print(f"   Pulling {model} (optional)...", end=" ", flush=True)
        try:
            subprocess.check_call(
                ["ollama", "pull", model],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print("✅")
        except Exception:
            print("⚠️  Skipped (not required for text chat).")


def launch():
    print("\n🚀 Launching DadBot...")
    time.sleep(1)
    try:
        # Route through Dad.py so app_runtime.main() enforces all startup
        # safety checks, contract validation, and resource guards.
        subprocess.Popen([sys.executable, "Dad.py"])
        print("✅ DadBot should open in your browser shortly at http://localhost:8501")
        print("   If it doesn't open automatically, navigate there manually.")
    except Exception as e:
        print(f"❌ Failed to launch: {e}")


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
