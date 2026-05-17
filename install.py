#!/usr/bin/env python3
"""DadBot installer and launcher.

Run ``python install.py`` for the minimal local UI install.
Add ``--with-service`` or ``--with-voice`` to opt into heavier extras.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REQUIRED_MODELS = ["llama3.2", "nomic-embed-text"]
OPTIONAL_MODELS = ["flux"]  # large image model — skip if slow


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Install DadBot and optionally launch the local UI")
    parser.add_argument("--with-service", action="store_true", help="Install API/Redis/Postgres service extras")
    parser.add_argument("--with-voice", action="store_true", help="Install voice/whisper extras")
    parser.add_argument("--with-dev", action="store_true", help="Install developer/test extras")
    parser.add_argument("--skip-model-pull", action="store_true", help="Do not pull Ollama models during setup")
    parser.add_argument("--no-launch", action="store_true", help="Install and prepare files without launching the UI")
    return parser.parse_args(argv)


def build_install_target(args) -> str:
    extras: list[str] = []
    if args.with_dev:
        extras.append("dev")
    if args.with_service:
        extras.append("service")
    if args.with_voice:
        extras.append("voice")
    if not extras:
        return "."
    return ".[" + ",".join(extras) + "]"


def check_python():
    if sys.version_info < (3, 13):
        print(f"❌ Python 3.13+ required (you have {sys.version_info.major}.{sys.version_info.minor})")
        print("   Download from: https://www.python.org/downloads/")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")


def install_dependencies(args):
    print("\n📦 Installing dependencies...")
    install_target = build_install_target(args)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", install_target])
    except subprocess.CalledProcessError as e:
        print("❌ Dependency installation failed.")
        print("   Error:", str(e))
        print("\n   Troubleshooting:")
        print("   1. Ensure you're using a fresh Python virtual environment")
        print("   2. Check that pip is up-to-date: python -m pip install --upgrade pip")
        print("   3. On Windows, you may need Visual Studio Build Tools for C++ support")
        print("   4. Re-run with only the extras you need, for example:")
        print("      python install.py --with-service")
        sys.exit(1)
    print("✅ Dependencies installed")
    if install_target == ".":
        print("   Installed base UI/runtime only. Use --with-service and/or --with-voice for optional features.")


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
        subprocess.check_call([
            sys.executable,
            "launch.py",
        ])
    except subprocess.CalledProcessError as e:
        print(f"❌ Launch failed with error code {e.returncode}")
        print("\n   Troubleshooting:")
        print("   • Check that Ollama is running: ollama serve")
        print("   • Port 8501 already in use? Check: lsof -i :8501")
        print("   • Try launching manually: python launch.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to launch: {e}")
        print("\n   Try manually: python launch.py")


def main(argv=None):
    args = parse_args(argv)
    print("🧔 DadBot Installer & Launcher\n" + "=" * 35)
    check_python()
    install_dependencies(args)
    setup_static_and_profile()
    if check_ollama():
        if not args.skip_model_pull:
            pull_models()
        if not args.no_launch:
            launch()
    else:
        print("\nAfter starting Ollama, run this script again or just run:")
        print("   python launch.py")


if __name__ == "__main__":
    main()
