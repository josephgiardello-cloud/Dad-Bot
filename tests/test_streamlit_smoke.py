import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from Dad import ensure_streamlit_app_file

pytestmark = pytest.mark.ui


def test_ensure_streamlit_app_file_creates_minimal_stub(tmp_path):
    target = tmp_path / "dad_streamlit.py"

    created = ensure_streamlit_app_file(target)

    assert created is True
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "Auto-generated minimal Streamlit chat" in content
    assert "DadBot" in content


def test_dad_streamlit_app_starts_without_streamlit_exceptions(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    sandbox_root = tmp_path / "app_sandbox"
    sandbox_root.mkdir()

    for file_name in ("Dad.py", "dad_streamlit.py", "dad_profile.json", "dad_profile.template.json"):
        shutil.copy2(repo_root / file_name, sandbox_root / file_name)

    shutil.copytree(repo_root / "dadbot", sandbox_root / "dadbot")
    shutil.copytree(repo_root / "dadbot_system", sandbox_root / "dadbot_system")
    shutil.copytree(repo_root / "static", sandbox_root / "static")
    shutil.copytree(repo_root / ".streamlit", sandbox_root / ".streamlit")

    smoke_code = textwrap.dedent(
        """
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file("dad_streamlit.py").run(timeout=15)
        button_labels = [button.label for button in app.button]
        radio_labels = [r.label for r in app.radio]

        assert len(app.exception) == 0, list(app.exception)
        assert len(app.error) == 0, [item.value for item in app.error]
        assert "New Thread" in button_labels, button_labels
        assert "Navigate" in radio_labels, radio_labels
        """
    )

    _dadbot_keys = {k for k in os.environ if k.startswith("DADBOT_")}
    _smoke_env = {k: v for k, v in os.environ.items() if k not in _dadbot_keys}
    _smoke_env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "-c", smoke_code],
        cwd=sandbox_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        env=_smoke_env,
    )

    assert result.returncode == 0, (
        f"dad_streamlit.py failed smoke startup.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_preferences_tab_shows_detected_cloud_llm_api_key_hint(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    sandbox_root = tmp_path / "app_sandbox"
    sandbox_root.mkdir()

    for file_name in ("Dad.py", "dad_streamlit.py", "dad_profile.json", "dad_profile.template.json"):
        shutil.copy2(repo_root / file_name, sandbox_root / file_name)

    shutil.copytree(repo_root / "dadbot", sandbox_root / "dadbot")
    shutil.copytree(repo_root / "dadbot_system", sandbox_root / "dadbot_system")
    shutil.copytree(repo_root / "static", sandbox_root / "static")
    shutil.copytree(repo_root / ".streamlit", sandbox_root / ".streamlit")

    profile_path = sandbox_root / "dad_profile.json"
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_payload["llm"] = {
        "provider": "openai",
        "model": "gpt-4o-mini",
    }
    profile_path.write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")

    smoke_code = textwrap.dedent(
        """
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file("dad_streamlit.py").run(timeout=15)

        assert len(app.exception) == 0, list(app.exception)
        assert len(app.error) == 0, [item.value for item in app.error]

        # Navigate to Dad's Workshop
        nav_radio = next(r for r in app.radio if r.label == "Navigate")
        app = nav_radio.set_value("workshop").run(timeout=15)

        # Select Preferences section in workshop
        workshop_radio = next(r for r in app.radio if r.label == "Workshop section")
        app = workshop_radio.set_value("Preferences").run(timeout=15)

        captions = [str(item.value) for item in app.caption]
        assert any("OPENAI_API_KEY" in caption and "detected" in caption for caption in captions), captions
        """
    )

    _dadbot_keys_pref = {k for k in os.environ if k.startswith("DADBOT_")}
    _pref_env = {k: v for k, v in os.environ.items() if k not in _dadbot_keys_pref}
    _pref_env["OPENAI_API_KEY"] = "test-openai-key"
    _pref_env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "-c", smoke_code],
        cwd=sandbox_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        env=_pref_env,
    )

    assert result.returncode == 0, (
        f"dad_streamlit.py failed LLM preferences hint smoke test.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_mobile_pwa_manifest_exists_and_points_at_static_icons():
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "static" / "dadbot-manifest.webmanifest"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["name"] == "Dad Bot"
    assert manifest["display"] == "standalone"
    assert any(icon["src"] == "/app/static/dadbot-icon.svg" for icon in manifest["icons"])


def test_sidebar_and_button_surface_accessible(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    sandbox_root = tmp_path / "app_sandbox"
    sandbox_root.mkdir()

    for file_name in ("Dad.py", "dad_streamlit.py", "dad_profile.json", "dad_profile.template.json"):
        shutil.copy2(repo_root / file_name, sandbox_root / file_name)

    shutil.copytree(repo_root / "dadbot", sandbox_root / "dadbot")
    shutil.copytree(repo_root / "dadbot_system", sandbox_root / "dadbot_system")
    shutil.copytree(repo_root / "static", sandbox_root / "static")
    shutil.copytree(repo_root / ".streamlit", sandbox_root / ".streamlit")

    smoke_code = textwrap.dedent(
        """
        from streamlit.testing.v1 import AppTest

        def collect_labels(app, buttons, checkboxes, selects):
            buttons.update(str(item.label) for item in app.button)
            checkboxes.update(str(item.label) for item in app.checkbox)
            selects.update(str(item.label) for item in app.selectbox)

        app = AppTest.from_file("dad_streamlit.py").run(timeout=20)
        assert len(app.exception) == 0, list(app.exception)
        assert len(app.error) == 0, [item.value for item in app.error]

        seen_buttons = set()
        seen_checkboxes = set()
        seen_selects = set()
        collect_labels(app, seen_buttons, seen_checkboxes, seen_selects)

        nav_radio = next(r for r in app.radio if r.label == "Navigate")
        for view in ("status", "workshop", "voice", "chat"):
            app = nav_radio.set_value(view).run(timeout=20)
            assert len(app.exception) == 0, list(app.exception)
            assert len(app.error) == 0, [item.value for item in app.error]
            collect_labels(app, seen_buttons, seen_checkboxes, seen_selects)
            nav_radio = next(r for r in app.radio if r.label == "Navigate")

            if view == "workshop":
                section_radio = next(r for r in app.radio if r.label == "Workshop section")
                for section in ("Status", "Preferences", "Data", "Mobile"):
                    app = section_radio.set_value(section).run(timeout=20)
                    assert len(app.exception) == 0, list(app.exception)
                    assert len(app.error) == 0, [item.value for item in app.error]
                    collect_labels(app, seen_buttons, seen_checkboxes, seen_selects)
                    nav_radio = next(r for r in app.radio if r.label == "Navigate")
                    section_radio = next(r for r in app.radio if r.label == "Workshop section")

        expected_buttons = {
            "Open Dad's Workshop",
            "Go to thread",
            "New Thread",
            "📸 Send Photo",
            "Daily Check-in",
            "Evolve Persona",
            "🧵 New Thread",
            "🎙️ Talk",
            "📷 Send Photo",
            "💚 Check-in",
            "💬 Chat",
            "🩺 Status",
            "🛠️ Workshop",
            "🎙️ Voice",
            "Force Consolidation",
            "Clear Semantic Index",
            "Export Memory",
            "Optimize Hardware",
            "Reset session context",
            "Start Mobile Thread",
            "Send Photo to Thread",
            "Switch to Selected Thread",
            "Generate new avatar",
            "Write this week's entry",
            "Search",
        }

        assert expected_buttons.issubset(seen_buttons), sorted(expected_buttons - seen_buttons)
        assert ("Enable Quiet Mode" in seen_buttons) or ("Disable Quiet Mode" in seen_buttons), sorted(seen_buttons)
        assert "Show open threads only" in seen_checkboxes, sorted(seen_checkboxes)
        assert "Switch thread" in seen_selects, sorted(seen_selects)
        """
    )

    # Strip any DADBOT_* env overrides so the subprocess always derives paths
    # from the sandbox root rather than inheriting leaked test-scoped env vars.
    _dadbot_keys = {k for k in os.environ if k.startswith("DADBOT_")}
    sandbox_env = {k: v for k, v in os.environ.items() if k not in _dadbot_keys}
    sandbox_env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "-c", smoke_code],
        cwd=sandbox_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env=sandbox_env,
    )

    assert result.returncode == 0, (
        "dad_streamlit.py failed sidebar/button surface smoke test.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
