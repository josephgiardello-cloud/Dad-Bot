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


def _copytree_ignore_runtime_artifacts(_src: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name == "__pycache__":
            ignored.add(name)
        elif name.endswith((".pyc", ".pyo", ".pyd")):
            ignored.add(name)
    return ignored


def _prepare_streamlit_sandbox(repo_root: Path, sandbox_root: Path) -> None:
    for file_name in ("Dad.py", "dad_streamlit.py", "dad_profile.json", "dad_profile.template.json"):
        shutil.copy2(repo_root / file_name, sandbox_root / file_name)

    shutil.copytree(repo_root / "dadbot", sandbox_root / "dadbot", ignore=_copytree_ignore_runtime_artifacts)
    shutil.copytree(repo_root / "dadbot_system", sandbox_root / "dadbot_system", ignore=_copytree_ignore_runtime_artifacts)
    shutil.copytree(repo_root / "static", sandbox_root / "static", ignore=_copytree_ignore_runtime_artifacts)

    source_streamlit_dir = repo_root / ".streamlit"
    target_streamlit_dir = sandbox_root / ".streamlit"
    if source_streamlit_dir.exists():
        shutil.copytree(source_streamlit_dir, target_streamlit_dir, ignore=_copytree_ignore_runtime_artifacts)
        return

    target_streamlit_dir.mkdir()
    (target_streamlit_dir / "config.toml").write_text(
        "[server]\nenableStaticServing = true\n",
        encoding="utf-8",
    )


def test_ensure_streamlit_app_file_rejects_missing_entrypoint(tmp_path):
    target = tmp_path / "dad_streamlit.py"

    with pytest.raises(FileNotFoundError, match="Required Streamlit app entrypoint is missing"):
        ensure_streamlit_app_file(target)

    assert not target.exists()


def test_ensure_streamlit_app_file_preserves_existing_entrypoint(tmp_path):
    target = tmp_path / "dad_streamlit.py"
    target.write_text("real application", encoding="utf-8")

    assert ensure_streamlit_app_file(target) is False
    assert target.read_text(encoding="utf-8") == "real application"


def test_dad_streamlit_app_starts_without_streamlit_exceptions(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    sandbox_root = tmp_path / "app_sandbox"
    sandbox_root.mkdir()
    _prepare_streamlit_sandbox(repo_root, sandbox_root)

    smoke_code = textwrap.dedent(
        """
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file("dad_streamlit.py").run(timeout=30)
        button_labels = [str(button.label) for button in app.button]
        radio_labels = [str(r.label) for r in app.radio]

        assert len(app.exception) == 0, list(app.exception)
        assert len(app.error) == 0, [item.value for item in app.error]
        assert "New chat" in button_labels, button_labels
        assert "Navigation" in radio_labels, radio_labels

        # Availability-only check: route load succeeds and workshop section is present.
        nav_radio = next(r for r in app.radio if r.label == "Navigation")
        app = nav_radio.set_value("Workshop").run(timeout=30)
        assert len(app.exception) == 0, list(app.exception)
        assert len(app.error) == 0, [item.value for item in app.error]

        workshop_labels = [str(button.label) for button in app.button]
        assert "Run live probe" in workshop_labels, workshop_labels
        """
    )

    _dadbot_keys = {k for k in os.environ if k.startswith("DADBOT_")}
    _smoke_env = {k: v for k, v in os.environ.items() if k not in _dadbot_keys}
    _smoke_env["PYTHONIOENCODING"] = "utf-8"
    _smoke_env["DADBOT_TENANT_ID"] = "tenant-streamlit-smoke"

    result = subprocess.run(
        [sys.executable, "-c", smoke_code],
        cwd=sandbox_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env=_smoke_env,
    )

    assert result.returncode == 0, (
        f"dad_streamlit.py failed smoke startup.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_preferences_tab_shows_detected_cloud_llm_api_key_hint(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    sandbox_root = tmp_path / "app_sandbox"
    sandbox_root.mkdir()
    _prepare_streamlit_sandbox(repo_root, sandbox_root)

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

        app = AppTest.from_string(
            "from dadbot.core.dadbot import DadBot\\n"
            "from dadbot.ui.preferences import render_preferences_tab\\n"
            "render_preferences_tab(DadBot())\\n"
        ).run(timeout=30)

        assert len(app.exception) == 0, list(app.exception)
        assert len(app.error) == 0, [item.value for item in app.error]

        captions = [str(item.value) for item in app.caption]
        assert any("OPENAI_API_KEY" in caption and "detected" in caption for caption in captions), captions
        """
    )

    _dadbot_keys_pref = {k for k in os.environ if k.startswith("DADBOT_")}
    _pref_env = {k: v for k, v in os.environ.items() if k not in _dadbot_keys_pref}
    _pref_env["OPENAI_API_KEY"] = "test-openai-key"
    _pref_env["PYTHONIOENCODING"] = "utf-8"
    _pref_env["DADBOT_TENANT_ID"] = "tenant-streamlit-preferences"

    result = subprocess.run(
        [sys.executable, "-c", smoke_code],
        cwd=sandbox_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
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
    _prepare_streamlit_sandbox(repo_root, sandbox_root)

    smoke_code = textwrap.dedent(
        """
        from streamlit.testing.v1 import AppTest

        def collect_labels(app, buttons, checkboxes, selects):
            buttons.update(str(item.label) for item in app.button)
            checkboxes.update(str(item.label) for item in app.checkbox)
            selects.update(str(item.label) for item in app.selectbox)

        app = AppTest.from_file("dad_streamlit.py").run(timeout=30)
        assert len(app.exception) == 0, list(app.exception)
        assert len(app.error) == 0, [item.value for item in app.error]

        seen_buttons = set()
        seen_checkboxes = set()
        seen_selects = set()
        collect_labels(app, seen_buttons, seen_checkboxes, seen_selects)

        nav_radio = next(r for r in app.radio if r.label == "Navigation")
        for view in ("Status", "Workshop", "Voice", "Chat"):
            app = nav_radio.set_value(view).run(timeout=30)
            assert len(app.exception) == 0, list(app.exception)
            assert len(app.error) == 0, [item.value for item in app.error]
            collect_labels(app, seen_buttons, seen_checkboxes, seen_selects)
            nav_radio = next(r for r in app.radio if r.label == "Navigation")

        expected_buttons = {
            "New chat",
            "Duplicate",
            "New",
            "Fork",
            "Probe one turn",
            "Run live probe",
            "Clear chat history",
            "Reset MQTT session",
            "Reset voice runtime state",
        }

        assert expected_buttons.issubset(seen_buttons), sorted(expected_buttons - seen_buttons)
        assert "Jor-El source mode" in seen_checkboxes, sorted(seen_checkboxes)
        assert "Active conversation" in seen_selects, sorted(seen_selects)
        """
    )

    # Strip any DADBOT_* env overrides so the subprocess always derives paths
    # from the sandbox root rather than inheriting leaked test-scoped env vars.
    _dadbot_keys = {k for k in os.environ if k.startswith("DADBOT_")}
    sandbox_env = {k: v for k, v in os.environ.items() if k not in _dadbot_keys}
    sandbox_env["PYTHONIOENCODING"] = "utf-8"
    sandbox_env["DADBOT_TENANT_ID"] = "tenant-streamlit-controls"

    result = subprocess.run(
        [sys.executable, "-c", smoke_code],
        cwd=sandbox_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=sandbox_env,
    )

    assert result.returncode == 0, (
        "dad_streamlit.py failed sidebar/button surface smoke test.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
