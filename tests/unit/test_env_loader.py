from __future__ import annotations

from pathlib import Path

import pytest

from dadbot.runtime.env_loader import (
    bootstrap_environment,
    load_env_file,
    validate_startup_environment,
)


pytestmark = pytest.mark.unit


def test_load_env_file_parses_and_preserves_existing_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "export DADBOT_ALPHA=one",
                "DADBOT_BETA='two words'",
                "DADBOT_GAMMA=three",
                "",
            ],
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("DADBOT_GAMMA", "already-set")
    loaded = load_env_file(env_file, override=False)

    assert loaded["DADBOT_ALPHA"] == "one"
    assert loaded["DADBOT_BETA"] == "two words"
    assert "DADBOT_GAMMA" not in loaded


def test_bootstrap_environment_skips_in_pytest_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "unit::x")
    monkeypatch.delenv("DADBOT_LOAD_ENV_IN_TESTS", raising=False)
    loaded = bootstrap_environment()
    assert loaded == {}


def test_validate_startup_environment_requires_token_for_api_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DADBOT_API_AUTH_REQUIRED", "1")
    monkeypatch.delenv("DADBOT_API_TOKEN_SECRET", raising=False)

    errors = validate_startup_environment(serve_api=True)
    assert any("DADBOT_API_TOKEN_SECRET" in message for message in errors)


def test_validate_startup_environment_rejects_short_api_token_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DADBOT_API_AUTH_REQUIRED", "1")
    monkeypatch.setenv("DADBOT_API_TOKEN_SECRET", "short-secret")

    errors = validate_startup_environment(serve_api=True)
    assert any("at least 32 characters" in message for message in errors)


def test_validate_startup_environment_requires_egress_allowlist_when_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DADBOT_EGRESS_ENFORCE", "1")
    monkeypatch.delenv("DADBOT_EGRESS_ALLOWLIST", raising=False)

    errors = validate_startup_environment(serve_api=False)
    assert any("DADBOT_EGRESS_ALLOWLIST" in message for message in errors)


def test_validate_startup_environment_rejects_invalid_story_mode_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DADBOT_STORY_MODE_PASSWORD_SHA256", "xyz")

    errors = validate_startup_environment(serve_api=False)
    assert any("DADBOT_STORY_MODE_PASSWORD_SHA256" in message for message in errors)
