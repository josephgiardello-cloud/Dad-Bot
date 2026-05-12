"""Tests for centralized persistence schema config resolution."""

import os
import pytest

from dadbot.core.persistence_schema_config import is_strict_persistence_schema_mode


class TestStrictPersistenceSchemaConfig:
    """Test centralized strict schema config resolution."""

    def test_service_flag_takes_precedence_over_env(self, monkeypatch):
        """Service strict_mode flag should take precedence over environment."""
        monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", "0")
        # Service flag True overrides env False
        assert is_strict_persistence_schema_mode(service_strict_mode=True) is True

    def test_service_flag_false_overrides_env_true(self, monkeypatch):
        """Service strict_mode=False should override env=true."""
        monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", "1")
        # Service flag False overrides env True
        assert is_strict_persistence_schema_mode(service_strict_mode=False) is False

    def test_env_flag_true_values(self, monkeypatch):
        """Environment flag should recognize truthy values."""
        for val in ("1", "true", "True", "TRUE", "yes", "Yes", "on", "ON"):
            monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", val)
            assert (
                is_strict_persistence_schema_mode(service_strict_mode=None) is True
            ), f"Failed for value: {val}"

    def test_env_flag_false_values(self, monkeypatch):
        """Environment flag should recognize falsy values."""
        for val in ("0", "false", "False", "FALSE", "no", "No", "off", "OFF", "", "random"):
            monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", val)
            assert (
                is_strict_persistence_schema_mode(service_strict_mode=None) is False
            ), f"Failed for value: {val}"

    def test_env_not_set_defaults_to_false(self, monkeypatch):
        """When env var is not set, should default to False (permissive)."""
        monkeypatch.delenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", raising=False)
        assert is_strict_persistence_schema_mode(service_strict_mode=None) is False

    def test_no_args_checks_only_env(self, monkeypatch):
        """When called with no args, should check only env and default."""
        monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", "1")
        assert is_strict_persistence_schema_mode() is True

        monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", "0")
        assert is_strict_persistence_schema_mode() is False

    def test_service_none_explicitly_checks_env(self, monkeypatch):
        """When service_strict_mode=None, should check env and default."""
        monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", "1")
        assert is_strict_persistence_schema_mode(service_strict_mode=None) is True

    def test_whitespace_handling_in_env(self, monkeypatch):
        """Environment flag should handle whitespace correctly."""
        monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", "  true  ")
        assert is_strict_persistence_schema_mode(service_strict_mode=None) is True

        monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", "  0  ")
        assert is_strict_persistence_schema_mode(service_strict_mode=None) is False
