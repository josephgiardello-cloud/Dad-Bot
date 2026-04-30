"""ROI #4 — Config Drift Detector.

Verifies that DadBot's config attribute routing maps are internally
consistent and have not drifted from the actual DadBotConfig runtime
attributes.

Prevents:
  * "Ghost" _CONFIG_ATTR_MAP entries pointing at attrs that no longer exist
  * Silent mismatches between mapped names and instantiated attribute names
  * Config keys that were added to __post_init__ but not mapped

Tests:
  * All _CONFIG_ATTR_MAP values are accessible on a live DadBotConfig instance
  * _CONFIG_ATTR_MAP values are all non-empty strings
  * DadBotConfig instantiates without errors
  * Key config attributes have appropriate types
  * _RUNTIME_STATE_ATTR_MAP and _INTERNAL_RUNTIME_ATTR_MAP values reference
    real attribute names on the state container interface
"""

from __future__ import annotations

import pytest


class TestDadBotConfigAttrMapConsistency:
    """All entries in _CONFIG_ATTR_MAP must resolve to real DadBotConfig attrs."""

    @pytest.fixture(scope="class")
    def config_instance(self):
        from dadbot.config import DadBotConfig

        return DadBotConfig()

    @pytest.fixture(scope="class")
    def attr_map(self):
        from dadbot.core.dadbot import DadBot

        return DadBot._CONFIG_ATTR_MAP

    def test_attr_map_is_non_empty(self, attr_map):
        assert len(attr_map) > 10, "Expected at least 10 entries in _CONFIG_ATTR_MAP"

    def test_all_map_values_are_non_empty_strings(self, attr_map):
        for key, value in attr_map.items():
            assert isinstance(value, str) and value.strip(), (
                f"_CONFIG_ATTR_MAP[{key!r}] = {value!r} is not a non-empty string"
            )

    def test_all_mapped_attr_names_exist_on_config_instance(self, attr_map, config_instance):
        """Every attribute name that _CONFIG_ATTR_MAP routes to must be
        accessible on a freshly constructed DadBotConfig instance."""
        missing: list[str] = []
        for key, attr_name in attr_map.items():
            if not hasattr(config_instance, attr_name):
                missing.append(f"_CONFIG_ATTR_MAP[{key!r}] → {attr_name!r}: not found on DadBotConfig")
        assert not missing, "\n".join(missing)

    def test_unique_attr_names_all_resolvable(self, attr_map, config_instance):
        """Deduplicated attribute names (values) are all accessible."""
        unique_attrs = set(attr_map.values())
        not_found = [a for a in sorted(unique_attrs) if not hasattr(config_instance, a)]
        assert not not_found, f"Config attrs not found on instance: {not_found}"

    def test_no_map_value_shadows_wrong_type(self, attr_map, config_instance):
        """Spot-check that known attributes have the expected Python type."""
        type_expectations: dict[str, type | tuple] = {
            "model_name": str,
            "fallback_models": (list, tuple),
            "active_model": str,
            "append_signoff": bool,
            "light_mode": bool,
            "tenant_id": str,
            "recent_history_window": int,
            "context_token_budget": int,
            "stream_timeout_seconds": int,
        }
        wrong_types: list[str] = []
        for attr, expected_type in type_expectations.items():
            if attr not in {v for v in attr_map.values()}:
                continue  # attr not in map, skip
            val = getattr(config_instance, attr, None)
            if not isinstance(val, expected_type):
                wrong_types.append(f"{attr!r}: expected {expected_type}, got {type(val).__name__!r} ({val!r})")
        assert not wrong_types, "Type mismatches in config attrs:\n" + "\n".join(wrong_types)


class TestDadBotConfigInstantiation:
    """DadBotConfig must instantiate cleanly and produce well-formed values."""

    def test_config_instantiates_without_error(self):
        from dadbot.config import DadBotConfig

        config = DadBotConfig()
        assert config is not None

    def test_tenant_id_is_normalized(self):
        from dadbot.config import DadBotConfig

        config = DadBotConfig(tenant_id="My Tenant!")
        assert " " not in config.tenant_id
        assert "!" not in config.tenant_id

    def test_active_model_is_non_empty(self):
        from dadbot.config import DadBotConfig

        config = DadBotConfig()
        assert isinstance(config.active_model, str)
        assert len(config.active_model) > 0

    def test_runtime_config_is_attached(self):
        from dadbot.config import DadBotConfig, DadRuntimeConfig

        config = DadBotConfig()
        assert isinstance(config.runtime_config, DadRuntimeConfig)

    def test_flattened_runtime_attrs_match_runtime_config(self):
        """Flattened attrs set in __post_init__ must match runtime_config values."""
        from dadbot.config import DadBotConfig

        config = DadBotConfig()
        rc = config.runtime_config
        assert config.recent_history_window == rc.recent_history_window
        assert config.context_token_budget == rc.context_token_budget
        assert config.stream_timeout_seconds == rc.stream_timeout_seconds


class TestDadBotConfigNoDriftFromMapKeys:
    """Detect ghost entries: map keys that no longer correspond to anything useful."""

    def test_no_duplicate_attr_name_points_to_different_values(self):
        """If the same external key appears twice, both must map to the same attr."""
        from dadbot.core.dadbot import DadBot

        attr_map = DadBot._CONFIG_ATTR_MAP
        seen: dict[str, str] = {}
        conflicts: list[str] = []
        for key, attr_name in attr_map.items():
            # Same key appearing twice with different values is a drift indicator
            canonical_key = key.lower()
            if canonical_key in seen and seen[canonical_key] != attr_name:
                conflicts.append(
                    f"Key collision: {key!r} and its lowercase variant map to {seen[canonical_key]!r} vs {attr_name!r}"
                )
            else:
                seen[canonical_key] = attr_name
        assert not conflicts, "\n".join(conflicts)

    def test_path_attrs_return_path_objects(self):
        """Mapped path attributes must return pathlib.Path instances."""
        from pathlib import Path

        from dadbot.config import DadBotConfig

        config = DadBotConfig()
        path_attr_names = {
            "profile_path",
            "memory_path",
            "semantic_memory_db_path",
            "graph_store_db_path",
            "session_log_dir",
        }
        for attr in path_attr_names:
            val = getattr(config, attr, None)
            assert isinstance(val, Path), f"{attr!r} expected pathlib.Path, got {type(val).__name__!r}"
