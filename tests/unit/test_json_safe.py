"""Unit tests for _json_safe — type coercion and fallback behaviour."""
from __future__ import annotations

import json

import pytest
pytestmark = pytest.mark.unit
from dadbot.core.graph import _json_safe


class TestJsonSafeScalars:
    def test_string_passthrough(self):
        assert _json_safe("hello") == "hello"

    def test_int_passthrough(self):
        assert _json_safe(42) == 42

    def test_float_passthrough(self):
        assert _json_safe(3.14) == pytest.approx(3.14)

    def test_bool_passthrough(self):
        assert _json_safe(True) is True
        assert _json_safe(False) is False

    def test_none_passthrough(self):
        assert _json_safe(None) is None


class TestJsonSafeBytes:
    def test_bytes_encoded_as_metadata_dict(self):
        result = _json_safe(b"hello world")
        assert result == {"type": "bytes", "size": 11}

    def test_bytes_non_utf8_still_metadata_dict(self):
        result = _json_safe(b"\xff\xfe")
        assert result == {"type": "bytes", "size": 2}

    def test_empty_bytes(self):
        assert _json_safe(b"") == {"type": "bytes", "size": 0}


class TestJsonSafeCollections:
    def test_set_converts_to_list(self):
        result = _json_safe({1, 2, 3})
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]

    def test_frozenset_falls_back_to_repr(self):
        result = _json_safe(frozenset({"a", "b"}))
        assert isinstance(result, str)
        assert "frozenset" in result

    def test_list_elements_recursed(self):
        result = _json_safe([b"bytes", {1, 2}])
        assert result[0] == {"type": "bytes", "size": 5}
        assert isinstance(result[1], list)

    def test_tuple_converts_to_list(self):
        result = _json_safe((1, "x", b"y"))
        assert isinstance(result, list)
        assert result == [1, "x", {"type": "bytes", "size": 1}]

    def test_dict_values_recursed(self):
        result = _json_safe({"key": b"val", "set": {9}})
        assert result["key"] == {"type": "bytes", "size": 3}
        assert isinstance(result["set"], list)

    def test_nested_dict_fully_recursed(self):
        nested = {"a": {"b": {"c": b"deep"}}}
        result = _json_safe(nested)
        assert result["a"]["b"]["c"] == {"type": "bytes", "size": 4}


class TestJsonSafeNonSerializable:
    def test_object_falls_back_to_repr(self):
        class _Custom:
            def __repr__(self):
                return "<Custom>"

        result = _json_safe(_Custom())
        assert isinstance(result, str)
        assert "<Custom>" in result

    def test_callable_falls_back_to_repr(self):
        result = _json_safe(lambda: None)
        assert isinstance(result, str)

    def test_complex_falls_back(self):
        result = _json_safe(1 + 2j)
        assert isinstance(result, str)


class TestJsonSafeEndToEnd:
    def test_deeply_mixed_structure_is_serialisable(self):
        mixed = {
            "text": "hello",
            "num": 99,
            "raw": b"\x00\x01",
            "collection": {4, 5, 6},
            "nested": {"inner": b"data", "set": frozenset({"x"})},
        }
        safe = _json_safe(mixed)
        serialised = json.dumps(safe)
        assert "hello" in serialised
        assert "99" in serialised or 99 in json.loads(serialised).values()

    def test_empty_dict_passthrough(self):
        assert _json_safe({}) == {}

    def test_empty_list_passthrough(self):
        assert _json_safe([]) == []
