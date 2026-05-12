"""Unit tests for filesystem_read_tool.

Tests:
  - Read file within root sandbox
  - List directory within root sandbox
  - Path traversal attacks are blocked
  - Absolute paths outside root are blocked
  - File-too-large guard
  - Not-a-file error on read
  - Not-a-directory error on list
  - Round-trip through DynamicToolRegistry / ExternalToolRuntime
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from dadbot.core.external_tool_runtime import DynamicToolRegistry, ExternalToolRuntime, ToolExecutionStatus
from dadbot.tools.filesystem_read_tool import _safe_resolve, build_filesystem_read_tool

pytestmark = pytest.mark.unit


class TestSafeResolve:
    def test_relative_path_inside_root(self, tmp_path):
        result = _safe_resolve(tmp_path, "subdir/file.txt")
        assert result == (tmp_path / "subdir" / "file.txt").resolve()

    def test_traversal_blocked(self, tmp_path):
        result = _safe_resolve(tmp_path, "../../etc/passwd")
        assert result is None

    def test_absolute_path_outside_root_blocked(self, tmp_path):
        result = _safe_resolve(tmp_path, "/etc/passwd")
        assert result is None

    def test_empty_path_returns_none(self, tmp_path):
        result = _safe_resolve(tmp_path, "")
        assert result is None

    def test_path_equal_to_root_is_ok(self, tmp_path):
        result = _safe_resolve(tmp_path, ".")
        assert result is not None
        assert result == tmp_path.resolve()


class TestFilesystemReadToolRead:
    def test_read_file_success(self, tmp_path):
        (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
        cap, handler = build_filesystem_read_tool(str(tmp_path))
        result = handler({"action": "read", "path": "hello.txt"})
        assert result.status == ToolExecutionStatus.OK
        assert result.output["content"] == "hello world"
        assert result.output["size_bytes"] == 11

    def test_read_file_in_subdirectory(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "data.json").write_text('{"x": 1}', encoding="utf-8")
        cap, handler = build_filesystem_read_tool(str(tmp_path))
        result = handler({"action": "read", "path": "sub/data.json"})
        assert result.status == ToolExecutionStatus.OK
        assert '{"x": 1}' in result.output["content"]

    def test_read_missing_file_returns_error(self, tmp_path):
        cap, handler = build_filesystem_read_tool(str(tmp_path))
        result = handler({"action": "read", "path": "nonexistent.txt"})
        assert result.status == ToolExecutionStatus.ERROR
        assert result.error == "path_not_found"

    def test_read_directory_returns_not_a_file_error(self, tmp_path):
        (tmp_path / "adir").mkdir()
        cap, handler = build_filesystem_read_tool(str(tmp_path))
        result = handler({"action": "read", "path": "adir"})
        assert result.status == ToolExecutionStatus.ERROR
        assert result.error == "not_a_file"

    def test_traversal_attack_blocked(self, tmp_path):
        cap, handler = build_filesystem_read_tool(str(tmp_path))
        result = handler({"action": "read", "path": "../../etc/passwd"})
        assert result.status == ToolExecutionStatus.ERROR
        assert result.error == "path_traversal_rejected"

    def test_file_too_large_blocked(self, tmp_path):
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * 1000)
        cap, handler = build_filesystem_read_tool(str(tmp_path), max_file_bytes=100)
        result = handler({"action": "read", "path": "big.bin"})
        assert result.status == ToolExecutionStatus.ERROR
        assert "file_too_large" in result.error


class TestFilesystemReadToolList:
    def test_list_directory_success(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "subdir").mkdir()
        cap, handler = build_filesystem_read_tool(str(tmp_path))
        result = handler({"action": "list", "path": "."})
        assert result.status == ToolExecutionStatus.OK
        names = {e["name"] for e in result.output["entries"]}
        assert "a.txt" in names
        assert "b.txt" in names
        assert "subdir" in names

    def test_list_missing_directory_returns_error(self, tmp_path):
        cap, handler = build_filesystem_read_tool(str(tmp_path))
        result = handler({"action": "list", "path": "nope"})
        assert result.status == ToolExecutionStatus.ERROR
        assert result.error == "path_not_found"

    def test_list_file_returns_not_a_directory_error(self, tmp_path):
        (tmp_path / "f.txt").write_text("hi")
        cap, handler = build_filesystem_read_tool(str(tmp_path))
        result = handler({"action": "list", "path": "f.txt"})
        assert result.status == ToolExecutionStatus.ERROR
        assert result.error == "not_a_directory"

    def test_list_traversal_blocked(self, tmp_path):
        cap, handler = build_filesystem_read_tool(str(tmp_path))
        result = handler({"action": "list", "path": "../../"})
        assert result.status == ToolExecutionStatus.ERROR
        assert result.error == "path_traversal_rejected"


class TestFilesystemReadInExternalToolRuntime:
    def test_round_trip_through_registry(self, tmp_path):
        (tmp_path / "greet.txt").write_text("hello from test")
        cap, handler = build_filesystem_read_tool(str(tmp_path))
        registry = DynamicToolRegistry()
        registry.register(cap, handler)
        runtime = ExternalToolRuntime(registry, sleeper=lambda _: None)
        result = runtime.execute("filesystem_read", {"action": "read", "path": "greet.txt"})
        assert result.status == ToolExecutionStatus.OK
        assert "hello from test" in result.output["content"]

    def test_traversal_blocked_via_runtime(self, tmp_path):
        cap, handler = build_filesystem_read_tool(str(tmp_path))
        registry = DynamicToolRegistry()
        registry.register(cap, handler)
        runtime = ExternalToolRuntime(registry, sleeper=lambda _: None)
        result = runtime.execute("filesystem_read", {"action": "read", "path": "../../../etc/passwd"})
        assert result.status == ToolExecutionStatus.ERROR
