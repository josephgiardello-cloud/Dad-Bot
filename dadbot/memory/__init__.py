"""Memory package placeholder for the DadBot split."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["MemoryManager"]

if TYPE_CHECKING:
	from .manager import MemoryManager as MemoryManager


def __getattr__(name: str):
	if name == "MemoryManager":
		from .manager import MemoryManager

		return MemoryManager
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
	return sorted(set(globals()) | {"MemoryManager"})
