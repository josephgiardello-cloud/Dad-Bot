from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


class StorageBackend(Protocol):
    """Interface for durable storage backends."""

    def save_atomic(self, data: dict[str, Any], target_path: Path) -> None: ...

    def create_corruption_snapshot(self, path: Path) -> Path | None: ...


class FileSystemAdapter:
    """Filesystem adapter preserving DadBot's atomic-save + corruption-snapshot semantics."""

    def save_atomic(self, data: dict[str, Any], target_path: Path) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Keep tempfile + shutil.move pattern to match prior behavior.
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            json.dump(data, handle, indent=4)
            temp_name = handle.name

        shutil.move(temp_name, target)

    def create_corruption_snapshot(self, path: Path) -> Path | None:
        source = Path(path)
        if not source.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = source.with_suffix(f".corrupt_{timestamp}{source.suffix}")
        try:
            shutil.copy(source, destination)
        except OSError:
            return None
        return destination
