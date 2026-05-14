from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ConfigSchema(BaseModel):
    model_candidates: list[str] = Field(default_factory=lambda: ["llama3.2"])
    embedding_model_candidates: list[str] = Field(
        default_factory=lambda: ["nomic-embed-text"],
    )
    light_mode: bool = False
    telemetry_level: str = "INFO"
    data_dir: str = "."
    feature_flags: dict[str, bool] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, config_path: str | Path) -> ConfigSchema:
        path = Path(config_path)
        if not path.exists():
            return cls()

        suffix = path.suffix.lower()
        if suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls.model_validate(raw)

        if suffix in {".yaml", ".yml"}:
            return cls.model_validate(_load_yaml_like(path))

        return cls()


def _load_yaml_like(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        yaml = None

    if yaml is not None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    # Minimal fallback parser for simple top-level key: value pairs.
    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = _coerce_yaml_scalar(value.strip())
    return payload


def _coerce_yaml_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")
