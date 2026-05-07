from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.models import PersistenceStatusSnapshot
from dadbot.utils import json_dumps, json_load
from dadbot_system import CompositeStateStore, PostgresStateStore


class RuntimeStorageManager:
    """Owns durable local JSON helpers plus profile persistence and storage status."""

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    @staticmethod
    def json_backup_path(destination):
        path = Path(destination)
        return path.with_name(f"{path.name}.bak")

    @staticmethod
    def corrupt_json_snapshot_path(destination):
        path = Path(destination)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return path.with_name(f"{path.stem}.corrupt-{timestamp}{path.suffix}")

    def capture_corrupt_json_snapshot(self, source_path):
        source = Path(source_path)
        if not source.exists():
            return None
        destination = self.corrupt_json_snapshot_path(source)
        try:
            shutil.copy2(source, destination)
        except OSError:
            return None
        return destination

    def write_json_atomically(self, destination, payload, *, backup=True):
        target_path = Path(destination)
        serialized_payload = json_dumps(payload, indent=2)

        def write_once():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = target_path.with_name(
                f".{target_path.name}.{uuid.uuid4().hex}.tmp",
            )
            try:
                with temp_path.open("w", encoding="utf-8") as handle:
                    handle.write(serialized_payload)
                    handle.flush()
                    os.fsync(handle.fileno())

                if backup and target_path.exists():
                    backup_path = self.json_backup_path(target_path)
                    shutil.copy2(target_path, backup_path)

                os.replace(temp_path, target_path)
                try:
                    directory_fd = os.open(str(target_path.parent), os.O_RDONLY)
                except OSError:
                    directory_fd = None
                if directory_fd is not None:
                    try:
                        os.fsync(directory_fd)
                    except OSError:
                        pass
                    finally:
                        os.close(directory_fd)
            finally:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

        try:
            write_once()
        except FileNotFoundError:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            write_once()

    def load_profile(self):
        if self.bot._tenant_document_store is not None:
            persisted = self.bot._tenant_document_store.load_session_state("profile")
            if isinstance(persisted, dict) and persisted:
                return persisted

            if self.bot.PROFILE_PATH.exists():
                with self.bot.PROFILE_PATH.open("r", encoding="utf-8") as profile_file:
                    profile = json_load(profile_file)
            else:
                profile = self.bot.default_profile()
            self.bot._tenant_document_store.save_session_state("profile", profile)
            return profile

        if not self.bot.PROFILE_PATH.exists():
            # Deterministic bootstrap: always initialize a missing profile artifact.
            self.bot.initialize_profile_file(
                profile_path=self.bot.PROFILE_PATH,
                force=False,
            )

        if not self.bot.PROFILE_PATH.exists():
            # Final fallback if custom initialization hook did not materialize the file.
            self.write_json_atomically(
                self.bot.PROFILE_PATH,
                self.bot.default_profile(),
                backup=False,
            )
        with self.bot.PROFILE_PATH.open("r", encoding="utf-8") as profile_file:
            return json_load(profile_file)

    def save_profile(self):
        with self.bot._io_lock:
            if self.bot._tenant_document_store is not None:
                self.bot._tenant_document_store.save_session_state(
                    "profile",
                    self.bot.PROFILE,
                )
                return
            self.write_json_atomically(
                self.bot.PROFILE_PATH,
                self.bot.PROFILE,
                backup=True,
            )

    def customer_persistence_status(self):
        store = self.bot._customer_state_store
        backend = "filesystem"
        acid_enabled = False
        enabled = self.bot._tenant_document_store is not None
        if isinstance(store, CompositeStateStore):
            backends = []
            if store.fast_store is not None:
                backends.append(type(store.fast_store).__name__)
            if store.durable_store is not None:
                backends.append(type(store.durable_store).__name__)
                acid_enabled = isinstance(store.durable_store, PostgresStateStore)
            backend = "+".join(backends) or "filesystem"
        elif store is not None:
            backend = type(store).__name__
            acid_enabled = isinstance(store, PostgresStateStore)
        validated = PersistenceStatusSnapshot.model_validate(
            {
                "tenant_id": self.bot.TENANT_ID,
                "backend": backend,
                "acid_enabled": acid_enabled,
                "enabled": enabled,
                "primary_store": "tenant_document_store" if enabled else "filesystem",
                "profile_backend": "tenant_document_store" if enabled else "json_file",
                "memory_backend": "tenant_document_store" if enabled else "json_file",
                "json_mirror_enabled": bool(enabled),
            },
        )
        return validated.model_dump(mode="python")

    def session_log_rotation_candidates(self) -> list[Path]:
        """Return rotatable session log files while preserving primary identity artifacts."""
        raw_log_dir = getattr(self.bot, "SESSION_LOG_DIR", None)
        if raw_log_dir in (None, ""):
            return []
        log_dir = Path(raw_log_dir)
        try:
            log_dir = log_dir.resolve()
        except OSError:
            return []
        if not log_dir.exists():
            return []

        protected = set()
        config = getattr(self.bot, "config", None)
        if config is not None:
            protected_names = getattr(config, "primary_identity_log_filenames", ())
            if isinstance(protected_names, (tuple, list, set)):
                protected = {str(name).strip() for name in protected_names if str(name).strip()}

        candidates: list[Path] = []
        for child in log_dir.iterdir():
            if not child.is_file():
                continue
            if child.name in protected:
                continue
            if child.suffix.lower() not in {".log", ".jsonl", ".txt", ".out"}:
                continue
            candidates.append(child)
        candidates.sort(key=lambda path: path.stat().st_mtime)
        return candidates

    def prune_session_logs(self, *, max_files: int = 200) -> list[Path]:
        """Delete oldest rotatable session logs, excluding protected identity logs."""
        limit = max(0, int(max_files or 0))
        candidates = self.session_log_rotation_candidates()
        if len(candidates) <= limit:
            return []
        removed: list[Path] = []
        for path in candidates[: len(candidates) - limit]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            removed.append(path)
        return removed


__all__ = ["RuntimeStorageManager"]
