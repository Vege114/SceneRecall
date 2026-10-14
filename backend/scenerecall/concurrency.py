"""One durable update for task defaults and connection concurrency limits."""
from __future__ import annotations

import copy
import os
from pathlib import Path

from .library import SCHEMA, read_json


def sync_directory(directory: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish_json(path: Path, value) -> None:
    from .providers import _atomic_json
    _atomic_json(path, value)
    sync_directory(path.parent)


def recover_concurrency(settings_dir: Path) -> None:
    """Replay a committed preferences update before any later profile/settings edit.

    Callers serialize this with the provider manager lock. The journal contains
    public connection metadata only; credentials remain in their existing store.
    """
    pending = settings_dir / "concurrency-pending.json"
    update = read_json(pending)
    if update is None:
        return
    if (not isinstance(update, dict) or update.get("schema_version") != SCHEMA
            or not isinstance(update.get("settings"), dict) or not isinstance(update.get("profiles"), dict)):
        raise ValueError("并发设置恢复记录损坏，请从备份恢复")
    publish_json(settings_dir / "profiles.json", update["profiles"])
    publish_json(settings_dir / "settings.json", update["settings"])
    pending.unlink()
    sync_directory(settings_dir)


def save_concurrency(library, providers, payload, current_settings):
    """Validate the entire edit before publishing either preferences file."""
    from .models import ExecutionOptions
    from .providers import Profile, ProviderError

    with library.lock, providers._lock:
        recover_concurrency(providers.settings_dir)
        profiles = providers._profiles()
        if set(payload.profiles) - set(profiles):
            raise ValueError("模型连接已变化，请刷新并发设置后重试")
        result = copy.deepcopy(current_settings())
        if payload.execution is not None:
            result["defaults"]["execution"] = ExecutionOptions.model_validate({
                **result["defaults"]["execution"], **payload.execution.model_dump(exclude_unset=True)}).model_dump()
        for profile_id, limit in payload.profiles.items():
            profiles[profile_id] = Profile.model_validate({**profiles[profile_id], "max_concurrency": limit}).model_dump()
        pending = providers.settings_dir / "concurrency-pending.json"
        try:
            publish_json(pending, {"schema_version": SCHEMA, "settings": result,
                                 "profiles": {"schema_version": 1, "profiles": list(profiles.values())}})
            recover_concurrency(providers.settings_dir)
            providers.concurrency_changed()
        except OSError:
            # A committed journal is retained until both files are published.
            # A retry or restart finishes that exact update before later edits.
            raise ProviderError("并发设置尚未完整保存，请重试；已提交的设置将在重试或重启时恢复") from None
        return {"settings": result, "provider_profiles": list(profiles.values())}
