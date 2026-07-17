from __future__ import annotations

import json
from pathlib import Path


def _load_json_like(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_task_spec(path: str | Path, override_path: str | Path | None = None, override: dict | None = None) -> dict:
    """
    Load a task spec and merge it with its referenced hardware spec.

    Task and hardware specs are stored as JSON-compatible YAML to avoid adding a
    YAML dependency at phase 1.
    """
    task_path = Path(path).resolve()
    task_spec = _load_json_like(task_path)

    hardware_rel = task_spec.get("hardware_spec")
    if not hardware_rel:
        merged = task_spec
    else:
        base_dir = task_path.parent.parent
        hardware_path = (base_dir / hardware_rel).resolve()
        hardware_spec = _load_json_like(hardware_path)
        merged = _deep_merge(hardware_spec, task_spec)
        merged["_hardware_spec_path"] = str(hardware_path)

    if override_path is not None:
        merged = _deep_merge(merged, _load_json_like(Path(override_path).resolve()))
        merged["_override_spec_path"] = str(Path(override_path).resolve())
    if override is not None:
        merged = _deep_merge(merged, override)

    merged["_task_spec_path"] = str(task_path)
    return merged
