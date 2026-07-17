from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ArtifactCheck:
    ok: bool
    missing: List[str] = field(default_factory=list)


def check_required_artifacts(run_dir: str | Path, task_spec: dict, mode: str) -> ArtifactCheck:
    if mode != "full":
        return ArtifactCheck(ok=True)

    required = (
        task_spec.get("artifact_requirements", {}).get("full_run_must_include")
        or task_spec.get("required_artifacts", [])
    )
    base = Path(run_dir)
    missing = [name for name in required if not (base / name).exists()]
    return ArtifactCheck(ok=not missing, missing=missing)
