from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_protected_files(repo_root: str | Path, protected_paths: List[str]) -> Dict[str, str]:
    root = Path(repo_root)
    snapshot: Dict[str, str] = {}
    for rel in protected_paths:
        path = root / rel
        if path.exists():
            snapshot[rel] = _sha256(path)
    return snapshot


@dataclass
class SourceGuardCheck:
    ok: bool
    changed: List[str] = field(default_factory=list)


def verify_protected_files_unchanged(repo_root: str | Path, before: Dict[str, str]) -> SourceGuardCheck:
    root = Path(repo_root)
    changed: List[str] = []
    for rel, digest in before.items():
        path = root / rel
        if not path.exists() or _sha256(path) != digest:
            changed.append(rel)
    return SourceGuardCheck(ok=not changed, changed=changed)
