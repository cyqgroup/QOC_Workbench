from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def ensure_run_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(path: str | Path, payload: dict) -> None:
    with Path(path).open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, payloads: Iterable[dict]) -> None:
    with Path(path).open("w") as f:
        for payload in payloads:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_summary_csv(path: str | Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
