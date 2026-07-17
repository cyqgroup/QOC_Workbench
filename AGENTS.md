# QOC Workbench Agent Guide

- Keep evaluator boundaries in `src/qoc_lbi/` fixed unless the task explicitly asks to change the physics model.
- Store new official runs under `artifacts/<timestamp>_<task>_<purpose>/` with a `README.md`, summary CSV/JSON, diagnostics, and any code snapshot needed for auditability.
- Store external literature notes in `knowledge_base/wiki/` or `reference/`; store internal experiment notes in `artifacts/`, `registry.jsonl`, or `summary.md`.
- Do not commit generated caches such as `__pycache__`, LaTeX aux files, or full exploratory sweeps unless they are needed for paper reproduction.
- Prefer compact reproducible tables/NPZ files for manuscript plotting, and document any expensive recomputation path in `docs/reproduction.md`.
