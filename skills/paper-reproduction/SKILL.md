---
name: paper-reproduction
description: Reproduce a published QOC/AQC paper result inside QOC Workbench by extracting Hamiltonian, controls, metrics, baselines, implementing or selecting a backend, running a timestamped artifact, and comparing reported versus reproduced values.
---

# Paper Reproduction

Use when the user asks to reproduce a paper, table, figure, baseline protocol, or reported AQC result.

## Workflow

1. Identify the exact claim to reproduce: Hamiltonian, system size, initial state, controls, total time, metric, reported value, and tolerances.
2. Add or update a concise note in `knowledge_base/wiki/04_Papers/` or `reference/`; store raw sources only when redistribution is allowed.
3. If no backend exists, follow `docs/new_task_backend_template.md`; otherwise reuse the closest protected backend.
4. Create a reproduction driver under `examples/`, with fixed seeds and documented approximations.
5. Run the reproduction into `artifacts/<timestamp>_<paper_or_task>_reproduction/`.
6. Write reported-vs-reproduced tables and discrepancy notes before using the result as a baseline for search.

## Required Outputs

- Literature note with Hamiltonian, protocol, metric, and assumptions.
- Reproduction script and command.
- Timestamped artifact with machine-readable results.
- Comparison table: reported value, reproduced value, tolerance, status.
- If reproduction fails, a failure note rather than a silent workaround.

## Case Lessons

- XXZ reproduction is the model pattern: first reconstruct paper-inspired SA/OI/AH/CD baselines, then use them as a reference for later catalyst/schedule search.
- Do not mix reported baselines with newly simulated candidate metrics without labeling the source of each value.
- Reproduction may require stricter optimization than search scripts; keep a dedicated reproduction driver when needed.

## Guardrails

- Do not tune unknown parameters until the reported value is matched without recording assumptions.
- Do not claim reproduction from copied table values alone; table values are references, not simulated results.
- If exact reproduction is impossible within budget, record the mismatch and continue only with a clearly labeled approximate baseline.
