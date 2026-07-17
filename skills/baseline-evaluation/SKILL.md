---
name: baseline-evaluation
description: Run and audit a baseline protocol for a QOC Workbench task before improvement search, ensuring constraints, backend execution, diagnostics, and artifact completeness are verified.
---

# Baseline Evaluation

Use when the user wants to test a task/backend, establish a reference metric, or generate the first artifact before search.

## Workflow

1. Load the task spec and baseline candidate from `configs/task_specs/`.
2. Validate protocol constraints, Hamiltonian terms, and time-series constraints.
3. Run the fixed evaluator/backend through `QOCEvaluator` or a task-specific driver.
4. Write a timestamped artifact under `artifacts/<timestamp>_<task>_baseline/`.
5. Check required files with `artifact_checker.py` or an equivalent task-specific check.
6. Record whether the baseline is valid for search and what metric should be improved.

## Required Outputs

- `README.md` with command, task, backend, assumptions, and headline metric.
- `summary.csv` or equivalent machine-readable summary.
- Baseline candidate payload, usually `best_protocol.json` or `baseline_protocol.json`.
- Hamiltonian form and diagnostic data.
- Artifact completeness status.

## Recommended Diagnostics

- State-preparation tasks: final fidelity, final energy, trajectory arrays.
- QOC path tasks: instantaneous overlap or adiabatic-following metric.
- Hardware-control tasks: sampled controls, endpoint checks, amplitude ranges.
- New backends: Hermiticity, endpoint consistency, state normalization, finite metric.

## Case Lessons

- Rydberg C10 baseline stores final fidelity, degeneracy, Hamiltonian form, and trajectory diagnostics before supplemental search.
- XXZ baseline reproduction separates paper reference values from newly simulated candidate values.
- TFIM neural-generator baselines must include both coefficient-quality metrics and downstream evolution metrics.

## Guardrails

- Do not start improvement search from an unvalidated backend.
- Do not hide failed baseline runs; record failure artifacts if useful.
- Do not compare future candidates against an undocumented or non-reproducible baseline.
