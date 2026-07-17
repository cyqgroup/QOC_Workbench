---
name: new-hamiltonian-onboarding
description: Add a previously unsupported Hamiltonian family to QOC Workbench by formalizing task and hardware specs, choosing representation and initial-state assumptions, implementing an executable backend, protecting validation code, and generating a baseline artifact.
---

# New Hamiltonian Onboarding

Use when the user provides a new Hamiltonian and wants QOC Workbench to design, test, or search protocols for it.

## Required Inputs or Assumptions

Collect or explicitly assume:

- Hilbert-space type: qubit, fermion, boson, qudit, constrained basis, etc.
- System size and boundary conditions.
- Target Hamiltonian and fixed problem instance.
- Initial Hamiltonian or initial-state rule; if absent, choose a conservative baseline and record it.
- Explicitly forbidden controls; allowed controls may be inferred conservatively.
- Primary metric, secondary diagnostics, and stop intent.

## Workflow

1. Choose names: `<task_id>`, `<platform>`, and backend file `src/qoc_lbi/<system_or_task>_qoc.py`.
2. Create `configs/task_specs/<task_id>.yaml` with objective metric, baseline candidate, Hamiltonian structure, stop rules, required artifacts, and protected paths.
3. Create or update `configs/hardware_specs/<platform>.yaml` with allowed channels, bases, bounds, forbidden terms, and CD/auxiliary permissions.
4. If no system-specific search policy exists yet, set `search_template: generic` or rely on the non-Rydberg generic run-dir fallback.
5. Implement the backend using `docs/new_task_backend_template.md`.
6. Add system-specific constraints or term checkers when YAML rules are insufficient.
7. Add `examples/run_<task>_baseline.py` and, if search is requested, `examples/run_<task>_search.py`.
8. Run baseline first; only start search after artifact checks pass.
9. Add a problem/platform note to `knowledge_base/wiki/` with assumptions and reusable motifs.

## Required Outputs

- Task spec and hardware spec.
- Executable backend with real Hamiltonian construction and metric computation.
- Baseline example driver.
- Timestamped baseline artifact.
- Generic or system-specific run-dir search template selected by the task spec.
- Protected paths covering backend, evaluator, constraints, and term checker.

## Case Lessons

- Rydberg shows hardware-native constraints: fixed interaction graph, bounded Rabi/detuning controls, and optional global-Y control.
- XXZ shows literature-guided onboarding: reproduce baseline path first, then add catalysts/CD/schedule deformation.
- TFIM shows backend growth can be algorithmic: weighted-CD labels and learned generators become part of the realization layer.

## Guardrails

- Do not insert a new Hamiltonian into Rydberg/XXZ files unless it is genuinely a small extension of those systems.
- Do not leave placeholder metrics or Hamiltonians in a backend used for claims.
- Do not let a generic first-round search invent channels, bases, CD kinds, or Hamiltonian terms absent from the task/hardware spec.
- Once the backend defines the task, protect it from LLM search modifications.
