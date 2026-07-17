# New Task Backend Template

This document defines the expected structure for adding a new Hamiltonian family to QOC Workbench. It is intended for cases where the workbench does not yet contain a backend for the requested system.

A new backend should make the task executable without changing the meaning of the target Hamiltonian, the evaluation metric, or the declared hardware/control constraints during search.

## When to Add a Backend

Add a new backend when a task cannot be represented by the existing Rydberg or XXZ evaluators. Typical cases include:

- a new Hilbert-space type, such as fermions, bosons, qudits, or constrained subspaces;
- a new target Hamiltonian family;
- a new hardware/control language;
- a new metric that cannot be computed from existing artifact bundles;
- a new simulation method, such as sparse exact evolution, tensor-network evolution, or a problem-specific solver.

Recommended file pattern:

```text
src/qoc_lbi/<system_or_task>_qoc.py
```

Recommended example entry points:

```text
examples/run_<task>_baseline.py
examples/run_<task>_search.py
```

For the first search round of a newly onboarded Hamiltonian, `run_dir.py` can generate a generic heuristic template when the task is not marked as `search_template: rydberg`. This generic template does not assume Rydberg controls such as `omega`, `delta`, or global-Y CD. It reads the declared `allowed_channels`, `allowed_parameterizations`, `allowed_cd_kinds`, `candidate_specs`, and `baseline_candidate`, then proposes conservative mutations for total time, allowed channel bases, and allowed auxiliary/CD kinds. A later agent-guided policy update may specialize this run-dir policy after backend evidence and failure records justify a system-specific search strategy.

## Required Evaluator Interface

A backend evaluator should expose a function with this shape:

```python
def evaluate_<task>(task_spec: dict, candidate: ProtocolCandidate, mode: str) -> dict:
    ...
```

The function must return a JSON-serializable dictionary containing at least:

```python
{
    "metric": float(...),
    "diagnostics": {...},
    "artifact_bundle": {...},
}
```

The returned `metric` is the scalar objective used by the search loop. The `diagnostics` dictionary should contain lightweight metadata useful for ranking, debugging, and failure analysis. The `artifact_bundle` should contain arrays and metadata needed to reproduce or audit the run.

## Backend Construction Checklist

A new backend should explicitly define the following items.

### 1. Hilbert Space and Operator Representation

Document and implement:

- Hilbert-space type, such as qubit, fermion, boson, qudit, or constrained basis;
- system size and basis ordering;
- dense, sparse, tensor-network, or problem-specific operator representation;
- any truncation, encoding, or symmetry-sector restriction.

### 2. Hamiltonian Definition

Construct the Hamiltonian components from the task specification:

- initial Hamiltonian or initial-state preparation rule;
- target/problem Hamiltonian;
- fixed hardware terms;
- allowed controllable terms;
- optional catalyst, auxiliary, or counterdiabatic terms;
- explicitly forbidden terms.

The backend must preserve the declared target Hamiltonian and must not modify the problem instance during candidate search.

### 3. Candidate Interpretation

Define how a `ProtocolCandidate` payload is converted into a concrete time-dependent Hamiltonian:

- map each named channel to a Hamiltonian term;
- map each channel basis and parameter dictionary to a time-dependent coefficient;
- interpret `candidate.cd` only if the task permits CD or auxiliary controls;
- reject or report unsupported channels, bases, or CD kinds.

The same candidate payload should always produce the same Hamiltonian path under the same task specification.

### 4. Initial State

Define how the initial state is prepared:

- exact ground state of the initial Hamiltonian;
- analytic product state;
- Hartree-Fock-like reference state;
- random or problem-specific state, if explicitly justified and seeded.

If the user does not provide an initial Hamiltonian, the backend should record the chosen baseline assumption in the run artifact.

### 5. Time Evolution or Solver

Implement a concrete evolution or solver method, such as:

- dense diagonalization and piecewise `exp(-i H dt)`;
- sparse `expm_multiply`;
- ODE integration;
- tensor-network evolution;
- variational or algebraic inner solver for control coefficients;
- trained generator inference followed by direct simulation.

Approximation choices, time steps, tolerances, truncations, and random seeds must be recorded in diagnostics or artifacts.

### 6. Metrics

Compute task-specific metrics from the simulated state or solver output. Examples include:

- final ground-subspace fidelity;
- final energy;
- normalized approximation ratio;
- adiabatic-following metric;
- leakage outside a constrained subspace;
- coefficient prediction error;
- custom metric declared by the task specification.

The metric used for ranking must be declared in the task spec and returned as `metric`.

### 7. Artifact Bundle

For a time-evolution backend, the artifact bundle should include as many of the following as apply:

```python
artifact_bundle = {
    "times": times,
    "target_fidelity": target_fidelity,
    "instantaneous_ground_overlap": instantaneous_ground_overlap,
    "control_values": control_values,
    "energy_values": energy_values,
    "hamiltonian_formula": hamiltonian_formula,
    "snapshot_times": snapshot_times,
    "hamiltonian_snapshots_real": hamiltonian_snapshots_real,
    "backend_metadata": backend_metadata,
}
```

If some fields are not meaningful for the task, document the replacement fields in the task spec and in the run `README.md`.

## Validation Requirements

A new backend should include checks appropriate for the system:

- Hermiticity of generated Hamiltonians;
- endpoint consistency with the declared initial and target Hamiltonians;
- preservation of required symmetries or conserved quantities;
- absence of explicitly forbidden terms;
- finite and normalized state vectors;
- metric finite and within the expected range;
- deterministic behavior under fixed seeds.

General constraints should be encoded in task or hardware specs where possible. System-specific checks may be implemented in a dedicated checker module, for example:

```text
src/qoc_lbi/<system>_constraints.py
src/qoc_lbi/<system>_term_checker.py
```

## Protected Infrastructure

Once a backend defines the task, it should be added to the task spec's `protected_paths`, together with shared validation code. Example:

```json
"protected_paths": [
  "src/qoc_lbi/<system_or_task>_qoc.py",
  "src/qoc_lbi/evaluator.py",
  "src/qoc_lbi/constraints.py",
  "src/qoc_lbi/hamiltonian_term_checker.py"
]
```

During protocol search, candidate improvements must come from admissible protocol modifications, not from changing the simulator, target Hamiltonian, forbidden terms, or evaluation rule.

## Minimal Backend Skeleton

```python
from __future__ import annotations

from typing import Any

import numpy as np

from qoc_lbi.protocol import ProtocolCandidate
from qoc_lbi.time_series_checker import check_time_series_constraints


def evaluate_<task>(task_spec: dict, candidate: ProtocolCandidate, mode: str) -> dict[str, Any]:
    defaults = task_spec.get("evaluator_defaults", {})
    n_steps = int(defaults.get("n_steps_full", 101))
    if mode == "probe":
        n_steps = int(defaults.get("n_steps_probe", max(21, n_steps // 4)))

    # 1. Build operators and Hamiltonian components from task_spec.
    # 2. Interpret candidate channels as time-dependent controls.
    # 3. Validate controls and forbidden terms before solving.
    # 4. Prepare the initial state.
    # 5. Evolve or solve the system.
    # 6. Compute metric and diagnostics.
    # 7. Return artifact_bundle for auditability.

    times = np.linspace(0.0, float(candidate.total_time), n_steps)
    artifact_bundle = {
        "times": times,
        "hamiltonian_formula": "TODO: explicit H(t) formula for this backend",
        "backend_metadata": {
            "mode": mode,
            "n_steps": n_steps,
        },
    }

    constraint_check = check_time_series_constraints(task_spec, artifact_bundle)
    if not constraint_check.ok:
        raise ValueError("time_series_constraints_failed: " + "; ".join(constraint_check.errors))

    metric = float("nan")  # Replace with a real simulated metric.
    diagnostics = {
        "mode": mode,
        "n_steps": n_steps,
    }
    return {
        "metric": metric,
        "diagnostics": diagnostics,
        "artifact_bundle": artifact_bundle,
    }
```

The skeleton is intentionally incomplete. A submitted backend must replace the placeholder Hamiltonian and metric with real task-specific simulation or solver logic before it is used for claims.

## Documentation and Artifact Expectations

When a new backend is added, also add or update:

- `configs/task_specs/<task>.yaml`;
- `configs/hardware_specs/<platform>.yaml`, if needed;
- `examples/run_<task>_baseline.py`;
- `examples/run_<task>_search.py`, if search is supported;
- `knowledge_base/wiki/01_Problems/<task>.md` or related notes;
- a timestamped baseline artifact under `artifacts/`.

The first baseline artifact should explain any assumptions introduced by the backend, especially when the user did not specify the initial Hamiltonian, allowed controls, or metric details.
