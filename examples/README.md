# Examples

This directory contains runnable entry points that demonstrate how QOC Workbench creates new auditable artifacts under `artifacts/`.

Each script follows the same pattern:

1. Load a task, baseline, or paper-reproduction setup.
2. Run a fixed simulator/evaluator or search procedure through the shared TensorCircuit-NG/JAX MVP ODE inner solver in `src/qoc_lbi/inner_solver.py`.
3. Create a timestamped run directory under `artifacts/`.
4. Write machine-readable results, diagnostic plots/data, and a short run summary.
5. Optionally copy the driver script into the run directory for auditability.

## Inner solver backend

The examples do not carry independent physics propagators. Rydberg and XXZ entry points call the task-specific backend in `src/qoc_lbi/`, and those backends construct `H(t)` and delegate state evolution to the common TensorCircuit-NG/JAX MVP ODE layer. Full-run artifacts record the solver backend, representation, ODE backend, ODE solver, and JIT flag so that downstream audits can verify that a result was not produced by a shortcut or stale exact-exponential path.

## Relation to the agent-assisted search loop

These examples implement the executable side of the workflow described in the manuscript. A new task begins from a task specification and a hardware specification. The workbench first evaluates a baseline, records diagnostic artifacts, and can then enter an LLM-assisted search loop. In that loop, previous artifacts, trial records, failure notes, and reusable control motifs provide structured context for proposing the next candidate protocol. Each accepted or rejected candidate is written back into the timestamped run directory, so later searches can audit and reuse both successes and failures.

## Scripts

| Script | Purpose | Artifact directory | Key outputs |
| --- | --- | --- | --- |
| `run_rydberg_baseline.py` | Run the Rydberg MIS C6 baseline candidate through the fixed evaluator. | `artifacts/<timestamp>_rydberg_mis_c6_baseline/` | `README.md`, `summary.csv`, `trials.jsonl`, `best_protocol.json`, `trajectory_diagnostics.npz`, `target_fidelity_vs_time.png`, `instantaneous_ground_overlap_vs_time.png`, `schedule_shapes.png`, `hamiltonian_form.md`, `hamiltonian_snapshots_real.png` |
| `run_rydberg_search.py` | Run the Rydberg MIS search loop with local run-directory policy modules and deterministic fallback updates. | `artifacts/<timestamp>_rydberg_mis_c6_search/` or `--run-dir <path>` | Search state, candidate modules, trial records, diagnostics, and summaries |
| `run_xxz_ring8_experiment.py` | Run an XXZ ring-8 LBI-style search/benchmark example. | `artifacts/<timestamp>_xxz_ring8_lbi_search/` | `paper_baseline_T10.csv`, `xxz_lbi_search_results.csv`, `xxz_lbi_best_by_delta.csv`, copied driver script, Markdown summary files, inner-solver metadata columns |
| `xxz_T10.py` | Reproduce selected XXZ paper table baselines with stricter optimization. | `artifacts/<timestamp>_xxz_paper_<T>_reproduction/` | CSV/JSON summaries for reproduced table entries, optimization results, and inner-solver metadata columns |

## Quick artifact generation

From the repository root:

```bash
python3 examples/run_rydberg_baseline.py
```

Then inspect the newest directory:

```bash
ls -lt artifacts | head
```

A valid Rydberg full-run artifact should contain at least:

```text
README.md
summary.csv
trials.jsonl
best_protocol.json
trajectory_diagnostics.npz
target_fidelity_vs_time.png
instantaneous_ground_overlap_vs_time.png
schedule_shapes.png
hamiltonian_form.md
hamiltonian_snapshots_real.png
```

The `trajectory_diagnostics.npz` file also records `inner_solver_backend`, `inner_solver_representation`, `inner_solver_ode_backend`, `inner_solver_ode_solver`, and `inner_solver_used_jit`.

## Creating artifacts for a new system

Use these examples as templates rather than editing them in place:

1. Add a task spec under `configs/task_specs/`.
2. Add or reuse a hardware spec under `configs/hardware_specs/`.
3. Add a fixed evaluator in `src/qoc_lbi/` or a dedicated driver in `examples/`.
4. Create the run directory with `ensure_run_dir(repo_root / "artifacts" / f"{timestamp}_{task_id}_{purpose}")`.
5. Write a `README.md`, machine-readable summary files, diagnostic plots/data, and any code snapshot needed to audit the run.
6. If the artifact will support a manuscript figure, copy or reduce it into compact data used by `manuscript/scripts_plot_*.py`.

See `docs/extension_workflow.md` for the broader workflow and `docs/reproduction.md` for paper-result reproduction.
