# QOC Workbench

QOC Workbench is a standalone research codebase for auditable quantum optimal
control (QOC) and adiabatic quantum computation (AQC) workflow experiments.

The open-source tree packages the reusable evaluator boundary, task and hardware
specifications, curated notes, runnable examples, and compact selected result
tables. Full local run directories and manuscript/private publishing material
are intentionally kept out of Git.

## Repository layout

- `src/qoc_lbi/` — reusable workbench package: candidate protocol schema, evaluator boundary, Rydberg and XXZ simulators, artifact checking, local search-loop support, and source guards.
- `configs/task_specs/` — task definitions for benchmark problems and future extensions.
- `configs/hardware_specs/` — hardware/control-language constraints.
- `skills/` and `docs/` — repo-local workflow skills, extension checklists, and backend templates.
- `knowledge_base/` — curated QOC/AQC notes, paper summaries, and permissible raw paper sources.
- `artifacts/` — tracked only as an empty `.gitkeep` placeholder; generated run directories are ignored.
- `results/` — compact selected result tables and lightweight plot/data summaries; generated HTML reports are ignored.
- `examples/` — runnable entry points for baseline/search/reproduction experiments; see `examples/README.md` for the artifact outputs each script creates.
- `docs/` — operational documentation for reproduction and extension.

## Paper-Facing Results

- Rydberg MIS C6: per-total-time hardware-compatible candidate selection improves the average final fidelity over the reproduced smooth ACQC baseline.
- Rydberg MIS C10: global-Y smooth beta-bump refinement improves the fixed-time C10 ACQC baseline and supplies trajectory diagnostics.
- XXZ ring: catalyst and schedule-design artifacts support the manuscript comparison and appendix diagnostics.
- TFIM weighted CD: MLP/GNN generator artifacts support the amortized weighted counterdiabatic coefficient-path study.

## Quick start

```bash
cd QOC_Workbench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For full numerical reproduction, install the scientific stack listed in `requirements.txt`. Rydberg runs additionally require JAX; choose the CPU/GPU wheel appropriate for your machine.

## Run core workflow examples

```bash
cd QOC_Workbench
python3 examples/run_rydberg_baseline.py
python3 examples/run_rydberg_search.py --max-rounds 1
python3 examples/run_xxz_ring8_experiment.py
```

The search example uses local run-directory policy modules and deterministic fallback updates. Agent-assisted exploration can still edit these modules during a normal coding-agent session, while the released mainline does not require an in-script model API endpoint.

Generated runs are written under `artifacts/` by default. The open-source tree
keeps only `artifacts/.gitkeep`; copy or summarize any run into `results/` only
when it is intentionally part of the released compact result set.

## Extend to new systems

1. Add a hardware/control spec under `configs/hardware_specs/`.
2. Add a task spec under `configs/task_specs/` with public inputs, physical parameters, baseline candidate, stop rules, and protected paths.
3. Add or reuse an evaluator module under `src/qoc_lbi/` with fixed simulation boundaries.
4. Add relevant paper notes under `knowledge_base/wiki/` and raw sources under `knowledge_base/raw/` when licensing permits.
5. Run an example/search entry point and keep outputs under `artifacts/` with `README.md`, diagnostics, summaries, and source snapshots.

See `docs/extension_workflow.md` for a more explicit checklist.

## Submission note

This repository intentionally excludes local exploratory artifacts, manuscript
submission bundles, generated HTML reports, Python caches, and private publishing
material. Raw paper PDFs are kept only when redistribution is considered
permissible for this working release; otherwise use links from the knowledge-base
notes instead of adding binaries.
