# Extension Workflow

## Add a new Hamiltonian task

1. Write a task specification in `configs/task_specs/<task_id>.yaml`.
2. Include public inputs, physical parameters, a baseline candidate, objective metric, stop rules, and protected source paths.
3. Add an evaluator under `src/qoc_lbi/` or an example driver under `examples/`.
4. Run a baseline first and check the generated artifact bundle.
5. Only then enable heuristic or LLM-driven search.

## Add a new hardware platform

1. Add a control-language file under `configs/hardware_specs/`.
2. Encode native channels, amplitude/rate limits, allowed auxiliary controls, and forbidden terms.
3. Update candidate validation/checking code if the platform requires a new control family.
4. Keep the simulator/evaluator boundary explicit and non-editable during LLM search.

## Add literature to the knowledge base

1. Save permissible raw source files under `knowledge_base/raw/papers/` or link them from notes if redistribution is not permitted.
2. Add concise notes under `knowledge_base/wiki/04_Papers/` and strategy/problem summaries under the relevant `knowledge_base/wiki/` subdirectories.
3. Record what the paper suggests, what constraints it assumes, and how it can seed candidate protocols.
4. When a literature idea is tested, store outcomes under `artifacts/` rather than inside `knowledge_base/`.

## Artifact standards

Each official run should include:

- `README.md` with task, method, command, and headline result.
- Machine-readable summary such as `summary.csv`, `trials.jsonl`, or `best_protocol.json`.
- Diagnostic plots or compact arrays sufficient to audit the result.
- Code snapshot or exact script path when the script is not stable.
