# Workbench Skills

This document defines reusable workflow skills for QOC Workbench. A skill is a repeatable procedure that maps a user intent to a sequence of workbench actions, files, checks, and artifacts. The purpose is to avoid ad hoc LLM behavior and to make future work at least as auditable and reproducible as the existing case studies.

## Skill Selection Principles

- Identify the user's intent before editing code or running simulations.
- Prefer the smallest skill that satisfies the request.
- Record assumptions when the user does not specify initial Hamiltonians, controls, metrics, or stopping rules.
- Keep external knowledge in `knowledge_base/` or `reference/` and internal run outcomes in `artifacts/`.
- Never improve a result by changing protected backend, target Hamiltonian, forbidden terms, or evaluation rules during search.

## Skill 1: Paper Reproduction

### User Intent

Use this skill when the user asks to reproduce a paper result, table, figure, baseline, or published protocol.

Example requests:

- "Reproduce this QOC/AQC paper baseline."
- "Check whether Table I can be reproduced."
- "Implement the protocol from this paper and compare with the reported result."

### Workflow

1. Ingest the paper or paper note into `knowledge_base/` when redistribution is allowed.
2. Extract Hamiltonian form, Hilbert space, system size, initial state, controls, metric, and reported values.
3. Create or update a task spec and backend if needed.
4. Implement a reproduction driver under `examples/`.
5. Run the reproduction and write a timestamped artifact.
6. Compare reproduced values with reported values using a declared tolerance.
7. Record discrepancies, assumptions, and possible causes of mismatch.

### Required Outputs

- paper note under `knowledge_base/wiki/04_Papers/` or `reference/`;
- reproduction script;
- timestamped artifact under `artifacts/`;
- reported-vs-reproduced comparison table;
- discrepancy report if reproduction is not successful.

## Skill 2: New Hamiltonian Onboarding

### User Intent

Use this skill when the user gives a new Hamiltonian family and asks QOC Workbench to design, test, or search protocols for it.

Example requests:

- "Here is a new Hamiltonian; design a QOC protocol."
- "Add this fermionic model to the workbench."
- "I only know which controls are forbidden; infer a conservative allowed control set."

### Workflow

1. Formalize the Hilbert space, operator representation, system size, target Hamiltonian, and boundary conditions.
2. Choose or document an initial Hamiltonian and initial state if the user did not specify them.
3. Encode allowed and explicitly forbidden controls in task/hardware specs.
4. Add a backend following `docs/new_task_backend_template.md` if no backend exists.
5. Add validation checks for Hermiticity, endpoint consistency, forbidden terms, conserved quantities, and metric range.
6. Add protected paths for the backend and validation code.
7. Run a baseline artifact before any improvement search.
8. Update knowledge notes with assumptions and reusable motifs.

### Required Outputs

- `configs/task_specs/<task>.yaml`;
- `configs/hardware_specs/<platform>.yaml`, if needed;
- `src/qoc_lbi/<system_or_task>_qoc.py`, if needed;
- `examples/run_<task>_baseline.py`;
- baseline artifact under `artifacts/`;
- knowledge note for the problem or platform.

## Skill 3: Baseline Evaluation

### User Intent

Use this skill when the user wants to test a Hamiltonian/backend, generate diagnostics, or establish a reference before search.

Example requests:

- "Run the baseline for this task."
- "Check whether this backend works."
- "Generate a diagnostic artifact for this Hamiltonian."

### Workflow

1. Load the task spec and baseline candidate.
2. Validate protocol constraints and Hamiltonian terms.
3. Run the fixed evaluator/backend.
4. Write required diagnostics and machine-readable results.
5. Check artifact completeness.
6. Record whether the baseline is suitable for improvement search.

### Required Outputs

- timestamped baseline artifact;
- `README.md` with command, assumptions, and headline result;
- `summary.csv` or equivalent;
- `best_protocol.json` or baseline candidate payload;
- trajectory/solver diagnostics;
- artifact completeness check.

## Skill 4: Cross-Paradigm Strategy Search

### User Intent

Use this skill when the user asks for a better protocol, cross-paradigm transfer, schedule/catalyst/CD search, or LLM-driven exploration.

Example requests:

- "Find a better protocol than the baseline."
- "Try schedule deformation, catalysts, and CD-like controls."
- "Use previous artifacts to propose new strategies."

### Workflow

1. Confirm baseline artifact and metric.
2. Retrieve relevant knowledge notes, previous artifacts, failed trials, and reusable motifs.
3. Define or confirm stop rules and budget.
4. Generate constrained candidate protocols.
5. Run probe evaluations when useful.
6. Promote selected candidates to full evaluation.
7. Record every accepted and rejected candidate.
8. Inspect failures and update search memory.
9. Stop only when declared stop rules or budget conditions are met.

### Required Outputs

- timestamped search artifact;
- trial records and candidate payloads;
- best protocol and comparison against baseline;
- failure notes and search memory;
- stop-rule evaluation;
- recommendation for next search if target is not reached.

## Skill 5: Knowledge Base Expansion

### User Intent

Use this skill when the user wants the workbench to learn from new papers, notes, codebases, or prior artifacts.

Example requests:

- "Add this paper to the knowledge base."
- "Extract useful QOC strategies from these notes."
- "Turn this failed run into reusable search memory."

### Workflow

1. Store permissible raw sources in `knowledge_base/raw/` or link them from notes.
2. Write concise wiki-style summaries of Hamiltonian form, control strategy, metric, baseline, results, and assumptions.
3. Extract reusable motifs, such as schedule shaping, catalysts, CD ansatz, initialization, or learned generators.
4. Record limitations and failure cases separately from successful motifs.
5. Link knowledge notes to reproduction or search artifacts when available.

### Required Outputs

- paper/problem/strategy note under `knowledge_base/wiki/`;
- extracted motifs and constraints;
- source metadata or citation note;
- links to artifacts that tested the idea.

## Skill 6: Artifact Audit

### User Intent

Use this skill when the user asks whether a result, figure, artifact, or code path is trustworthy.

Example requests:

- "Check whether this figure is reproducible."
- "Audit this artifact for shortcuts."
- "Verify that the result came from real simulation."

### Workflow

1. Identify the claim, figure, or metric being audited.
2. Trace it to source data, candidate payload, evaluator/backend, and artifact directory.
3. Check for cached values, hardcoded metrics, stale paths, missing diagnostics, or modified protected files.
4. Verify that the evaluator performs real physical computation or documented solver logic.
5. Compare figure captions, manuscript claims, and artifact values.
6. Write an audit report with pass/fail items and recommended fixes.

### Required Outputs

- audit report or checklist;
- list of traced source files and artifacts;
- identified issues and severity;
- recommended fixes before publication or release.

## Skill 7: Metric and Stop-Rule Design

### User Intent

Use this skill when the user asks how to judge success, when to stop, or how much improvement over baseline is required.

Example requests:

- "Use a metric appropriate for this Hamiltonian."
- "Stop when it beats baseline by 5%."
- "Only reproduce the baseline; do not search."

### Workflow

1. Identify user intent: reproduction, baseline test, improvement search, robustness, or exploration.
2. Choose a primary metric using `docs/metrics_and_stop_rules.md`.
3. Add secondary diagnostics for validity and interpretability.
4. Define stop rules in the task spec or driver.
5. Record baseline value and target threshold before search.
6. Ensure stop-rule evaluation is written into the artifact.

### Required Outputs

- declared objective metric;
- baseline metric;
- stop rules;
- secondary diagnostics;
- explanation of why these metrics match the task.

## Skill Composition

Common composite workflows:

### New Paper to Searchable Strategy

```text
Paper Reproduction
→ Knowledge Base Expansion
→ New Hamiltonian Onboarding, if needed
→ Baseline Evaluation
→ Cross-Paradigm Strategy Search
→ Artifact Audit
```

### New Hamiltonian from User Specification

```text
Metric and Stop-Rule Design
→ New Hamiltonian Onboarding
→ Baseline Evaluation
→ Cross-Paradigm Strategy Search, if requested
→ Knowledge Base Expansion
```

### Existing Result Verification

```text
Artifact Audit
→ Reproduction, if missing
→ Metric and Stop-Rule Design, if claims are ambiguous
```

## Skill Invocation Record

For substantial work, the run artifact or report should record which skill was used, why it was selected, and what assumptions were made. This keeps the workbench from depending on hidden conversational context.
