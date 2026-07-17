# QOC Workbench Repo-Local Skills

This directory contains workflow skills that make QOC Workbench reusable without relying on hidden conversational context. Each skill is a focused procedure with trigger conditions, workflow steps, required outputs, case lessons, and guardrails.

## Skills

- `paper-reproduction/`: reproduce published baselines, tables, or protocols before using them as search references.
- `new-hamiltonian-onboarding/`: add a previously unsupported Hamiltonian family, backend, task spec, and baseline artifact.
- `baseline-evaluation/`: run and audit a baseline before improvement search.
- `cross-paradigm-strategy-search/`: run constrained LLM-driven protocol search using prior artifacts and motifs.
- `knowledge-base-expansion/`: add papers, strategies, code resources, and failed/successful motifs to the knowledge base.
- `artifact-audit/`: verify that a result, figure, or artifact is traceable to real computation and consistent claims.
- `metric-stop-rule-design/`: choose objective metrics, secondary diagnostics, baseline comparisons, and stopping rules.

## Typical Compositions

```text
New Hamiltonian request:
metric-stop-rule-design
→ new-hamiltonian-onboarding
→ baseline-evaluation
→ cross-paradigm-strategy-search, if requested
→ knowledge-base-expansion
```

```text
New paper request:
paper-reproduction
→ knowledge-base-expansion
→ new-hamiltonian-onboarding, if needed
→ baseline-evaluation
→ cross-paradigm-strategy-search, if requested
→ artifact-audit
```

```text
Existing result verification:
artifact-audit
→ paper-reproduction or baseline-evaluation, if evidence is missing
→ metric-stop-rule-design, if success criteria are ambiguous
```

## Operating Rule

For substantial runs, record which skill or skill composition was used in the artifact `README.md` or audit report. This preserves the workbench procedure as an explicit artifact rather than an implicit chat history.
