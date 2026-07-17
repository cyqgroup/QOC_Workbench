---
name: artifact-audit
description: Audit QOC Workbench figures, artifacts, and code paths for reproducibility, real physical computation, stale paths, hardcoded metrics, missing diagnostics, and consistency with manuscript claims.
---

# Artifact Audit

Use when the user asks whether a result, figure, artifact, backend, or example is trustworthy.

## Workflow

1. Identify the claim, metric, figure, or artifact being audited.
2. Trace it to source data, candidate payload, evaluator/backend, script, and artifact directory.
3. Check whether values come from real simulation, reported literature references, or cached plotting data; label each source.
4. Search for stale paths, hardcoded metrics, placeholder code, `np.load`/pickle caches, and copied table values.
5. Verify backend integrity: Hamiltonian construction, time evolution/solver, metric computation, and protected paths.
6. Compare manuscript text, captions, plotting scripts, and artifact values.
7. Write pass/fail findings with severity and recommended fixes.

## Required Outputs

- Audit checklist or report.
- File and artifact trace for each audited claim.
- List of issues: missing artifact, stale path, hardcoded value, cache dependency, caption mismatch, backend shortcut, or unprotected code.
- Recommended fixes before publication or release.

## Case Lessons

- Manuscript figures should cite compact copied data, but the source artifact must remain traceable.
- Rydberg C10 diagnostics use a reduced trajectory cache; the cache is acceptable only because the artifact/source candidate is documented.
- XXZ captions must not overclaim when a candidate improves one metric but not all metrics.
- TFIM learned-generator claims must distinguish coefficient prediction quality from downstream evolution performance.

## Guardrails

- Do not accept a figure as reproducible just because the PDF exists.
- Do not accept a backend if it reads final metrics from previous artifacts instead of simulating.
- Do not silently fix manuscript claims during audit unless the user asks for edits.
