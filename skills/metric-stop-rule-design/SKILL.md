---
name: metric-stop-rule-design
description: Choose task-appropriate objective metrics, secondary diagnostics, baseline comparisons, and stopping rules for QOC Workbench reproduction, baseline testing, improvement search, robustness testing, or exploration.
---

# Metric and Stop-Rule Design

Use when the user asks how to judge success, when to stop, how much to beat a baseline, or which metrics suit a Hamiltonian task.

## Workflow

1. Determine user intent: reproduction, baseline test, improvement search, robustness test, or open-ended exploration.
2. Choose a primary metric using `docs/metrics_and_stop_rules.md`.
3. Add secondary diagnostics for physical validity and interpretability.
4. Define baseline comparison: absolute threshold, relative improvement, reproduction tolerance, robustness target, or budget-limited exploration.
5. Write metric and stop rules into the task spec when possible.
6. Ensure run artifacts record baseline metric, current best metric, stop-rule status, and reason for continuing or stopping.

## Metric Defaults

- Unique ground-state preparation: `final_ground_state_fidelity`.
- Degenerate target space: `final_ground_subspace_fidelity`.
- Unknown target state: `final_energy`, `residual_energy`, or `approximation_ratio`.
- Adiabatic-following study: final metric plus `adiabatic_fidelity` or instantaneous overlap.
- Hardware pulse search: task final metric plus amplitude/smoothness/endpoint checks.
- Learned generator: downstream quantum metric plus coefficient RMSE/R2.

## Stop Modes

- Reproduction: stop when reported metric is matched within tolerance or discrepancy is documented.
- Baseline test: stop after a complete baseline artifact passes validation.
- Improvement search: stop when absolute target or relative improvement over baseline is reached, or budget is exhausted.
- Robustness: stop when mean, variance, and worst-case targets are satisfied.
- Exploration: stop after declared budget and strategy diversity are reached.

## Required Outputs

- Objective metric and rationale.
- Secondary diagnostics.
- Baseline value and comparison rule.
- Stop rules and budget.
- Artifact field documenting stop-rule evaluation.

## Case Lessons

- Rydberg uses final ground-subspace fidelity as the primary metric, with trajectory diagnostics for auditability.
- XXZ needs multiple metrics: normalized energy and adiabatic-following can tell different stories, so captions and ranking must be precise.
- TFIM neural generators require both coefficient metrics and downstream evolution metrics; coefficient accuracy alone is insufficient.

## Guardrails

- Do not choose or change the success metric after seeing candidate results.
- Do not stop on a soft preference metric if the physical objective fails.
- Do not ignore hard constraint violations even when the objective improves.
