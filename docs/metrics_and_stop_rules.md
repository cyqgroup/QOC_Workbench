# Metrics and Stop Rules

This document standardizes how QOC Workbench chooses metrics, compares candidates, and decides when to stop or continue a search. Metrics and stop rules should be declared in the task specification whenever possible, so that the LLM-driven search loop does not redefine success after seeing results.

## Core Principles

- The primary objective metric must be declared before search starts.
- Baseline performance should be measured and stored as an artifact before improvement search begins.
- Secondary metrics should be recorded when they affect physical validity, robustness, or interpretability.
- Stop rules should match the user intent: reproduction, baseline testing, improvement search, robustness testing, or open-ended exploration.
- A candidate should never be considered successful if it improves the objective by violating declared Hamiltonian, hardware, symmetry, endpoint, or evaluation constraints.

## Metric Categories

### 1. State-Preparation Metrics

Use these when the goal is to prepare a target eigenstate or target subspace.

- `final_ground_state_fidelity`: overlap with a unique final ground state.
- `final_ground_subspace_fidelity`: total overlap with a degenerate or near-degenerate target ground subspace.
- `target_state_overlap`: overlap with a user-specified target state.
- `leakage`: probability outside an allowed subspace or code space.

Recommended use:

- Use subspace fidelity when the target ground space is degenerate.
- Track leakage whenever the Hilbert space has constraints or encoded sectors.

### 2. Energy and Optimization Metrics

Use these when exact target states are expensive, unknown, or less meaningful than energy quality.

- `final_energy`: expectation value of the target Hamiltonian at final time.
- `residual_energy`: final energy above the target ground energy.
- `normalized_energy_distance`: dimensionless energy improvement from initial to final state.
- `approximation_ratio`: optimization-quality metric for classical objective encodings.

Recommended use:

- Use final energy or residual energy when ground-state fidelity is unavailable.
- Use normalized metrics when comparing across instances with different energy scales.

### 3. Path-Following Metrics

Use these when the desired behavior includes adiabatic following, not only final performance.

- `instantaneous_ground_overlap`: overlap with the instantaneous ground subspace along the path.
- `adiabatic_fidelity`: time-averaged adiabatic-following score.
- `minimum_gap`: spectral gap diagnostic, when feasible.
- `diabatic_transition_indicator`: task-specific transition or leakage proxy.

Recommended use:

- Record path-following diagnostics for finite-time QOC protocols.
- Do not use path-following metrics alone if the final state quality is the actual objective.

### 4. Control and Hardware Metrics

Use these to enforce or prefer practical protocols.

- `control_amplitude_max`: maximum absolute control amplitude.
- `control_smoothness`: penalty or score for waveform roughness.
- `bandwidth_proxy`: finite-difference or Fourier-content proxy.
- `endpoint_error`: violation of required initial/final Hamiltonian endpoints.
- `pulse_area`: integrated control strength, when relevant.

Recommended use:

- Treat hard hardware constraints as validation failures, not soft preferences.
- Use smoothness or simplicity only as tie-breakers unless the user declares them as objectives.

### 5. Learned-Generator Metrics

Use these when an inner neural or statistical generator is part of the protocol.

- `coefficient_rmse`: coefficient prediction error against teacher labels.
- `coefficient_r2`: coefficient prediction coefficient of determination.
- `transfer_fidelity`: downstream fidelity under generated controls on unseen instances.
- `generalization_gap`: train/validation/test performance gap.

Recommended use:

- Always evaluate learned controls by downstream quantum evolution, not only by coefficient error.

## Choosing the Primary Metric

Use the following default policy unless the user specifies otherwise.

| Task Goal | Recommended Primary Metric | Required Secondary Checks |
| --- | --- | --- |
| Ground-state preparation with unique ground state | `final_ground_state_fidelity` | final energy, endpoint consistency |
| Degenerate ground-space preparation | `final_ground_subspace_fidelity` | degeneracy metadata, leakage |
| Optimization with unknown exact state | `final_energy`, `residual_energy`, or `approximation_ratio` | constraint violations, final-state diagnostics |
| Adiabatic-following study | `adiabatic_fidelity` plus final metric | final fidelity or energy |
| Hardware pulse design | task final metric | amplitude, smoothness, endpoint constraints |
| Learned coefficient generator | downstream final metric | coefficient RMSE/R2, transfer diagnostics |
| Paper reproduction | reported paper metric | reproduction tolerance and parameter match |

## Baseline Comparison Modes

A task may define one or more baseline comparisons.

### Absolute Target

Stop when a metric exceeds a fixed threshold:

```json
{"rule": "target_metric_reached", "metric": "final_ground_subspace_fidelity", "threshold": 0.99}
```

### Relative Improvement over Baseline

Use when the user asks for improvement by a percentage over a baseline:

```json
{
  "rule": "relative_improvement_reached",
  "metric": "final_ground_subspace_fidelity",
  "baseline_metric": 0.882,
  "relative_gain": 0.05
}
```

If this rule is not implemented in code for a task, record it in the task spec and enforce it in the example/search driver before making claims.

### Reproduction Tolerance

Use when reproducing a published result:

```json
{
  "rule": "reproduction_within_tolerance",
  "metric": "final_energy",
  "reported_value": -12.345,
  "absolute_tolerance": 0.01
}
```

### Budget-Limited Exploration

Use when the goal is to explore strategy families rather than reach a fixed threshold:

```json
{"rule": "max_rounds", "value": 8}
{"rule": "max_sim_cost", "value": 500000}
{"rule": "min_distinct_strategies_attempted", "count": 4, "mode": "full"}
```

## Stop Rule Modes by User Intent

### 1. Reproduction Mode

Intent: reproduce a paper, table, figure, or baseline.

Stop when:

- reported metric is matched within tolerance; or
- reproduction fails after the declared budget, with failure diagnostics recorded.

Required artifacts:

- reported values and source citation/note;
- reproduced values;
- parameter match table;
- discrepancy analysis if reproduction fails.

### 2. Baseline-Test Mode

Intent: test whether a new Hamiltonian/backend works.

Stop when:

- baseline artifact is generated;
- validation checks pass;
- required diagnostics are present.

Required artifacts:

- baseline candidate;
- Hamiltonian form;
- trajectory or solver diagnostics;
- metric and validation summary.

### 3. Improvement-Search Mode

Intent: find a protocol better than baseline.

Stop when:

- target absolute metric is reached; or
- requested relative improvement over baseline is reached; or
- budget is exhausted after attempting required strategy diversity.

Required artifacts:

- baseline artifact;
- all trial records;
- best protocol;
- failed strategy notes;
- comparison table against baseline.

### 4. Robustness Mode

Intent: find a protocol that performs reliably across instances, perturbations, or times.

Stop when:

- mean metric meets threshold;
- variance or worst-case metric meets threshold;
- all hard constraints remain satisfied.

Required artifacts:

- per-instance metrics;
- aggregate mean/std/worst-case table;
- robustness perturbation description.

### 5. Open-Ended Exploration Mode

Intent: map possible strategies without requiring improvement.

Stop when:

- maximum rounds, simulation budget, or walltime is reached;
- required number of strategy families has been explored;
- failure modes have been summarized.

Required artifacts:

- strategy catalog;
- candidate family table;
- negative-result notes;
- recommendations for the next focused search.

## Ranking and Tie-Breaking

Candidate ranking should use the primary metric first. Tie-breakers may include:

- lower constraint violation;
- better path-following diagnostic;
- simpler protocol;
- smoother controls;
- lower simulation/control cost;
- better transfer or robustness.

Tie-breakers must be declared in task preferences or documented in the run artifact.

## What Must Be Recorded

Every official search artifact should record:

- primary metric name and value;
- baseline metric and comparison rule;
- secondary diagnostics;
- stop rule evaluation;
- whether stopping criteria were satisfied;
- whether any hard constraints failed;
- reason for continuing or stopping.

## Implementation Notes

Current code supports `target_metric_reached` and `min_distinct_strategies_attempted` in `src/qoc_lbi/stop_rules.py`. Additional stop rules, such as relative improvement and reproduction tolerance, should be added to `stop_rules.py` when they are used repeatedly across tasks. Until then, task-specific drivers may enforce them explicitly, but the rule must still be documented in the task spec and artifact report.
