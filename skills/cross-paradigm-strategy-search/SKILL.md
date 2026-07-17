---
name: cross-paradigm-strategy-search
description: Run an auditable LLM-driven QOC strategy search that retrieves prior artifacts and knowledge motifs, generates constrained candidate protocols, simulates them with protected backends, records failures, and stops by declared metrics and budgets.
---

# Cross-Paradigm Strategy Search

Use when the user asks to improve over a baseline, transfer strategy motifs, explore schedules/catalysts/CD controls, or run an LLM-driven protocol search.

## Preconditions

- A valid task spec and protected backend exist.
- A baseline artifact exists and declares the baseline metric.
- Objective metric, stop rules, and budget are declared or confirmed.
- Hard constraints and forbidden terms are encoded before search starts.

## Workflow

1. Retrieve compact context: task spec, baseline artifact, prior failed trials, knowledge notes, and reusable motifs.
2. Choose strategy families consistent with constraints: schedule shaping, endpoint-vanishing catalysts, allowed auxiliary controls, approximate CD, learned generators, or initialization changes.
3. Generate candidate `ProtocolCandidate` payloads; validate them before simulation.
4. Run probe evaluations when cheap; promote selected candidates to full evaluation.
5. Record every candidate, including failures, rejected channels, and constraint violations.
6. Rank by the declared primary metric, then declared tie-breakers.
7. Update search memory with what worked, what failed, and what should be tried next.
8. Stop only by declared stop rules, budget, or user instruction.

## Required Outputs

- Timestamped search artifact.
- `trials.jsonl`, candidate payloads, and best protocol.
- Search memory, failure notes, and comparison against baseline.
- Stop-rule evaluation and next-step recommendation if target not reached.

## Case Lessons

- Rydberg C6 succeeded by evaluating families per total time rather than forcing one waveform to win everywhere.
- Rydberg C10 used structured low-dimensional follow-up families after the baseline: no-Y, five-knot global-Y, seven-knot, beta-bump.
- XXZ transferred motifs by role, not algebraic form: catalyst, schedule deformation, and CD were rewritten for the spin-chain control language.
- TFIM escalated from per-instance weighted-CD solves to amortized neural generation when repeated coefficient optimization became the bottleneck.

## Guardrails

- Do not modify protected backends, target Hamiltonians, metrics, or forbidden terms during search.
- Do not treat a copied table or cached figure as a new simulation.
- Do not discard negative results; failures are part of workbench memory.
- Do not use smoothness/simplicity as the primary objective unless the task declares it.
