# Floquet VCD

## Type

AQC Strategy Note

## Purpose

This entry records a reusable control or schedule strategy that can seed candidate protocols in the search loop.

## Workbench-Relevant Content

- Identify the Hamiltonian terms, control channels, and constraints before running a search.
- Specify objective metrics and stop rules explicitly in the task specification.
- Store accepted and rejected candidates as auditable artifacts.
- Prefer backend-verified simulations over unvalidated heuristic claims.

## Preserved Mathematical Snippets

$$\mathcal{G}(\mathcal{A}^*) = \int_0^T dt \, \text{Tr}[G^2(t)]$$

$$G(t) = \partial_\lambda H + i[\mathcal{A}^*, H] - \partial_t \mathcal{A}^*$$


## Related Entries

- [Counterdiabatic Driving for Periodically Driven Systems](../04_Papers/arXiv_2310_02728_Floquet_CD.md)
- [Shortcuts to adiabaticity (Torrontegui, 2013)](../04_Papers/sta_review_2013.md)

## Status

English scaffold generated during repository language cleanup. Expand with task-specific details when this note is used in a new search run.
