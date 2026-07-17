# QUBO Ising Models

## Type

Problem and Hamiltonian Model

## Purpose

This entry records a Hamiltonian family or optimization problem that can be formalized as a QOC Workbench task specification.

## Workbench-Relevant Content

- Identify the Hamiltonian terms, control channels, and constraints before running a search.
- Specify objective metrics and stop rules explicitly in the task specification.
- Store accepted and rejected candidates as auditable artifacts.
- Prefer backend-verified simulations over unvalidated heuristic claims.

## Preserved Mathematical Snippets

$$f(x) = \sum_i Q_{ii} x_i + \sum_{i<j} Q_{ij} x_i x_j$$

$$H = \sum_i h_i s_i + \sum_{i<j} J_{ij} s_i s_j$$

$$H_B = -\sum_i \Delta_i \sigma_i^x$$


## Related Entries

- [Ising formulations of many NP problems (Lucas, 2014)](../04_Papers/lucas_2014.md)
- [Transformer-Based Neural Quantum Digital Twins...](../04_Papers/arXiv_2505_15662_NQDT.md)

## Status

English scaffold generated during repository language cleanup. Expand with task-specific details when this note is used in a new search run.
