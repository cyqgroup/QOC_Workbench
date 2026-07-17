# MWIS

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

$$H_P = -\sum_{i \in V} w_i \hat{n}_i + \sum_{(i,j) \in E} U_{ij} \hat{n}_i \hat{n}_j$$

$$H_B = -\sum_{i \in V} \sigma_i^x$$


## Related Entries

- [Enhancement of quantum annealing via n-local catalysts](../04_Papers/arXiv_2409_13029_nLocal_Catalysts.md)
- [Ising formulations of many NP problems (Lucas, 2014)](../04_Papers/lucas_2014.md)

## Status

English scaffold generated during repository language cleanup. Expand with task-specific details when this note is used in a new search run.
