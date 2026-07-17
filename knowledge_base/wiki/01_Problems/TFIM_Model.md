# TFIM Model

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

$$H = -J \sum_{i=1}^{n-1} \sigma_i^z \sigma_{i+1}^z - h \sum_{i=1}^n \sigma_i^x$$

$$H_{CD}(s) = \dot{h}(s) \cdot \alpha(s) \sum_i \sigma_i^y$$


## Related Entries

- [Minimizing Irreversible Losses in Quantum Systems by Local Counter-Diabatic Driving](../04_Papers/arXiv_1607_05687_Local_CD.md)
- [Quantum phase transitions (Sachdev, 2011)](https://doi.org/10.1017/CBO9780511973765)

## Status

English scaffold generated during repository language cleanup. Expand with task-specific details when this note is used in a new search run.
