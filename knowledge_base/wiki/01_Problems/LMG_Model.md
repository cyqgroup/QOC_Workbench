# LMG Model

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

$$H = -\frac{J}{N} (S_z^2 + \gamma S_x^2) - h S_z$$

$$H_P = -\frac{1}{n} \sum_{i<j} \sigma_i^z \sigma_j^z - h \sum_i \sigma_i^x$$


## Related Entries

- [Minimizing Irreversible Losses in Quantum Systems by Local Counter-Diabatic Driving](../04_Papers/arXiv_1607_05687_Local_CD.md)
- [Lipkin-Meshkov-Glick model in quantum information (2005)](https://arxiv.org/abs/quant-ph/0501110)

## Status

English scaffold generated during repository language cleanup. Expand with task-specific details when this note is used in a new search run.
