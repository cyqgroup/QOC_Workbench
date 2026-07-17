# Grover Search

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

$$H(s) = (1-s) H_{B} + s H_{P}$$

$$H_B = \sum_q [\mathbb{I} - X_q]/2$$

$$H_B = \mathbb{I} - |\psi_0\rangle\langle\psi_0|$$

$$\Delta_{min} \propto \frac{1}{\sqrt{N}}$$


## Related Entries

- [Quantum Adiabatic Algorithm Design using Reinforcement Learning](../04_Papers/arXiv_1812_10797.md)
- [Local adiabatic evolution in quantum computation](../04_Papers/local_adiabatic_paper.md)

## Status

English scaffold generated during repository language cleanup. Expand with task-specific details when this note is used in a new search run.
