# Bang Bang Control

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

$$\mathcal{H} = \text{Re}(\langle p | (-i H(f)) | \psi \rangle)$$

$$f(t) = \begin{cases} 1 & \text{if } \text{Switching Function } > 0 \\ 0 & \text{if } \text{Switching Function } < 0 \end{cases}$$


## Related Entries

- [Bang-bang control as a design principle (arXiv:1812.02746)](../04_Papers/arXiv_1812_02746_Bang-Bang_Principle.md)
- [Quantum Approximate Optimization Algorithm (Farhi, 2014)](../04_Papers/farhi_2014_qaoa.md)

## Status

English scaffold generated during repository language cleanup. Expand with task-specific details when this note is used in a new search run.
