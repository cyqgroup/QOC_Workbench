# Hamming Weight Spike

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

$$V(w) = w + V_{spike}(w)$$

$$V_{spike}(w) = 
\begin{cases} 
H, & w = w^* \\
0, & w \neq w^* 
\end{cases}$$

$$H_B = -\sum_{i=1}^n \sigma_i^x$$


## Related Entries

- [Deep reinforcement learning for quantum adiabatic algorithm design](../04_Papers/NMI_2020_RL_QA.md)
- [Quantum Adiabatic Evolution Algorithms with Application to 3-SAT](../04_Papers/farhi_2001.md)

## Status

English scaffold generated during repository language cleanup. Expand with task-specific details when this note is used in a new search run.
