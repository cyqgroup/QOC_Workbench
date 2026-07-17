# Counter diabatic Driving

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

$$H(t) = H_0(s) + H_{CD}(s)$$

$$H_{CD}(s) = \dot{s} A_s$$

$$[i\partial_s H_0, H_0] = [H_0, [H_0, A_s]]$$


## Related Entries

- [Digitized-Counterdiabatic Quantum Optimization (DCQO, arXiv:2201.00790)](../04_Papers/arXiv_2201_00790_DCQO.md)
- [Counterdiabatic driving in the quantum annealing of the p-spin model: a variational approach (arXiv:1912.09711)](../04_Papers/arXiv_1912_09711_pSpin_CD.md)
- [Fighting Exponentially Small Gaps by Counterdiabatic Driving (arXiv:2410.02520)](../04_Papers/arXiv_2410_02520_CD_Small_Gaps.md)
- [Minimizing Irreversible Losses in Quantum Systems by Local Counter-Diabatic Driving (arXiv:1607.05687)](../04_Papers/arXiv_1607_05687_Local_CD.md)
- [Physics-Informed Neural Networks for an optimal counterdiabatic quantum computation](../04_Papers/arXiv_2309_04434.md)
- [Shortcuts to adiabaticity](../04_Papers/sta_review.md)

## Status

English scaffold generated during repository language cleanup. Expand with task-specific details when this note is used in a new search run.
