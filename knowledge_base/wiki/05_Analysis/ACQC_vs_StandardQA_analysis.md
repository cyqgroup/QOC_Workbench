# ACQC vs StandardQA analysis

## Type

Analysis Note

## Purpose

This entry records reusable knowledge for the QOC Workbench workflow.

## Workbench-Relevant Content

- Identify the Hamiltonian terms, control channels, and constraints before running a search.
- Specify objective metrics and stop rules explicitly in the task specification.
- Store accepted and rejected candidates as auditable artifacts.
- Prefer backend-verified simulations over unvalidated heuristic claims.

## Preserved Mathematical Snippets

$$H(t) = (1-s(t))\,H_B + s(t)\,H_P, \qquad s(0)=0,\;s(T)=1$$

$$H(t) = \frac{\Omega(t)}{2}\sum X_i - \Delta(t)\sum n_i + V\sum_{(i,j)\in E} n_i n_j$$

$$\Omega(t) = \Omega_0\sin^2\!\left(\frac{\pi}{2}\sin\frac{\pi t}{T}\right), \qquad \Delta(t) = -\Delta_0\cos\frac{\pi t}{T}$$

$$H_{\rm CD}(t) = f_y(t)\sum_i Y_i$$

$$f_y(t) = -\frac{\Omega\,\dot\Delta - \Delta\,\dot\Omega}{2(\Omega^2+\Delta^2)}$$

$$H_P = -\sum_i n_i + 2\sum_{(i,j)\in E} n_i n_j = \text{const} + \sum_i c_i Z_i + \frac{1}{2}\sum_{(i,j)\in E} Z_i Z_j$$


## Status

English scaffold generated during repository language cleanup. Expand with task-specific details when this note is used in a new search run.
