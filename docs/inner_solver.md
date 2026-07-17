# Inner Solver Standard

QOC Workbench uses `src/qoc_lbi/inner_solver.py` as the shared numerical evolution boundary for formal backends.

## Standard Contract

A backend should separate two responsibilities:

1. construct the task-specific time-dependent Hamiltonian action `H(t) @ psi` from the task spec and candidate protocol;
2. evolve the state through the shared TensorCircuit NG ODE interface.

The shared solver uses:

- TensorCircuit NG installed in the active Python environment and imported as `tensorcircuit`;
- JAX backend via `tc.set_backend("jax")`;
- matrix-vector-product evolution rather than hidden dense-exponential propagation;
- `tc.timeevol.ode_evol_global(..., mode="raw")` for raw MVP right-hand sides;
- `jax.jit` around the right-hand side passed to the ODE solver.

## Provided Entrypoints

- `make_pauli_mvp(terms)`: build a TensorCircuit `PauliStringSum2MVP` from Pauli-string terms.
- `evolve_mvp_ode(components, initial_state, times, ...)`: evolve a sum of scheduled MVP components.
- `evolve_raw_mvp_ode(rhs, initial_state, times, ...)`: evolve a backend-defined raw MVP right-hand side.

## Current Backends

- Rydberg MIS constructs Pauli-string MVP components for `X`, `Y`, `n`, and `n_i n_j` terms, then evolves through `evolve_mvp_ode`.
- XXZ uses the same ODE boundary through `evolve_raw_mvp_ode`; the right-hand side applies `H(t) @ psi` directly and supports schedule, auxiliary-field, and CD terms.
- TFIM weighted-CD support in `src/qoc_lbi/tfim_weighted_cd.py` uses `evolve_raw_mvp_ode` for the interpolation Hamiltonian plus optional one-body weighted-CD generator path.

Historical code snapshots under local `artifacts/` directories are preserved outside Git for auditability and may show the original implementation used to create archived results. New formal backend code should use this shared inner solver boundary.
