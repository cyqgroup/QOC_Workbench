from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

import tensorcircuit as tc
from tensorcircuit import timeevol
from tensorcircuit.quantum import PauliStringSum2MVP

K = tc.set_backend("jax")
tc.set_dtype("complex64")

PauliTerm = tuple[Sequence[int], Union[complex, float]]
ScheduleFn = Callable[[Any], Any]


@dataclass(frozen=True)
class MVPComponent:
    name: str
    mvp: Callable
    coefficient: ScheduleFn


@dataclass(frozen=True)
class InnerSolverResult:
    times: np.ndarray
    states: np.ndarray
    backend: str
    representation: str
    ode_backend: str
    solver: str
    used_jit: bool


def make_pauli_mvp(terms: Sequence[PauliTerm]) -> Callable:
    structures = [list(structure) for structure, _ in terms]
    weights = [complex(weight) for _, weight in terms]
    return PauliStringSum2MVP(structures, weights)


def make_dense_mvp(matrix: np.ndarray | jnp.ndarray) -> Callable:
    op = jnp.asarray(matrix, dtype=jnp.complex64)

    @jax.jit
    def mvp(psi):
        return K.matmul(op, psi)

    return mvp


def combine_mvp_components(components: Sequence[MVPComponent]) -> Callable:
    def rhs(psi, time):
        total = jnp.zeros_like(psi)
        for component in components:
            coeff = component.coefficient(time)
            total = total + jnp.asarray(coeff, dtype=jnp.complex64) * component.mvp(psi)
        return -1.0j * total

    return rhs


def evolve_mvp_ode(
    components: Sequence[MVPComponent],
    initial_state: np.ndarray | jnp.ndarray,
    times: np.ndarray | jnp.ndarray,
    *,
    ode_backend: str = "diffrax",
    solver: str = "Tsit5",
    max_steps: int = 2_000_000,
    rtol: float = 1e-7,
    atol: float = 1e-7,
) -> InnerSolverResult:
    psi0 = jnp.asarray(initial_state, dtype=jnp.complex64)
    time_grid = jnp.asarray(times, dtype=jnp.float32)
    rhs = combine_mvp_components(components)
    rhs_jit = jax.jit(rhs)
    states = timeevol.ode_evol_global(
        rhs_jit,
        psi0,
        time_grid,
        mode="raw",
        ode_backend=ode_backend,
        solver=solver,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
    )
    states = states / jnp.linalg.norm(states, axis=1, keepdims=True)
    return InnerSolverResult(
        times=np.asarray(time_grid),
        states=np.asarray(states),
        backend="tensorcircuit-jax",
        representation="mvp",
        ode_backend=ode_backend,
        solver=solver,
        used_jit=True,
    )


def evolve_raw_mvp_ode(
    rhs: Callable,
    initial_state: np.ndarray | jnp.ndarray,
    times: np.ndarray | jnp.ndarray,
    *,
    ode_backend: str = "diffrax",
    solver: str = "Tsit5",
    max_steps: int = 2_000_000,
    rtol: float = 1e-7,
    atol: float = 1e-7,
) -> InnerSolverResult:
    psi0 = jnp.asarray(initial_state, dtype=jnp.complex64)
    time_grid = jnp.asarray(times, dtype=jnp.float32)
    rhs_jit = jax.jit(rhs)
    states = timeevol.ode_evol_global(
        rhs_jit,
        psi0,
        time_grid,
        mode="raw",
        ode_backend=ode_backend,
        solver=solver,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
    )
    states = states / jnp.linalg.norm(states, axis=1, keepdims=True)
    return InnerSolverResult(
        times=np.asarray(time_grid),
        states=np.asarray(states),
        backend="tensorcircuit-jax",
        representation="raw-mvp",
        ode_backend=ode_backend,
        solver=solver,
        used_jit=True,
    )


def dense_hamiltonian_from_mvp(mvp: Callable, dim: int) -> np.ndarray:
    eye = jnp.eye(dim, dtype=jnp.complex64)

    @jax.jit
    def apply_columns(columns):
        return jax.vmap(mvp, in_axes=1, out_axes=1)(columns)

    return np.asarray(apply_columns(eye))
