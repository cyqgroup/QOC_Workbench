from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.linalg import eigh

from .inner_solver import evolve_raw_mvp_ode

Array = np.ndarray

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


@dataclass(frozen=True)
class TFIMInstance:
    h: Array
    edges: tuple[tuple[int, int], ...]
    couplings: Array

    @property
    def n(self) -> int:
        return int(len(self.h))


def kron_embed(op: Array, site: int, n_qubits: int) -> Array:
    ops = [_I2] * n_qubits
    ops[site] = op
    out = ops[0]
    for item in ops[1:]:
        out = np.kron(out, item)
    return out


@lru_cache(maxsize=64)
def _cached_site_ops(n_qubits: int) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[Array, ...]]:
    xs = tuple(kron_embed(_X, site, n_qubits) for site in range(n_qubits))
    ys = tuple(kron_embed(_Y, site, n_qubits) for site in range(n_qubits))
    zs = tuple(kron_embed(_Z, site, n_qubits) for site in range(n_qubits))
    return xs, ys, zs


def site_ops(n_qubits: int) -> dict[str, tuple[Array, ...]]:
    xs, ys, zs = _cached_site_ops(int(n_qubits))
    return {"X": xs, "Y": ys, "Z": zs}


def tfim_problem_hamiltonian(instance: TFIMInstance) -> Array:
    ops = site_ops(instance.n)
    h = np.zeros((2**instance.n, 2**instance.n), dtype=complex)
    for field, z_op in zip(instance.h, ops["Z"]):
        h = h + float(field) * z_op
    for coupling, (i, j) in zip(instance.couplings, instance.edges):
        h = h - float(coupling) * (ops["Z"][i] @ ops["Z"][j])
    return h


def tfim_driver_hamiltonian(instance: TFIMInstance) -> Array:
    ops = site_ops(instance.n)
    return sum(ops["X"])


def tfim_hamiltonian(instance: TFIMInstance, lam: float) -> Array:
    return (1.0 - float(lam)) * tfim_driver_hamiltonian(instance) + float(lam) * tfim_problem_hamiltonian(instance)


def tfim_dh_dlambda(instance: TFIMInstance) -> Array:
    return tfim_problem_hamiltonian(instance) - tfim_driver_hamiltonian(instance)


def ground_subspace(hamiltonian: Array, tol: float = 1e-8) -> tuple[float, Array, Array]:
    evals, evecs = eigh(hamiltonian, check_finite=False)
    e0 = float(evals[0].real)
    mask = np.abs(evals - e0) <= tol * max(1.0, abs(e0))
    return e0, evecs[:, mask], evals


def subspace_fidelity(psi: Array, subspace: Array) -> float:
    amps = subspace.conj().T @ psi
    return float(np.real(np.vdot(amps, amps)))


def energy_shift_for_ground_weight(evals: Array, order: int) -> float:
    if int(order) == 1:
        return 0.0
    lo = float(evals.max() + 1e-5)
    hi = float(evals.max() + 2.0 * (evals.max() - evals.min()) + 1.0)
    grid = np.linspace(lo, hi, 220)
    power = 2 * int(order) - 2
    weights = np.power(evals[None, :] - grid[:, None], power)
    values = (weights @ evals) / np.sum(weights, axis=1)
    return float(grid[int(np.argmin(values))])


def polynomial_derivative_operator(hamiltonian: Array, derivative: Array, order: int, energy_shift: float) -> Array:
    if int(order) == 1:
        return derivative
    eye = np.eye(hamiltonian.shape[0], dtype=complex)
    shifted = hamiltonian - float(energy_shift) * eye
    powers = [eye]
    for _ in range(int(order)):
        powers.append(powers[-1] @ shifted)
    out = np.zeros_like(hamiltonian)
    for ell in range(int(order)):
        out = out + powers[ell] @ derivative @ powers[int(order) - 1 - ell]
    return out


def weighted_cd_coefficients(instance: TFIMInstance, lam: float, order: int = 3) -> Array:
    hamiltonian = tfim_hamiltonian(instance, lam)
    derivative = tfim_dh_dlambda(instance)
    _, _, evals = ground_subspace(hamiltonian)
    energy_shift = energy_shift_for_ground_weight(evals, int(order))
    eye = np.eye(hamiltonian.shape[0], dtype=complex)
    weighted_h = hamiltonian - energy_shift * eye
    if int(order) > 1:
        weighted_h = np.linalg.matrix_power(weighted_h, int(order))
    weighted_derivative = polynomial_derivative_operator(hamiltonian, derivative, int(order), energy_shift)
    y_ops = site_ops(instance.n)["Y"]
    basis = [1j * (weighted_h @ y_op - y_op @ weighted_h) for y_op in y_ops]
    matrix = np.empty((instance.n, instance.n), dtype=float)
    vector = np.empty(instance.n, dtype=float)
    for row in range(instance.n):
        vector[row] = np.real(np.trace(basis[row].conj().T @ weighted_derivative))
        for col in range(instance.n):
            matrix[row, col] = np.real(np.trace(basis[row].conj().T @ basis[col]))
    ridge = 1e-10 * max(1.0, np.linalg.norm(matrix))
    return np.linalg.solve(matrix + ridge * np.eye(instance.n), vector).real


def simulate_tfim_weighted_cd_ode(
    instance: TFIMInstance,
    coeff_table: Array | None,
    lambda_grid: Sequence[float],
    *,
    total_time: float,
    n_steps: int,
) -> dict[str, object]:
    _, initial_subspace, _ = ground_subspace(tfim_hamiltonian(instance, 0.0))
    psi0 = initial_subspace[:, 0]
    _, target_subspace, _ = ground_subspace(tfim_hamiltonian(instance, 1.0))
    driver = tfim_driver_hamiltonian(instance)
    problem = tfim_problem_hamiltonian(instance)
    y_ops = site_ops(instance.n)["Y"]
    times = np.linspace(0.0, float(total_time), int(n_steps) + 1)
    lambda_grid_np = np.asarray(lambda_grid, dtype=np.float32)
    coeff_table_np = None if coeff_table is None else np.asarray(coeff_table, dtype=np.float32)

    import jax.numpy as jnp

    driver_j = jnp.asarray(driver, dtype=jnp.complex64)
    problem_j = jnp.asarray(problem, dtype=jnp.complex64)
    y_ops_j = jnp.asarray(np.stack(y_ops), dtype=jnp.complex64)
    lambda_grid_j = jnp.asarray(lambda_grid_np, dtype=jnp.float32)
    coeff_table_j = None if coeff_table_np is None else jnp.asarray(coeff_table_np, dtype=jnp.float32)
    inv_total_time = jnp.asarray(1.0 / float(total_time), dtype=jnp.float32)

    def rhs(psi, time):
        lam = jnp.clip(time / float(total_time), 0.0, 1.0)
        hamiltonian = (1.0 - lam) * driver_j + lam * problem_j
        if coeff_table_j is not None:
            coeffs = jnp.asarray([jnp.interp(lam, lambda_grid_j, coeff_table_j[:, site]) for site in range(instance.n)])
            cd_hamiltonian = jnp.tensordot(coeffs, y_ops_j, axes=1)
            hamiltonian = hamiltonian + inv_total_time * cd_hamiltonian
        return -1.0j * (hamiltonian @ psi)

    result = evolve_raw_mvp_ode(rhs, psi0, times, rtol=1e-7, atol=1e-7)
    states = result.states
    inst_overlap = []
    target_overlap = []
    for time, psi in zip(times, states):
        lam = float(np.clip(time / float(total_time), 0.0, 1.0))
        _, inst_subspace, _ = ground_subspace(tfim_hamiltonian(instance, lam))
        inst_overlap.append(math.sqrt(max(0.0, subspace_fidelity(psi, inst_subspace))))
        target_overlap.append(math.sqrt(max(0.0, subspace_fidelity(psi, target_subspace))))
    final_state = states[-1]
    return {
        "final_fidelity": subspace_fidelity(final_state, target_subspace),
        "final_energy": float(np.real(np.vdot(final_state, tfim_hamiltonian(instance, 1.0) @ final_state))),
        "F_ad": float(np.trapz(inst_overlap, times) / float(total_time)),
        "times": times,
        "instantaneous_overlap": np.asarray(inst_overlap),
        "target_overlap": np.asarray(target_overlap),
        "inner_solver": {
            "backend": result.backend,
            "representation": result.representation,
            "ode_backend": result.ode_backend,
            "ode_solver": result.solver,
            "used_jit": result.used_jit,
        },
    }
