from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import jax.numpy as jnp
import numpy as np

from .inner_solver import MVPComponent, make_pauli_mvp
from .controls.rydberg_registry import (
    ScalarControl,
    build_rydberg_cd_control,
    build_rydberg_channel_control,
)


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


@dataclass
class RydbergControlBundle:
    omega: ScalarControl
    delta: ScalarControl
    cd: ScalarControl
    active_terms: list[str]
    hamiltonian_formula: str


def kron_embed(op_2x2: np.ndarray, site: int, n_qubits: int) -> np.ndarray:
    ops = [_I2] * n_qubits
    ops[site] = op_2x2
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def build_rydberg_ops(n_qubits: int, edges: Iterable[Tuple[int, int]]) -> dict:
    dim = 2**n_qubits
    eye = np.eye(dim, dtype=complex)
    ni = [(eye - kron_embed(_Z, idx, n_qubits)) / 2.0 for idx in range(n_qubits)]

    x_sum = sum(kron_embed(_X, idx, n_qubits) for idx in range(n_qubits))
    y_sum = sum(kron_embed(_Y, idx, n_qubits) for idx in range(n_qubits))
    n_sum = sum(ni)
    edge_list = list(edges)
    nn_sum = (
        sum(ni[i] @ ni[j] for (i, j) in edge_list)
        if edge_list
        else np.zeros((dim, dim), dtype=complex)
    )

    return {
        "X_sum": jnp.array(x_sum, dtype=jnp.complex64),
        "Y_sum": jnp.array(y_sum, dtype=jnp.complex64),
        "n_sum": jnp.array(n_sum, dtype=jnp.complex64),
        "nn_sum": jnp.array(nn_sum, dtype=jnp.complex64),
        "ni": ni,
    }


def build_rydberg_mvp_components(task_spec, candidate) -> tuple[list[MVPComponent], RydbergControlBundle, dict]:
    controls = build_rydberg_control_bundle(task_spec, candidate)
    n_qubits = int(task_spec.n_qubits)
    interaction_v = float(task_spec.interaction_v)
    delta0 = float(task_spec.delta0)
    edges = [tuple(edge) for edge in task_spec.edges]

    x_terms = []
    y_terms = []
    n_terms = []
    for site in range(n_qubits):
        x = [0] * n_qubits
        x[site] = 1
        x_terms.append((x, 1.0))
        y = [0] * n_qubits
        y[site] = 2
        y_terms.append((y, 1.0))
        n_terms.append(([0] * n_qubits, 0.5))
        z = [0] * n_qubits
        z[site] = 3
        n_terms.append((z, -0.5))

    nn_terms = []
    for i, j in edges:
        nn_terms.append(([0] * n_qubits, 0.25))
        zi = [0] * n_qubits
        zi[i] = 3
        nn_terms.append((zi, -0.25))
        zj = [0] * n_qubits
        zj[j] = 3
        nn_terms.append((zj, -0.25))
        zizj = [0] * n_qubits
        zizj[i] = 3
        zizj[j] = 3
        nn_terms.append((zizj, 0.25))

    components = [
        MVPComponent("omega_x_sum", make_pauli_mvp(x_terms), lambda t: controls.omega.value(t) / 2.0),
        MVPComponent("cd_y_sum", make_pauli_mvp(y_terms), lambda t: controls.cd.value(t)),
        MVPComponent("minus_delta_n_sum", make_pauli_mvp(n_terms), lambda t: -controls.delta.value(t)),
        MVPComponent("interaction_nn_sum", make_pauli_mvp(nn_terms), lambda t: interaction_v),
    ]
    ops = build_rydberg_ops(n_qubits, edges)
    _, gs_meta = make_ground_subspace_fidelity_fn(
        np.array(ops["n_sum"]), np.array(ops["nn_sum"]), delta0, interaction_v
    )
    return components, controls, gs_meta


def make_initial_state(n_qubits: int) -> jnp.ndarray:
    dim = 2**n_qubits
    return jnp.zeros(dim, dtype=jnp.complex64).at[0].set(1.0)


def omega_trapezoid(t: float, t_max: float, omega0: float) -> float:
    tau = t / t_max
    return float(omega0 * jnp.clip(jnp.minimum(4.0 * tau, 4.0 * (1.0 - tau)), 0.0, 1.0))


def delta_linear(t: float, t_max: float, delta0: float) -> float:
    return float(delta0 * (2.0 * t / t_max - 1.0))


def omega_smooth(t: float, t_max: float, omega0: float) -> float:
    tau = t / t_max
    return float(omega0 * jnp.sin(jnp.pi / 2.0 * jnp.sin(jnp.pi * tau)) ** 2)


def domega_smooth_dt(t: float, t_max: float, omega0: float) -> float:
    tau = t / t_max
    inner = jnp.pi * tau
    deriv_tau = jnp.sin(jnp.pi * jnp.sin(inner)) * (jnp.pi**2 / 2.0) * jnp.cos(inner)
    return float(omega0 * deriv_tau / t_max)


def delta_smooth(t: float, t_max: float, delta0: float) -> float:
    return float(-delta0 * jnp.cos(jnp.pi * t / t_max))


def ddelta_smooth_dt(t: float, t_max: float, delta0: float) -> float:
    tau = t / t_max
    return float(delta0 * jnp.sin(jnp.pi * tau) * (jnp.pi / t_max))


def fy_acqc_smooth(t: float, t_max: float, omega0: float, delta0: float) -> float:
    om = omega_smooth(t, t_max, omega0)
    de = delta_smooth(t, t_max, delta0)
    dom = domega_smooth_dt(t, t_max, omega0)
    dde = ddelta_smooth_dt(t, t_max, delta0)
    numer = om * dde - de * dom
    denom = 2.0 * (om**2 + de**2)
    return float(-numer / denom) if denom > 1e-12 else 0.0


def make_ground_subspace_fidelity_fn(
    n_sum_np: np.ndarray,
    nn_sum_np: np.ndarray,
    delta0: float,
    interaction_v: float,
) -> tuple:
    h_final = -delta0 * n_sum_np + interaction_v * nn_sum_np
    evals, evecs = np.linalg.eigh(h_final.real)
    e0 = evals[0]
    n_degen = int(np.sum(np.abs(evals - e0) < 1e-4 * max(abs(e0), 1.0)))
    gs = jnp.array(evecs[:, :n_degen], dtype=jnp.complex64)

    def fidelity(psi: jnp.ndarray) -> float:
        psi_n = psi / jnp.linalg.norm(psi)
        overlaps = jnp.einsum("i,ij->j", jnp.conj(psi_n), gs)
        return float(jnp.sum(jnp.abs(overlaps) ** 2))

    return fidelity, {"ground_energy": float(e0), "degeneracy": n_degen}


def _channel_map(candidate) -> Dict[str, object]:
    return {channel.name: channel for channel in candidate.channels}


def build_rydberg_control_bundle(task_spec, candidate) -> RydbergControlBundle:
    channels = _channel_map(candidate)
    if "omega" not in channels or "delta" not in channels:
        raise ValueError("Rydberg candidate requires 'omega' and 'delta' channels")

    omega_control = build_rydberg_channel_control("omega", channels["omega"], task_spec, candidate)
    delta_control = build_rydberg_channel_control("delta", channels["delta"], task_spec, candidate)
    cd_control = build_rydberg_cd_control(candidate.cd, omega_control, delta_control, task_spec, candidate)

    active_terms = [
        "Omega(t)/2 * Sum_i X_i",
        "-Delta(t) * Sum_i n_i",
    ]
    if candidate.cd.kind != "none":
        active_terms.append("f_cd(t) * Sum_i Y_i")
    active_terms.append("V * Sum_(i,j in E) n_i n_j")

    hamiltonian_formula = (
        "H(t) = Omega(t)/2 * Sum_i X_i + f_cd(t) * Sum_i Y_i - Delta(t) * Sum_i n_i + "
        "V * Sum_(i,j in E) n_i n_j\n\n"
        f"{omega_control.formula}\n"
        f"{delta_control.formula}\n"
        f"{cd_control.formula}\n"
    )
    return RydbergControlBundle(
        omega=omega_control,
        delta=delta_control,
        cd=cd_control,
        active_terms=active_terms,
        hamiltonian_formula=hamiltonian_formula,
    )


def build_rydberg_model(task_spec, candidate) -> dict:
    ops = build_rydberg_ops(task_spec.n_qubits, task_spec.edges)
    x_sum = ops["X_sum"]
    y_sum = ops["Y_sum"]
    n_sum = ops["n_sum"]
    nn_sum = ops["nn_sum"]

    controls = build_rydberg_control_bundle(task_spec, candidate)
    interaction_v = float(task_spec.interaction_v)
    delta0 = float(task_spec.delta0)

    def hamiltonian_fn(t, *_):
        om = controls.omega.value(t)
        de = controls.delta.value(t)
        fy = controls.cd.value(t)
        return om / 2.0 * x_sum + fy * y_sum - de * n_sum + interaction_v * nn_sum

    fidelity_fn, gs_meta = make_ground_subspace_fidelity_fn(
        np.array(n_sum), np.array(nn_sum), delta0, interaction_v
    )
    meta = {
        "omega_basis": controls.omega.basis,
        "delta_basis": controls.delta.basis,
        "cd_kind": candidate.cd.kind,
        "omega_params": dict(controls.omega.params),
        "delta_params": dict(controls.delta.params),
        "cd_params": dict(controls.cd.params),
        "omega_formula": controls.omega.formula,
        "delta_formula": controls.delta.formula,
        "cd_formula": controls.cd.formula,
        "n_qubits": int(task_spec.n_qubits),
        "n_edges": int(len(task_spec.edges)),
        **gs_meta,
    }
    return {
        "hamiltonian_fn": hamiltonian_fn,
        "fidelity_fn": fidelity_fn,
        "meta": meta,
        "controls": controls,
    }


def build_rydberg_hamiltonian_fn(task_spec, candidate):
    model = build_rydberg_model(task_spec, candidate)
    return model["hamiltonian_fn"], model["fidelity_fn"], model["meta"]
