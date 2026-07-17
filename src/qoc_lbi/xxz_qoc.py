from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csc_matrix, eye as sparse_eye, kron as sparse_kron
from scipy.sparse.linalg import eigsh
from scipy.optimize import differential_evolution, minimize

import jax.numpy as jnp

from .inner_solver import evolve_raw_mvp_ode

Array = np.ndarray
SparseArray = csc_matrix

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def kron_embed(op: Array, site: int, n_sites: int) -> Array:
    ops = [_I2] * n_sites
    ops[site] = op
    out = ops[0]
    for item in ops[1:]:
        out = np.kron(out, item)
    return out


def sparse_kron_embed(op: Array, site: int, n_sites: int) -> SparseArray:
    op_sparse = csc_matrix(op)
    ident = sparse_eye(2, format="csc", dtype=complex)
    out = csc_matrix([[1.0 + 0.0j]])
    for idx in range(n_sites):
        out = sparse_kron(out, op_sparse if idx == site else ident, format="csc")
    return out


@dataclass(frozen=True)
class XXZOperators:
    n_sites: int
    x: Tuple[Array, ...]
    y: Tuple[Array, ...]
    z: Tuple[Array, ...]
    xx_sum: Array
    yy_sum: Array
    zz_sum: Array
    x_sum: Array
    y_sum: Array
    z_sum: Array


@dataclass(frozen=True)
class SparseXXZOperators:
    n_sites: int
    x: Tuple[SparseArray, ...]
    y: Tuple[SparseArray, ...]
    z: Tuple[SparseArray, ...]
    xx_sum: SparseArray
    yy_sum: SparseArray
    zz_sum: SparseArray
    x_sum: SparseArray
    y_sum: SparseArray
    z_sum: SparseArray


def build_xxz_operators(n_sites: int = 8) -> XXZOperators:
    x_ops = tuple(kron_embed(_X, j, n_sites) for j in range(n_sites))
    y_ops = tuple(kron_embed(_Y, j, n_sites) for j in range(n_sites))
    z_ops = tuple(kron_embed(_Z, j, n_sites) for j in range(n_sites))
    dim = 2**n_sites
    xx_sum = np.zeros((dim, dim), dtype=complex)
    yy_sum = np.zeros_like(xx_sum)
    zz_sum = np.zeros_like(xx_sum)
    for j in range(n_sites):
        k = (j + 1) % n_sites
        xx_sum += x_ops[j] @ x_ops[k]
        yy_sum += y_ops[j] @ y_ops[k]
        zz_sum += z_ops[j] @ z_ops[k]
    return XXZOperators(
        n_sites=n_sites,
        x=x_ops,
        y=y_ops,
        z=z_ops,
        xx_sum=xx_sum,
        yy_sum=yy_sum,
        zz_sum=zz_sum,
        x_sum=sum(x_ops),
        y_sum=sum(y_ops),
        z_sum=sum(z_ops),
    )


def build_sparse_xxz_operators(n_sites: int = 8) -> SparseXXZOperators:
    x_ops = tuple(sparse_kron_embed(_X, j, n_sites) for j in range(n_sites))
    y_ops = tuple(sparse_kron_embed(_Y, j, n_sites) for j in range(n_sites))
    z_ops = tuple(sparse_kron_embed(_Z, j, n_sites) for j in range(n_sites))
    dim = 2**n_sites
    zero = csc_matrix((dim, dim), dtype=complex)
    xx_sum = zero.copy()
    yy_sum = zero.copy()
    zz_sum = zero.copy()
    for j in range(n_sites):
        k = (j + 1) % n_sites
        xx_sum = xx_sum + x_ops[j] @ x_ops[k]
        yy_sum = yy_sum + y_ops[j] @ y_ops[k]
        zz_sum = zz_sum + z_ops[j] @ z_ops[k]
    return SparseXXZOperators(
        n_sites=n_sites,
        x=x_ops,
        y=y_ops,
        z=z_ops,
        xx_sum=xx_sum,
        yy_sum=yy_sum,
        zz_sum=zz_sum,
        x_sum=sum(x_ops, zero.copy()),
        y_sum=sum(y_ops, zero.copy()),
        z_sum=sum(z_ops, zero.copy()),
    )


def xxz_target(ops: XXZOperators, delta: float, j_coupling: float = 1.0) -> Array:
    return j_coupling * (ops.xx_sum + ops.yy_sum + delta * ops.zz_sum)


def sparse_xxz_target(ops: SparseXXZOperators, delta: float, j_coupling: float = 1.0) -> SparseArray:
    return (j_coupling * (ops.xx_sum + ops.yy_sum + delta * ops.zz_sum)).tocsc()


def lambda_smooth(s: float) -> float:
    return float(np.sin((np.pi / 2.0) * np.sin(np.pi * s / 2.0) ** 2) ** 2)


def dlambda_smooth_ds(s: float) -> float:
    inner = (np.pi / 2.0) * np.sin(np.pi * s / 2.0) ** 2
    return float(
        np.sin(2.0 * inner)
        * (np.pi / 2.0)
        * np.sin(np.pi * s)
        * (np.pi / 2.0)
    )


def make_power_schedule(power: float) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    p = float(power)

    def schedule(s: float) -> float:
        s_clip = float(np.clip(s, 0.0, 1.0))
        if p <= 0.0:
            return s_clip
        a = s_clip**p
        b = (1.0 - s_clip) ** p
        denom = a + b
        return float(a / denom) if denom > 1e-15 else s_clip

    def deriv(s: float) -> float:
        s_clip = float(np.clip(s, 1e-9, 1.0 - 1e-9))
        a = s_clip**p
        b = (1.0 - s_clip) ** p
        ap = p * s_clip ** (p - 1.0)
        bp = -p * (1.0 - s_clip) ** (p - 1.0)
        denom = a + b
        return float((ap * denom - a * (ap + bp)) / (denom * denom))

    return schedule, deriv


def make_smooth_mixed_schedule(power: float, mix: float) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    power_schedule, power_deriv = make_power_schedule(power)
    eta = float(mix)

    def schedule(s: float) -> float:
        return float((1.0 - eta) * lambda_smooth(s) + eta * power_schedule(s))

    def deriv(s: float) -> float:
        return float((1.0 - eta) * dlambda_smooth_ds(s) + eta * power_deriv(s))

    return schedule, deriv


def product_state_from_angles(theta: Array, phi: Array, sign: float = -1.0) -> Array:
    state = np.array([1.0 + 0.0j])
    for th, ph in zip(theta, phi):
        direction = np.array([
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            np.cos(th),
        ])
        local_h = sign * (direction[0] * _X + direction[1] * _Y + direction[2] * _Z)
        evals, evecs = np.linalg.eigh(local_h)
        local = evecs[:, int(np.argmin(evals))]
        state = np.kron(state, local)
    return state / np.linalg.norm(state)


def initial_hamiltonian_from_angles(ops: XXZOperators, theta: Array, phi: Array, epsilon: float = 1.0, sign: float = -1.0) -> Array:
    out = np.zeros_like(ops.x_sum)
    for j, (th, ph) in enumerate(zip(theta, phi)):
        out += sign * epsilon * (
            np.sin(th) * np.cos(ph) * ops.x[j]
            + np.sin(th) * np.sin(ph) * ops.y[j]
            + np.cos(th) * ops.z[j]
        )
    return out


def sparse_initial_hamiltonian_from_angles(ops: SparseXXZOperators, theta: Array, phi: Array, epsilon: float = 1.0, sign: float = -1.0) -> SparseArray:
    dim = 2**ops.n_sites
    out = csc_matrix((dim, dim), dtype=complex)
    for j, (th, ph) in enumerate(zip(theta, phi)):
        out = out + sign * epsilon * (
            np.sin(th) * np.cos(ph) * ops.x[j]
            + np.sin(th) * np.sin(ph) * ops.y[j]
            + np.cos(th) * ops.z[j]
        )
    return out.tocsc()


def optimize_product_initial_state(hf: Array, n_sites: int, seed: int = 260315794, maxiter: int = 120) -> Tuple[Array, Array, float]:
    def objective(params: Array) -> float:
        theta = params[:n_sites]
        phi = params[n_sites:]
        psi = product_state_from_angles(theta, phi)
        return float(np.real(np.vdot(psi, hf @ psi)))

    bounds = [(0.0, np.pi)] * n_sites + [(0.0, 2.0 * np.pi)] * n_sites
    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=10,
        polish=False,
        tol=1e-7,
        updating="immediate",
        workers=1,
    )
    polished = minimize(objective, result.x, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-9, "fatol": 1e-9})
    params = polished.x if polished.fun <= result.fun else result.x
    return params[:n_sites], np.mod(params[n_sites:], 2.0 * np.pi), min(float(result.fun), float(polished.fun))


def optimize_aux_fields(
    ops: XXZOperators,
    hi: Array,
    hf: Array,
    total_time: float,
    schedule: Callable[[float], float],
    initial_state: Array,
    seed: int = 260315794,
    maxiter: int = 80,
    bound: float = 2.5,
) -> Tuple[Array, float]:
    def objective(fields: Array) -> float:
        result = simulate_xxz_protocol(
            ops=ops,
            hi=hi,
            hf=hf,
            total_time=total_time,
            schedule=schedule,
            initial_state=initial_state,
            aux_fields=fields,
            n_steps=120,
            metrics=False,
        )
        return float(result.final_energy)

    bounds = [(-bound, bound)] * ops.n_sites
    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=8,
        polish=False,
        tol=1e-6,
        updating="immediate",
        workers=1,
    )
    polished = minimize(objective, result.x, method="Nelder-Mead", options={"maxiter": 900, "xatol": 1e-7, "fatol": 1e-7})
    fields = polished.x if polished.fun <= result.fun else result.x
    return fields, min(float(result.fun), float(polished.fun))


def commutator_cd_matrix(hi: Array, hf: Array, lam_dot: float, alpha: float) -> Array:
    dh = hf - hi
    had_mid = None
    _ = had_mid
    return None  # placeholder for type checkers


@dataclass
class SimulationResult:
    protocol_id: str
    delta: float
    total_time: float
    normalized_energy_distance: float
    adiabatic_fidelity: float
    final_ground_fidelity: float
    final_energy: float
    initial_energy: float
    ground_energy: float
    schedule_values: Array = field(default_factory=lambda: np.array([]))
    instantaneous_overlap: Array = field(default_factory=lambda: np.array([]))
    target_fidelity: Array = field(default_factory=lambda: np.array([]))
    times: Array = field(default_factory=lambda: np.array([]))
    spectrum_low: Array = field(default_factory=lambda: np.array([]))
    inner_solver: Dict[str, object] = field(default_factory=dict)


def evolve_xxz_mvp_ode(
    hi: Array,
    hf: Array,
    aux: Array,
    ops: XXZOperators,
    schedule: Callable[[float], float],
    initial_state: Array,
    total_time: float,
    n_steps: int,
    *,
    cd_alpha: float | None = None,
    cd_mode: str = "commutator",
    cd_scale_fn: Callable[[float], float] | None = None,
) -> Tuple[Array, Array, Dict[str, object]]:
    times = np.linspace(0.0, float(total_time), int(n_steps) + 1)
    s_grid = np.clip(times / float(total_time), 0.0, 1.0)
    lam_grid = np.asarray([float(schedule(float(s))) for s in s_grid], dtype=np.float32)
    dlam_grid = np.gradient(lam_grid, times, edge_order=1).astype(np.float32)
    cd_shape_grid = np.ones_like(lam_grid, dtype=np.float32)
    if cd_scale_fn is not None:
        cd_shape_grid = np.asarray([float(cd_scale_fn(float(s))) for s in s_grid], dtype=np.float32)

    hi_j = jnp.asarray(hi, dtype=jnp.complex64)
    hf_j = jnp.asarray(hf, dtype=jnp.complex64)
    aux_j = jnp.asarray(aux, dtype=jnp.complex64)
    dh_j = hf_j - hi_j
    y_sum_j = jnp.asarray(ops.y_sum, dtype=jnp.complex64)
    y_staggered = np.zeros_like(hf)
    for site, y_op in enumerate(ops.y):
        y_staggered += ((-1.0) ** site) * y_op
    y_staggered_j = jnp.asarray(y_staggered, dtype=jnp.complex64)
    time_grid_j = jnp.asarray(times, dtype=jnp.float32)
    lam_grid_j = jnp.asarray(lam_grid, dtype=jnp.float32)
    dlam_grid_j = jnp.asarray(dlam_grid, dtype=jnp.float32)
    cd_shape_grid_j = jnp.asarray(cd_shape_grid, dtype=jnp.float32)
    alpha = jnp.asarray(0.0 if cd_alpha is None else float(cd_alpha), dtype=jnp.float32)

    def rhs(psi, time):
        lam = jnp.interp(time, time_grid_j, lam_grid_j)
        dlam_dt = jnp.interp(time, time_grid_j, dlam_grid_j)
        cd_shape = jnp.interp(time, time_grid_j, cd_shape_grid_j)
        h = (1.0 - lam) * hi_j + lam * hf_j + lam * (1.0 - lam) * aux_j
        if cd_alpha is not None and abs(float(cd_alpha)) > 1e-14:
            if cd_mode == "global_y":
                h = h + dlam_dt * alpha * cd_shape * y_sum_j
            elif cd_mode == "local_y_staggered":
                h = h + dlam_dt * alpha * cd_shape * y_staggered_j
            else:
                d_h_dlam = dh_j + (1.0 - 2.0 * lam) * aux_j
                h = h + 1.0j * dlam_dt * alpha * cd_shape * (h @ d_h_dlam - d_h_dlam @ h)
        return -1.0j * (h @ psi)

    result = evolve_raw_mvp_ode(
        rhs,
        initial_state,
        times,
        ode_backend="diffrax",
        solver="Tsit5",
        max_steps=2_000_000,
        rtol=1e-7,
        atol=1e-7,
    )
    return result.times, result.states, {
        "backend": result.backend,
        "representation": result.representation,
        "ode_backend": result.ode_backend,
        "ode_solver": result.solver,
        "used_jit": result.used_jit,
    }


def ground_subspace(h: Array, tol: float = 1e-8) -> Tuple[float, Array]:
    evals, evecs = np.linalg.eigh(h)
    e0 = float(evals[0].real)
    mask = np.abs(evals - e0) <= tol * max(1.0, abs(e0))
    return e0, evecs[:, mask]


def subspace_fidelity(psi: Array, subspace: Array) -> float:
    overlaps = np.conj(psi) @ subspace
    return float(np.real(np.sum(np.abs(overlaps) ** 2)))


def simulate_xxz_protocol(
    ops: XXZOperators,
    hi: Array,
    hf: Array,
    total_time: float,
    schedule: Callable[[float], float],
    initial_state: Array | None = None,
    aux_fields: Array | None = None,
    aux_matrix: Array | None = None,
    cd_alpha: float | None = None,
    cd_mode: str = "commutator",
    cd_scale_fn: Callable[[float], float] | None = None,
    n_steps: int = 260,
    protocol_id: str = "protocol",
    delta: float = 0.0,
    metrics: bool = True,
) -> SimulationResult:
    if initial_state is None:
        _, ground = ground_subspace(hi)
        initial_state = ground[:, 0]
    e0, hf_ground = ground_subspace(hf)
    initial_energy = float(np.real(np.vdot(initial_state, hf @ initial_state)))
    aux = np.zeros_like(hf)
    if aux_matrix is not None:
        aux += np.asarray(aux_matrix, dtype=complex)
    elif aux_fields is not None:
        aux_array = np.asarray(aux_fields)
        if aux_array.shape == hf.shape:
            aux += aux_array.astype(complex)
        else:
            for field, z_op in zip(aux_array, ops.z):
                aux += float(field) * z_op
    dh = hf - hi

    def hamiltonian_at(t: float) -> Array:
        s = float(np.clip(t / total_time, 0.0, 1.0))
        lam = schedule(s)
        h = (1.0 - lam) * hi + lam * hf + lam * (1.0 - lam) * aux
        if cd_alpha is not None and abs(cd_alpha) > 1e-14:
            eps = 1e-5
            lp = schedule(min(1.0, s + eps))
            lm = schedule(max(0.0, s - eps))
            dlam_ds = (lp - lm) / (min(1.0, s + eps) - max(0.0, s - eps)) if eps < s < 1.0 - eps else 0.0
            dlam_dt = dlam_ds / total_time
            cd_shape = 1.0 if cd_scale_fn is None else float(cd_scale_fn(s))
            if cd_mode == "global_y":
                h = h + dlam_dt * float(cd_alpha) * cd_shape * ops.y_sum
            elif cd_mode == "local_y_staggered":
                y_aux = np.zeros_like(hf)
                for j, y_op in enumerate(ops.y):
                    y_aux += ((-1.0) ** j) * y_op
                h = h + dlam_dt * float(cd_alpha) * cd_shape * y_aux
            else:
                d_h_dlam = dh + (1.0 - 2.0 * lam) * aux
                h = h + 1.0j * dlam_dt * float(cd_alpha) * cd_shape * (h @ d_h_dlam - d_h_dlam @ h)
        return h

    times, states, inner_solver = evolve_xxz_mvp_ode(
        hi,
        hf,
        aux,
        ops,
        schedule,
        initial_state,
        total_time,
        n_steps,
        cd_alpha=cd_alpha,
        cd_mode=cd_mode,
        cd_scale_fn=cd_scale_fn,
    )
    final_state = states[-1]
    final_energy = float(np.real(np.vdot(final_state, hf @ final_state)))
    denom = initial_energy - e0
    normalized = float((initial_energy - final_energy) / denom) if abs(denom) > 1e-12 else 0.0
    normalized = float(np.clip(normalized, -10.0, 1.0))
    final_ground = subspace_fidelity(final_state, hf_ground)

    if not metrics:
        return SimulationResult(
            protocol_id,
            delta,
            total_time,
            normalized,
            0.0,
            final_ground,
            final_energy,
            initial_energy,
            e0,
            inner_solver=inner_solver,
        )

    inst = []
    target = []
    schedules = []
    low_spectrum = []
    stride = max(1, len(times) // 160)
    for idx, (t, psi) in enumerate(zip(times, states)):
        s = float(np.clip(t / total_time, 0.0, 1.0))
        schedules.append(schedule(s))
        target.append(subspace_fidelity(psi, hf_ground))
        h = hamiltonian_at(float(t))
        evals, evecs = np.linalg.eigh(h)
        e0_t = float(evals[0].real)
        mask = np.abs(evals - e0_t) <= 1e-8 * max(1.0, abs(e0_t))
        # Paper Eq. (15) defines F_ad with |<phi(t)|Psi(t)>|, not its square.
        # For a degenerate instantaneous ground subspace we use the square root
        # of the total subspace probability as the corresponding amplitude.
        inst.append(float(np.sqrt(max(0.0, subspace_fidelity(psi, evecs[:, mask])))))
        if idx % stride == 0 or idx == len(times) - 1:
            low_spectrum.append(evals[:8].real)
    ad_fid = float(np.trapz(inst, times) / total_time)
    return SimulationResult(
        protocol_id=protocol_id,
        delta=delta,
        total_time=total_time,
        normalized_energy_distance=normalized,
        adiabatic_fidelity=ad_fid,
        final_ground_fidelity=final_ground,
        final_energy=final_energy,
        initial_energy=initial_energy,
        ground_energy=e0,
        schedule_values=np.array(schedules),
        instantaneous_overlap=np.array(inst),
        target_fidelity=np.array(target),
        times=times,
        spectrum_low=np.array(low_spectrum),
        inner_solver=inner_solver,
    )


def optimize_cd_alpha(
    ops: XXZOperators,
    hi: Array,
    hf: Array,
    total_time: float,
    schedule: Callable[[float], float],
    initial_state: Array,
    aux_fields: Array | None = None,
    bound: float = 10.0,
) -> Tuple[float, float]:
    def eval_alpha(alpha: float) -> float:
        result = simulate_xxz_protocol(
            ops=ops,
            hi=hi,
            hf=hf,
            total_time=total_time,
            schedule=schedule,
            initial_state=initial_state,
            aux_fields=aux_fields,
            cd_alpha=float(alpha),
            n_steps=32,
            metrics=False,
        )
        return result.final_energy

    grid = np.array([-bound, -0.6 * bound, -0.25 * bound, -0.08 * bound, 0.0, 0.08 * bound, 0.25 * bound, 0.6 * bound, bound])
    values = np.array([eval_alpha(float(alpha)) for alpha in grid])
    best_idx = int(np.argmin(values))
    alpha = float(grid[best_idx])
    if 0 < best_idx < len(grid) - 1:
        local_grid = np.linspace(grid[best_idx - 1], grid[best_idx + 1], 7)
        local_values = np.array([eval_alpha(float(item)) for item in local_grid])
        local_idx = int(np.argmin(local_values))
        alpha = float(local_grid[local_idx])
        return alpha, float(local_values[local_idx])
    return alpha, float(values[best_idx])

@dataclass
class FastSimulationResult:
    normalized_energy_distance: float
    adiabatic_fidelity: float
    final_ground_fidelity: float
    final_energy: float
    initial_energy: float
    ground_energy: float
    inner_solver: Dict[str, object] = field(default_factory=dict)


def _lowest_subspace_sparse(h: SparseArray, k: int = 4, tol: float = 1e-8) -> tuple[float, Array]:
    dim = h.shape[0]
    if dim <= 4:
        evals, evecs = np.linalg.eigh(h.toarray())
    else:
        evals, evecs = eigsh(h, k=min(k, dim - 2), which="SA", tol=1e-10)
        order = np.argsort(evals.real)
        evals = evals[order]
        evecs = evecs[:, order]
    e0 = float(evals[0].real)
    mask = np.abs(evals.real - e0) <= tol * max(1.0, abs(e0))
    if not np.any(mask):
        mask[0] = True
    return e0, evecs[:, mask]


def simulate_xxz_protocol_fast(
    ops: SparseXXZOperators,
    hi: SparseArray,
    hf: SparseArray,
    total_time: float,
    schedule: Callable[[float], float],
    initial_state: Array,
    aux_fields: Array | None = None,
    cd_alpha: float | None = None,
    cd_mode: str = "commutator",
    n_steps: int = 80,
    metrics: bool = True,
) -> FastSimulationResult:
    dense_ops = build_xxz_operators(ops.n_sites)
    result = simulate_xxz_protocol(
        ops=dense_ops,
        hi=hi.toarray(),
        hf=hf.toarray(),
        total_time=total_time,
        schedule=schedule,
        initial_state=np.asarray(initial_state, dtype=complex),
        aux_fields=aux_fields,
        cd_alpha=cd_alpha,
        cd_mode=cd_mode,
        n_steps=n_steps,
        metrics=metrics,
    )
    return FastSimulationResult(
        normalized_energy_distance=result.normalized_energy_distance,
        adiabatic_fidelity=result.adiabatic_fidelity,
        final_ground_fidelity=result.final_ground_fidelity,
        final_energy=result.final_energy,
        initial_energy=result.initial_energy,
        ground_energy=result.ground_energy,
        inner_solver=result.inner_solver,
    )
