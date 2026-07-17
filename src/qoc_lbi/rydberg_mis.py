from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

from .inner_solver import evolve_mvp_ode  # noqa: E402
from .rydberg_backend import (  # noqa: E402
    build_rydberg_ops,
    build_rydberg_model,
    build_rydberg_mvp_components,
    make_initial_state,
)
from .time_series_checker import check_time_series_constraints  # noqa: E402


def rydberg_sim_cost_fn(candidate, mode: str) -> int:
    """
    Coarse accounting rule for phase 1.

    We bill by time-step count so probe/full runs are distinguishable without
    coupling budget logic to wall-clock fluctuations.
    """
    if mode == "probe":
        return 40
    return 200


def _make_task_view(task_spec: dict):
    public = task_spec["public_inputs"]
    phys = task_spec["physical_params"]

    class _TaskView:
        n_qubits = int(public["n_qubits"])
        edges = [tuple(edge) for edge in public["edges"]]
        omega0 = float(phys["omega0"])
        delta0 = float(phys["delta0"])
        interaction_v = float(phys["interaction_v"])

    return _TaskView()


def _control_trace_summary(times_np: np.ndarray, omega_values: np.ndarray, delta_values: np.ndarray, cd_values: np.ndarray) -> dict[str, Any]:
    return {
        "time_grid_size": int(len(times_np)),
        "omega_range": [float(np.min(omega_values)), float(np.max(omega_values))],
        "delta_range": [float(np.min(delta_values)), float(np.max(delta_values))],
        "cd_range": [float(np.min(cd_values)), float(np.max(cd_values))],
    }


def _sample_control_artifact_bundle(candidate, controls, n_steps: int) -> dict[str, Any]:
    times_np = np.linspace(0.0, float(candidate.total_time), int(n_steps))
    omega_values = np.array([float(controls.omega.value(float(t))) for t in times_np])
    delta_values = np.array([float(controls.delta.value(float(t))) for t in times_np])
    cd_values = np.array([float(controls.cd.value(float(t))) for t in times_np])
    return {
        "times": times_np,
        "schedule_values": np.array(times_np / float(candidate.total_time)),
        "omega_values": omega_values,
        "delta_values": delta_values,
        "cd_values": cd_values,
    }


def _smoothness_score(*series: np.ndarray) -> float:
    roughness_parts = []
    for values in series:
        if len(values) < 2:
            continue
        span = float(np.max(values) - np.min(values))
        scale = span * span if span > 1e-12 else 1.0
        diffs = np.diff(values)
        roughness = float(np.mean(diffs * diffs) / scale)
        roughness_parts.append(roughness)
    if not roughness_parts:
        return 1.0
    return float(1.0 / (1.0 + np.mean(roughness_parts)))


def _translation_symmetry_score(candidate) -> float:
    for channel in candidate.channels:
        params = channel.params or {}
        if any(key in params for key in ("site", "sites", "per_site", "site_values")):
            return 0.0
    return 1.0


def _simplicity_score(candidate) -> float:
    complexity = float(len(candidate.channels))
    if candidate.cd.kind != "none":
        complexity += 1.0
    param_count = sum(len(channel.params or {}) for channel in candidate.channels)
    if candidate.cd.params:
        param_count += len(candidate.cd.params)
    complexity += 0.1 * param_count
    return float(1.0 / (1.0 + complexity))


def evaluate_rydberg_mis(task_spec: dict, candidate, mode: str) -> Dict[str, Any]:
    """
    Concrete evaluator backend for fixed-graph Rydberg MIS tasks.
    """
    defaults = task_spec["evaluator_defaults"]
    task_view = _make_task_view(task_spec)
    model = build_rydberg_model(task_view, candidate)
    hamiltonian_fn = model["hamiltonian_fn"]
    fidelity_fn = model["fidelity_fn"]
    meta = model["meta"]
    mvp_components, controls, _ = build_rydberg_mvp_components(task_view, candidate)
    psi0 = make_initial_state(task_view.n_qubits)

    if mode == "probe":
        n_steps = int(defaults["n_steps_probe"])
        rtol = float(defaults["rtol_probe"])
        atol = float(defaults["atol_probe"])
    else:
        n_steps = int(defaults["n_steps_full"])
        rtol = float(defaults["rtol_full"])
        atol = float(defaults["atol_full"])

    precheck_cfg = task_spec.get("pre_solver_validation", {})
    precheck_points = int(precheck_cfg.get("time_series_sample_points", max(64, min(n_steps, 256))))
    precheck_bundle = _sample_control_artifact_bundle(candidate, controls, n_steps=precheck_points)
    precheck = check_time_series_constraints(task_spec, precheck_bundle)
    if not precheck.ok:
        raise ValueError("pre_solver_time_series_constraints_failed: " + "; ".join(precheck.errors))

    times = jnp.linspace(0.0, float(candidate.total_time), n_steps)
    solver_result = evolve_mvp_ode(
        mvp_components,
        psi0,
        times,
        ode_backend="diffrax",
        solver="Tsit5",
        max_steps=2_000_000,
        rtol=rtol,
        atol=atol,
    )
    states = jnp.asarray(solver_result.states, dtype=jnp.complex64)
    psi_f = states[-1] / jnp.linalg.norm(states[-1])
    metric = fidelity_fn(psi_f)
    trace_bundle = _sample_control_artifact_bundle(candidate, controls, n_steps=n_steps)
    times_np = np.array(trace_bundle["times"])
    omega_values = np.array(trace_bundle["omega_values"])
    delta_values = np.array(trace_bundle["delta_values"])
    cd_values = np.array(trace_bundle["cd_values"])
    preference_metrics = {
        "smoothness_score": _smoothness_score(omega_values, delta_values, cd_values),
        "translation_symmetry_score": _translation_symmetry_score(candidate),
        "simplicity_score": _simplicity_score(candidate),
    }
    hamiltonian_formula = controls.hamiltonian_formula

    artifact_bundle = None
    if mode == "full":
        states_np = np.array(states)
        ops = build_rydberg_ops(task_view.n_qubits, task_view.edges)
        n_sum_np = np.array(ops["n_sum"])
        nn_sum_np = np.array(ops["nn_sum"])
        h_final = -task_view.delta0 * n_sum_np + task_view.interaction_v * nn_sum_np
        evals_f, evecs_f = np.linalg.eigh(h_final.real)
        e0_f = evals_f[0]
        degen_f = int(np.sum(np.abs(evals_f - e0_f) < 1e-4 * max(abs(e0_f), 1.0)))
        gs_final = evecs_f[:, :degen_f]

        target_fidelity = []
        inst_ground_overlap = []
        snapshot_specs = [0.0, float(candidate.total_time) / 2.0, float(candidate.total_time)]
        snapshot_mats = []

        for idx, t in enumerate(times_np):
            psi_t = states_np[idx]
            psi_t = psi_t / np.linalg.norm(psi_t)

            overlaps_target = np.conj(psi_t) @ gs_final
            target_fidelity.append(float(np.sum(np.abs(overlaps_target) ** 2)))

            h_t = np.array(hamiltonian_fn(float(t)))
            evals_t, evecs_t = np.linalg.eigh(h_t.real)
            e0_t = evals_t[0]
            degen_t = int(np.sum(np.abs(evals_t - e0_t) < 1e-4 * max(abs(e0_t), 1.0)))
            gs_inst = evecs_t[:, :degen_t]
            overlaps_inst = np.conj(psi_t) @ gs_inst
            inst_ground_overlap.append(float(np.sum(np.abs(overlaps_inst) ** 2)))

        for t in snapshot_specs:
            snapshot_mats.append(np.array(hamiltonian_fn(t)).real)

        artifact_bundle = {
            "times": times_np,
            "target_fidelity": np.array(target_fidelity),
            "instantaneous_ground_overlap": np.array(inst_ground_overlap),
            "schedule_values": np.array(trace_bundle["schedule_values"]),
            "omega_values": omega_values,
            "delta_values": delta_values,
            "cd_values": cd_values,
            "hamiltonian_formula": hamiltonian_formula,
            "snapshot_times": np.array(snapshot_specs),
            "hamiltonian_snapshots_real": np.array(snapshot_mats),
        }

    return {
        "metric": float(metric),
        "diagnostics": {
            "backend": "rydberg_mis",
            "n_steps": n_steps,
            "rtol": rtol,
            "atol": atol,
            "inner_solver_backend": solver_result.backend,
            "inner_solver_representation": solver_result.representation,
            "inner_solver_ode_backend": solver_result.ode_backend,
            "inner_solver_ode_solver": solver_result.solver,
            "inner_solver_used_jit": solver_result.used_jit,
            "builder_meta": meta,
            "translation_symmetry_score": preference_metrics["translation_symmetry_score"],
            "control_trace_summary": _control_trace_summary(times_np, omega_values, delta_values, cd_values),
        },
        "hamiltonian_formula": hamiltonian_formula,
        "preference_metrics": preference_metrics,
        "artifact_bundle": artifact_bundle,
    }
