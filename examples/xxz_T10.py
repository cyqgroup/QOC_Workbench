#!/usr/bin/env python3
"""Dedicated reproduction driver for `papers/2603.15794v1.pdf` Tables I--III.

This script is intentionally separate from `run_xxz_ring8_experiment.py`.
The LBI search script records paper table baselines and performs fast heuristic
search; this driver should be used for strict reproduction with longer
continuous optimization of AH fields and CD alpha.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qoc-lbi")

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qoc_lbi.xxz_qoc import (  # noqa: E402
    build_sparse_xxz_operators,
    lambda_smooth,
    product_state_from_angles,
    simulate_xxz_protocol_fast,
    sparse_initial_hamiltonian_from_angles,
    sparse_xxz_target,
)


def optimize_cd_alpha(ops, hi, hf, psi0, total_time, steps, mode, alpha_bound, aux_fields=None):
    def objective(alpha):
        result = simulate_xxz_protocol_fast(
            ops,
            hi,
            hf,
            total_time,
            lambda_smooth,
            psi0,
            aux_fields=aux_fields,
            cd_alpha=float(alpha),
            cd_mode=mode,
            n_steps=steps,
            metrics=False,
        )
        return result.final_energy

    result = minimize_scalar(objective, bounds=(-alpha_bound, alpha_bound), method="bounded", options={"xatol": 1e-3})
    grid = np.array([-alpha_bound, -0.5 * alpha_bound, 0.0, 0.5 * alpha_bound, alpha_bound])
    candidates = [(float(result.fun), float(result.x))]
    candidates.extend((float(objective(alpha)), float(alpha)) for alpha in grid)
    _, alpha = min(candidates, key=lambda item: item[0])
    return alpha


def optimize_ah_fields(ops, hi, hf, psi0, delta, total_time, steps, maxiter, method="powell"):
    def objective(fields):
        result = simulate_xxz_protocol_fast(
            ops,
            hi,
            hf,
            total_time,
            lambda_smooth,
            psi0,
            aux_fields=np.asarray(fields, dtype=float),
            n_steps=steps,
            metrics=False,
        )
        return result.final_energy

    alt = np.array([(-1.0) ** j for j in range(ops.n_sites)], dtype=float)
    seeds = [np.zeros(ops.n_sites), alt, -alt, 2.0 * alt, -2.0 * alt]
    best_x = seeds[0]
    best_val = objective(best_x)
    if method == "de":
        result = differential_evolution(
            objective,
            bounds=[(-5.0, 5.0)] * ops.n_sites,
            seed=260315794 + int(100 * delta),
            maxiter=maxiter,
            popsize=5,
            polish=False,
            tol=1e-4,
            workers=1,
        )
        best_x, best_val = np.asarray(result.x, dtype=float), float(result.fun)
    else:
        for seed in seeds:
            result = minimize(
                objective,
                seed,
                method="Powell",
                bounds=[(-5.0, 5.0)] * ops.n_sites,
                options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-3},
            )
            if float(result.fun) < best_val:
                best_x, best_val = np.asarray(result.x, dtype=float), float(result.fun)
    return best_x


def optimize_oi_angles(ops, hf, delta, maxiter, method="de"):
    def objective(params):
        theta = np.asarray(params[: ops.n_sites], dtype=float)
        phi = np.asarray(params[ops.n_sites :], dtype=float)
        psi = product_state_from_angles(theta, phi, sign=-1.0)
        return float(np.real(np.vdot(psi, hf @ psi)))

    theta_x = np.full(ops.n_sites, np.pi / 2.0)
    phi_x = np.zeros(ops.n_sites)
    theta_staggered = np.array([0.0 if j % 2 == 0 else np.pi for j in range(ops.n_sites)])
    seeds = [np.concatenate([theta_x, phi_x]), np.concatenate([theta_staggered, phi_x])]
    bounds = [(0.0, np.pi)] * ops.n_sites + [(0.0, 2.0 * np.pi)] * ops.n_sites
    best_x = seeds[0]
    best_val = objective(best_x)
    if method == "de":
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=260315794 + int(1000 * delta),
            maxiter=maxiter,
            popsize=5,
            polish=False,
            tol=1e-5,
            workers=1,
        )
        best_x, best_val = np.asarray(result.x, dtype=float), float(result.fun)
    for seed in seeds + [best_x]:
        result = minimize(
            objective,
            seed,
            method="Powell",
            bounds=bounds,
            options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-4},
        )
        if float(result.fun) < best_val:
            best_x, best_val = np.asarray(result.x, dtype=float), float(result.fun)
    theta = np.asarray(best_x[: ops.n_sites], dtype=float)
    phi = np.mod(np.asarray(best_x[ops.n_sites :], dtype=float), 2.0 * np.pi)
    return theta, phi, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--total-time", type=float, default=10.0)
    parser.add_argument("--ah-maxiter", type=int, default=60)
    parser.add_argument("--ah-method", choices=["powell", "de"], default="powell")
    parser.add_argument("--oi-maxiter", type=int, default=80)
    parser.add_argument("--oi-method", choices=["powell", "de"], default="de")
    parser.add_argument("--oi-mode", choices=["optimize", "fixed-staggered"], default="optimize")
    parser.add_argument("--alpha-bound", type=float, default=10.0)
    parser.add_argument("--skip-ah", action="store_true", help="Skip expensive AH field optimization and write SA+AH as not_run")
    parser.add_argument("--no-metrics", action="store_true", help="Skip F_ad instantaneous eigensolves for speed")
    parser.add_argument("--cd-mode", choices=["commutator", "global_y", "local_y_staggered"], default="commutator")
    args = parser.parse_args()

    time_tag = f"T{args.total_time:g}".replace(".", "p")
    run_dir = ROOT / "artifacts" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_xxz_paper_{time_tag}_reproduction"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    ops = build_sparse_xxz_operators(8)
    rows = []
    params = []
    out_csv = run_dir / "reproduction_results.csv"

    for delta in [0.5, 1.0, 1.5]:
        print(f"[repro] delta={delta}", flush=True)
        hf = sparse_xxz_target(ops, delta)
        theta_x = np.full(ops.n_sites, np.pi / 2.0)
        phi_x = np.zeros(ops.n_sites)
        hi_sa = sparse_initial_hamiltonian_from_angles(ops, theta_x, phi_x, sign=-1.0)
        psi_sa = product_state_from_angles(theta_x, phi_x, sign=-1.0)

        if args.oi_mode == "optimize":
            theta_oi, phi_oi, oi_energy = optimize_oi_angles(ops, hf, delta, args.oi_maxiter, args.oi_method)
        else:
            theta_oi = np.array([0.0 if j % 2 == 0 else np.pi for j in range(ops.n_sites)])
            phi_oi = np.zeros(ops.n_sites)
            psi_tmp = product_state_from_angles(theta_oi, phi_oi, sign=-1.0)
            oi_energy = float(np.real(np.vdot(psi_tmp, hf @ psi_tmp)))
        hi_oi = sparse_initial_hamiltonian_from_angles(ops, theta_oi, phi_oi, sign=-1.0)
        psi_oi = product_state_from_angles(theta_oi, phi_oi, sign=-1.0)

        aux_sa = None if args.skip_ah else optimize_ah_fields(ops, hi_sa, hf, psi_sa, delta, args.total_time, args.steps, args.ah_maxiter, args.ah_method)
        aux_oi = None if args.skip_ah else optimize_ah_fields(ops, hi_oi, hf, psi_oi, delta, args.total_time, args.steps, args.ah_maxiter, args.ah_method)
        alpha_sa = optimize_cd_alpha(ops, hi_sa, hf, psi_sa, args.total_time, args.steps, args.cd_mode, args.alpha_bound)
        alpha_oi = optimize_cd_alpha(ops, hi_oi, hf, psi_oi, args.total_time, args.steps, args.cd_mode, args.alpha_bound)
        alpha_oi_ah = None if aux_oi is None else optimize_cd_alpha(ops, hi_oi, hf, psi_oi, args.total_time, args.steps, args.cd_mode, args.alpha_bound, aux_fields=aux_oi)
        params.append({
            "delta": delta,
            "oi_energy": oi_energy,
            "theta_oi": list(map(float, theta_oi)),
            "phi_oi": list(map(float, phi_oi)),
            "aux_sa": None if aux_sa is None else list(map(float, aux_sa)),
            "aux_oi": None if aux_oi is None else list(map(float, aux_oi)),
            "alpha_sa": alpha_sa,
            "alpha_oi": alpha_oi,
            "alpha_oi_ah": alpha_oi_ah,
        })
        (run_dir / "optimized_parameters.json").write_text(json.dumps(params, indent=2))

        specs = [
            ("SA", hi_sa, psi_sa, None, None),
            ("SA+AH", hi_sa, psi_sa, aux_sa, None),
            ("SA+CD", hi_sa, psi_sa, None, alpha_sa),
            ("OI", hi_oi, psi_oi, None, None),
            ("OI+AH", hi_oi, psi_oi, aux_oi, None),
            ("OI+CD", hi_oi, psi_oi, None, alpha_oi),
            ("OI+AH+CD", hi_oi, psi_oi, aux_oi, alpha_oi_ah),
        ]
        for name, hi, psi0, aux_fields, alpha in specs:
            if aux_fields is None and "AH" in name:
                continue
            result = simulate_xxz_protocol_fast(
                ops,
                hi,
                hf,
                args.total_time,
                lambda_smooth,
                psi0,
                aux_fields=aux_fields,
                cd_alpha=alpha,
                cd_mode=args.cd_mode,
                n_steps=args.steps,
                metrics=not args.no_metrics,
            )
            rows.append({
                "delta": delta,
                "protocol": name,
                "N": result.normalized_energy_distance,
                "F_ad": result.adiabatic_fidelity,
                "final_ground_fidelity": result.final_ground_fidelity,
                "final_energy": result.final_energy,
                "alpha": "" if alpha is None else alpha,
                "aux_fields": "" if aux_fields is None else list(map(float, aux_fields)),
                "inner_solver_backend": result.inner_solver.get("backend", ""),
                "inner_solver_representation": result.inner_solver.get("representation", ""),
                "inner_solver_ode_backend": result.inner_solver.get("ode_backend", ""),
                "inner_solver_ode_solver": result.inner_solver.get("ode_solver", ""),
                "inner_solver_used_jit": result.inner_solver.get("used_jit", ""),
            })
            print(rows[-1], flush=True)
            with out_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(run_dir)


if __name__ == "__main__":
    main()
