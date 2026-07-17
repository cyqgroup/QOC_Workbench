#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qoc-lbi")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qoc_lbi.xxz_qoc import (  # noqa: E402
    build_xxz_operators,
    ground_subspace,
    initial_hamiltonian_from_angles,
    lambda_smooth,
    make_power_schedule,
    make_smooth_mixed_schedule,
    product_state_from_angles,
    simulate_xxz_protocol,
    xxz_target,
)


PAPER_BASELINE = {
    0.5: {
        "SA": (0.20, 0.40),
        "SA+AH": (0.94, 0.87),
        "SA+CD": (1.00, 0.99),
        "OI": (0.98, 0.99),
        "OI+CD": (0.99, 0.96),
    },
    1.0: {
        "SA": (0.00, 0.37),
        "SA+AH": (0.90, 0.71),
        "SA+CD": (0.00, 0.37),
        "OI": (1.00, 1.00),
        "OI+CD": (1.00, 0.94),
    },
    1.5: {
        "SA": (0.12, 0.37),
        "SA+AH": (0.94, 0.82),
        "SA+CD": (1.00, 0.99),
        "OI": (0.99, 0.99),
        "OI+CD": (0.99, 0.96),
    },
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def paper_baseline_rows() -> list[dict]:
    rows = []
    for delta, protocols in PAPER_BASELINE.items():
        for protocol, (n_value, f_ad) in protocols.items():
            rows.append(
                {
                    "delta": delta,
                    "protocol": protocol,
                    "N": n_value,
                    "F_ad": f_ad,
                    "source": "Table I-III of papers/2603.15794v1.pdf",
                }
            )
    return rows


def best_baseline(delta: float) -> tuple[str, float, float, float]:
    items = PAPER_BASELINE[delta]
    best_name = max(items, key=lambda name: 0.5 * (items[name][0] + items[name][1]))
    n_value, f_ad = items[best_name]
    return best_name, n_value, f_ad, 0.5 * (n_value + f_ad)


def relative_gain(value: float, baseline: float) -> float:
    if abs(baseline) < 1e-12:
        return float("inf") if value > baseline else 0.0
    return 100.0 * (value / baseline - 1.0)


def plot_protocol(run_dir: Path, result, label: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    axes[0].plot(result.times, result.target_fidelity, label="target ground-subspace fidelity")
    axes[0].set_ylabel("F_target")
    axes[0].grid(alpha=0.3)
    axes[1].plot(result.times, result.instantaneous_overlap, color="tab:orange", label="instantaneous ground overlap")
    axes[1].set_ylabel("inst. overlap")
    axes[1].grid(alpha=0.3)
    axes[2].plot(result.times, result.schedule_values, color="tab:green")
    axes[2].set_ylabel("lambda")
    axes[2].set_xlabel("time")
    axes[2].grid(alpha=0.3)
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(run_dir / f"{label}_diagnostics.png", dpi=170)
    plt.close(fig)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "artifacts" / f"{timestamp}_xxz_ring8_lbi_search"
    run_dir.mkdir(parents=True, exist_ok=False)

    ops = build_xxz_operators(8)
    total_time = 10.0
    eps = 1.0
    rows_baseline = paper_baseline_rows()
    write_csv(run_dir / "paper_baseline_T10.csv", rows_baseline)

    experiment_rows: list[dict] = []
    best_rows: list[dict] = []
    best_payload: dict[str, dict] = {}

    for delta in [0.5, 1.0, 1.5]:
        print(f"[xxz] delta={delta}: building candidate family", flush=True)
        hf = xxz_target(ops, delta)
        theta_x = np.full(ops.n_sites, np.pi / 2.0)
        phi_x = np.zeros(ops.n_sites)
        hi_sa = initial_hamiltonian_from_angles(ops, theta_x, phi_x, epsilon=eps, sign=-1.0)
        psi_sa = product_state_from_angles(theta_x, phi_x, sign=-1.0)

        # Fast deterministic OI surrogate for the antiferromagnetic XXZ ring:
        # a staggered product state regularizes low-energy crossings while keeping
        # the initial Hamiltonian local and separable.  We test both z-staggered
        # and canted x/z-staggered variants and keep the lower target energy.
        oi_candidates = []
        for cant in [0.0, 0.18, 0.35, 0.55]:
            theta = np.array([cant if j % 2 == 0 else np.pi - cant for j in range(ops.n_sites)])
            phi = np.zeros(ops.n_sites)
            psi = product_state_from_angles(theta, phi, sign=-1.0)
            energy = float(np.real(np.vdot(psi, hf @ psi)))
            oi_candidates.append((energy, theta, phi, f"staggered_cant_{cant:g}"))
        oi_initial_energy, theta_oi, phi_oi, oi_label = min(oi_candidates, key=lambda item: item[0])
        hi_oi = initial_hamiltonian_from_angles(ops, theta_oi, phi_oi, epsilon=eps, sign=-1.0)
        psi_oi = product_state_from_angles(theta_oi, phi_oi, sign=-1.0)

        # Endpoint-vanishing Zeeman terms.  The alternating field is motivated by
        # the antiferromagnetic ordering tendency; the small defect fields break
        # residual translation/parity degeneracies.
        aux_alt = np.array([(-1.0) ** j for j in range(ops.n_sites)])
        aux_defect = np.array([1.0, -0.7, 0.45, -0.3, 0.18, -0.12, 0.08, -0.05])
        aux_sa = 0.6 * aux_alt + 0.12 * aux_defect
        aux_oi = 0.8 * aux_alt + 0.08 * aux_defect
        alpha_grid = [-6.0, -2.5, -1.0, -0.35, 0.35, 1.0, 2.5, 6.0]

        candidate_specs = [
            ("local_SA_repro", hi_sa, psi_sa, lambda_smooth, None, None, "local simulation of standard paper schedule"),
            ("local_SA_AH_repro", hi_sa, psi_sa, lambda_smooth, aux_sa, None, "optimized site-dependent Zeeman auxiliary field"),
            ("local_SA_CD_repro", hi_sa, psi_sa, lambda_smooth, None, 1.0, "first-order nested-commutator CD"),
            ("local_OI_repro", hi_oi, psi_oi, lambda_smooth, None, None, "optimized separable initial Hamiltonian"),
            ("local_OI_CD_repro", hi_oi, psi_oi, lambda_smooth, None, 1.0, "optimized initial Hamiltonian plus CD"),
            ("LBI_OI_AH_CD", hi_oi, psi_oi, lambda_smooth, aux_oi, 1.0, "spectral engineering plus CD"),
        ]

        for alpha in [-2.5, -0.35, 0.35, 2.5]:
            candidate_specs.append((f"LBI_OI_AH_CD_alpha_{alpha:g}", hi_oi, psi_oi, lambda_smooth, aux_oi, alpha, f"OI+AH+CD with fixed alpha={alpha:g}"))

        for power in [0.55, 1.4, 2.4]:
            schedule, _ = make_power_schedule(power)
            for alpha in [-0.35, 0.35]:
                candidate_specs.append((f"LBI_OI_AH_CD_power_{power:g}_a{alpha:g}", hi_oi, psi_oi, schedule, aux_oi, alpha, f"OI+AH+CD with power schedule p={power:g}, alpha={alpha:g}"))
        for power, mix in [(0.6, 0.5), (2.0, 0.5)]:
            schedule, _ = make_smooth_mixed_schedule(power, mix)
            for alpha in [0.35]:
                candidate_specs.append((f"LBI_OI_AH_CD_mixed_p{power:g}_m{mix:g}_a{alpha:g}", hi_oi, psi_oi, schedule, aux_oi, alpha, f"smooth/power mixed schedule p={power:g}, mix={mix:g}, alpha={alpha:g}"))

        for aux_scale in [0.55, 1.35]:
            for alpha in [0.35]:
                candidate_specs.append((f"LBI_OI_AHscale_{aux_scale:g}_CD_a{alpha:g}", hi_oi, psi_oi, lambda_smooth, aux_scale * aux_oi, alpha, f"OI+scaled auxiliary field, scale={aux_scale:g}, alpha={alpha:g}"))

        local_best = None
        for name, hi, psi0, schedule, aux, alpha, notes in candidate_specs:
            print(f"[xxz] delta={delta}: evaluating {name}", flush=True)
            result = simulate_xxz_protocol(
                ops=ops,
                hi=hi,
                hf=hf,
                total_time=total_time,
                schedule=schedule,
                initial_state=psi0,
                aux_fields=aux,
                cd_alpha=alpha,
                n_steps=24,
                protocol_id=name,
                delta=delta,
                metrics=True,
            )
            base_name, base_n, base_f, base_score = best_baseline(delta)
            score = 0.5 * (result.normalized_energy_distance + result.adiabatic_fidelity)
            row = {
                "delta": delta,
                "protocol": name,
                "N": result.normalized_energy_distance,
                "F_ad": result.adiabatic_fidelity,
                "final_ground_fidelity": result.final_ground_fidelity,
                "final_energy": result.final_energy,
                "best_paper_baseline": base_name,
                "paper_baseline_N": base_n,
                "paper_baseline_F_ad": base_f,
                "N_gain_percent": relative_gain(result.normalized_energy_distance, base_n),
                "F_ad_gain_percent": relative_gain(result.adiabatic_fidelity, base_f),
                "score": score,
                "score_gain_percent": relative_gain(score, base_score),
                "notes": notes,
                "cd_alpha": "" if alpha is None else alpha,
                "aux_fields": "" if aux is None else json.dumps([float(x) for x in aux]),
                "oi_theta": json.dumps([float(x) for x in theta_oi]),
                "oi_phi": json.dumps([float(x) for x in phi_oi]),
                "oi_initial_energy": oi_initial_energy,
                "oi_label": oi_label,
                "inner_solver_backend": result.inner_solver.get("backend", ""),
                "inner_solver_representation": result.inner_solver.get("representation", ""),
                "inner_solver_ode_backend": result.inner_solver.get("ode_backend", ""),
                "inner_solver_ode_solver": result.inner_solver.get("ode_solver", ""),
                "inner_solver_used_jit": result.inner_solver.get("used_jit", ""),
            }
            experiment_rows.append(row)
            if local_best is None or row["score"] > local_best[0]["score"]:
                local_best = (row, result)
        assert local_best is not None
        best_rows.append(local_best[0])
        best_payload[str(delta)] = local_best[0]
        plot_protocol(run_dir, local_best[1], f"best_delta_{str(delta).replace('.', 'p')}")

    write_csv(run_dir / "xxz_lbi_search_results.csv", experiment_rows)
    write_csv(run_dir / "xxz_lbi_best_by_delta.csv", best_rows)
    (run_dir / "best_protocols.json").write_text(json.dumps(best_payload, ensure_ascii=False, indent=2))
    shutil.copy2(Path(__file__), run_dir / "run_xxz_ring8_experiment.py")

    (run_dir / "hamiltonian_form.md").write_text(
        "# XXZ ring-8 Hamiltonian and LBI search forms\n\n"
        "Target Hamiltonian:\n\n"
        "```text\n"
        "H_f = J sum_{j=1}^8 (X_j X_{j+1} + Y_j Y_{j+1} + Delta Z_j Z_{j+1})\n"
        "periodic boundary: sigma_9 = sigma_1, J=1, Delta in {0.5, 1.0, 1.5}\n"
        "```\n\n"
        "Paper standard interpolation:\n\n"
        "```text\n"
        "H_ad(t) = [1 - lambda(t/T)] H_i + lambda(t/T) H_f\n"
        "lambda(s) = sin^2[(pi/2) sin^2(pi s/2)]\n"
        "H_i = - sum_j X_j\n"
        "```\n\n"
        "Auxiliary spectral-engineering term:\n\n"
        "```text\n"
        "H(t) = H_ad(t) + lambda(t/T)[1-lambda(t/T)] sum_j omega_j Z_j\n"
        "```\n\n"
        "Optimized initial Hamiltonian:\n\n"
        "```text\n"
        "H_i = - sum_j u_j . sigma_j,  u_j=(sin theta_j cos phi_j, sin theta_j sin phi_j, cos theta_j)\n"
        "```\n\n"
        "Approximate CD term used in the local search:\n\n"
        "```text\n"
        "H_cd(t) = i dot(lambda) alpha [H(t), dH/dlambda]\n"
        "```\n",
    )

    (run_dir / "strategy.md").write_text(
        "# Strategy relation to the Rydberg MIS example\n\n"
        "The previous `rydberg_mis_c6` example searches Rydberg problem-encoding controls "
        "Omega(t), Delta(t), and local CD corrections for graph optimization. This `xxz_ring8` "
        "example keeps the LBI idea but changes the paradigm to many-body ground-state preparation. "
        "The literature lesson from `papers/2603.15794v1.pdf` is that spectral engineering should "
        "precede CD: optimize the initial separable Hamiltonian and/or add endpoint-vanishing auxiliary "
        "terms before adding approximate counterdiabatic driving. Recent CD literature motivates testing "
        "weighted or performance-guaranteed CD variants, but this first implementation keeps a bounded "
        "nested-commutator CD ansatz and broadens the search with schedule deformation.\n\n"
        "Search families attempted:\n\n"
        "1. local reproduction of SA, SA+AH, SA+CD, OI, OI+CD.\n"
        "2. OI+AH+CD spectral engineering.\n"
        "3. OI+AH+CD with power schedules.\n"
        "4. OI+AH+CD with smooth/power mixed schedules.\n"
    )

    print(json.dumps({"run_dir": str(run_dir), "best": best_payload}, indent=2))


if __name__ == "__main__":
    main()
