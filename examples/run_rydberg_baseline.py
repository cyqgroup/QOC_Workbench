#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qoc_lbi.artifacts import ensure_run_dir
from qoc_lbi.artifact_checker import check_required_artifacts
from qoc_lbi.budget import BudgetTracker
from qoc_lbi.evaluator import QOCEvaluator, EvaluatorConfig
from qoc_lbi.hamiltonian_term_checker import check_hamiltonian_structure
from qoc_lbi.loop import SearchLoopState, build_trial_record, record_trial, update_best_trial, write_summary
from qoc_lbi.protocol import CDConfig, ChannelConfig, ProtocolCandidate
from qoc_lbi.rydberg_mis import evaluate_rydberg_mis, rydberg_sim_cost_fn
from qoc_lbi.source_guard import snapshot_protected_files, verify_protected_files_unchanged
from qoc_lbi.stop_rules import evaluate_stop_rules, recommend_continue_actions
from qoc_lbi.task_loader import load_task_spec
from qoc_lbi.time_series_checker import check_time_series_constraints


def candidate_from_dict(task_id: str, payload: dict) -> ProtocolCandidate:
    return ProtocolCandidate(
        candidate_id=payload["candidate_id"],
        task_id=task_id,
        family=payload["family"],
        hardware=payload["hardware"],
        total_time=float(payload["total_time"]),
        channels=[ChannelConfig(**channel) for channel in payload.get("channels", [])],
        cd=CDConfig(**payload.get("cd", {})),
        constraints=payload.get("constraints", {}),
        provenance=payload.get("provenance", {}),
        notes=payload.get("notes", []),
    )


def write_protocol_py(run_dir: Path, candidate: ProtocolCandidate) -> None:
    body = f"""# Auto-generated phase-1 baseline protocol
PROTOCOL = {repr(candidate.to_dict())}
"""
    (run_dir / "protocol.py").write_text(body)


def write_readme(run_dir: Path, task_spec: dict, candidate: ProtocolCandidate, result) -> None:
    continue_actions = recommend_continue_actions(task_spec, result.metric)
    text = (
        f"# Baseline Run\n\n"
        f"- task_id: {task_spec['task_id']}\n"
        f"- hardware: {task_spec['hardware']}\n"
        f"- objective_metric: {task_spec['objective_metric']}\n"
        f"- candidate_id: {candidate.candidate_id}\n"
        f"- metric: {result.metric:.6f}\n"
        f"- cumulative_sim_cost: {result.cumulative_sim_cost}\n"
        f"- runtime_sec: {result.runtime_sec:.3f}\n"
        f"- mode: {result.mode}\n"
        f"- continue_actions_if_not_solved: {continue_actions}\n"
    )
    (run_dir / "README.md").write_text(text)


def write_failure_notes(run_dir: Path) -> None:
    (run_dir / "failure_notes.md").write_text(
        "# Failure Notes\n\n"
        "Phase 1 baseline run only. No search iterations yet.\n"
    )


def write_diagnostics(run_dir: Path, artifact_bundle: dict | None) -> None:
    if not artifact_bundle:
        return

    times = np.array(artifact_bundle["times"])
    target_fidelity = np.array(artifact_bundle["target_fidelity"])
    inst_overlap = np.array(artifact_bundle["instantaneous_ground_overlap"])
    omega_values = np.array(artifact_bundle["omega_values"])
    delta_values = np.array(artifact_bundle["delta_values"])
    cd_values = np.array(artifact_bundle["cd_values"])
    snapshot_times = np.array(artifact_bundle["snapshot_times"])
    snapshots = np.array(artifact_bundle["hamiltonian_snapshots_real"])

    (run_dir / "hamiltonian_form.md").write_text(
        "# Instantaneous Hamiltonian Form\n\n" + artifact_bundle["hamiltonian_formula"]
    )

    np.savez(
        run_dir / "trajectory_diagnostics.npz",
        times=times,
        target_fidelity=target_fidelity,
        instantaneous_ground_overlap=inst_overlap,
        omega_values=omega_values,
        delta_values=delta_values,
        cd_values=cd_values,
        snapshot_times=snapshot_times,
        hamiltonian_snapshots_real=snapshots,
        inner_solver_backend=artifact_bundle.get("inner_solver_backend", ""),
        inner_solver_representation=artifact_bundle.get("inner_solver_representation", ""),
        inner_solver_ode_backend=artifact_bundle.get("inner_solver_ode_backend", ""),
        inner_solver_ode_solver=artifact_bundle.get("inner_solver_ode_solver", ""),
        inner_solver_used_jit=bool(artifact_bundle.get("inner_solver_used_jit", False)),
    )

    plt.figure(figsize=(6, 4))
    plt.plot(times, target_fidelity, lw=2)
    plt.xlabel("time t")
    plt.ylabel("overlap with final ground subspace")
    plt.title("Target Fidelity vs Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "target_fidelity_vs_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(times, inst_overlap, lw=2, color="tab:green")
    plt.xlabel("time t")
    plt.ylabel("overlap with instantaneous ground subspace")
    plt.title("Instantaneous Ground Overlap vs Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "instantaneous_ground_overlap_vs_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(times, omega_values, lw=2, label="Omega(t)")
    plt.plot(times, delta_values, lw=2, label="Delta(t)")
    if np.max(np.abs(cd_values)) > 1e-10:
        plt.plot(times, cd_values, lw=2, label="f_cd(t)")
    plt.xlabel("time t")
    plt.ylabel("control value")
    plt.title("Schedule / Control Shapes")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "schedule_shapes.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, len(snapshot_times), figsize=(4.8 * len(snapshot_times), 4))
    if len(snapshot_times) == 1:
        axes = [axes]
    for ax, t_snap, mat in zip(axes, snapshot_times, snapshots):
        im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
        ax.set_title(f"Re[H(t)] at t={t_snap:.3f}")
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(run_dir / "hamiltonian_snapshots_real.png", dpi=150)
    plt.close()


def write_efficiency_plot(run_dir: Path, rows: list[dict]) -> None:
    xs = [row["cumulative_sim_cost"] for row in rows]
    ys = [row["metric"] for row in rows]
    running_best = []
    best = float("-inf")
    for y in ys:
        best = max(best, y)
        running_best.append(best)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label="metric")
    plt.plot(xs, running_best, marker="s", label="running_best")
    plt.xlabel("cumulative_sim_cost")
    plt.ylabel("metric")
    plt.title("Sample Efficiency")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "sample_efficiency.png", dpi=150)
    plt.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root
    task_spec_path = repo_root / "configs" / "task_specs" / "rydberg_mis_c6.yaml"
    task_spec = load_task_spec(task_spec_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_run_dir(base_dir / "artifacts" / f"{timestamp}_{task_spec['task_id']}_baseline")

    candidate = candidate_from_dict(task_spec["task_id"], task_spec["baseline_candidate"])
    term_check = check_hamiltonian_structure(task_spec, candidate)
    if not term_check.ok:
        raise RuntimeError(f"hamiltonian term check failed: {term_check.errors}")

    evaluator = QOCEvaluator(
        config=EvaluatorConfig(
            metric_name=task_spec["objective_metric"],
            sim_cost_fn=rydberg_sim_cost_fn,
        ),
        eval_fn=evaluate_rydberg_mis,
    )
    budget = BudgetTracker(max_sim_cost=int(task_spec["sim_budget"]))
    protected_snapshot = snapshot_protected_files(repo_root, task_spec.get("protected_paths", []))
    loop_state = SearchLoopState()
    trial_records: list[dict] = []

    sim_cost = rydberg_sim_cost_fn(candidate, mode="full")
    if not budget.can_afford(sim_cost):
        raise RuntimeError("baseline run exceeds budget before execution")

    result, payload = evaluator.evaluate_with_payload(
        task_spec=task_spec,
        candidate=candidate,
        mode="full",
        trial_index=0,
        cumulative_sim_cost=budget.cumulative_sim_cost + sim_cost,
    )
    budget.charge(result.sim_cost)
    trial_record = build_trial_record(
        task_spec=task_spec,
        candidate=candidate,
        eval_result=result,
        payload=payload,
        active_terms=term_check.active_terms,
        continue_actions=recommend_continue_actions(task_spec, result.metric),
        stop_decision={},
        term_check_ok=term_check.ok,
    )
    is_best = update_best_trial(task_spec, loop_state, trial_record)
    stop_decision = evaluate_stop_rules(
        task_spec,
        budget,
        None if loop_state.best_trial_record is None else float(loop_state.best_trial_record["metric"]),
        trial_records=[trial_record],
    )
    trial_record["stop_decision"] = stop_decision
    record_trial(run_dir, trial_record)
    trial_records.append(trial_record)
    write_summary(run_dir, trial_records)
    write_failure_notes(run_dir)
    write_efficiency_plot(run_dir, trial_records)

    if is_best:
        write_protocol_py(run_dir, candidate)
        (run_dir / "best_protocol.json").write_text(json.dumps(candidate.to_dict(), indent=2))
        write_readme(run_dir, task_spec, candidate, result)
        write_diagnostics(run_dir, payload.get("artifact_bundle"))

    artifact_check = check_required_artifacts(run_dir, task_spec, mode="full")
    if not artifact_check.ok:
        raise RuntimeError(f"missing required artifacts: {artifact_check.missing}")

    ts_check = check_time_series_constraints(task_spec, payload.get("artifact_bundle"))
    if not ts_check.ok:
        raise RuntimeError(f"time-series constraint violation: {ts_check.errors}")

    source_guard = verify_protected_files_unchanged(repo_root, protected_snapshot)
    if not source_guard.ok:
        raise RuntimeError(f"protected source files changed during run: {source_guard.changed}")

    print(f"task_id={task_spec['task_id']}")
    print(f"candidate_id={candidate.candidate_id}")
    print(f"metric={result.metric:.6f}")
    print(f"cumulative_sim_cost={budget.cumulative_sim_cost}")
    print(f"active_terms={term_check.active_terms}")
    print(f"preference_metrics={trial_record['preference_metrics']}")
    print(f"ranking={trial_record['ranking']}")
    print(f"artifact_check={artifact_check.ok}")
    print(f"time_series_check={ts_check.ok}")
    print(f"source_guard={source_guard.ok}")
    print(f"stop_decision={stop_decision}")
    print(f"continue_actions={recommend_continue_actions(task_spec, result.metric)}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
