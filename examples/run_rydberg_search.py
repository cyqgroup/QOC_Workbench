#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qoc_lbi.artifact_checker import check_required_artifacts
from qoc_lbi.artifacts import ensure_run_dir, write_jsonl
from qoc_lbi.budget import BudgetTracker
from qoc_lbi.evaluator import QOCEvaluator, EvaluatorConfig
from qoc_lbi.hamiltonian_term_checker import check_hamiltonian_structure
from qoc_lbi.loop import SearchLoopState, build_trial_record, record_trial, update_best_trial, write_summary
from qoc_lbi.protocol import ProtocolCandidate
from qoc_lbi.run_dir import (
    ensure_default_run_dir_system,
    load_python_module,
    load_heuristic_search_module,
    load_protocol_candidate,
    update_heuristic_search_context,
    write_protocol_module,
    write_search_memory,
    write_search_state,
)
from qoc_lbi.rydberg_mis import evaluate_rydberg_mis, rydberg_sim_cost_fn
from qoc_lbi.source_guard import snapshot_protected_files, verify_protected_files_unchanged
from qoc_lbi.stop_rules import (
    collect_distinct_strategy_signatures,
    evaluate_stop_rules,
    recommend_continue_actions,
    strategy_signature,
)
from qoc_lbi.task_loader import load_task_spec
from qoc_lbi.time_series_checker import check_time_series_constraints

SIMPLIFICATION_DEFAULT_TOLERANCE = 1e-3
EDITABLE_RUN_DIR_FILES = (
    "detector.py",
    "heuristic_search.py",
    "failure_inspector.py",
    "simplifier.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unattended multi-round Rydberg MIS search.")
    parser.add_argument(
        "--task-spec",
        default="configs/task_specs/rydberg_mis_c6.yaml",
        help="Path to the task spec.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=4,
        help="Maximum search rounds after the initial seed full evaluation.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional existing or new run directory.",
    )
    parser.add_argument(
        "--user-total-time",
        type=float,
        default=None,
        help="Optional user-specified total evolution time T. If provided, it has higher priority than fallback time-scaling heuristics.",
    )
    return parser.parse_args()


def default_seed_candidate(task_spec: dict) -> ProtocolCandidate:
    baseline = task_spec.get("baseline_candidate")
    if isinstance(baseline, dict):
        return ProtocolCandidate.from_dict(baseline, task_id=task_spec["task_id"])
    return ProtocolCandidate.from_dict(
        {
            "candidate_id": "seed_default",
            "task_id": task_spec["task_id"],
            "family": "mc_itm",
            "hardware": task_spec["hardware"],
            "total_time": 2.0,
            "channels": [
                {"name": "omega", "basis": "trapezoid", "params": {}},
                {"name": "delta", "basis": "linear", "params": {}},
            ],
            "cd": {"kind": "none", "ansatz": None, "order": None, "params": {}},
            "constraints": {"omega_nonnegative": True},
            "provenance": {"source": "default_seed"},
            "notes": ["generated default seed"],
        }
    )


def apply_user_total_time_override(candidate: ProtocolCandidate, user_total_time: float | None) -> ProtocolCandidate:
    if user_total_time is None:
        return candidate
    payload = candidate.to_dict()
    payload["total_time"] = float(user_total_time)
    payload["candidate_id"] = f"{payload['candidate_id']}_userT{str(round(float(user_total_time), 3)).replace('.', 'p')}"
    notes = list(payload.get("notes", []))
    notes.append(f"user_total_time_override={float(user_total_time)}")
    payload["notes"] = notes
    return ProtocolCandidate.from_dict(payload, task_id=candidate.task_id)


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


def build_full_run_context(candidate: ProtocolCandidate, trial_record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    artifact_bundle = payload.get("artifact_bundle") or {}
    target = np.array(artifact_bundle.get("target_fidelity", []), dtype=float)
    inst = np.array(artifact_bundle.get("instantaneous_ground_overlap", []), dtype=float)
    omega = np.array(artifact_bundle.get("omega_values", []), dtype=float)
    delta = np.array(artifact_bundle.get("delta_values", []), dtype=float)
    cd = np.array(artifact_bundle.get("cd_values", []), dtype=float)

    def _tail_gain(values: np.ndarray) -> float | None:
        if values.size < 4:
            return None
        start_idx = max(int(0.9 * len(values)), 0)
        return float(values[-1] - values[start_idx])

    return {
        "trial_index": trial_record.get("trial_index"),
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "hardware": candidate.hardware,
        "total_time": float(candidate.total_time),
        "channel_bases": {channel.name: channel.basis for channel in candidate.channels},
        "cd_kind": candidate.cd.kind,
        "metric": float(trial_record.get("metric", 0.0)),
        "smoothness_score": float(trial_record.get("preference_metrics", {}).get("smoothness_score", 0.0)),
        "translation_symmetry_score": float(trial_record.get("preference_metrics", {}).get("translation_symmetry_score", 0.0)),
        "target_fidelity_end": None if target.size == 0 else float(target[-1]),
        "target_fidelity_max": None if target.size == 0 else float(np.max(target)),
        "target_fidelity_gain_last_window": _tail_gain(target),
        "instantaneous_ground_overlap_end": None if inst.size == 0 else float(inst[-1]),
        "instantaneous_ground_overlap_min": None if inst.size == 0 else float(np.min(inst)),
        "instantaneous_ground_overlap_mean": None if inst.size == 0 else float(np.mean(inst)),
        "omega_range": None if omega.size == 0 else [float(np.min(omega)), float(np.max(omega))],
        "delta_range": None if delta.size == 0 else [float(np.min(delta)), float(np.max(delta))],
        "cd_range": None if cd.size == 0 else [float(np.min(cd)), float(np.max(cd))],
        "hamiltonian_formula": trial_record.get("hamiltonian_formula"),
        "active_hamiltonian_terms": list(trial_record.get("active_hamiltonian_terms", [])),
    }


def write_full_run_context(run_dir: Path, candidate: ProtocolCandidate, trial_record: dict[str, Any], payload: dict[str, Any]) -> None:
    context = build_full_run_context(candidate, trial_record, payload)
    (run_dir / "latest_full_run_context.json").write_text(json.dumps(context, indent=2))


def write_efficiency_plot(run_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
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


def write_failure_notes(run_dir: Path, trial_records: list[dict]) -> None:
    lines = ["# Failure Notes", ""]
    failures = [row for row in trial_records if row.get("error_msg") or not row.get("constraint_ok", True)]
    if not failures:
        lines.append("- no hard failures recorded so far")
    else:
        for row in failures[-10:]:
            lines.append(
                f"- trial={row.get('trial_index')} candidate={row.get('candidate_id')} "
                f"mode={row.get('mode')} error={row.get('error_msg')}"
            )
    lines.extend(["", "## Recent Search Rationales", ""])
    for row in trial_records[-10:]:
        lines.append(
            f"- trial={row.get('trial_index')} stage={row.get('proposal_stage')} "
            f"candidate={row.get('candidate_id')} rationale={row.get('proposal_rationale', '')}"
        )
    (run_dir / "failure_notes.md").write_text("\n".join(lines) + "\n")


def write_readme(
    run_dir: Path,
    task_spec: dict,
    loop_state: SearchLoopState,
    trial_records: list[dict],
    budget: BudgetTracker,
    stop_decision: dict,
) -> None:
    best_trial = loop_state.best_trial_record
    if best_trial is None:
        text = "# Search Run\n\nNo successful best trial recorded.\n"
    else:
        text = (
            "# Search Run\n\n"
            f"- task_id: {task_spec['task_id']}\n"
            f"- hardware: {task_spec['hardware']}\n"
            f"- objective_metric: {task_spec['objective_metric']}\n"
            f"- best_candidate_id: {best_trial['candidate_id']}\n"
            f"- best_metric: {best_trial['metric']:.6f}\n"
            f"- best_mode: {best_trial['mode']}\n"
            f"- total_trials: {len(trial_records)}\n"
            f"- cumulative_sim_cost: {budget.cumulative_sim_cost}\n"
            f"- stop_decision: {stop_decision}\n"
        )
    (run_dir / "README.md").write_text(text)


def compact_trial_context(trial: dict[str, Any] | None) -> dict[str, Any] | None:
    if trial is None:
        return None
    return {
        "trial_index": trial.get("trial_index"),
        "candidate_id": trial.get("candidate_id"),
        "mode": trial.get("mode"),
        "metric": trial.get("metric"),
        "total_time": trial.get("total_time"),
        "channels": trial.get("channel_bases"),
        "cd_kind": trial.get("cd_kind"),
        "proposal_stage": trial.get("proposal_stage"),
        "proposal_rationale": trial.get("proposal_rationale"),
        "continue_actions": trial.get("continue_actions"),
        "error_msg": trial.get("error_msg"),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return repr(value)


def write_stage_report(run_dir: Path, stage_name: str, iteration: int, payload: dict[str, Any]) -> Path:
    stage_dir = ensure_run_dir(run_dir / stage_name)
    stage_path = stage_dir / f"iter_{iteration:02d}.json"
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
    stage_path.write_text(serialized)
    (stage_dir / "latest.json").write_text(serialized)
    return stage_path


def load_run_dir_stage_module(run_dir: Path, filename: str, iteration: int):
    return load_python_module(run_dir / filename, f"aqc_run_stage_{Path(filename).stem}_{iteration}")


def sanitize_edit_targets(targets: list[Any] | None) -> list[str]:
    sanitized: list[str] = []
    for target in targets or []:
        target_str = str(target)
        if target_str not in EDITABLE_RUN_DIR_FILES:
            continue
        if target_str not in sanitized:
            sanitized.append(target_str)
    return sanitized


def build_edit_plan(
    run_dir: Path,
    task_spec: dict,
    detector_report: dict[str, Any] | None,
    failure_report: dict[str, Any] | None,
    best_trial: dict[str, Any] | None,
    iteration: int,
) -> dict[str, Any]:
    _ = task_spec
    plan = None if failure_report is None else failure_report.get("edit_plan")
    if isinstance(plan, dict):
        target_files = sanitize_edit_targets(plan.get("target_files"))
        rationale = plan.get("rationale") or "failure_inspector supplied an explicit edit plan."
        raw_file_objectives = plan.get("file_objectives")
        if isinstance(raw_file_objectives, dict):
            file_objectives = {
                str(target): str(objective)
                for target, objective in raw_file_objectives.items()
                if str(target) in EDITABLE_RUN_DIR_FILES and isinstance(objective, str) and str(objective).strip()
            }
        else:
            file_objectives = {}
        raw_function_objectives = plan.get("function_objectives")
        if isinstance(raw_function_objectives, dict):
            function_objectives = sanitize_function_objectives(raw_function_objectives)
        else:
            function_objectives = {}
    else:
        preferred = None if failure_report is None else failure_report.get("preferred_edit_targets")
        target_files = sanitize_edit_targets(preferred if isinstance(preferred, list) else None)
        if not target_files:
            failure_modes = [] if failure_report is None else list(failure_report.get("failure_modes", []))
            detector_patterns = [] if detector_report is None else list(detector_report.get("detected_patterns", []))
            if "runtime_or_constraint_failures_present" in failure_modes:
                target_files.append("heuristic_search.py")
            if "probe_stage_found_no_improvement" in failure_modes or "no_probe_trials_generated" in detector_patterns:
                target_files.extend(["heuristic_search.py", "detector.py"])
            if "target_metric_not_reached" in failure_modes and "heuristic_search.py" not in target_files:
                target_files.append("heuristic_search.py")
            if "no_obvious_failure_mode" in failure_modes:
                target_files.append("failure_inspector.py")
            if best_trial is not None and best_trial.get("proposal_stage") == "simplify":
                target_files.append("simplifier.py")
            target_files = sanitize_edit_targets(target_files)
        rationale = (
            "failure_inspector preferred edit targets"
            if target_files
            else "defaulted to heuristic_search.py because no explicit edit target was available"
        )
        file_objectives = {}
        function_objectives = {}
    if not target_files:
        target_files = ["heuristic_search.py"]
    if not file_objectives:
        file_objectives = default_file_objectives_for_targets(target_files)
    if not function_objectives:
        function_objectives = default_function_objectives_for_targets(target_files)

    edit_plan = {
        "iteration": iteration,
        "primary_target": target_files[0],
        "target_files": target_files,
        "rationale": rationale,
        "file_objectives": file_objectives,
        "function_objectives": function_objectives,
        "failure_modes": [] if failure_report is None else list(failure_report.get("failure_modes", [])),
        "detector_patterns": [] if detector_report is None else list(detector_report.get("detected_patterns", [])),
    }
    write_stage_report(run_dir, "edit_plan", iteration, edit_plan)
    return edit_plan


def simplification_metric_tolerance(task_spec: dict) -> float:
    prefs = task_spec.get("search_preferences", {})
    return float(prefs.get("simplification_metric_tolerance", SIMPLIFICATION_DEFAULT_TOLERANCE))


def rewrite_trial_logs(run_dir: Path, trial_records: list[dict[str, Any]]) -> None:
    write_jsonl(run_dir / "trials.jsonl", trial_records)
    write_summary(run_dir, trial_records)


def default_file_objectives_for_targets(target_files: list[str]) -> dict[str, str]:
    objectives: dict[str, str] = {}
    for target in target_files:
        if target == "heuristic_search.py":
            objectives[target] = "Improve policy logic so the next probe batch explores more informative valid candidates."
        elif target == "detector.py":
            objectives[target] = "Improve probe-state extraction so full-eval selection uses stronger signals."
        elif target == "failure_inspector.py":
            objectives[target] = "Improve failure diagnosis so the next edit plan points at the real bottleneck."
        elif target == "simplifier.py":
            objectives[target] = "Improve simplification proposals so simpler candidates preserve best full-run performance."
    return objectives


def default_function_objectives_for_targets(target_files: list[str]) -> dict[str, dict[str, str]]:
    objectives: dict[str, dict[str, str]] = {}
    for target in target_files:
        if target == "heuristic_search.py":
            objectives[target] = {
                "generate_proposals": "Improve proposal generation so probe candidates are more informative, diverse, and valid.",
                "choose_full_eval_count": "Use detector evidence to allocate a disciplined number of full evaluations.",
            }
        elif target == "detector.py":
            objectives[target] = {
                "analyze_probe_results": "Improve probe-state summaries so promising candidates are easier to identify.",
            }
        elif target == "failure_inspector.py":
            objectives[target] = {
                "inspect_failures": "Improve failure diagnosis and the quality of the next ordered edit plan.",
            }
        elif target == "simplifier.py":
            objectives[target] = {
                "propose_simplifications": "Generate simpler candidates that are likely to preserve the best full-run metric.",
            }
    return objectives


def sanitize_function_objectives(raw: dict[Any, Any]) -> dict[str, dict[str, str]]:
    cleaned: dict[str, dict[str, str]] = {}
    for target, mapping in raw.items():
        target_name = str(target)
        if target_name not in EDITABLE_RUN_DIR_FILES or not isinstance(mapping, dict):
            continue
        target_mapping = {
            str(name): str(objective)
            for name, objective in mapping.items()
            if isinstance(name, str) and isinstance(objective, str) and str(objective).strip()
        }
        if target_mapping:
            cleaned[target_name] = target_mapping
    return cleaned


def apply_fallback_edit_plan(
    *,
    run_dir: Path,
    seed_candidate: ProtocolCandidate,
    task_spec: dict,
    edit_plan: dict[str, Any] | None,
    failure_report: dict[str, Any] | None,
    detector_report: dict[str, Any] | None,
    user_total_time: float | None,
    iteration: int,
) -> list[str]:
    rewritten: list[str] = []
    targets = ["heuristic_search.py"] if edit_plan is None else sanitize_edit_targets(edit_plan.get("target_files"))
    if not targets:
        targets = ["heuristic_search.py"]

    if "heuristic_search.py" in targets:
        (run_dir / "heuristic_search.py").write_text(
            _fallback_heuristic_search_template(
                seed_candidate=seed_candidate,
                task_spec=task_spec,
                edit_plan=edit_plan,
                failure_report=failure_report,
                detector_report=detector_report,
                user_total_time=user_total_time,
            )
        )
        rewritten.append("heuristic_search.py")

    if "detector.py" in targets:
        (run_dir / "detector.py").write_text(_fallback_detector_template())
        rewritten.append("detector.py")

    return rewritten


def _fallback_heuristic_search_template(
    *,
    seed_candidate: ProtocolCandidate,
    task_spec: dict,
    edit_plan: dict[str, Any] | None,
    failure_report: dict[str, Any] | None,
    detector_report: dict[str, Any] | None,
    user_total_time: float | None,
) -> str:
    seed_payload = repr(seed_candidate.to_dict())
    failure_payload = repr(failure_report or {})
    detector_payload = repr(detector_report or {})
    edit_plan_payload = repr(edit_plan or {})
    allowed_parameterizations = repr(task_spec.get("run_search_directive_defaults", {}).get("allowed_parameterizations", {}))
    allowed_cd_kinds = repr(task_spec.get("run_search_directive_defaults", {}).get("allowed_cd_kinds", []))
    candidate_specs_payload = repr(task_spec.get("candidate_specs", {}))
    return f'''from __future__ import annotations

import json

from qoc_lbi.protocol import ProtocolCandidate

SEED_PROTOCOL = {seed_payload}
FAILURE_REPORT = {failure_payload}
DETECTOR_REPORT = {detector_payload}
EDIT_PLAN = {edit_plan_payload}
USER_TOTAL_TIME = {repr(user_total_time)}
ALLOWED_PARAMETERIZATIONS = {allowed_parameterizations}
ALLOWED_CD_KINDS = {allowed_cd_kinds}
CANDIDATE_SPECS = {candidate_specs_payload}

SEARCH_NOTES = [
    "Fallback no-LLM policy generated from edit_plan and failure diagnostics.",
    "This policy exists so the loop can keep moving even when no external LLM is configured.",
]

# @@SEARCH_CONTEXT_BEGIN
SEARCH_CONTEXT = {{}}
# @@SEARCH_CONTEXT_END


def seed_protocol(task_spec):
    return ProtocolCandidate.from_dict(SEED_PROTOCOL)


def generate_proposals(task_spec, current_best, trials, iteration, continue_actions):
    _ = continue_actions
    base = dict(current_best or SEED_PROTOCOL)
    base["task_id"] = task_spec["task_id"]
    proposals = []
    seen_trial_signatures = _collect_seen_signatures(trials, mode_scope="all")
    seen_full_signatures = _collect_seen_signatures(trials, mode_scope="full")
    batch_signatures = set()
    required_full_distinct = _required_distinct_full(task_spec)
    needs_more_diversity = len(seen_full_signatures) < required_full_distinct

    def add_proposal(payload, rationale):
        signature = _strategy_signature(payload)
        if signature in seen_trial_signatures or signature in batch_signatures:
            return
        batch_signatures.add(signature)
        proposals.append({{"candidate": payload, "rationale": rationale, "stage": "probe"}})

    max_proposals = 6 if needs_more_diversity else 4
    current_t = float(base["total_time"])
    for target_t in _candidate_total_times(current_t):
        for schedule_variant in _ordered_schedule_variants(base):
            payload = _clone_payload(base)
            payload["total_time"] = float(target_t)
            _set_channel(payload, "omega", schedule_variant["omega"]["basis"], schedule_variant["omega"]["params"])
            _set_channel(payload, "delta", schedule_variant["delta"]["basis"], schedule_variant["delta"]["params"])
            payload["cd"] = _cd_payload("none", {{}})
            payload["candidate_id"] = _candidate_name(iteration, target_t, schedule_variant, payload["cd"])
            payload["notes"] = list(payload.get("notes", [])) + ["fallback_executor:diversity_enumeration"]
            add_proposal(
                payload,
                "Fallback enumerates a new runnable schedule/CD combination to expand distinct protocol coverage.",
            )
            if len(proposals) >= max_proposals:
                return proposals[:max_proposals]

            for cd_payload in _cd_variants_for_schedule(schedule_variant):
                if cd_payload["kind"] == "none":
                    continue
                payload_cd = _clone_payload(payload)
                payload_cd["cd"] = cd_payload
                payload_cd["candidate_id"] = _candidate_name(iteration, target_t, schedule_variant, cd_payload)
                payload_cd["notes"] = list(payload_cd.get("notes", [])) + ["fallback_executor:diversity_enumeration_cd"]
                add_proposal(
                    payload_cd,
                    "Fallback enumerates an allowed auxiliary term on top of a runnable schedule to increase protocol diversity.",
                )
                if len(proposals) >= max_proposals:
                    return proposals[:max_proposals]

    return proposals[:max_proposals]


def choose_full_eval_count(task_spec, probe_trial_records, iteration):
    _ = (task_spec, iteration)
    if not probe_trial_records:
        return 0
    return min(4, len(probe_trial_records))


def _clone_payload(payload):
    return {{
        "candidate_id": payload["candidate_id"],
        "task_id": payload["task_id"],
        "family": payload["family"],
        "hardware": payload["hardware"],
        "total_time": float(payload["total_time"]),
        "channels": [{{"name": channel["name"], "basis": channel["basis"], "params": dict(channel.get("params", {{}}))}} for channel in payload.get("channels", [])],
        "cd": {{
            "kind": payload.get("cd", {{}}).get("kind", "none"),
            "ansatz": payload.get("cd", {{}}).get("ansatz"),
            "order": payload.get("cd", {{}}).get("order"),
            "params": dict(payload.get("cd", {{}}).get("params", {{}})),
        }},
        "constraints": dict(payload.get("constraints", {{}})),
        "provenance": dict(payload.get("provenance", {{}})),
        "notes": list(payload.get("notes", [])),
    }}


def _time_token(value):
    return str(round(float(value), 3)).replace(".", "p")


def _preferred_cd_order():
    allowed_orders = list((CANDIDATE_SPECS.get("cd_specs", {{}}) or {{}}).get("allowed_orders", []))
    if "first_order" in allowed_orders:
        return "first_order"
    if "unspecified" in allowed_orders:
        return "unspecified"
    return None


def _cd_payload(kind, params):
    if kind == "none":
        return {{"kind": "none", "ansatz": None, "order": None, "params": {{}}}}
    ansatz = "sum_y" if kind in ("acqc_j0", "acqc_j0_scaled", "y_sum_parameterized") else None
    return {{
        "kind": kind,
        "ansatz": ansatz,
        "order": _preferred_cd_order(),
        "params": dict(params),
    }}


def _collect_seen_signatures(trials, mode_scope="all"):
    signatures = set()
    for row in trials:
        if mode_scope != "all" and row.get("mode") != mode_scope:
            continue
        candidate = row.get("candidate")
        if isinstance(candidate, dict):
            signatures.add(_strategy_signature(candidate))
    return signatures


def _strategy_signature(payload):
    canonical = {{
        "family": payload.get("family"),
        "hardware": payload.get("hardware"),
        "total_time": round(float(payload.get("total_time", 0.0)), 12),
        "channels": sorted(
            [
                {{
                    "name": channel.get("name"),
                    "basis": channel.get("basis"),
                    "params": _normalize_value(channel.get("params", {{}})),
                }}
                for channel in payload.get("channels", [])
            ],
            key=lambda item: str(item["name"]),
        ),
        "cd": _normalize_value(payload.get("cd", {{}})),
        "constraints": _normalize_value(payload.get("constraints", {{}})),
    }}
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _normalize_value(value):
    if isinstance(value, dict):
        return {{str(key): _normalize_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def _required_distinct_full(task_spec):
    for rule in task_spec.get("stop_rules", []):
        if isinstance(rule, dict) and rule.get("rule") == "min_distinct_strategies_attempted":
            if rule.get("mode_scope", "full") == "full":
                return int(rule.get("count", 0))
    return 0


def _candidate_total_times(current_t):
    if USER_TOTAL_TIME is not None:
        return [float(USER_TOTAL_TIME)]
    times = [float(current_t)]
    if FAILURE_REPORT.get("is_time_scale_likely_too_short"):
        times.extend([float(current_t) * 1.5, float(current_t) * 2.0])
    elif DETECTOR_REPORT.get("improved_over_current_best") is False:
        times.append(float(current_t) * 1.25)
    return _unique_preserve_order(times)


def _ordered_schedule_variants(base):
    current = {{
        "name": "current",
        "omega": _channel_descriptor(base, "omega"),
        "delta": _channel_descriptor(base, "delta"),
    }}
    catalog = [
        current,
        {{
            "name": "smooth_pair",
            "omega": {{"basis": "smooth", "params": {{}}}},
            "delta": {{"basis": "smooth", "params": {{}}}},
        }},
        {{
            "name": "smooth_beta_pair",
            "omega": {{"basis": "smooth_beta", "params": {{"p": 2.0, "q": 2.0}}}},
            "delta": {{"basis": "smooth_beta", "params": {{"a": 2.0, "b": 2.0}}}},
        }},
        {{
            "name": "piecewise_pair",
            "omega": {{
                "basis": "piecewise_linear",
                "params": {{"knots": [0.0, 0.25, 0.5, 0.75, 1.0], "values": [0.0, 0.6, 1.0, 0.6, 0.0]}},
            }},
            "delta": {{
                "basis": "piecewise_linear",
                "params": {{"knots": [0.0, 0.25, 0.5, 0.75, 1.0], "values": [-1.0, -0.55, 0.0, 0.55, 1.0]}},
            }},
        }},
        {{
            "name": "fourier_omega",
            "omega": {{
                "basis": "fourier_enveloped",
                "params": {{"order": 1, "a_cos": [0.05], "b_sin": [0.08], "envelope_power": 1.5, "offset": 0.85}},
            }},
            "delta": {{"basis": "smooth", "params": {{}}}},
        }},
        {{
            "name": "fourier_delta",
            "omega": {{"basis": "smooth", "params": {{}}}},
            "delta": {{
                "basis": "fourier_enveloped",
                "params": {{"order": 1, "a_cos": [0.04], "b_sin": [0.06], "envelope_power": 1.5, "offset": 0.0}},
            }},
        }},
    ]
    if FAILURE_REPORT.get("is_schedule_shape_likely_bad"):
        preferred_names = ["smooth_beta_pair", "piecewise_pair", "fourier_omega", "fourier_delta", "smooth_pair", "current"]
    else:
        preferred_names = ["current", "smooth_pair", "smooth_beta_pair", "piecewise_pair", "fourier_omega", "fourier_delta"]
    ordered = []
    for name in preferred_names:
        for variant in catalog:
            if variant["name"] == name:
                ordered.append(variant)
    allowed_omega = set(ALLOWED_PARAMETERIZATIONS.get("omega", []))
    allowed_delta = set(ALLOWED_PARAMETERIZATIONS.get("delta", []))
    filtered = []
    seen = set()
    for variant in ordered:
        omega_basis = variant["omega"]["basis"]
        delta_basis = variant["delta"]["basis"]
        if omega_basis not in allowed_omega or delta_basis not in allowed_delta:
            continue
        signature = json.dumps(variant, sort_keys=True, separators=(",", ":"))
        if signature in seen:
            continue
        seen.add(signature)
        filtered.append(variant)
    return filtered


def _cd_variants_for_schedule(schedule_variant):
    variants = [_cd_payload("none", {{}})]
    schedule_bases = {{
        schedule_variant["omega"]["basis"],
        schedule_variant["delta"]["basis"],
    }}
    smoothish = {{"trapezoid", "linear", "smooth", "smooth_beta"}}
    if "acqc_j0" in ALLOWED_CD_KINDS and schedule_bases.issubset(smoothish):
        variants.append(_cd_payload("acqc_j0", {{}}))
    if "acqc_j0_scaled" in ALLOWED_CD_KINDS and schedule_bases.issubset(smoothish):
        variants.append(_cd_payload("acqc_j0_scaled", {{"alpha": 0.35}}))
    if "y_sum_parameterized" in ALLOWED_CD_KINDS:
        variants.append(
            _cd_payload(
                "y_sum_parameterized",
                {{
                    "basis": "smooth_beta",
                    "scale": 0.08,
                    "p": 2.0,
                    "q": 2.0,
                }},
            )
        )
        variants.append(
            _cd_payload(
                "y_sum_parameterized",
                {{
                    "basis": "piecewise_linear",
                    "scale": 0.25,
                    "knots": [0.0, 0.5, 1.0],
                    "values": [0.0, 0.12, 0.0],
                }},
            )
        )
        variants.append(
            _cd_payload(
                "y_sum_parameterized",
                {{
                    "basis": "fourier_enveloped",
                    "scale": 0.12,
                    "order": 1,
                    "a_cos": [0.02],
                    "b_sin": [0.08],
                    "envelope_power": 1.5,
                }},
            )
        )
    seen = set()
    filtered = []
    for payload in variants:
        signature = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if signature in seen:
            continue
        seen.add(signature)
        filtered.append(payload)
    return filtered


def _channel_descriptor(payload, name):
    for channel in payload.get("channels", []):
        if channel.get("name") == name:
            return {{
                "basis": channel.get("basis"),
                "params": dict(channel.get("params", {{}})),
            }}
    return {{"basis": "smooth", "params": {{}}}}


def _candidate_name(iteration, target_t, schedule_variant, cd_payload):
    cd_token = str(cd_payload["kind"])
    cd_params = dict(cd_payload.get("params", {{}}))
    if cd_token == "acqc_j0_scaled":
        cd_token = f"{{cd_token}}-a{{_param_token(cd_params.get('alpha'))}}"
    elif cd_token == "y_sum_parameterized":
        cd_token = f"{{cd_token}}-{{cd_params.get('basis', 'unknown')}}"
    return (
        f"fallback_i{{iteration}}"
        f"_T{{_time_token(target_t)}}"
        f"_sched-{{schedule_variant['name']}}"
        f"_om-{{schedule_variant['omega']['basis']}}"
        f"_de-{{schedule_variant['delta']['basis']}}"
        f"_cd-{{cd_token}}"
    )


def _set_channel(payload, name, basis, params):
    for channel in payload.get("channels", []):
        if channel.get("name") == name:
            channel["basis"] = basis
            channel["params"] = dict(params or {{}})
            return
    payload.setdefault("channels", []).append({{"name": name, "basis": basis, "params": dict(params or {{}})}})


def _param_token(value):
    if value is None:
        return "none"
    if isinstance(value, float):
        return str(round(float(value), 3)).replace(".", "p")
    return str(value).replace(".", "p")


def _unique_preserve_order(values):
    seen = set()
    out = []
    for value in values:
        key = round(float(value), 12)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return out
'''


def _fallback_detector_template() -> str:
    return '''from __future__ import annotations

SEARCH_NOTES = [
    "Fallback no-LLM detector generated from edit_plan.",
]


def analyze_probe_results(task_spec, probe_trial_records, best_trial, iteration):
    best_metric_before = 0.0 if best_trial is None else float(best_trial.get("metric", 0.0))
    metrics = [float(row.get("metric", 0.0)) for row in probe_trial_records]
    total_times = [float(row.get("total_time", 0.0)) for row in probe_trial_records]
    return {
        "iteration": iteration,
        "n_probe_trials": len(probe_trial_records),
        "best_probe_metric": max(metrics) if metrics else None,
        "mean_probe_metric": (sum(metrics) / len(metrics)) if metrics else None,
        "best_probe_total_time": total_times[metrics.index(max(metrics))] if metrics else None,
        "improved_over_current_best": bool(metrics and max(metrics) > best_metric_before),
        "detected_patterns": ["fallback_detector_active"],
        "suggested_actions": ["promote_top_probe_candidates_to_full"],
    }
'''


def run_detector_stage(
    run_dir: Path,
    task_spec: dict,
    probe_trial_records: list[dict[str, Any]],
    best_trial: dict[str, Any] | None,
    iteration: int,
) -> dict[str, Any]:
    module = load_run_dir_stage_module(run_dir, "detector.py", iteration)
    if not hasattr(module, "analyze_probe_results"):
        report = {"iteration": iteration, "error": "detector.py missing analyze_probe_results"}
    else:
        report = module.analyze_probe_results(task_spec, probe_trial_records, best_trial, iteration) or {}
    report.setdefault("iteration", iteration)
    write_stage_report(run_dir, "detector", iteration, report)
    return report


def run_failure_inspector_stage(
    run_dir: Path,
    task_spec: dict,
    trial_records: list[dict[str, Any]],
    probe_trial_records: list[dict[str, Any]],
    full_trial_records: list[dict[str, Any]],
    best_trial: dict[str, Any] | None,
    detector_report: dict[str, Any] | None,
    iteration: int,
) -> dict[str, Any]:
    module = load_run_dir_stage_module(run_dir, "failure_inspector.py", iteration)
    if not hasattr(module, "inspect_failures"):
        report = {"iteration": iteration, "error": "failure_inspector.py missing inspect_failures"}
    else:
        report = (
            module.inspect_failures(
                task_spec,
                trial_records,
                probe_trial_records,
                full_trial_records,
                best_trial,
                detector_report,
                iteration,
            )
            or {}
        )
    report.setdefault("iteration", iteration)
    write_stage_report(run_dir, "inspect", iteration, report)
    return report


def run_simplifier_stage(
    run_dir: Path,
    task_spec: dict,
    best_candidate: ProtocolCandidate,
    best_trial: dict[str, Any],
    iteration: int,
) -> list[dict[str, Any]]:
    module = load_run_dir_stage_module(run_dir, "simplifier.py", iteration)
    if not hasattr(module, "propose_simplifications"):
        report = {"iteration": iteration, "proposals": [], "error": "simplifier.py missing propose_simplifications"}
        write_stage_report(run_dir, "simplify", iteration, report)
        return []
    proposals = module.propose_simplifications(task_spec, best_candidate, best_trial, iteration) or []
    report = {
        "iteration": iteration,
        "n_proposals": len(proposals),
        "candidate_ids": [proposal.get("candidate", {}).get("candidate_id") for proposal in proposals if isinstance(proposal, dict)],
    }
    write_stage_report(run_dir, "simplify", iteration, report)
    return proposals


def materialize_candidate_source(
    run_dir: Path,
    candidate: ProtocolCandidate,
    *,
    trial_index: int,
    stage: str,
    rationale: str,
    parent_candidate_id: str | None,
) -> Path:
    candidates_dir = ensure_run_dir(run_dir / "candidates")
    candidate_path = candidates_dir / f"{trial_index:04d}_{candidate.candidate_id}_{stage}.py"
    write_protocol_module(
        candidate_path,
        candidate,
        rationale=rationale,
        metadata={
            "stage": stage,
            "parent_candidate_id": parent_candidate_id,
        },
    )
    return candidate_path


def distinct_full_strategy_target(task_spec: dict) -> int:
    for rule in task_spec.get("stop_rules", []):
        if isinstance(rule, dict) and rule.get("rule") == "min_distinct_strategies_attempted":
            if rule.get("mode_scope", "full") == "full":
                return int(rule.get("count", 0))
    return 0


def filter_generated_proposals(
    proposals: list[dict[str, Any]],
    trial_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    seen_signatures = collect_distinct_strategy_signatures(
        trial_records,
        mode_scope="all",
        require_constraint_ok=False,
    )
    batch_signatures: set[str] = set()
    filtered: list[dict[str, Any]] = []
    skipped_candidate_ids: list[str] = []
    for proposal in proposals:
        candidate_payload = proposal.get("candidate")
        if not isinstance(candidate_payload, dict):
            continue
        signature = strategy_signature(candidate_payload)
        if signature in seen_signatures or signature in batch_signatures:
            skipped_candidate_ids.append(str(candidate_payload.get("candidate_id", "unknown")))
            continue
        batch_signatures.add(signature)
        filtered.append(proposal)
    return filtered, skipped_candidate_ids


def choose_top_probe_candidates(task_spec: dict, probe_records: list[dict], top_k: int) -> list[dict]:
    ranked = sorted(
        probe_records,
        key=lambda row: (
            float(row.get("metric", 0.0)),
            tuple(row.get("ranking", {}).get("preference_tie_break_key", [])),
            -float(row.get("cumulative_sim_cost", 0.0)),
            -int(row.get("trial_index", 0)),
        ),
        reverse=True,
    )
    filtered: list[dict] = []
    for row in ranked:
        if not row.get("constraint_ok", True):
            continue
        filtered.append(row)
        if len(filtered) >= top_k:
            break
    return filtered


def choose_probe_records_for_full_eval(
    heuristic_module,
    task_spec: dict,
    trial_records: list[dict[str, Any]],
    probe_records: list[dict],
    iteration: int,
    detector_report: dict[str, Any] | None,
) -> list[dict]:
    existing_full_signatures = collect_distinct_strategy_signatures(
        trial_records,
        mode_scope="full",
        require_constraint_ok=True,
    )
    required_full_distinct = distinct_full_strategy_target(task_spec)
    need_more_full_diversity = len(existing_full_signatures) < required_full_distinct

    def _filter_unseen_full(rows: list[dict]) -> list[dict]:
        unseen: list[dict] = []
        for row in rows:
            candidate_payload = row.get("candidate")
            if not isinstance(candidate_payload, dict):
                continue
            if strategy_signature(candidate_payload) in existing_full_signatures:
                continue
            unseen.append(row)
        return unseen

    if hasattr(heuristic_module, "choose_full_eval_candidates"):
        chosen_ids = heuristic_module.choose_full_eval_candidates(task_spec, probe_records, iteration, detector_report) or []
        if not isinstance(chosen_ids, list):
            raise ValueError("choose_full_eval_candidates must return a list of candidate ids")
        chosen_id_set = {str(candidate_id) for candidate_id in chosen_ids}
        chosen_rows = [row for row in probe_records if row.get("candidate_id") in chosen_id_set]
        if need_more_full_diversity:
            chosen_rows = _filter_unseen_full(chosen_rows)
        return chosen_rows
    full_eval_count = heuristic_module.choose_full_eval_count(task_spec, probe_records, iteration)
    if need_more_full_diversity:
        remaining = max(required_full_distinct - len(existing_full_signatures), 0)
        full_eval_count = max(int(full_eval_count), min(max(remaining, 1), len(probe_records), 4))
        probe_records = _filter_unseen_full(probe_records)
    return choose_top_probe_candidates(task_spec, probe_records, top_k=full_eval_count)


def force_install_best_trial(
    *,
    run_dir: Path,
    candidate: ProtocolCandidate,
    trial_record: dict[str, Any],
    payload: dict[str, Any],
    loop_state: SearchLoopState,
    trial_records: list[dict[str, Any]],
    rationale: str,
) -> None:
    previous_best = loop_state.best_trial_record
    if previous_best is not None and previous_best is not trial_record:
        previous_best["is_best_so_far"] = False
    trial_record["is_best_so_far"] = True
    trial_record["forced_best_by_simplification"] = True
    loop_state.best_trial_record = trial_record
    loop_state.running_best = max(loop_state.running_best, float(trial_record.get("metric", 0.0)))
    write_protocol_module(run_dir / "protocol.py", candidate, rationale=rationale, metadata={"mode": trial_record.get("mode")})
    (run_dir / "best_protocol.json").write_text(json.dumps(candidate.to_dict(), indent=2))
    write_diagnostics(run_dir, payload.get("artifact_bundle"))
    rewrite_trial_logs(run_dir, trial_records)


def record_evaluation(
    *,
    run_dir: Path,
    task_spec: dict,
    evaluator: QOCEvaluator,
    budget: BudgetTracker,
    loop_state: SearchLoopState,
    trial_records: list[dict],
    candidate: ProtocolCandidate,
    mode: str,
    rationale: str,
    proposal_stage: str,
    parent_candidate_id: str | None,
    source_file: Path,
) -> tuple[dict, dict]:
    sim_cost = rydberg_sim_cost_fn(candidate, mode=mode)
    if not budget.can_afford(sim_cost):
        raise RuntimeError(f"budget exhausted before evaluating {candidate.candidate_id} in {mode} mode")

    term_check = check_hamiltonian_structure(task_spec, candidate)
    result, payload = evaluator.evaluate_with_payload(
        task_spec=task_spec,
        candidate=candidate,
        mode=mode,
        trial_index=len(trial_records),
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
        extra_fields={
            "proposal_stage": proposal_stage,
            "proposal_rationale": rationale,
            "source_file": str(source_file.relative_to(run_dir)),
            "parent_candidate_id": parent_candidate_id,
        },
    )

    if mode == "full":
        ts_check = check_time_series_constraints(task_spec, payload.get("artifact_bundle"))
        trial_record["time_series_check_ok"] = ts_check.ok
        if not ts_check.ok:
            trial_record["constraint_ok"] = False
            trial_record["metric"] = 0.0
            errmsg = trial_record.get("error_msg")
            ts_error = "; ".join(ts_check.errors)
            trial_record["error_msg"] = ts_error if not errmsg else f"{errmsg}; {ts_error}"
        write_full_run_context(run_dir, candidate, trial_record, payload)

    is_best = False
    if trial_record.get("constraint_ok", True):
        is_best = update_best_trial(task_spec, loop_state, trial_record)
    trial_record["stop_decision"] = evaluate_stop_rules(
        task_spec,
        budget,
        None if loop_state.best_trial_record is None else float(loop_state.best_trial_record["metric"]),
        trial_records=trial_records + [trial_record],
    )

    record_trial(run_dir, trial_record)
    trial_records.append(trial_record)
    write_summary(run_dir, trial_records)
    write_efficiency_plot(run_dir, trial_records)
    write_failure_notes(run_dir, trial_records)

    if is_best and mode == "full":
        write_protocol_module(run_dir / "protocol.py", candidate, rationale=rationale, metadata={"mode": mode})
        (run_dir / "best_protocol.json").write_text(json.dumps(candidate.to_dict(), indent=2))
        write_diagnostics(run_dir, payload.get("artifact_bundle"))

    return trial_record, payload


def load_or_initialize_seed(run_dir: Path, task_spec: dict) -> ProtocolCandidate:
    protocol_path = run_dir / "protocol.py"
    if protocol_path.exists():
        return load_protocol_candidate(protocol_path)
    seed = default_seed_candidate(task_spec)
    write_protocol_module(protocol_path, seed, rationale="initial seed protocol")
    return seed


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    task_spec = load_task_spec(Path(args.task_spec))
    task_spec["_user_total_time"] = args.user_total_time

    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ensure_run_dir(repo_root / "artifacts" / f"{timestamp}_{task_spec['task_id']}_search")
    else:
        run_dir = ensure_run_dir(Path(args.run_dir))

    seed_candidate = load_or_initialize_seed(run_dir, task_spec)
    seed_candidate = apply_user_total_time_override(seed_candidate, args.user_total_time)
    write_protocol_module(run_dir / "protocol.py", seed_candidate, rationale="initial seed protocol with user override applied if requested")
    ensure_default_run_dir_system(run_dir, seed_candidate, task_spec)

    evaluator = QOCEvaluator(
        config=EvaluatorConfig(
            metric_name=task_spec["objective_metric"],
            sim_cost_fn=rydberg_sim_cost_fn,
        ),
        eval_fn=evaluate_rydberg_mis,
    )
    budget = BudgetTracker(max_sim_cost=int(task_spec["sim_budget"]))
    loop_state = SearchLoopState()
    trial_records: list[dict] = []
    last_detector_report: dict[str, Any] | None = None
    last_failure_report: dict[str, Any] | None = None
    last_edit_plan: dict[str, Any] | None = None
    protected_snapshot = snapshot_protected_files(repo_root, task_spec.get("protected_paths", []))

    seed_source = materialize_candidate_source(
        run_dir,
        seed_candidate,
        trial_index=0,
        stage="seed_full",
        rationale="Initial seed protocol loaded from run_dir or default task seed.",
        parent_candidate_id=None,
    )
    record_evaluation(
        run_dir=run_dir,
        task_spec=task_spec,
        evaluator=evaluator,
        budget=budget,
        loop_state=loop_state,
        trial_records=trial_records,
        candidate=seed_candidate,
        mode="full",
        rationale="Initial seed protocol loaded from run_dir or default task seed.",
        proposal_stage="seed_full",
        parent_candidate_id=None,
        source_file=seed_source,
    )
    last_detector_report = {
        "iteration": 0,
        "n_probe_trials": 0,
        "detected_patterns": ["seed_only_no_probe_phase_yet"],
        "suggested_actions": ["generate_probe_candidates_next_iteration"],
    }
    write_stage_report(
        run_dir,
        "detector",
        0,
        last_detector_report,
    )
    write_stage_report(
        run_dir,
        "simplify",
        0,
        {
            "iteration": 0,
            "n_proposals": 0,
            "candidate_ids": [],
            "reason": "no_new_best_beyond_seed_yet",
        },
    )
    last_failure_report = run_failure_inspector_stage(
        run_dir,
        task_spec,
        trial_records,
        [],
        [trial_records[-1]],
        loop_state.best_trial_record,
        last_detector_report,
        0,
    )
    last_edit_plan = build_edit_plan(
        run_dir,
        task_spec,
        last_detector_report,
        last_failure_report,
        loop_state.best_trial_record,
        0,
    )
    update_heuristic_search_context(
        run_dir / "heuristic_search.py",
        {
            "iteration": 0,
            "best_trial": compact_trial_context(loop_state.best_trial_record),
            "recent_trials": [compact_trial_context(row) for row in trial_records[-5:]],
            "continue_actions": recommend_continue_actions(
                task_spec,
                None if loop_state.best_trial_record is None else float(loop_state.best_trial_record["metric"]),
            ),
            "detector_report": last_detector_report,
            "failure_report": last_failure_report,
            "edit_plan": last_edit_plan,
        },
    )

    final_stop_decision = evaluate_stop_rules(
        task_spec,
        budget,
        None if loop_state.best_trial_record is None else float(loop_state.best_trial_record["metric"]),
        trial_records=trial_records,
    )
    last_completed_iteration = 0

    for iteration in range(1, args.max_rounds + 1):
        if final_stop_decision["should_stop"]:
            break

        update_heuristic_search_context(
            run_dir / "heuristic_search.py",
            {
                "iteration": iteration,
                "best_trial": compact_trial_context(loop_state.best_trial_record),
                "recent_trials": [compact_trial_context(row) for row in trial_records[-8:]],
                "continue_actions": recommend_continue_actions(
                    task_spec,
                    None if loop_state.best_trial_record is None else float(loop_state.best_trial_record["metric"]),
                ),
                "detector_report": last_detector_report,
                "failure_report": last_failure_report,
                "edit_plan": last_edit_plan,
            },
        )
        apply_fallback_edit_plan(
            run_dir=run_dir,
            seed_candidate=seed_candidate,
            task_spec=task_spec,
            edit_plan=last_edit_plan,
            failure_report=last_failure_report,
            detector_report=last_detector_report,
            user_total_time=args.user_total_time,
            iteration=iteration,
        )
        heuristic_module = load_heuristic_search_module(run_dir / "heuristic_search.py", iteration)
        current_best_payload = None
        current_best_id = None
        if loop_state.best_trial_record is not None:
            current_best_payload = loop_state.best_trial_record["candidate"]
            current_best_id = loop_state.best_trial_record["candidate_id"]
        continue_actions = recommend_continue_actions(
            task_spec,
            None if loop_state.best_trial_record is None else float(loop_state.best_trial_record["metric"]),
        )
        proposals = heuristic_module.generate_proposals(
            task_spec,
            current_best_payload,
            trial_records,
            iteration,
            continue_actions,
        )
        proposals, skipped_duplicate_proposals = filter_generated_proposals(proposals, trial_records)
        if not proposals:
            if skipped_duplicate_proposals:
                print(
                    "all_generated_proposals_were_duplicate_signatures="
                    + ",".join(skipped_duplicate_proposals[:10])
                )
            break

        probe_records: list[dict] = []
        full_records: list[dict] = []
        simplifier_triggered = False
        for proposal in proposals:
            candidate = ProtocolCandidate.from_dict(proposal["candidate"], task_id=task_spec["task_id"])
            source_file = materialize_candidate_source(
                run_dir,
                candidate,
                trial_index=len(trial_records),
                stage="probe",
                rationale=proposal.get("rationale", ""),
                parent_candidate_id=current_best_id,
            )
            probe_record, _ = record_evaluation(
                run_dir=run_dir,
                task_spec=task_spec,
                evaluator=evaluator,
                budget=budget,
                loop_state=loop_state,
                trial_records=trial_records,
                candidate=candidate,
                mode="probe",
                rationale=proposal.get("rationale", ""),
                proposal_stage="probe",
                parent_candidate_id=current_best_id,
                source_file=source_file,
            )
            probe_records.append(probe_record)
            if budget.cumulative_sim_cost >= budget.max_sim_cost:
                break

        last_detector_report = run_detector_stage(
            run_dir,
            task_spec,
            probe_records,
            loop_state.best_trial_record,
            iteration,
        )
        top_probe_records = choose_probe_records_for_full_eval(
            heuristic_module,
            task_spec,
            trial_records,
            probe_records,
            iteration,
            last_detector_report,
        )

        for probe_record in top_probe_records:
            candidate = ProtocolCandidate.from_dict(probe_record["candidate"], task_id=task_spec["task_id"])
            existing_full_signatures = collect_distinct_strategy_signatures(
                trial_records,
                mode_scope="full",
                require_constraint_ok=False,
            )
            if strategy_signature(candidate.to_dict()) in existing_full_signatures:
                continue
            source_file = materialize_candidate_source(
                run_dir,
                candidate,
                trial_index=len(trial_records),
                stage="full",
                rationale=probe_record.get("proposal_rationale", ""),
                parent_candidate_id=probe_record.get("parent_candidate_id"),
            )
            record_evaluation(
                run_dir=run_dir,
                task_spec=task_spec,
                evaluator=evaluator,
                budget=budget,
                loop_state=loop_state,
                trial_records=trial_records,
                candidate=candidate,
                mode="full",
                rationale=probe_record.get("proposal_rationale", ""),
                proposal_stage="full",
                parent_candidate_id=probe_record.get("parent_candidate_id"),
                source_file=source_file,
            )
            parent_full_trial = trial_records[-1]
            full_records.append(parent_full_trial)
            if parent_full_trial.get("is_best_so_far") and parent_full_trial.get("proposal_stage") != "simplify":
                simplifier_triggered = True
                simplifier_parent_id = parent_full_trial.get("candidate_id")
                simplifier_parent_metric = float(parent_full_trial.get("metric", 0.0))
                simplification_proposals = run_simplifier_stage(
                    run_dir,
                    task_spec,
                    candidate,
                    parent_full_trial,
                    iteration,
                )
                simplification_results: list[dict[str, Any]] = []
                for simplification in simplification_proposals:
                    simple_candidate = ProtocolCandidate.from_dict(
                        simplification["candidate"],
                        task_id=task_spec["task_id"],
                    )
                    existing_full_signatures = collect_distinct_strategy_signatures(
                        trial_records,
                        mode_scope="full",
                        require_constraint_ok=False,
                    )
                    if strategy_signature(simple_candidate.to_dict()) in existing_full_signatures:
                        simplification_results.append(
                            {
                                "candidate_id": simple_candidate.candidate_id,
                                "skipped": True,
                                "reason": "duplicate_full_strategy_signature",
                            }
                        )
                        continue
                    simple_source = materialize_candidate_source(
                        run_dir,
                        simple_candidate,
                        trial_index=len(trial_records),
                        stage="simplify",
                        rationale=simplification.get("rationale", ""),
                        parent_candidate_id=simplifier_parent_id,
                    )
                    simple_trial, simple_payload = record_evaluation(
                        run_dir=run_dir,
                        task_spec=task_spec,
                        evaluator=evaluator,
                        budget=budget,
                        loop_state=loop_state,
                        trial_records=trial_records,
                        candidate=simple_candidate,
                        mode="full",
                        rationale=simplification.get("rationale", ""),
                        proposal_stage="simplify",
                        parent_candidate_id=simplifier_parent_id,
                        source_file=simple_source,
                    )
                    simple_trial["simplification_parent_metric"] = simplifier_parent_metric
                    preserved_metric = bool(
                        simple_trial.get("constraint_ok", True)
                        and float(simple_trial.get("metric", 0.0)) + simplification_metric_tolerance(task_spec) >= simplifier_parent_metric
                    )
                    simple_trial["simplification_preserved_metric"] = preserved_metric
                    if preserved_metric:
                        simple_trial["simplification_adopted_as_best"] = True
                        force_install_best_trial(
                            run_dir=run_dir,
                            candidate=simple_candidate,
                            trial_record=simple_trial,
                            payload=simple_payload,
                            loop_state=loop_state,
                            trial_records=trial_records,
                            rationale=simplification.get("rationale", ""),
                        )
                    else:
                        simple_trial["simplification_adopted_as_best"] = False
                        rewrite_trial_logs(run_dir, trial_records)
                    simplification_results.append(
                        {
                            "candidate_id": simple_trial.get("candidate_id"),
                            "metric": simple_trial.get("metric"),
                            "parent_metric": simplifier_parent_metric,
                            "preserved_metric": preserved_metric,
                            "adopted_as_best": simple_trial.get("simplification_adopted_as_best", False),
                        }
                    )
                    full_records.append(simple_trial)
                    if budget.cumulative_sim_cost >= budget.max_sim_cost:
                        break
                write_stage_report(
                    run_dir,
                    "simplify",
                    iteration,
                    {
                        "iteration": iteration,
                        "parent_candidate_id": simplifier_parent_id,
                        "parent_metric": simplifier_parent_metric,
                        "n_proposals": len(simplification_proposals),
                        "results": simplification_results,
                    },
                )
            if budget.cumulative_sim_cost >= budget.max_sim_cost:
                break
        if not simplifier_triggered:
            write_stage_report(
                run_dir,
                "simplify",
                iteration,
                {
                    "iteration": iteration,
                    "n_proposals": 0,
                    "candidate_ids": [],
                    "reason": "no_new_best_full_trial_triggered_simplification",
                },
            )

        last_failure_report = run_failure_inspector_stage(
            run_dir,
            task_spec,
            trial_records,
            probe_records,
            full_records,
            loop_state.best_trial_record,
            last_detector_report,
            iteration,
        )
        last_edit_plan = build_edit_plan(
            run_dir,
            task_spec,
            last_detector_report,
            last_failure_report,
            loop_state.best_trial_record,
            iteration,
        )
        final_stop_decision = evaluate_stop_rules(
            task_spec,
            budget,
            None if loop_state.best_trial_record is None else float(loop_state.best_trial_record["metric"]),
            trial_records=trial_records,
        )
        write_search_memory(run_dir / "search_memory.md", trial_records, loop_state.best_trial_record)
        write_search_state(
            run_dir / "search_state.json",
            {
                "iteration": iteration,
                "n_trials": len(trial_records),
                "cumulative_sim_cost": budget.cumulative_sim_cost,
                "best_trial": loop_state.best_trial_record,
                "stop_decision": final_stop_decision,
                "edit_plan": last_edit_plan,
            },
        )
        last_completed_iteration = iteration

    write_readme(run_dir, task_spec, loop_state, trial_records, budget, final_stop_decision)
    write_search_memory(run_dir / "search_memory.md", trial_records, loop_state.best_trial_record)
    write_search_state(
        run_dir / "search_state.json",
        {
            "iteration": last_completed_iteration,
            "n_trials": len(trial_records),
            "cumulative_sim_cost": budget.cumulative_sim_cost,
            "best_trial": loop_state.best_trial_record,
            "stop_decision": final_stop_decision,
            "edit_plan": last_edit_plan,
        },
    )

    artifact_check = check_required_artifacts(run_dir, task_spec, mode="full")
    if not artifact_check.ok:
        raise RuntimeError(f"missing required artifacts: {artifact_check.missing}")

    source_guard = verify_protected_files_unchanged(repo_root, protected_snapshot)
    if not source_guard.ok:
        raise RuntimeError(f"protected source files changed during run: {source_guard.changed}")

    print(f"task_id={task_spec['task_id']}")
    print(f"run_dir={run_dir}")
    print(f"n_trials={len(trial_records)}")
    print(f"cumulative_sim_cost={budget.cumulative_sim_cost}")
    print(f"best_trial={None if loop_state.best_trial_record is None else loop_state.best_trial_record['candidate_id']}")
    print(f"best_metric={None if loop_state.best_trial_record is None else loop_state.best_trial_record['metric']}")
    print(f"stop_decision={final_stop_decision}")
    print(f"artifact_check={artifact_check.ok}")
    print(f"source_guard={source_guard.ok}")


if __name__ == "__main__":
    main()
