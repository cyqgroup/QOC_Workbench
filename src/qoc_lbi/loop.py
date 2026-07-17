from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .artifacts import append_jsonl, write_summary_csv
from .ranking import compare_trial_records, preference_metrics_from_payload, ranking_snapshot


@dataclass
class SearchLoopState:
    trial_index: int = 0
    running_best: float = float("-inf")
    best_trial_record: dict[str, Any] | None = None


def build_trial_record(
    task_spec: dict,
    candidate,
    eval_result,
    payload: dict | None = None,
    active_terms: list[str] | None = None,
    continue_actions: list[str] | None = None,
    stop_decision: dict | None = None,
    term_check_ok: bool | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = payload or {}
    artifact_bundle = payload.get("artifact_bundle") or {}
    record = eval_result.to_dict()
    record["family"] = candidate.family
    record["hardware"] = candidate.hardware
    record["total_time"] = float(candidate.total_time)
    record["channel_bases"] = {channel.name: channel.basis for channel in candidate.channels}
    record["cd_kind"] = candidate.cd.kind
    record["cd_order"] = candidate.cd.order
    record["candidate"] = candidate.to_dict()
    record["hamiltonian_formula"] = payload.get("hamiltonian_formula") or artifact_bundle.get("hamiltonian_formula")
    record["active_hamiltonian_terms"] = list(active_terms or [])
    record["term_check_ok"] = term_check_ok
    record["continue_actions"] = list(continue_actions or [])
    record["stop_decision"] = stop_decision or {}
    record["artifact_bundle_summary"] = summarize_artifact_bundle(artifact_bundle)
    record["search_preferences"] = task_spec.get("search_preferences", {})
    record["preference_metrics"] = preference_metrics_from_payload(candidate, payload)
    record["ranking"] = ranking_snapshot(task_spec, record)
    record["is_best_so_far"] = False
    if extra_fields:
        record.update(extra_fields)
    return record


def record_trial(run_dir: str, trial_record: dict[str, Any]) -> None:
    append_jsonl(f"{run_dir}/trials.jsonl", trial_record)


def write_summary(run_dir: str, trial_records: list[dict[str, Any]]) -> None:
    write_summary_csv(f"{run_dir}/summary.csv", [build_summary_row(row) for row in trial_records])


def update_best_trial(task_spec: dict, state: SearchLoopState, trial_record: dict[str, Any]) -> bool:
    is_better = compare_trial_records(task_spec, trial_record, state.best_trial_record) > 0
    trial_record["is_best_so_far"] = is_better
    state.running_best = max(state.running_best, float(trial_record.get("metric", float("-inf"))))
    if is_better:
        state.best_trial_record = trial_record
    return is_better


def build_summary_row(trial_record: dict[str, Any]) -> dict[str, Any]:
    preference_metrics = trial_record.get("preference_metrics", {})
    return {
        "trial_index": trial_record.get("trial_index"),
        "candidate_id": trial_record.get("candidate_id"),
        "task_id": trial_record.get("task_id"),
        "mode": trial_record.get("mode"),
        "metric": trial_record.get("metric"),
        "sim_cost": trial_record.get("sim_cost"),
        "cumulative_sim_cost": trial_record.get("cumulative_sim_cost"),
        "runtime_sec": trial_record.get("runtime_sec"),
        "constraint_ok": trial_record.get("constraint_ok"),
        "term_check_ok": trial_record.get("term_check_ok"),
        "family": trial_record.get("family"),
        "hardware": trial_record.get("hardware"),
        "total_time": trial_record.get("total_time"),
        "channels": _format_channel_bases(trial_record.get("channel_bases", {})),
        "cd_kind": trial_record.get("cd_kind"),
        "cd_order": trial_record.get("cd_order"),
        "proposal_stage": trial_record.get("proposal_stage"),
        "proposal_rationale": trial_record.get("proposal_rationale"),
        "source_file": trial_record.get("source_file"),
        "parent_candidate_id": trial_record.get("parent_candidate_id"),
        "smoothness_score": preference_metrics.get("smoothness_score"),
        "translation_symmetry_score": preference_metrics.get("translation_symmetry_score"),
        "simplicity_score": preference_metrics.get("simplicity_score"),
        "active_hamiltonian_terms": " | ".join(trial_record.get("active_hamiltonian_terms", [])),
        "continue_actions": " | ".join(trial_record.get("continue_actions", [])),
        "is_best_so_far": trial_record.get("is_best_so_far", False),
        "hamiltonian_formula": trial_record.get("hamiltonian_formula"),
        "candidate_json": json.dumps(trial_record.get("candidate", {}), ensure_ascii=False),
        "error_msg": trial_record.get("error_msg"),
    }


def summarize_artifact_bundle(artifact_bundle: dict | None) -> dict[str, Any]:
    if not artifact_bundle:
        return {}
    summary: dict[str, Any] = {}
    for name in ("times", "target_fidelity", "instantaneous_ground_overlap", "omega_values", "delta_values", "cd_values"):
        values = artifact_bundle.get(name)
        if values is None:
            continue
        values = [float(v) for v in values]
        if not values:
            continue
        summary[name] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "start": values[0],
            "end": values[-1],
        }
    if "snapshot_times" in artifact_bundle:
        summary["snapshot_times"] = [float(v) for v in artifact_bundle["snapshot_times"]]
    if "hamiltonian_formula" in artifact_bundle:
        summary["hamiltonian_formula_present"] = True
    return summary


def _format_channel_bases(channel_bases: dict[str, str]) -> str:
    parts = [f"{name}:{basis}" for name, basis in sorted(channel_bases.items())]
    return ";".join(parts)
