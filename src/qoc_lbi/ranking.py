from __future__ import annotations

from typing import Any


def metric_tie_tolerance(task_spec: dict) -> float:
    prefs = task_spec.get("search_preferences", {})
    return float(prefs.get("metric_tie_tolerance", 1e-6))


def preference_metrics_from_payload(candidate, payload: dict | None) -> dict[str, float]:
    payload = payload or {}
    provided = payload.get("preference_metrics")
    if isinstance(provided, dict):
        return {
            "smoothness_score": float(provided.get("smoothness_score", 0.0)),
            "translation_symmetry_score": float(provided.get("translation_symmetry_score", 0.0)),
            "simplicity_score": float(provided.get("simplicity_score", _simplicity_score(candidate))),
        }

    artifact_bundle = payload.get("artifact_bundle") or {}
    smoothness = _smoothness_from_artifact_bundle(artifact_bundle)
    if smoothness is None:
        smoothness = 0.0
    return {
        "smoothness_score": float(smoothness),
        "translation_symmetry_score": float(_translation_symmetry_score(candidate, payload)),
        "simplicity_score": float(_simplicity_score(candidate)),
    }


def compare_trial_records(task_spec: dict, left: dict | None, right: dict | None) -> int:
    if left is None and right is None:
        return 0
    if left is None:
        return -1
    if right is None:
        return 1

    tol = metric_tie_tolerance(task_spec)
    left_metric = float(left.get("metric", 0.0))
    right_metric = float(right.get("metric", 0.0))
    if left_metric > right_metric + tol:
        return 1
    if right_metric > left_metric + tol:
        return -1

    left_key = preference_tie_break_key(task_spec, left)
    right_key = preference_tie_break_key(task_spec, right)
    if left_key > right_key:
        return 1
    if right_key > left_key:
        return -1

    left_cost = float(left.get("cumulative_sim_cost", float("inf")))
    right_cost = float(right.get("cumulative_sim_cost", float("inf")))
    if left_cost < right_cost:
        return 1
    if right_cost < left_cost:
        return -1

    left_index = int(left.get("trial_index", 0))
    right_index = int(right.get("trial_index", 0))
    if left_index < right_index:
        return 1
    if right_index < left_index:
        return -1
    return 0


def preference_tie_break_key(task_spec: dict, trial_record: dict) -> tuple[float, ...]:
    prefs = task_spec.get("search_preferences", {})
    metrics = trial_record.get("preference_metrics", {})

    key: list[float] = []
    order = prefs.get(
        "ranking_tie_break_order",
        ["translation_symmetry_score", "smoothness_score", "simplicity_score"],
    )
    for name in order:
        if name == "translation_symmetry_score" and not prefs.get(
            "prefer_translation_symmetric_pauli_controls",
            False,
        ):
            continue
        if name == "smoothness_score" and not prefs.get("prefer_smooth_hamiltonian_path", False):
            continue
        key.append(float(metrics.get(name, 0.0)))
    if "simplicity_score" not in order:
        key.append(float(metrics.get("simplicity_score", 0.0)))
    return tuple(key)


def ranking_snapshot(task_spec: dict, trial_record: dict) -> dict[str, Any]:
    return {
        "metric_tie_tolerance": metric_tie_tolerance(task_spec),
        "preference_tie_break_key": list(preference_tie_break_key(task_spec, trial_record)),
    }


def _smoothness_from_artifact_bundle(artifact_bundle: dict) -> float | None:
    traces = []
    for name in ("omega_values", "delta_values", "cd_values"):
        values = artifact_bundle.get(name)
        if values is None:
            continue
        values = [float(v) for v in values]
        if len(values) < 2:
            continue
        span = max(values) - min(values)
        scale = span * span if span > 1e-12 else 1.0
        roughness = sum((values[idx + 1] - values[idx]) ** 2 for idx in range(len(values) - 1))
        roughness /= max(len(values) - 1, 1)
        traces.append(roughness / scale)
    if not traces:
        return None
    mean_roughness = sum(traces) / len(traces)
    return 1.0 / (1.0 + mean_roughness)


def _translation_symmetry_score(candidate, payload: dict) -> float:
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict) and "translation_symmetry_score" in diagnostics:
        return float(diagnostics["translation_symmetry_score"])

    for channel in candidate.channels:
        params = channel.params or {}
        if any(key in params for key in ("site", "sites", "per_site", "site_values")):
            return 0.0
    return 1.0


def _simplicity_score(candidate) -> float:
    param_count = 0
    for channel in candidate.channels:
        param_count += len(channel.params or {})
    if candidate.cd.params:
        param_count += len(candidate.cd.params)

    complexity = float(len(candidate.channels))
    if candidate.cd.kind != "none":
        complexity += 1.0
    complexity += 0.1 * param_count
    return 1.0 / (1.0 + complexity)
