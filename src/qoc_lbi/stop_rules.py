from __future__ import annotations

import json
from typing import Any


def evaluate_stop_rules(
    task_spec: dict,
    budget_tracker,
    best_metric: float | None,
    trial_records: list[dict[str, Any]] | None = None,
) -> dict:
    """
    Evaluate configured stop rules against the current search state.

    Returns a dictionary that can be recorded in logs or surfaced to the LLM.
    """
    trial_records = trial_records or []
    decisions = {
        "should_stop": False,
        "reasons": [],
        "blockers": [],
        "context": {},
    }

    if budget_tracker.cumulative_sim_cost >= budget_tracker.max_sim_cost:
        decisions["should_stop"] = True
        decisions["reasons"].append("budget_exhausted")

    for rule in task_spec.get("stop_rules", []):
        rule_name = rule["rule"] if isinstance(rule, dict) else rule

        if rule_name == "budget_exhausted":
            continue

        if rule_name == "min_distinct_strategies_attempted":
            mode_scope = rule.get("mode_scope", "full")
            require_constraint_ok = bool(rule.get("require_constraint_ok", True))
            required_count = int(rule["count"])
            signatures = collect_distinct_strategy_signatures(
                trial_records,
                mode_scope=mode_scope,
                require_constraint_ok=require_constraint_ok,
            )
            attempted_count = len(signatures)
            decisions["context"]["distinct_strategies"] = {
                "count": attempted_count,
                "required_count": required_count,
                "mode_scope": mode_scope,
                "require_constraint_ok": require_constraint_ok,
            }
            if attempted_count < required_count:
                decisions["blockers"].append(
                    f"min_distinct_strategies_attempted:{attempted_count}/{required_count}:{mode_scope}"
                )
            continue

        if rule_name == "target_metric_reached":
            threshold = float(rule["threshold"])
            decisions["context"]["target_metric"] = {
                "best_metric": best_metric,
                "threshold": threshold,
            }
            if best_metric is not None and best_metric >= threshold:
                if decisions["blockers"]:
                    decisions["reasons"].append(f"target_metric_reached_but_blocked:{threshold}")
                else:
                    decisions["should_stop"] = True
                    decisions["reasons"].append(f"target_metric_reached:{threshold}")

    return decisions


def recommend_continue_actions(task_spec: dict, metric: float | None) -> list[str]:
    actions: list[str] = []
    if metric is None:
        return actions
    for rule in task_spec.get("continue_rules", []):
        threshold = rule.get("if_metric_below")
        if threshold is not None and metric < float(threshold):
            actions.extend(rule.get("actions", []))
    return actions


def collect_distinct_strategy_signatures(
    trial_records: list[dict[str, Any]],
    mode_scope: str = "full",
    require_constraint_ok: bool = True,
) -> set[str]:
    signatures: set[str] = set()
    for row in trial_records:
        if mode_scope != "all" and row.get("mode") != mode_scope:
            continue
        if require_constraint_ok and not row.get("constraint_ok", True):
            continue
        candidate = row.get("candidate")
        if not isinstance(candidate, dict):
            continue
        signatures.add(strategy_signature(candidate))
    return signatures


def strategy_signature(candidate_payload: dict[str, Any]) -> str:
    canonical = {
        "family": candidate_payload.get("family"),
        "hardware": candidate_payload.get("hardware"),
        "total_time": _normalize_value(candidate_payload.get("total_time")),
        "channels": sorted(
            [
                {
                    "name": channel.get("name"),
                    "basis": channel.get("basis"),
                    "params": _normalize_value(channel.get("params", {})),
                }
                for channel in candidate_payload.get("channels", [])
                if isinstance(channel, dict)
            ],
            key=lambda item: str(item["name"]),
        ),
        "cd": _normalize_value(candidate_payload.get("cd", {})),
        "constraints": _normalize_value(candidate_payload.get("constraints", {})),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    return value
