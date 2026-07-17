from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .protocol import ProtocolCandidate


def load_python_module(path: str | Path, module_name: str) -> ModuleType:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    module_dir = str(module_path.parent.resolve())
    added_to_sys_path = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added_to_sys_path = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added_to_sys_path and sys.path and sys.path[0] == module_dir:
            sys.path.pop(0)
    return module


def write_protocol_module(
    path: str | Path,
    candidate: ProtocolCandidate,
    *,
    rationale: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = candidate.to_dict()
    text = (
        "from __future__ import annotations\n\n"
        "from qoc_lbi.protocol import ProtocolCandidate\n\n"
        f"PROTOCOL_PAYLOAD = {repr(payload)}\n"
        f"SEARCH_RATIONALE = {rationale!r}\n"
        f"RUNTIME_METADATA = {repr(metadata or {})}\n\n"
        "def build_candidate() -> ProtocolCandidate:\n"
        "    return ProtocolCandidate.from_dict(PROTOCOL_PAYLOAD)\n"
    )
    Path(path).write_text(text)


def load_protocol_candidate(path: str | Path) -> ProtocolCandidate:
    module = load_python_module(path, f"aqc_run_protocol_{Path(path).stem}")
    if hasattr(module, "build_candidate"):
        candidate = module.build_candidate()
        if isinstance(candidate, ProtocolCandidate):
            return candidate
    payload = getattr(module, "PROTOCOL_PAYLOAD", None)
    if isinstance(payload, dict):
        return ProtocolCandidate.from_dict(payload)
    current = getattr(module, "CURRENT_PROTOCOL", None) or getattr(module, "PROTOCOL", None)
    if isinstance(current, ProtocolCandidate):
        return current
    if isinstance(current, dict):
        return ProtocolCandidate.from_dict(current)
    raise ValueError(f"no protocol candidate found in {path}")


def ensure_default_heuristic_search(path: str | Path, seed_candidate: ProtocolCandidate, task_spec: dict) -> None:
    out = Path(path)
    if out.exists():
        return
    out.write_text(_default_heuristic_template(seed_candidate, task_spec))


def load_heuristic_search_module(path: str | Path, iteration: int) -> ModuleType:
    return load_python_module(path, f"aqc_run_heuristic_search_{iteration}")


def update_heuristic_search_context(path: str | Path, context: dict[str, Any]) -> None:
    target = Path(path)
    text = target.read_text()
    begin = "# @@SEARCH_CONTEXT_BEGIN"
    end = "# @@SEARCH_CONTEXT_END"
    replacement = f'{begin}\nSEARCH_CONTEXT = {repr(context)}\n{end}'
    if begin in text and end in text:
        start = text.index(begin)
        finish = text.index(end) + len(end)
        text = text[:start] + replacement + text[finish:]
        target.write_text(text)


def write_search_memory(path: str | Path, trial_records: list[dict[str, Any]], best_trial: dict[str, Any] | None) -> None:
    lines = ["# Search Memory", ""]
    if best_trial is None:
        lines.append("- no successful trial yet")
    else:
        lines.extend(
            [
                f"- best_candidate_id: {best_trial.get('candidate_id')}",
                f"- best_metric: {best_trial.get('metric')}",
                f"- best_mode: {best_trial.get('mode')}",
                f"- best_formula: {best_trial.get('hamiltonian_formula')}",
                "",
            ]
        )
    lines.append("## Recent Trials")
    lines.append("")
    for trial in trial_records[-12:]:
        lines.append(
            f"- trial={trial.get('trial_index')} mode={trial.get('mode')} "
            f"candidate={trial.get('candidate_id')} metric={trial.get('metric'):.6f} "
            f"stage={trial.get('proposal_stage')} rationale={trial.get('proposal_rationale', '')}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def write_search_state(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2))


def ensure_default_run_dir_system(run_dir: str | Path, seed_candidate: ProtocolCandidate, task_spec: dict) -> None:
    root = Path(run_dir)
    ensure_default_heuristic_search(root / "heuristic_search.py", seed_candidate, task_spec)
    _write_if_missing(root / "detector.py", _default_detector_template())
    _write_if_missing(root / "failure_inspector.py", _default_failure_inspector_template())
    _write_if_missing(root / "simplifier.py", _default_simplifier_template())


def _default_heuristic_template(seed_candidate: ProtocolCandidate, task_spec: dict) -> str:
    template_kind = str(task_spec.get("search_template") or task_spec.get("hardware") or "generic")
    if template_kind == "rydberg":
        return _default_rydberg_heuristic_template(seed_candidate)
    return _default_generic_heuristic_template(seed_candidate, template_kind)


def _default_generic_heuristic_template(seed_candidate: ProtocolCandidate, template_kind: str = "generic") -> str:
    seed_payload = repr(seed_candidate.to_dict())
    return f'''from __future__ import annotations

from copy import deepcopy

from qoc_lbi.protocol import ProtocolCandidate

SEED_PROTOCOL = {seed_payload}
TEMPLATE_KIND = {template_kind!r}

SEARCH_NOTES = [
    "Editable generic run_dir heuristic system for a newly formalized QOC task.",
    "This template does not assume Rydberg omega/delta controls or global-Y CD terms.",
    "It only proposes conservative mutations declared by task_spec allowed channels, bases, CD kinds, and baseline_candidate.",
    "The constraints checker and evaluator backend decide whether generated candidates are admissible and useful.",
]

# @@SEARCH_CONTEXT_BEGIN
SEARCH_CONTEXT = {{}}
# @@SEARCH_CONTEXT_END


def seed_protocol(task_spec):
    payload = task_spec.get("baseline_candidate") or SEED_PROTOCOL
    return ProtocolCandidate.from_dict(payload, task_id=task_spec.get("task_id"))


def generate_proposals(task_spec, current_best, trials, iteration, continue_actions):
    baseline = _candidate_payload(task_spec, current_best)
    proposals = []
    seen = set()

    def add(payload, rationale, stage="generic_probe"):
        key = _proposal_key(payload)
        if key in seen:
            return
        seen.add(key)
        proposals.append({{"candidate": payload, "rationale": rationale, "stage": stage}})

    for scale in _time_scales(task_spec, iteration, continue_actions):
        payload = deepcopy(baseline)
        payload["candidate_id"] = _candidate_id(baseline, f"T{{scale:.2f}}", iteration)
        payload["total_time"] = max(1e-9, float(baseline.get("total_time", 1.0)) * scale)
        payload["provenance"] = _with_provenance(payload, "generic_total_time_mutation", iteration)
        payload["notes"] = list(payload.get("notes", [])) + [f"generic template: total_time scaled by {{scale:.2f}}"]
        add(payload, "Probe a conservative total-time perturbation allowed by the task specification.")

    for channel_name, basis_name, params in _allowed_channel_basis_trials(task_spec):
        payload = deepcopy(baseline)
        payload["candidate_id"] = _candidate_id(baseline, f"{{channel_name}}_{{basis_name}}", iteration)
        payload["channels"] = _replace_or_append_channel(payload.get("channels", []), channel_name, basis_name, params)
        payload["provenance"] = _with_provenance(payload, "generic_channel_basis_mutation", iteration)
        payload["notes"] = list(payload.get("notes", [])) + [
            f"generic template: use allowed basis {{basis_name}} for channel {{channel_name}}"
        ]
        add(payload, f"Probe allowed channel {{channel_name}} with allowed basis {{basis_name}}.")

    for cd_payload in _allowed_cd_trials(task_spec, baseline):
        payload = deepcopy(baseline)
        cd_kind = cd_payload.get("kind", "none")
        payload["candidate_id"] = _candidate_id(baseline, f"cd_{{cd_kind}}", iteration)
        payload["cd"] = cd_payload
        payload["provenance"] = _with_provenance(payload, "generic_cd_mutation", iteration)
        payload["notes"] = list(payload.get("notes", [])) + [f"generic template: toggle allowed CD kind {{cd_kind}}"]
        add(payload, f"Probe allowed auxiliary/CD kind {{cd_kind}} without adding undeclared Hamiltonian terms.")

    return proposals[:12]


def choose_full_eval_count(task_spec, probe_trial_records, iteration):
    _ = (task_spec, iteration)
    successful = [row for row in probe_trial_records if not row.get("error_msg")]
    if not successful:
        return 0
    return min(2, len(successful))


def _candidate_payload(task_spec, current_best):
    if isinstance(current_best, ProtocolCandidate):
        return current_best.to_dict()
    if isinstance(current_best, dict):
        if isinstance(current_best.get("candidate"), dict):
            return deepcopy(current_best["candidate"])
        if "candidate_id" in current_best:
            return deepcopy(current_best)
    return deepcopy(task_spec.get("baseline_candidate") or SEED_PROTOCOL)


def _directive_defaults(task_spec):
    return task_spec.get("run_search_directive_defaults", {{}}) or {{}}


def _allowed_channels(task_spec):
    explicit = list(task_spec.get("allowed_channels", []) or [])
    if explicit:
        return explicit
    channel_specs = (task_spec.get("candidate_specs", {{}}) or {{}}).get("channel_specs", {{}}) or {{}}
    if channel_specs:
        return list(channel_specs.keys())
    allowed_parameterizations = task_spec.get("allowed_parameterizations") or _directive_defaults(task_spec).get("allowed_parameterizations", {{}})
    if isinstance(allowed_parameterizations, dict):
        return list(allowed_parameterizations.keys())
    baseline = task_spec.get("baseline_candidate") or SEED_PROTOCOL
    return [channel.get("name") for channel in baseline.get("channels", []) if channel.get("name")]


def _allowed_bases(task_spec, channel_name):
    channel_specs = (task_spec.get("candidate_specs", {{}}) or {{}}).get("channel_specs", {{}}) or {{}}
    channel_spec = channel_specs.get(channel_name, {{}}) or {{}}
    bases = list(channel_spec.get("allowed_bases", []) or [])
    if bases:
        return bases
    allowed_parameterizations = task_spec.get("allowed_parameterizations") or _directive_defaults(task_spec).get("allowed_parameterizations", {{}})
    if isinstance(allowed_parameterizations, dict):
        return list(allowed_parameterizations.get(channel_name, []) or [])
    return []


def _allowed_channel_basis_trials(task_spec):
    trials = []
    baseline_channels = {{channel.get("name"): channel for channel in (task_spec.get("baseline_candidate") or SEED_PROTOCOL).get("channels", [])}}
    for channel_name in _allowed_channels(task_spec):
        for basis_name in _allowed_bases(task_spec, channel_name):
            current_basis = (baseline_channels.get(channel_name) or {{}}).get("basis")
            if basis_name == current_basis:
                continue
            params = _default_params_for_basis(task_spec, channel_name, basis_name)
            trials.append((channel_name, basis_name, params))
    return trials


def _default_params_for_basis(task_spec, channel_name, basis_name):
    channel_specs = (task_spec.get("candidate_specs", {{}}) or {{}}).get("channel_specs", {{}}) or {{}}
    schema = (((channel_specs.get(channel_name, {{}}) or {{}}).get("param_schemas", {{}}) or {{}}).get(basis_name, {{}}) or {{}})
    params = {{}}
    for key, rules in schema.items():
        if str(key).startswith("__"):
            continue
        params[key] = _default_schema_value(key, rules, params)
    return params


def _default_schema_value(key, rules, params):
    typ = rules.get("type")
    value_range = rules.get("range")
    if typ == "float_list":
        length = int(rules.get("min_length") or 3)
        if rules.get("same_length_as") in params and isinstance(params[rules["same_length_as"]], list):
            length = len(params[rules["same_length_as"]])
        fixed_first = rules.get("fixed_first")
        fixed_last = rules.get("fixed_last")
        if rules.get("strictly_increasing"):
            lo, hi = value_range or [0.0, 1.0]
            values = [lo + (hi - lo) * idx / max(1, length - 1) for idx in range(length)]
        else:
            lo, hi = value_range or [-0.1, 0.1]
            mid = 0.5 * (lo + hi)
            values = [mid for _ in range(length)]
        if fixed_first is not None:
            values[0] = fixed_first
        if fixed_last is not None:
            values[-1] = fixed_last
        return values
    if typ == "int":
        lo, hi = value_range or [1, 1]
        return int(max(lo, min(hi, lo)))
    if typ == "enum":
        choices = rules.get("choices", []) or []
        return choices[0] if choices else None
    lo, hi = value_range or [0.0, 0.0]
    if rules.get("fixed_first") is not None and key in {{"value", "start"}}:
        return rules["fixed_first"]
    return float(0.5 * (lo + hi))


def _allowed_cd_trials(task_spec, baseline):
    candidate_specs = task_spec.get("candidate_specs", {{}}) or {{}}
    cd_specs = candidate_specs.get("cd_specs", {{}}) or {{}}
    kinds = list(cd_specs.get("allowed_kinds", []) or task_spec.get("allowed_cd_kinds", []) or _directive_defaults(task_spec).get("allowed_cd_kinds", []) or [])
    orders = list(cd_specs.get("allowed_orders", []) or task_spec.get("allowed_cd_orders", []) or _directive_defaults(task_spec).get("allowed_cd_orders", []) or ["unspecified"])
    current_kind = (baseline.get("cd") or {{}}).get("kind", "none")
    trials = []
    for kind in kinds:
        if kind == current_kind or kind == "none":
            continue
        order = orders[0] if orders else "unspecified"
        schema = ((cd_specs.get("param_schemas", {{}}) or {{}}).get(kind, {{}}) or {{}})
        params = {{key: _default_schema_value(key, rules, {{}}) for key, rules in schema.items() if not str(key).startswith("__")}}
        trials.append({{"kind": kind, "ansatz": None, "order": order, "params": params}})
    return trials


def _replace_or_append_channel(channels, channel_name, basis_name, params):
    updated = []
    replaced = False
    for channel in channels:
        if channel.get("name") == channel_name:
            updated.append({{"name": channel_name, "basis": basis_name, "params": params}})
            replaced = True
        else:
            updated.append(deepcopy(channel))
    if not replaced:
        updated.append({{"name": channel_name, "basis": basis_name, "params": params}})
    return updated


def _time_scales(task_spec, iteration, continue_actions):
    if not (task_spec.get("time_scale_design_space", {{}}) or {{}}).get("allow_total_time_search", True):
        return []
    actions = " ".join(str(action) for action in (continue_actions or []))
    if "longer" in actions or "time" in actions:
        return [1.10, 1.25]
    if iteration <= 1:
        return [0.90, 1.10]
    return [1.05, 1.15]


def _with_provenance(payload, source, iteration):
    provenance = dict(payload.get("provenance", {{}}) or {{}})
    provenance.update({{"source": source, "template_kind": TEMPLATE_KIND, "iteration": iteration}})
    return provenance


def _candidate_id(baseline, suffix, iteration):
    root = baseline.get("candidate_id", "candidate")
    safe_suffix = str(suffix).replace(".", "p").replace("-", "m")
    return f"{{root}}_generic_i{{iteration}}_{{safe_suffix}}"


def _proposal_key(payload):
    channels = tuple((channel.get("name"), channel.get("basis"), repr(channel.get("params", {{}}))) for channel in payload.get("channels", []))
    cd = payload.get("cd", {{}}) or {{}}
    return (round(float(payload.get("total_time", 0.0)), 12), channels, cd.get("kind"), repr(cd.get("params", {{}})))
'''


def _default_rydberg_heuristic_template(seed_candidate: ProtocolCandidate) -> str:
    seed_payload = repr(seed_candidate.to_dict())
    return f'''from __future__ import annotations

from qoc_lbi.protocol import ProtocolCandidate

SEED_PROTOCOL = {seed_payload}

SEARCH_NOTES = [
    "Editable run_dir heuristic system for Rydberg MIS.",
    "This file is the search object: the loop loads it every round to decide what to try next.",
    "This default template is intentionally neutral and does not bake in ACQC-specific proposals.",
    "Edit this file to add search logic. By default it proposes nothing beyond the seed protocol.",
]

# @@SEARCH_CONTEXT_BEGIN
SEARCH_CONTEXT = {{}}
# @@SEARCH_CONTEXT_END


def seed_protocol(task_spec):
    return ProtocolCandidate.from_dict(SEED_PROTOCOL)


def generate_proposals(task_spec, current_best, trials, iteration, continue_actions):
    _ = (task_spec, current_best, trials, iteration, continue_actions)
    return []


def choose_full_eval_count(task_spec, probe_trial_records, iteration):
    _ = (task_spec, probe_trial_records, iteration)
    return 0
'''


def _default_detector_template() -> str:
    return '''from __future__ import annotations

from statistics import mean

SEARCH_NOTES = [
    "Detector stage: summarize what probe evaluations revealed before full runs.",
    "An agent may update this file to extract better state signals from probe records.",
]


def analyze_probe_results(task_spec, probe_trial_records, best_trial, iteration):
    metrics = [float(row.get("metric", 0.0)) for row in probe_trial_records]
    current_best = 0.0 if best_trial is None else float(best_trial.get("metric", 0.0))
    report = {
        "iteration": iteration,
        "n_probe_trials": len(probe_trial_records),
        "current_best_metric_before_probe": current_best,
        "best_probe_metric": max(metrics) if metrics else None,
        "mean_probe_metric": mean(metrics) if metrics else None,
        "improved_over_current_best": any(metric > current_best for metric in metrics),
        "detected_patterns": [],
        "suggested_actions": [],
    }
    if not probe_trial_records:
        report["detected_patterns"].append("no_probe_trials_generated")
        report["suggested_actions"].append("edit_policy_to_propose_candidates")
        return report
    if report["improved_over_current_best"]:
        report["detected_patterns"].append("some_probe_candidates_improved")
        report["suggested_actions"].append("promote_top_probe_candidates_to_full")
    else:
        report["detected_patterns"].append("probe_plateau")
        report["suggested_actions"].append("broaden_policy_search")
    if (report["best_probe_metric"] or 0.0) < 0.99:
        report["suggested_actions"].append("if_allowed_consider_longer_T_or_new_family")
    return report
'''


def _default_failure_inspector_template() -> str:
    return '''from __future__ import annotations

import json
from pathlib import Path

SEARCH_NOTES = [
    "Failure inspector stage: translate logs and full-run diagnostics into hypotheses for the next edit.",
    "An agent may update this file to perform more specific failure analysis.",
    "This module is allowed to decide an ordered multi-file edit plan for the next iteration.",
    "It may assign distinct edit objectives to each target file and to priority functions inside each file.",
]


def inspect_failures(task_spec, trial_records, probe_trial_records, full_trial_records, best_trial, detector_report, iteration):
    recent = list(full_trial_records or probe_trial_records)
    full_ctx = _load_latest_full_run_context()
    report = {
        "iteration": iteration,
        "n_recent_probe_trials": len(probe_trial_records),
        "n_recent_full_trials": len(full_trial_records),
        "best_metric": None if best_trial is None else float(best_trial.get("metric", 0.0)),
        "failure_modes": [],
        "edit_hypotheses": [],
        "preferred_edit_targets": [],
        "dominant_failure_cause": "undetermined",
        "is_time_scale_likely_too_short": False,
        "is_schedule_shape_likely_bad": False,
        "recommended_protocol_actions": [],
        "edit_plan": None,
        "full_run_context": full_ctx,
    }
    if not recent:
        report["failure_modes"].append("no_recent_trials_to_inspect")
        report["edit_hypotheses"].append("write_or_expand_policy_logic")
        report["preferred_edit_targets"] = ["heuristic_search.py"]
        report["dominant_failure_cause"] = "no_actionable_search_step_executed"
        report["recommended_protocol_actions"] = ["generate_probe_candidates"]
        report["edit_plan"] = {
            "target_files": ["heuristic_search.py"],
            "rationale": "No recent trials were available, so the next step is to expand policy logic first.",
            "file_objectives": {
                "heuristic_search.py": "Generate at least one valid probe candidate and keep the policy logic auditable.",
            },
            "function_objectives": {
                "heuristic_search.py": {
                    "generate_proposals": "Generate at least one valid probe candidate consistent with the current task and hardware constraints.",
                    "choose_full_eval_count": "Return a conservative non-negative full-eval count consistent with the probe evidence.",
                }
            },
        }
        return report

    if any(row.get("error_msg") for row in recent):
        report["failure_modes"].append("runtime_or_constraint_failures_present")
        report["edit_hypotheses"].append("repair_invalid_candidate_generation")
        report["preferred_edit_targets"].append("heuristic_search.py")

    best_recent = max(float(row.get("metric", 0.0)) for row in recent)
    if best_recent < 0.99:
        report["failure_modes"].append("target_metric_not_reached")
        report["edit_hypotheses"].append("continue_search_and_change_design")
        if "heuristic_search.py" not in report["preferred_edit_targets"]:
            report["preferred_edit_targets"].append("heuristic_search.py")

    if detector_report and not detector_report.get("improved_over_current_best", False):
        report["failure_modes"].append("probe_stage_found_no_improvement")
        report["edit_hypotheses"].append("change_policy_more_aggressively")
        if "detector.py" not in report["preferred_edit_targets"]:
            report["preferred_edit_targets"].append("detector.py")

    _diagnose_from_full_run_context(task_spec, best_trial, full_ctx, report)

    if not report["failure_modes"]:
        report["failure_modes"].append("no_obvious_failure_mode")
        report["edit_hypotheses"].append("continue_with_small_edits")
        report["preferred_edit_targets"] = ["failure_inspector.py"]

    simplify_trials = [row for row in recent if row.get("proposal_stage") == "simplify"]
    if simplify_trials and not any(row.get("simplification_preserved_metric") for row in simplify_trials):
        if "simplifier.py" not in report["preferred_edit_targets"]:
            report["preferred_edit_targets"].append("simplifier.py")

    report["preferred_edit_targets"] = _dedupe(report["preferred_edit_targets"])
    file_objectives = {}
    function_objectives = {}
    for target in report["preferred_edit_targets"]:
        if target == "heuristic_search.py":
            actions = report["recommended_protocol_actions"]
            if "increase_total_time" in actions:
                file_objectives[target] = (
                    "Modify proposal-generation logic so the next probe batch explicitly explores longer total evolution times."
                )
                function_objectives[target] = {
                    "generate_proposals": (
                        "Generate valid proposals that first prioritize increasing total_time T while respecting any user-specified T override."
                    ),
                    "choose_full_eval_count": (
                        "Advance a small number of the most promising longer-T proposals to full evaluation."
                    ),
                }
            elif "change_schedule_shape" in actions:
                file_objectives[target] = (
                    "Modify proposal-generation logic so the next probe batch explores better schedule shapes or channel bases."
                )
                function_objectives[target] = {
                    "generate_proposals": (
                        "Generate proposals that change schedule parameterization or channel basis to repair likely path-shape issues."
                    ),
                    "choose_full_eval_count": (
                        "Promote only the strongest schedule-shape repairs to full evaluation."
                    ),
                }
            elif "consider_cd_or_sta" in actions:
                file_objectives[target] = (
                    "Modify proposal-generation logic so the next probe batch explores adding allowed auxiliary CD/STA structure."
                )
                function_objectives[target] = {
                    "generate_proposals": (
                        "Generate proposals that introduce allowed auxiliary CD/STA structure without violating hardware constraints."
                    ),
                    "choose_full_eval_count": (
                        "Promote a disciplined subset of CD/STA-enhanced proposals to full evaluation."
                    ),
                }
            else:
                file_objectives[target] = (
                    "Change proposal-generation logic so the next probe batch explores more informative protocol variations while staying inside hardware and task constraints."
                )
                function_objectives[target] = {
                    "generate_proposals": (
                        "Prioritize improving candidate generation diversity and validity using the latest detector and failure signals."
                    ),
                    "choose_full_eval_count": (
                        "Set the number of full evaluations so promising probe candidates are advanced without wasting budget."
                    ),
                }
        elif target == "detector.py":
            file_objectives[target] = (
                "Extract more useful probe-state summaries that help the loop decide which candidates deserve full evaluation."
            )
            function_objectives[target] = {
                "analyze_probe_results": (
                    "Produce stronger probe-state summaries and detector signals that help rank and filter candidates for full evaluation."
                )
            }
        elif target == "failure_inspector.py":
            file_objectives[target] = (
                "Refine failure analysis so future edit plans target the real bottleneck instead of using generic diagnoses."
            )
            function_objectives[target] = {
                "inspect_failures": (
                    "Improve full-run diagnosis, ordered edit-plan generation, and per-file/per-function objective assignment."
                )
            }
        elif target == "simplifier.py":
            file_objectives[target] = (
                "Propose simpler candidate variants whose full-regression metric is likely to preserve the current best performance."
            )
            function_objectives[target] = {
                "propose_simplifications": (
                    "Generate simpler candidates that are most likely to preserve the current best full-run metric under forced regression."
                )
            }
    report["edit_plan"] = {
        "target_files": list(report["preferred_edit_targets"]),
        "rationale": (
            "Failure inspection selected an ordered multi-file edit plan using full-run diagnostic evidence. "
            "Rewrite these files in sequence during the next iteration."
        ),
        "file_objectives": file_objectives,
        "function_objectives": function_objectives,
    }
    return report


def _load_latest_full_run_context():
    path = Path(__file__).resolve().with_name("latest_full_run_context.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _diagnose_from_full_run_context(task_spec, best_trial, full_ctx, report):
    if not isinstance(full_ctx, dict):
        report["dominant_failure_cause"] = "insufficient_full_run_diagnostics"
        report["recommended_protocol_actions"].append("generate_probe_candidates")
        return

    metric = float(full_ctx.get("metric", 0.0) or 0.0)
    target_end = float(full_ctx.get("target_fidelity_end", 0.0) or 0.0)
    target_gain_last = float(full_ctx.get("target_fidelity_gain_last_window", 0.0) or 0.0)
    inst_min = float(full_ctx.get("instantaneous_ground_overlap_min", 1.0) or 1.0)
    inst_end = float(full_ctx.get("instantaneous_ground_overlap_end", 1.0) or 1.0)
    smoothness = float(full_ctx.get("smoothness_score", 1.0) or 1.0)
    cd_kind = str(full_ctx.get("cd_kind", "none") or "none")

    report["is_time_scale_likely_too_short"] = bool(metric < 0.99 and target_gain_last > 0.05)
    report["is_schedule_shape_likely_bad"] = bool(
        metric < 0.99 and (
            smoothness < 0.995
            or (inst_min < 0.35 and target_gain_last <= 0.05)
            or (inst_end < 0.8 and target_end < 0.8 and target_gain_last <= 0.05)
        )
    )

    if report["is_time_scale_likely_too_short"]:
        report["dominant_failure_cause"] = "time_scale_too_short"
        report["recommended_protocol_actions"].append("increase_total_time")
    elif report["is_schedule_shape_likely_bad"]:
        report["dominant_failure_cause"] = "schedule_shape_likely_bad"
        report["recommended_protocol_actions"].append("change_schedule_shape")
    elif metric < 0.99 and cd_kind == "none":
        report["dominant_failure_cause"] = "missing_auxiliary_adiabatic_assistance"
        report["recommended_protocol_actions"].append("consider_cd_or_sta")
    elif metric < 0.99:
        report["dominant_failure_cause"] = "needs_broader_protocol_search"
        report["recommended_protocol_actions"].append("consider_protocol_family_switch")
    else:
        report["dominant_failure_cause"] = "metric_satisfactory"

    if metric < 0.99 and "generate_probe_candidates" not in report["recommended_protocol_actions"]:
        report["recommended_protocol_actions"].append("generate_probe_candidates")

    if report["is_time_scale_likely_too_short"] and "heuristic_search.py" not in report["preferred_edit_targets"]:
        report["preferred_edit_targets"].append("heuristic_search.py")
    if report["is_schedule_shape_likely_bad"] and "heuristic_search.py" not in report["preferred_edit_targets"]:
        report["preferred_edit_targets"].append("heuristic_search.py")
    if "consider_cd_or_sta" in report["recommended_protocol_actions"] and "heuristic_search.py" not in report["preferred_edit_targets"]:
        report["preferred_edit_targets"].append("heuristic_search.py")
'''


def _default_simplifier_template() -> str:
    return '''from __future__ import annotations

from qoc_lbi.protocol import ProtocolCandidate

SEARCH_NOTES = [
    "Simplifier stage: when a new best appears, try a simpler equivalent candidate and regress it.",
    "This default simplifier is intentionally conservative.",
]


def propose_simplifications(task_spec, best_candidate, best_trial, iteration):
    _ = (task_spec, best_trial, iteration)
    if isinstance(best_candidate, ProtocolCandidate):
        candidate = best_candidate
    else:
        candidate = ProtocolCandidate.from_dict(best_candidate)
    proposals = []
    if candidate.cd.kind != "none":
        payload = candidate.to_dict()
        payload["candidate_id"] = f"{candidate.candidate_id}_drop_cd"
        payload["cd"] = {"kind": "none", "ansatz": None, "order": None, "params": {}}
        payload["notes"] = list(payload.get("notes", [])) + ["simplifier: removed optional CD term"]
        proposals.append(
            {
                "candidate": payload,
                "rationale": "Simplify the best candidate by removing the optional auxiliary CD term.",
                "stage": "simplify",
            }
        )
    return proposals
'''


def _write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content)
