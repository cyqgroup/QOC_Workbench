from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class TimeSeriesCheck:
    ok: bool
    errors: List[str] = field(default_factory=list)


def check_time_series_constraints(task_spec: dict, artifact_bundle: dict | None) -> TimeSeriesCheck:
    if not artifact_bundle:
        return TimeSeriesCheck(ok=True)

    rules = task_spec.get("time_series_constraints", {})
    series_map: Dict[str, np.ndarray] = {
        "omega": np.array(artifact_bundle.get("omega_values", [])),
        "delta": np.array(artifact_bundle.get("delta_values", [])),
        "cd": np.array(artifact_bundle.get("cd_values", [])),
        "schedule": np.array(artifact_bundle.get("schedule_values", [])),
        "cd_strength": np.array(artifact_bundle.get("cd_values", [])),
    }
    errors: List[str] = []
    for name, series in series_map.items():
        if series.size == 0 or name not in rules:
            continue
        spec = rules[name]
        if not np.all(np.isfinite(series)):
            errors.append(f"{name}(t) contains non-finite values")
            continue
        if spec.get("nonnegative_over_time", False) and np.min(series) < -1e-10:
            errors.append(f"{name}(t) becomes negative")
        if "value_range_over_time" in spec:
            lo, hi = spec["value_range_over_time"]
            if np.min(series) < lo - 1e-10 or np.max(series) > hi + 1e-10:
                errors.append(f"{name}(t) violates range [{lo}, {hi}]")
        if "start_value_range" in spec:
            lo, hi = spec["start_value_range"]
            if series[0] < lo - 1e-10 or series[0] > hi + 1e-10:
                errors.append(f"{name}(0) violates range [{lo}, {hi}]")
        if "end_value_range" in spec:
            lo, hi = spec["end_value_range"]
            if series[-1] < lo - 1e-10 or series[-1] > hi + 1e-10:
                errors.append(f"{name}(T) violates range [{lo}, {hi}]")
        if "max_abs_step" in spec and series.size >= 2:
            max_abs_step = float(spec["max_abs_step"])
            if np.max(np.abs(np.diff(series))) > max_abs_step + 1e-10:
                errors.append(f"{name}(t) has adjacent-step jump larger than {max_abs_step}")
        if "max_abs_second_step" in spec and series.size >= 3:
            max_abs_second_step = float(spec["max_abs_second_step"])
            second = np.diff(series, n=2)
            if np.max(np.abs(second)) > max_abs_second_step + 1e-10:
                errors.append(f"{name}(t) has second-difference larger than {max_abs_second_step}")
    return TimeSeriesCheck(ok=not errors, errors=errors)
