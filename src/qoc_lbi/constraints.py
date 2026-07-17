from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class ConstraintCheck:
    ok: bool
    errors: List[str] = field(default_factory=list)


def check_protocol_constraints(task_spec: dict, candidate) -> ConstraintCheck:
    errors: List[str] = []

    if candidate.total_time <= 0:
        errors.append("total_time must be positive")

    spec_hardware = task_spec.get("hardware")
    if spec_hardware and candidate.hardware != spec_hardware:
        errors.append(f"candidate hardware mismatch: {candidate.hardware} != {spec_hardware}")

    allowed_families = set(task_spec.get("allowed_protocol_families", []))
    if allowed_families and candidate.family not in allowed_families:
        errors.append(f"family not allowed: {candidate.family}")

    allowed_channels = set(task_spec.get("allowed_channels", []))
    allowed_parameterizations = task_spec.get("allowed_parameterizations", {})
    candidate_specs = task_spec.get("candidate_specs", {})
    channel_specs = candidate_specs.get("channel_specs", {})
    for channel in candidate.channels:
        if allowed_channels and channel.name not in allowed_channels:
            errors.append(f"channel not allowed: {channel.name}")
        channel_spec = channel_specs.get(channel.name, {})
        allowed_bases = channel_spec.get("allowed_bases", []) or allowed_parameterizations.get(channel.name, [])
        if allowed_bases and channel.basis not in allowed_bases:
            errors.append(f"{channel.name} basis not allowed: {channel.basis}")
        param_schema = (channel_spec.get("param_schemas", {}) or {}).get(channel.basis, {})
        _check_param_schema(f"{channel.name}.{channel.basis}", channel.params or {}, param_schema, errors)
        _check_parameterized_channel(channel.name, channel.basis, channel.params, channel_spec, errors)

    _check_cd_constraints(task_spec, candidate, errors)
    _check_physical_params(task_spec.get("physical_params", {}), task_spec.get("physical_param_constraints", {}), errors)

    return ConstraintCheck(ok=not errors, errors=errors)


def _check_channel_params(channel_name: str, params: dict, rules: dict, errors: List[str]) -> None:
    if not params or not rules:
        return

    value_range = rules.get("value_range")
    nonnegative = bool(rules.get("nonnegative", False))

    for key, value in params.items():
        if not isinstance(value, (int, float)):
            continue
        if nonnegative and value < 0:
            errors.append(f"{channel_name}.{key} must be nonnegative")
        if value_range is not None:
            lo, hi = value_range
            if value < lo or value > hi:
                errors.append(f"{channel_name}.{key} out of range [{lo}, {hi}]")


def _check_physical_params(physical_params: dict, rules: dict, errors: List[str]) -> None:
    for name, value in physical_params.items():
        if not isinstance(value, (int, float)):
            continue
        spec = rules.get(name, {})
        if spec.get("nonnegative", False) and value < 0:
            errors.append(f"physical param {name} must be nonnegative")
        if "range" in spec:
            lo, hi = spec["range"]
            if value < lo or value > hi:
                errors.append(f"physical param {name} out of range [{lo}, {hi}]")


def _check_cd_constraints(task_spec: dict, candidate, errors: List[str]) -> None:
    candidate_specs = task_spec.get("candidate_specs", {})
    cd_specs = candidate_specs.get("cd_specs", {})
    allowed_cd_kinds = cd_specs.get("allowed_kinds", []) or task_spec.get("allowed_cd_kinds", [])
    allowed_cd_orders = cd_specs.get("allowed_orders", []) or task_spec.get("allowed_cd_orders", [])

    if allowed_cd_kinds and candidate.cd.kind not in allowed_cd_kinds:
        errors.append(f"cd kind not allowed: {candidate.cd.kind}")

    if allowed_cd_orders and candidate.cd.kind != "none":
        order = candidate.cd.order or "unspecified"
        if order not in allowed_cd_orders:
            errors.append(f"cd order not allowed: {order}")

    param_schema = (cd_specs.get("param_schemas", {}) or {}).get(candidate.cd.kind, {})
    _check_param_schema(f"cd.{candidate.cd.kind}", candidate.cd.params or {}, param_schema, errors)

    if candidate.cd.kind == "acqc_j0_scaled":
        alpha = (candidate.cd.params or {}).get("alpha")
        if alpha is not None and not isinstance(alpha, (int, float)):
            errors.append("cd alpha must be numeric")

    if candidate.cd.kind == "y_sum_parameterized":
        params = candidate.cd.params or {}
        basis = params.get("basis", "piecewise_linear")
        if basis not in {"piecewise_linear", "smooth_beta", "fourier_enveloped"}:
            errors.append(f"unsupported y_sum_parameterized basis: {basis}")
        _check_shape_params("cd", basis, params, errors)


def _check_parameterized_channel(
    channel_name: str,
    basis_name: str,
    params: dict,
    channel_spec: dict,
    errors: List[str],
) -> None:
    _ = channel_spec
    if basis_name in {"smooth_beta", "fourier_enveloped", "piecewise_linear"}:
        _check_shape_params(channel_name, basis_name, params or {}, errors)


def _check_shape_params(prefix: str, basis_name: str, params: dict, errors: List[str]) -> None:
    if basis_name == "smooth_beta":
        if prefix == "delta":
            names = ("a", "b")
        else:
            names = ("p", "q")
        for key in names:
            value = params.get(key)
            if value is None:
                errors.append(f"{prefix}.{basis_name}.{key} is required")
                continue
            if not isinstance(value, (int, float)) or float(value) <= 0:
                errors.append(f"{prefix}.{basis_name}.{key} must be positive")
        return

    if basis_name == "fourier_enveloped":
        order = params.get("order")
        if order is None:
            errors.append(f"{prefix}.fourier_enveloped.order is required")
        elif not isinstance(order, int) or order <= 0:
            errors.append(f"{prefix}.fourier_enveloped.order must be a positive int")
        for key in ("a_cos", "b_sin"):
            values = params.get(key)
            if values is None:
                errors.append(f"{prefix}.fourier_enveloped.{key} is required")
                continue
            if not isinstance(values, list) or not all(isinstance(v, (int, float)) for v in values):
                errors.append(f"{prefix}.fourier_enveloped.{key} must be a numeric list")
        return

    if basis_name == "piecewise_linear":
        knots = params.get("knots")
        values = params.get("values")
        if knots is None:
            errors.append(f"{prefix}.piecewise_linear.knots is required")
        if values is None:
            errors.append(f"{prefix}.piecewise_linear.values is required")
        if knots is None or values is None:
            return
        if not isinstance(knots, list) or not isinstance(values, list):
            errors.append(f"{prefix}.piecewise_linear knots/values must be lists")
            return
        if len(knots) != len(values):
            errors.append(f"{prefix}.piecewise_linear knots/values length mismatch")
            return
        if len(knots) < 2:
            errors.append(f"{prefix}.piecewise_linear requires at least two knots")
            return
        if any(not isinstance(v, (int, float)) for v in knots + values):
            errors.append(f"{prefix}.piecewise_linear knots/values must be numeric")
            return
        if any(float(knots[i]) >= float(knots[i + 1]) for i in range(len(knots) - 1)):
            errors.append(f"{prefix}.piecewise_linear knots must be strictly increasing")


def _check_param_schema(prefix: str, params: dict[str, Any], schema: dict[str, Any], errors: List[str]) -> None:
    if not schema:
        if params:
            errors.append(f"{prefix} does not accept params: {sorted(params.keys())}")
        return
    allow_extra_keys = bool(schema.get("__allow_extra_keys__", False))
    schema_fields = {str(key): rules for key, rules in schema.items() if not str(key).startswith("__")}
    if not allow_extra_keys:
        extra_keys = sorted(str(key) for key in params.keys() if str(key) not in schema_fields)
        if extra_keys:
            errors.append(f"{prefix} got unsupported params: {extra_keys}")
    for key, rules in schema_fields.items():
        value = params.get(key)
        if rules.get("required", False) and value is None:
            errors.append(f"{prefix}.{key} is required")
            continue
        if value is None:
            continue
        _check_schema_value(prefix, key, value, rules, params, errors)


def _check_schema_value(
    prefix: str,
    key: str,
    value: Any,
    rules: dict[str, Any],
    params: dict[str, Any],
    errors: List[str],
) -> None:
    typ = rules.get("type")
    label = f"{prefix}.{key}"
    if typ == "float":
        if not isinstance(value, (int, float)):
            errors.append(f"{label} must be numeric")
            return
        _check_numeric_range(label, float(value), rules, errors)
        return
    if typ == "int":
        if not isinstance(value, int):
            errors.append(f"{label} must be an int")
            return
        _check_numeric_range(label, float(value), rules, errors)
        return
    if typ == "enum":
        allowed = set(rules.get("choices", []))
        if allowed and value not in allowed:
            errors.append(f"{label} must be one of {sorted(allowed)}")
        return
    if typ == "float_list":
        if not isinstance(value, list) or not all(isinstance(v, (int, float)) for v in value):
            errors.append(f"{label} must be a numeric list")
            return
        _check_list_shape(label, value, rules, params, errors)
        for item in value:
            _check_numeric_range(label, float(item), rules, errors)
        return


def _check_list_shape(label: str, values: list[Any], rules: dict[str, Any], params: dict[str, Any], errors: List[str]) -> None:
    min_length = rules.get("min_length")
    max_length = rules.get("max_length")
    same_length_as = rules.get("same_length_as")
    if min_length is not None and len(values) < int(min_length):
        errors.append(f"{label} length must be >= {min_length}")
    if max_length is not None and len(values) > int(max_length):
        errors.append(f"{label} length must be <= {max_length}")
    if same_length_as is not None:
        peer = params.get(str(same_length_as))
        if isinstance(peer, list) and len(values) != len(peer):
            errors.append(f"{label} length must match {same_length_as}")
    if rules.get("strictly_increasing", False):
        if any(float(values[i]) >= float(values[i + 1]) for i in range(len(values) - 1)):
            errors.append(f"{label} must be strictly increasing")
    if "fixed_first" in rules and values:
        if abs(float(values[0]) - float(rules["fixed_first"])) > 1e-9:
            errors.append(f"{label} first element must equal {rules['fixed_first']}")
    if "fixed_last" in rules and values:
        if abs(float(values[-1]) - float(rules["fixed_last"])) > 1e-9:
            errors.append(f"{label} last element must equal {rules['fixed_last']}")


def _check_numeric_range(label: str, value: float, rules: dict[str, Any], errors: List[str]) -> None:
    if "range" in rules:
        lo, hi = rules["range"]
        if value < float(lo) or value > float(hi):
            errors.append(f"{label} out of range [{lo}, {hi}]")
