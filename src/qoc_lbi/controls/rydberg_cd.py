from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .rydberg_basis import (
    _beta_bump_deriv_tau,
    _beta_bump_value,
    _fourier_correction,
    _fourier_correction_deriv_tau,
    _piecewise_data,
    _piecewise_deriv_tau,
    _piecewise_value,
    _tau,
)
from .rydberg_registry import ScalarControl, register_cd_kind


def _j0_value(omega_control: ScalarControl, delta_control: ScalarControl, t: float) -> jnp.ndarray:
    om = omega_control.value(t)
    de = delta_control.value(t)
    dom = omega_control.deriv(t)
    dde = delta_control.deriv(t)
    numer = om * dde - de * dom
    denom = 2.0 * (om * om + de * de)
    return jnp.where(denom > 1e-12, -numer / denom, 0.0)


@register_cd_kind("none")
def build_cd_none(cd_config: Any, omega_control: ScalarControl, delta_control: ScalarControl, task_view: Any, candidate: Any) -> ScalarControl:
    _ = (cd_config, omega_control, delta_control, task_view, candidate)
    return ScalarControl(
        name="cd_strength",
        basis="none",
        formula="f_cd(t) = 0",
        params={},
        value_fn=lambda t: jnp.asarray(0.0),
        deriv_fn=lambda t: jnp.asarray(0.0),
    )


@register_cd_kind("acqc_j0")
def build_cd_acqc_j0(cd_config: Any, omega_control: ScalarControl, delta_control: ScalarControl, task_view: Any, candidate: Any) -> ScalarControl:
    _ = (cd_config, task_view, candidate)
    return ScalarControl(
        name="cd_strength",
        basis="acqc_j0",
        formula="f_cd(t) = -(Omega dDelta/dt - Delta dOmega/dt) / (2 (Omega^2 + Delta^2))",
        params=dict(getattr(cd_config, "params", {}) or {}),
        value_fn=lambda t: _j0_value(omega_control, delta_control, t),
        deriv_fn=None,
    )


@register_cd_kind("acqc_j0_scaled")
def build_cd_acqc_j0_scaled(cd_config: Any, omega_control: ScalarControl, delta_control: ScalarControl, task_view: Any, candidate: Any) -> ScalarControl:
    _ = (task_view, candidate)
    params = dict(getattr(cd_config, "params", {}) or {})
    alpha = float(params.get("alpha", 1.0))
    return ScalarControl(
        name="cd_strength",
        basis="acqc_j0_scaled",
        formula=f"f_cd(t) = {alpha} * f_j0(t)",
        params=params,
        value_fn=lambda t: alpha * _j0_value(omega_control, delta_control, t),
        deriv_fn=None,
    )


@register_cd_kind("y_sum_parameterized")
def build_cd_y_sum_parameterized(
    cd_config: Any,
    omega_control: ScalarControl,
    delta_control: ScalarControl,
    task_view: Any,
    candidate: Any,
) -> ScalarControl:
    _ = (omega_control, delta_control, task_view)
    params = dict(getattr(cd_config, "params", {}) or {})
    basis = str(params.get("basis", "piecewise_linear"))
    scale = float(params.get("scale", 1.0))
    t_max = float(candidate.total_time)

    if basis == "piecewise_linear":
        knots, values = _piecewise_data("cd_strength", params)
        return ScalarControl(
            name="cd_strength",
            basis="y_sum_parameterized",
            formula="f_cd(t) = scale * interp(t/T; knots, values)",
            params=params,
            value_fn=lambda t: scale * _piecewise_value(_tau(t, t_max), knots, values),
            deriv_fn=lambda t: scale * _piecewise_deriv_tau(_tau(t, t_max), knots, values) / t_max,
            metadata={"shape_basis": basis},
        )

    if basis == "smooth_beta":
        p = float(params.get("p", 2.0))
        q = float(params.get("q", 2.0))
        return ScalarControl(
            name="cd_strength",
            basis="y_sum_parameterized",
            formula=f"f_cd(t) = scale * normalized_beta_bump(t/T; p={p}, q={q})",
            params=params,
            value_fn=lambda t: scale * _beta_bump_value(_tau(t, t_max), p, q),
            deriv_fn=lambda t: scale * _beta_bump_deriv_tau(_tau(t, t_max), p, q) / t_max,
            metadata={"shape_basis": basis},
        )

    if basis == "fourier_enveloped":
        power = float(params.get("envelope_power", 1.0))

        def value_fn(t: float) -> jnp.ndarray:
            tau = _tau(t, t_max)
            envelope = jnp.sin(jnp.pi * tau) ** power
            return scale * envelope * _fourier_correction(tau, params)

        def deriv_fn(t: float) -> jnp.ndarray:
            tau = _tau(t, t_max)
            s = jnp.sin(jnp.pi * tau)
            c = jnp.cos(jnp.pi * tau)
            envelope = jnp.sin(jnp.pi * tau) ** power
            denv_tau = jnp.where(
                jnp.abs(s) < 1e-8,
                0.0,
                power * jnp.power(jnp.clip(s, 1e-8, 1.0), power - 1.0) * c * jnp.pi,
            )
            corr = _fourier_correction(tau, params)
            dcorr_tau = _fourier_correction_deriv_tau(tau, params)
            return scale * (denv_tau * corr + envelope * dcorr_tau) / t_max

        return ScalarControl(
            name="cd_strength",
            basis="y_sum_parameterized",
            formula="f_cd(t) = scale * sin(pi t / T)^p * Fourier(t/T)",
            params=params,
            value_fn=value_fn,
            deriv_fn=deriv_fn,
            metadata={"shape_basis": basis},
        )

    raise ValueError(f"unsupported y_sum_parameterized basis: {basis}")
