from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import jax.scipy as jsp

from .rydberg_registry import ScalarControl, register_channel_basis


_EPS = 1e-8


def _tau(t: float, t_max: float) -> jnp.ndarray:
    return jnp.clip(jnp.asarray(t) / float(t_max), 0.0, 1.0)


def _safe_pow(x: jnp.ndarray, power: float) -> jnp.ndarray:
    return jnp.power(jnp.clip(x, _EPS, 1.0), power)


def _beta_bump_peak(p: float, q: float) -> float:
    total = p + q
    tau_peak = p / total
    return float((tau_peak**p) * ((1.0 - tau_peak) ** q))


def _beta_bump_value(tau: jnp.ndarray, p: float, q: float) -> jnp.ndarray:
    peak = max(_beta_bump_peak(p, q), _EPS)
    return (_safe_pow(tau, p) * _safe_pow(1.0 - tau, q)) / peak


def _beta_bump_deriv_tau(tau: jnp.ndarray, p: float, q: float) -> jnp.ndarray:
    peak = max(_beta_bump_peak(p, q), _EPS)
    tau_c = jnp.clip(tau, _EPS, 1.0 - _EPS)
    raw = _safe_pow(tau_c, p) * _safe_pow(1.0 - tau_c, q)
    return (raw / peak) * (p / tau_c - q / (1.0 - tau_c))


def _beta_cdf_value(tau: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    return jsp.special.betainc(a, b, tau)


def _beta_pdf_value(tau: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    tau_c = jnp.clip(tau, _EPS, 1.0 - _EPS)
    log_pdf = (a - 1.0) * jnp.log(tau_c) + (b - 1.0) * jnp.log1p(-tau_c) - jsp.special.betaln(a, b)
    return jnp.exp(log_pdf)


def _harmonic_coeff(params: dict[str, Any], key: str, k: int) -> float:
    values = list(params.get(key, []))
    if k - 1 < len(values):
        return float(values[k - 1])
    return 0.0


def _fourier_order(params: dict[str, Any]) -> int:
    explicit = int(params.get("order", 0) or 0)
    n_cos = len(list(params.get("a_cos", [])))
    n_sin = len(list(params.get("b_sin", [])))
    return max(explicit, n_cos, n_sin, 1)


def _fourier_correction(tau: jnp.ndarray, params: dict[str, Any]) -> jnp.ndarray:
    order = _fourier_order(params)
    out = jnp.asarray(float(params.get("offset", 0.0)))
    for k in range(1, order + 1):
        angle = k * jnp.pi * tau
        out = out + _harmonic_coeff(params, "a_cos", k) * jnp.cos(angle)
        out = out + _harmonic_coeff(params, "b_sin", k) * jnp.sin(angle)
    return out


def _fourier_correction_deriv_tau(tau: jnp.ndarray, params: dict[str, Any]) -> jnp.ndarray:
    order = _fourier_order(params)
    out = jnp.asarray(0.0)
    for k in range(1, order + 1):
        angle = k * jnp.pi * tau
        factor = k * jnp.pi
        out = out - _harmonic_coeff(params, "a_cos", k) * factor * jnp.sin(angle)
        out = out + _harmonic_coeff(params, "b_sin", k) * factor * jnp.cos(angle)
    return out


def _sin_envelope(tau: jnp.ndarray, power: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    s = jnp.sin(jnp.pi * tau)
    c = jnp.cos(jnp.pi * tau)
    env = jnp.power(jnp.clip(s, 0.0, 1.0), power)
    deriv_tau = jnp.where(
        jnp.abs(s) < _EPS,
        0.0,
        power * jnp.power(jnp.clip(s, _EPS, 1.0), power - 1.0) * c * jnp.pi,
    )
    return env, deriv_tau


def _default_piecewise(channel_name: str) -> tuple[list[float], list[float]]:
    if channel_name == "omega":
        return [0.0, 0.5, 1.0], [0.0, 1.0, 0.0]
    return [0.0, 1.0], [-1.0, 1.0]


def _piecewise_data(channel_name: str, params: dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray]:
    default_knots, default_values = _default_piecewise(channel_name)
    knots = jnp.asarray(list(params.get("knots", default_knots)), dtype=jnp.float32)
    values = jnp.asarray(list(params.get("values", default_values)), dtype=jnp.float32)
    return knots, values


def _piecewise_value(tau: jnp.ndarray, knots: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    return jnp.interp(tau, knots, values)


def _piecewise_deriv_tau(tau: jnp.ndarray, knots: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    idx = jnp.clip(jnp.searchsorted(knots, tau, side="right") - 1, 0, len(knots) - 2)
    dx = knots[idx + 1] - knots[idx]
    dy = values[idx + 1] - values[idx]
    return jnp.where(jnp.abs(dx) < _EPS, 0.0, dy / dx)


def _omega_scale(task_view: Any) -> float:
    return float(task_view.omega0)


def _delta_scale(task_view: Any) -> float:
    return float(task_view.delta0)


@register_channel_basis("omega", "trapezoid")
@register_channel_basis("omega", "trapezoid_v1")
def build_omega_trapezoid(channel_name: str, channel: Any, task_view: Any, candidate: Any) -> ScalarControl:
    _ = (channel_name, channel)
    t_max = float(candidate.total_time)
    scale = _omega_scale(task_view)

    def value_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        return scale * jnp.clip(jnp.minimum(4.0 * tau, 4.0 * (1.0 - tau)), 0.0, 1.0)

    def deriv_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        deriv_tau = jnp.where(
            tau < 0.25,
            4.0,
            jnp.where(tau < 0.5, -4.0, jnp.where(tau < 0.75, -4.0, 4.0)),
        )
        active = jnp.where((tau <= 0.0) | (tau >= 1.0), 0.0, deriv_tau)
        return scale * active / t_max

    return ScalarControl(
        name="omega",
        basis=str(channel.basis),
        formula="Omega(t) = Omega0 * clip(min(4t/T, 4(1-t/T)), 0, 1)",
        params=dict(channel.params or {}),
        value_fn=value_fn,
        deriv_fn=deriv_fn,
    )


@register_channel_basis("delta", "linear")
@register_channel_basis("delta", "linear_v1")
def build_delta_linear(channel_name: str, channel: Any, task_view: Any, candidate: Any) -> ScalarControl:
    _ = (channel_name, channel)
    t_max = float(candidate.total_time)
    scale = _delta_scale(task_view)

    def value_fn(t: float) -> jnp.ndarray:
        return scale * (2.0 * _tau(t, t_max) - 1.0)

    def deriv_fn(t: float) -> jnp.ndarray:
        _ = t
        return jnp.asarray(2.0 * scale / t_max)

    return ScalarControl(
        name="delta",
        basis=str(channel.basis),
        formula="Delta(t) = Delta0 * (2t/T - 1)",
        params=dict(channel.params or {}),
        value_fn=value_fn,
        deriv_fn=deriv_fn,
    )


@register_channel_basis("omega", "smooth")
@register_channel_basis("omega", "smooth_sine_v1")
def build_omega_smooth(channel_name: str, channel: Any, task_view: Any, candidate: Any) -> ScalarControl:
    _ = (channel_name, channel)
    t_max = float(candidate.total_time)
    scale = _omega_scale(task_view)

    def value_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        return scale * jnp.sin(jnp.pi / 2.0 * jnp.sin(jnp.pi * tau)) ** 2

    def deriv_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        inner = jnp.pi * tau
        deriv_tau = jnp.sin(jnp.pi * jnp.sin(inner)) * (jnp.pi**2 / 2.0) * jnp.cos(inner)
        return scale * deriv_tau / t_max

    return ScalarControl(
        name="omega",
        basis=str(channel.basis),
        formula="Omega(t) = Omega0 * sin^2((pi/2) * sin(pi t / T))",
        params=dict(channel.params or {}),
        value_fn=value_fn,
        deriv_fn=deriv_fn,
    )


@register_channel_basis("delta", "smooth")
@register_channel_basis("delta", "smooth_cos_v1")
def build_delta_smooth(channel_name: str, channel: Any, task_view: Any, candidate: Any) -> ScalarControl:
    _ = (channel_name, channel)
    t_max = float(candidate.total_time)
    scale = _delta_scale(task_view)

    def value_fn(t: float) -> jnp.ndarray:
        return -scale * jnp.cos(jnp.pi * _tau(t, t_max))

    def deriv_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        return scale * jnp.sin(jnp.pi * tau) * (jnp.pi / t_max)

    return ScalarControl(
        name="delta",
        basis=str(channel.basis),
        formula="Delta(t) = -Delta0 * cos(pi t / T)",
        params=dict(channel.params or {}),
        value_fn=value_fn,
        deriv_fn=deriv_fn,
    )


@register_channel_basis("omega", "smooth_beta")
def build_omega_smooth_beta(channel_name: str, channel: Any, task_view: Any, candidate: Any) -> ScalarControl:
    _ = channel_name
    params = dict(channel.params or {})
    p = float(params.get("p", 2.0))
    q = float(params.get("q", 2.0))
    t_max = float(candidate.total_time)
    scale = _omega_scale(task_view)

    def value_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        return scale * _beta_bump_value(tau, p, q)

    def deriv_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        return scale * _beta_bump_deriv_tau(tau, p, q) / t_max

    return ScalarControl(
        name="omega",
        basis="smooth_beta",
        formula=f"Omega(t) = Omega0 * normalized_beta_bump(t/T; p={p}, q={q})",
        params=params,
        value_fn=value_fn,
        deriv_fn=deriv_fn,
    )


@register_channel_basis("delta", "smooth_beta")
def build_delta_smooth_beta(channel_name: str, channel: Any, task_view: Any, candidate: Any) -> ScalarControl:
    _ = channel_name
    params = dict(channel.params or {})
    a = float(params.get("a", 2.0))
    b = float(params.get("b", 2.0))
    t_max = float(candidate.total_time)
    scale = _delta_scale(task_view)

    def value_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        return scale * (2.0 * _beta_cdf_value(tau, a, b) - 1.0)

    def deriv_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        return scale * (2.0 * _beta_pdf_value(tau, a, b)) / t_max

    return ScalarControl(
        name="delta",
        basis="smooth_beta",
        formula=f"Delta(t) = Delta0 * (2 * I_(t/T)({a}, {b}) - 1)",
        params=params,
        value_fn=value_fn,
        deriv_fn=deriv_fn,
    )


@register_channel_basis("omega", "fourier_enveloped")
def build_omega_fourier(channel_name: str, channel: Any, task_view: Any, candidate: Any) -> ScalarControl:
    _ = channel_name
    params = dict(channel.params or {})
    t_max = float(candidate.total_time)
    scale = _omega_scale(task_view)
    power = float(params.get("envelope_power", 1.0))

    def value_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        env, _ = _sin_envelope(tau, power)
        raw = _fourier_correction(tau, params)
        return scale * env * raw * raw

    def deriv_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        env, denv_tau = _sin_envelope(tau, power)
        raw = _fourier_correction(tau, params)
        draw_tau = _fourier_correction_deriv_tau(tau, params)
        return scale * (denv_tau * raw * raw + env * 2.0 * raw * draw_tau) / t_max

    return ScalarControl(
        name="omega",
        basis="fourier_enveloped",
        formula="Omega(t) = Omega0 * sin(pi t / T)^p * (offset + Fourier(t/T))^2",
        params=params,
        value_fn=value_fn,
        deriv_fn=deriv_fn,
    )


@register_channel_basis("delta", "fourier_enveloped")
def build_delta_fourier(channel_name: str, channel: Any, task_view: Any, candidate: Any) -> ScalarControl:
    _ = channel_name
    params = dict(channel.params or {})
    t_max = float(candidate.total_time)
    scale = _delta_scale(task_view)
    power = float(params.get("envelope_power", 1.0))

    def value_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        env, _ = _sin_envelope(tau, power)
        corr = _fourier_correction(tau, params)
        return scale * ((2.0 * tau - 1.0) + env * corr)

    def deriv_fn(t: float) -> jnp.ndarray:
        tau = _tau(t, t_max)
        env, denv_tau = _sin_envelope(tau, power)
        corr = _fourier_correction(tau, params)
        dcorr_tau = _fourier_correction_deriv_tau(tau, params)
        return scale * (2.0 + denv_tau * corr + env * dcorr_tau) / t_max

    return ScalarControl(
        name="delta",
        basis="fourier_enveloped",
        formula="Delta(t) = Delta0 * ((2t/T - 1) + sin(pi t / T)^p * Fourier(t/T))",
        params=params,
        value_fn=value_fn,
        deriv_fn=deriv_fn,
    )


@register_channel_basis("omega", "piecewise_linear")
def build_omega_piecewise(channel_name: str, channel: Any, task_view: Any, candidate: Any) -> ScalarControl:
    params = dict(channel.params or {})
    t_max = float(candidate.total_time)
    scale = _omega_scale(task_view)
    knots, values = _piecewise_data(channel_name, params)

    def value_fn(t: float) -> jnp.ndarray:
        return scale * _piecewise_value(_tau(t, t_max), knots, values)

    def deriv_fn(t: float) -> jnp.ndarray:
        return scale * _piecewise_deriv_tau(_tau(t, t_max), knots, values) / t_max

    return ScalarControl(
        name="omega",
        basis="piecewise_linear",
        formula="Omega(t) = Omega0 * interp(t/T; knots, values)",
        params=params,
        value_fn=value_fn,
        deriv_fn=deriv_fn,
    )


@register_channel_basis("delta", "piecewise_linear")
def build_delta_piecewise(channel_name: str, channel: Any, task_view: Any, candidate: Any) -> ScalarControl:
    params = dict(channel.params or {})
    t_max = float(candidate.total_time)
    scale = _delta_scale(task_view)
    knots, values = _piecewise_data(channel_name, params)

    def value_fn(t: float) -> jnp.ndarray:
        return scale * _piecewise_value(_tau(t, t_max), knots, values)

    def deriv_fn(t: float) -> jnp.ndarray:
        return scale * _piecewise_deriv_tau(_tau(t, t_max), knots, values) / t_max

    return ScalarControl(
        name="delta",
        basis="piecewise_linear",
        formula="Delta(t) = Delta0 * interp(t/T; knots, values)",
        params=params,
        value_fn=value_fn,
        deriv_fn=deriv_fn,
    )
