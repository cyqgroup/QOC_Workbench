from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax.numpy as jnp


ScalarFn = Callable[[float], jnp.ndarray]
ChannelBuilder = Callable[[str, Any, Any, Any], "ScalarControl"]
CDBuilder = Callable[[Any, "ScalarControl", "ScalarControl", Any, Any], "ScalarControl"]


@dataclass
class ScalarControl:
    name: str
    basis: str
    formula: str
    params: dict[str, Any] = field(default_factory=dict)
    value_fn: ScalarFn | None = None
    deriv_fn: ScalarFn | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def value(self, t: float) -> jnp.ndarray:
        if self.value_fn is None:
            raise ValueError(f"{self.name}:{self.basis} missing value_fn")
        return self.value_fn(t)

    def deriv(self, t: float) -> jnp.ndarray:
        if self.deriv_fn is None:
            return jnp.array(0.0, dtype=jnp.float32)
        return self.deriv_fn(t)


CHANNEL_BASIS_REGISTRY: dict[tuple[str, str], ChannelBuilder] = {}
CD_REGISTRY: dict[str, CDBuilder] = {}
_REGISTRATIONS_LOADED = False


def register_channel_basis(channel_name: str, basis_name: str) -> Callable[[ChannelBuilder], ChannelBuilder]:
    def decorator(builder: ChannelBuilder) -> ChannelBuilder:
        CHANNEL_BASIS_REGISTRY[(channel_name, basis_name)] = builder
        return builder

    return decorator


def register_cd_kind(kind: str) -> Callable[[CDBuilder], CDBuilder]:
    def decorator(builder: CDBuilder) -> CDBuilder:
        CD_REGISTRY[kind] = builder
        return builder

    return decorator


def ensure_rydberg_registrations() -> None:
    global _REGISTRATIONS_LOADED
    if _REGISTRATIONS_LOADED:
        return
    from . import rydberg_basis  # noqa: F401
    from . import rydberg_cd  # noqa: F401

    _REGISTRATIONS_LOADED = True


def build_rydberg_channel_control(channel_name: str, channel: Any, task_view: Any, candidate: Any) -> ScalarControl:
    ensure_rydberg_registrations()
    key = (channel_name, channel.basis)
    if key not in CHANNEL_BASIS_REGISTRY:
        raise ValueError(f"unsupported {channel_name} basis: {channel.basis}")
    return CHANNEL_BASIS_REGISTRY[key](channel_name, channel, task_view, candidate)


def build_rydberg_cd_control(
    cd_config: Any,
    omega_control: ScalarControl,
    delta_control: ScalarControl,
    task_view: Any,
    candidate: Any,
) -> ScalarControl:
    ensure_rydberg_registrations()
    kind = str(getattr(cd_config, "kind", "none") or "none")
    if kind not in CD_REGISTRY:
        raise ValueError(f"unsupported Rydberg cd kind: {kind}")
    return CD_REGISTRY[kind](cd_config, omega_control, delta_control, task_view, candidate)
