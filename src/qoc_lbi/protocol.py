from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChannelConfig:
    name: str
    basis: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CDConfig:
    kind: str = "none"
    ansatz: Optional[str] = None
    order: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolCandidate:
    candidate_id: str
    task_id: str
    family: str
    hardware: str
    total_time: float
    channels: List[ChannelConfig] = field(default_factory=list)
    cd: CDConfig = field(default_factory=CDConfig)
    constraints: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], task_id: str | None = None) -> "ProtocolCandidate":
        return cls(
            candidate_id=payload["candidate_id"],
            task_id=task_id or payload["task_id"],
            family=payload["family"],
            hardware=payload["hardware"],
            total_time=float(payload["total_time"]),
            channels=[ChannelConfig(**channel) for channel in payload.get("channels", [])],
            cd=CDConfig(**payload.get("cd", {})),
            constraints=dict(payload.get("constraints", {})),
            provenance=dict(payload.get("provenance", {})),
            notes=list(payload.get("notes", [])),
        )
