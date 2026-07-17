from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvalResult:
    trial_index: int
    candidate_id: str
    task_id: str
    mode: str
    metric: float
    sim_cost: int
    cumulative_sim_cost: int
    runtime_sec: float
    constraint_ok: bool = True
    error_msg: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

