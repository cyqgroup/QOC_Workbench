from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class RegressionCase:
    case_id: str
    description: str
    check_fn: Callable[[object], bool]


@dataclass
class RegressionSuite:
    cases: List[RegressionCase] = field(default_factory=list)

    def run(self, protocol_module) -> dict:
        results = {}
        for case in self.cases:
            results[case.case_id] = bool(case.check_fn(protocol_module))
        return results

