from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetTracker:
    max_sim_cost: int
    cumulative_sim_cost: int = 0
    n_evaluations: int = 0

    def can_afford(self, sim_cost: int) -> bool:
        return self.cumulative_sim_cost + sim_cost <= self.max_sim_cost

    def charge(self, sim_cost: int) -> None:
        self.cumulative_sim_cost += sim_cost
        self.n_evaluations += 1

