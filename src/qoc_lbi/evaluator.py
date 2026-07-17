from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .constraints import check_protocol_constraints
from .result_schema import EvalResult


@dataclass
class EvaluatorConfig:
    metric_name: str
    sim_cost_fn: Callable[[Any, str], int]


class QOCEvaluator:
    """
    Fixed evaluator boundary.

    Phase 1 provides the contract only. Concrete backends will be plugged in
    later by binding `eval_fn`.
    """

    def __init__(self, config: EvaluatorConfig, eval_fn: Callable[[dict, Any, str], dict]):
        self.config = config
        self.eval_fn = eval_fn

    def evaluate_with_payload(
        self,
        task_spec: dict,
        candidate,
        mode: str,
        trial_index: int,
        cumulative_sim_cost: int,
    ) -> tuple[EvalResult, dict]:
        t0 = time.time()
        check = check_protocol_constraints(task_spec, candidate)
        sim_cost = self.config.sim_cost_fn(candidate, mode)

        if not check.ok:
            result = EvalResult(
                trial_index=trial_index,
                candidate_id=candidate.candidate_id,
                task_id=candidate.task_id,
                mode=mode,
                metric=0.0,
                sim_cost=sim_cost,
                cumulative_sim_cost=cumulative_sim_cost,
                runtime_sec=time.time() - t0,
                constraint_ok=False,
                error_msg="; ".join(check.errors),
            )
            return result, {}

        try:
            payload = self.eval_fn(task_spec, candidate, mode)
        except Exception as exc:  # noqa: BLE001
            result = EvalResult(
                trial_index=trial_index,
                candidate_id=candidate.candidate_id,
                task_id=candidate.task_id,
                mode=mode,
                metric=0.0,
                sim_cost=sim_cost,
                cumulative_sim_cost=cumulative_sim_cost,
                runtime_sec=time.time() - t0,
                constraint_ok=False,
                error_msg=f"{type(exc).__name__}: {exc}",
                diagnostics={"exception_type": type(exc).__name__},
            )
            return result, {}
        result = EvalResult(
            trial_index=trial_index,
            candidate_id=candidate.candidate_id,
            task_id=candidate.task_id,
            mode=mode,
            metric=float(payload["metric"]),
            sim_cost=sim_cost,
            cumulative_sim_cost=cumulative_sim_cost,
            runtime_sec=time.time() - t0,
            diagnostics=payload.get("diagnostics", {}),
        )
        return result, payload

    def evaluate(self, task_spec: dict, candidate, mode: str, trial_index: int, cumulative_sim_cost: int) -> EvalResult:
        result, _ = self.evaluate_with_payload(
            task_spec=task_spec,
            candidate=candidate,
            mode=mode,
            trial_index=trial_index,
            cumulative_sim_cost=cumulative_sim_cost,
        )
        return result
