from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class HamiltonianTermCheck:
    ok: bool
    active_terms: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def infer_candidate_terms(task_spec: dict, candidate) -> List[str]:
    hardware = task_spec.get("hardware")
    channels = {ch.name: ch for ch in candidate.channels}
    terms: List[str] = []

    if hardware == "rydberg":
        if "omega" in channels:
            terms.append("Omega(t)/2 * Sum_i X_i")
        if "delta" in channels:
            terms.append("-Delta(t) * Sum_i n_i")
        if candidate.cd.kind != "none":
            terms.append("f_cd(t) * Sum_i Y_i")
        terms.append("V * Sum_(i,j in E) n_i n_j")
        return terms

    if hardware == "abstract_spin":
        if "schedule" in channels:
            terms.append("(1-s(t)) * H_B + s(t) * H_P")
        if candidate.cd.kind != "none":
            terms.append("A_cd(t)")
        terms.extend(["H_B(transverse-field driver)", "H_P(3SAT instance seed=58)"])
        return terms

    return terms


def check_hamiltonian_structure(task_spec: dict, candidate) -> HamiltonianTermCheck:
    structure = task_spec.get("hamiltonian_structure", {})
    active = infer_candidate_terms(task_spec, candidate)
    allowed = set(structure.get("problem_fixed_terms", []))
    allowed |= set(structure.get("hardware_fixed_terms", []))
    allowed |= set(structure.get("controllable_terms", []))
    allowed |= set(structure.get("optional_auxiliary_terms", []))
    forbidden = set(structure.get("forbidden_controllable_terms", []))

    errors: List[str] = []
    for term in active:
        if allowed and term not in allowed:
            errors.append(f"active term not declared in hamiltonian_structure: {term}")
        if term in forbidden:
            errors.append(f"forbidden active term: {term}")

    return HamiltonianTermCheck(ok=not errors, active_terms=active, errors=errors)
