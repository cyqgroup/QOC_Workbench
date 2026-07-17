---
name: knowledge-base-expansion
description: Add papers, notes, code resources, failed attempts, and reusable QOC motifs to the QOC Workbench knowledge base without mixing external literature with internal experiment artifacts.
---

# Knowledge Base Expansion

Use when the user asks the workbench to learn from papers, notes, codebases, prior artifacts, or failed searches.

## Workflow

1. Classify the source: external literature, external code, internal artifact, failure note, or strategy motif.
2. Store raw sources in `knowledge_base/raw/` only when redistribution is allowed; otherwise record citation and path notes.
3. Write concise wiki notes under `knowledge_base/wiki/`:
   - `01_Problems/` for Hamiltonian/problem families;
   - `02_Strategies/` for reusable controls or algorithmic motifs;
   - `04_Papers/` for paper summaries;
   - other folders for tools, tutorials, and analysis.
4. Extract design ingredients: Hamiltonian form, path, auxiliary controls, baseline, metrics, assumptions, backend choices, and failure modes.
5. Link tested ideas to artifacts, not just claims.
6. Keep internal experiment outcomes in `artifacts/`, `summary.md`, or registry files, not as external references.

## Required Outputs

- Knowledge note with source, Hamiltonian, strategy, metric, result, and assumptions.
- Extracted reusable motifs and constraints.
- Links to reproduction/search artifacts when available.
- Clear separation of successful motifs and known failure modes.

## Case Lessons

- Rydberg artifacts taught that residual-CD variants can fail and that smooth beta-bump global-Y controls can be useful in constrained follow-up searches.
- XXZ artifacts taught that endpoint behavior matters: boundary-cleaning follow-ups can change which schedule deformations are defensible.
- TFIM artifacts taught that workflow-level bottlenecks can become new strategy motifs, such as learned coefficient generators.

## Guardrails

- Do not store internal run diaries in `knowledge_base/` as if they were external literature.
- Do not copy copyrighted papers into public releases unless allowed.
- Do not let an untested motif become a manuscript claim without an artifact.
