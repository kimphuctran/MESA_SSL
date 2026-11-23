# HCLM decision audit: choosing the NEXT experiment, not the best TEST model

Date: 23 November 2026. 
**Purpose:** use HCLM to govern the assistant's research actions.

The input is the completed frozen risk-gated BioPhysics benchmark
(4 UCR datasets, 2 seeds, 1/3/5-shot, 19 methods). The HCLM research
governor made decisions from actual source records with matching
dataset/seed/support IDs and exploratory paired cluster bootstrap.

**Observed difference = BioPhysics Risk-Gated − Dual MultiRocket episodic:**

| Labels/class | Δ Clean | Δ Affine | Δ Jitter | Action |
|---|---:|---:|---:|---|
| 1-shot | -0.0146 | -0.0234 | +0.0952 | RUN_SANDBOX |
| 3-shot | -0.0086 | -0.0275 | +0.1272 | RUN_SANDBOX |
| 5-shot | -0.0096 | -0.0270 | +0.1349 | RUN_SANDBOX |

All three label budgets have the same outcome:
**REJECT promotion; authorize the sandbox experiment**
`AFFINE_FALLBACK_CALIBRATION`.

The decision does NOT claim Risk-Gated failed everywhere; it recognizes
that improvements under jitter coexist with mean losses under affine drift.

## Main gate failures

- insufficient untouched independent datasets
- insufficient independent paired seeds
- mean affine Macro-F1 non-inferiority failed
- at least one dataset exceeds allowed affine degradation
- full-strength matched baselines missing: MOMENT, MultiRocket full, TS2Vec, TimeMAE

A four-task two-seed study is too small for a scientifically reliable
population guarantee. Its exploratory 95% bootstrap sensitivity
intervals for five-shot differences are:
- clean_macro_f1: [-0.0228, +0.0013]
- affine_macro_f1: [-0.0788, +0.0003]
- jitter_macro_f1: [+0.0007, +0.2765]

**These intervals are descriptive, NOT confirmatory confidence
intervals**: archives are nonrandom and have been inspected.

## What the controller authorized

It did NOT authorize another unconstrained architecture search or a
`master` merge. Instead it authorized a single new experiment,
frozen in advance on four new UCR datasets with fixed seeds,
support budgets, acceptance thresholds and prior action check.

Source code: `MESA_SSL/hclm_search_governor.py`.

Historical failures: `MESA_SSL/hclm_failure_ledger_2026-11-23.json`.

Decision audit JSON: `MESA_SSL/HCLM_SEARCH_DECISION_2026-11-23.json`.

Workflow: https://github.com/kimphuctran/MESA_SSL/actions/runs/45283247551

HCLM here is an observable search-governance procedure, inspired by
Tran's control–calibration–governed-action programme. It does not alter
ChatGPT's internal neural weights, nor pretend to implement the entire
differentiable entropy-force theory. Human approval is required
before any promotion; TEST-based oracle selection is prohibited.
