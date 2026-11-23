# HCLM-governed MESA: Frozen Affine-Fallback Results

**Research date:** 23 November 2026  
**Project:** https://github.com/kimphuctran/MESA_SSL  
**Branch:** `hclm-affine-fallback-sandbox-2026`  
**Predeclared contract:** `MESA_SSL/HCLM_FROZEN_AFFINE_EXPERIMENT_2026-11-23.json`  
**Executed experiments:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45283668253 — 8/8 successful  
**HCLM acceptance check:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45284438890 — successful  
**Total records:** 480 (4 datasets × 2 seeds × 3 budgets × 20 methods).

## Purpose

This is **HCLM used as the assistant's scientific research controller**,
NOT a new HCLM layer inside an ML network. The `hclm_search_governor.py`
read 16 archived failure modes, calibrated paired uncertainty and rejected
premature promotion of BioPhysics. It selected an action:
`AFFINE_FALLBACK_CALIBRATION` in a sandbox branch, with all thresholds,
dataset names and label budgets frozen before TEST.

The governed model uses predeclared 0.05-conformal nuisance alerts,
TRAIN-only evidence margins 0.04, and safe fallback when affine and
morphology charts conflict. Its claimed purpose is bounded
risk reduction, not universal accuracy dominance.

## Official UCR TEST Macro-F1, dataset-equal mean, 2 seeds

| Shot | Method | Clean | Affine drift | Waveform jitter |
|---|---|---:|---:|---:|
| 1 | Dual MR episodic | 0.5761 | 0.5656 | 0.4342 |
| 1 | BioPhysics HCLM-guarded | 0.5765 | 0.5660 | 0.4892 |
| 1 | BioPhysics risk-gated | 0.5889 | 0.5755 | 0.5138 |
| 1 | BioPhysics full | 0.5405 | 0.5196 | 0.5116 |
| 1 | Raw MR ridge | 0.5748 | 0.5473 | 0.4560 |
| 3 | Dual MR episodic | 0.6761 | 0.6688 | 0.4954 |
| 3 | BioPhysics HCLM-guarded | 0.6728 | 0.6694 | 0.5528 |
| 3 | BioPhysics risk-gated | 0.6807 | 0.6701 | 0.5702 |
| 3 | BioPhysics full | 0.6306 | 0.6083 | 0.5674 |
| 3 | Raw MR ridge | 0.6841 | 0.6524 | 0.5291 |
| 5 | Dual MR episodic | 0.7077 | 0.7030 | 0.5196 |
| 5 | BioPhysics HCLM-guarded | 0.7058 | 0.7040 | 0.5884 |
| 5 | BioPhysics risk-gated | 0.7083 | 0.7081 | 0.5980 |
| 5 | BioPhysics full | 0.6621 | 0.6364 | 0.5965 |
| 5 | Raw MR ridge | 0.7129 | 0.6598 | 0.4999 |

## HCLM-guarded model minus original Dual MR baseline

- 1-shot: clean 0.000357,
  affine 0.000358,
  jitter 0.055021.
- 3-shot: clean -0.003304,
  affine 0.000583,
  jitter 0.057413.
- 5-shot: clean -0.001897,
  affine 0.001033,
  jitter 0.068842.

## 5-shot per-dataset evidence

| Dataset | Dual clean | HCLM clean | Dual affine | HCLM affine | Dual jitter | HCLM jitter |
|---|---:|---:|---:|---:|---:|---:|
| Computers | 0.5871 | 0.6002 | 0.5792 | 0.5834 | 0.3962 | 0.4360 |
| Strawberry | 0.7658 | 0.7451 | 0.7691 | 0.7691 | 0.3914 | 0.6270 |
| ToeSegmentation1 | 0.7568 | 0.7568 | 0.7460 | 0.7460 | 0.5571 | 0.5571 |
| ToeSegmentation2 | 0.7212 | 0.7212 | 0.7177 | 0.7177 | 0.7336 | 0.7336 |

## Predeclared HCLM thresholds and decisions

The independent post-run audit found **no fixed-threshold failures**
for 1/3/5-shot. The explicit gates are mean clean difference >= -0.01,
mean affine >= -0.01, minimum jitter +0.04, and no dataset clean or
affine difference worse than -0.04. These are finite experimental
acceptance rules, **not** guarantees of future generalization.

The post-run controller issued `RESEARCH_SANDBOX_CANDIDATE`, while
`promotion=FORBIDDEN` at each budget.

## Important negative comparative evidence

The new HCLM-guarded algorithm does **not** outperform the preceding
`BioPhysics risk-gated` variant by average TEST Macro-F1 on this
evaluation group. The actual guarded minus prior risk-gated changes are:

- 1-shot: clean -0.012485,
  affine -0.009552,
  jitter -0.024611.
- 3-shot: clean -0.007946,
  affine -0.000744,
  jitter -0.017325.
- 5-shot: clean -0.002515,
  affine -0.004049,
  jitter -0.009520.

The value of HCLM here is **a reproducible process control policy**
that prevents risky promotion, preserves the frozen thresholds and
prevents TEST-selected oracle claims. It does not imply that the
new model gets the highest score in every experimental condition.
On Strawberry at 5-shot, HCLM-guarded preserves mean affine accuracy
but loses some clean accuracy relative to Dual MR; on ToeSegmentation1,
conservative fallback often yields identical scores rather than gains.
All raw comparisons are preserved.

## Inference limits

The new group consists of Strawberry, ToeSegmentation1,
ToeSegmentation2, Computers. The design used exactly seeds 5601/5602
and nested 1/3/5 labels/class, on official UCR splits. All synthetic
perturbations are **assumed** label preserving, and no sequential
SPC ARL guarantee can be inferred.

Only four datasets and two seeds were evaluated; MultiRocket is a compact
336-kernel baseline, and full-strength TS2Vec, TimeMAE, MOMENT,
Hydra+MultiRocket comparisons are not matched here.
The TEST datasets just examined now become development data.
Further claims require a *new* frozen independent validation set.

## Reproducibility

- Research governor: `MESA_SSL/hclm_search_governor.py`
- Persistent failure memory: `MESA_SSL/hclm_failure_ledger_2026-11-23.json`
- Predeclared study contract: `MESA_SSL/HCLM_FROZEN_AFFINE_EXPERIMENT_2026-11-23.json`
- Governed MESA code: `MESA_SSL/mesa_biophysics_memory.py`
- Raw 480 records: `MESA_SSL/HCLM_AFFINE_FROZEN_RUN_2026-11-23.json`
- Freeze-and-run workflow: `.github/workflows/hclm-affine-sandbox.yml`
- Post-run HCLM audit code: `MESA_SSL/hclm_sandbox_post_audit.py`
- Post-run gate workflow: `.github/workflows/hclm-postrun-audit.yml`
- Verified HCLM 1/3/5-shot prior decision:
  `MESA_SSL/HCLM_SEARCH_DECISION_2026-11-23.md` on
  `hclm-governed-research-search-2026`.

**Decision: passes frozen sandbox risk thresholds; no promotion, merge or
state-of-the-art claim authorized.**
