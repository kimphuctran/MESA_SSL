# HCLM as Research-Search Governor: Two Executed Feedback Cycles

**23 November 2026 · MESA-SSL · Kim Phuc Tran**

This document is about HCLM governing the **researcher's/assistant's search
process**, NOT adding an HCLM loss to another neural network. It records
observable decisions, experiments and feedback that changed the next
research action.

## Control loop actually executed

**Cycle 1 — HCLM observes earlier failures.**

Input: 4 tasks × 2 seeds × 1/3/5 labels per class from the risk-gated
BioPhysics archive, and a permanent ledger of 16 prior failures.

The previous model gained 0.13491 five-shot Macro-F1 under synthetic
waveform jitter relative to Dual MR episodic, but lost 0.00955 clean and
0.02703 affine. The governor applied preset risk constraints and returned:

```text
PROMOTION DENIED
ALLOW: research sandbox
NEXT: AFFINE_FALLBACK_CALIBRATION
```

The controller also blocked broad claims due to only 4 task groups,
2 seeds per task, and missing matched full MultiRocket,
TS2Vec, TimeMAE and MOMENT.

Machine-executed decision:
https://github.com/kimphuctran/MESA_SSL/actions/runs/45283247551

**Cycle 1 — Governed action.**

The selected study was frozen **before TEST** in
`HCLM_FROZEN_AFFINE_EXPERIMENT_2026-11-23.json`:
Strawberry, ToeSegmentation1, ToeSegmentation2, Computers;
seeds 5601/5602; 1-, 3-, 5-shot.
The controller was **run and its action checked before model training**.
It constrained clean decisions to Dual MR and used more conservative
TRAIN-supported affine/waveform fallbacks with p-value limits 0.05.
It reverted to the simple baseline when both charts conflicted.

All 8 workflows passed: 480 executed evaluations.
https://github.com/kimphuctran/MESA_SSL/actions/runs/45283668253

### Results: HCLM-sanctioned candidate versus frozen Dual MR baseline

| Budget | Dual clean | HCLM-guarded clean | Dual affine | HCLM-guarded affine | Dual jitter | HCLM-guarded jitter |
|---|---:|---:|---:|---:|---:|---:|
| 1-shot | 0.5761 | 0.5765 | 0.5656 | 0.5660 | 0.4342 | 0.4892 |
| 3-shot | 0.6761 | 0.6728 | 0.6688 | 0.6694 | 0.4954 | 0.5528 |
| 5-shot | 0.7077 | 0.7058 | 0.7030 | 0.7040 | 0.5196 | 0.5884 |

All means are equal-weighted over the 4 datasets and average 2 seeds each.

**Predeclared sandbox contract passed** on 1/3/5-shot:
mean clean and affine changes >= -0.01, mean jitter gain >= +0.04,
and maximum per-task harm limits.

It does NOT follow that this improves the previous incumbent. Against
the **existing risk-gated model on the same exact new tasks and seeds**,
the five-shot HCLM-guarded policy has:
- clean Macro-F1 difference **-0.00251**,
- affine Macro-F1 difference **-0.00405**,
- jitter Macro-F1 difference **-0.00952**.

The HCLM controller therefore marked it a **research sandbox candidate**,
not a new state of the art or a model that should replace the incumbent.
All negative and positive outcomes remain available; no post-hoc oracle
best-on-TEST score was substituted for the chosen model.

The independently executed frozen acceptance workflow:
https://github.com/kimphuctran/MESA_SSL/actions/runs/45284438890

**Cycle 2 — Feedback alters subsequent research action.**

The previous four TEST datasets are now marked *development*.
A new failure entry **F17** was written to the 16-item ledger:
the conservative fallback passed the baseline safety constraints,
but still failed to improve the incumbent head-to-head.

The research governor was changed to **forbid repeating completed
experiments**. Running it again with the complete feedback archive
produced a DIFFERENT action:

```text
PROMOTION STILL DENIED
BLOCKED REPEAT: AFFINE_FALLBACK_CALIBRATION
NEXT ALLOWED RESEARCH ACTION: MATCHED_FULL_BASELINE_AUDIT
```

The updated HCLM control-loop verification was successful:
https://github.com/kimphuctran/MESA_SSL/actions/runs/45284816647

Thus HCLM was actually used to drive the decision about what the
assistant should try NEXT, instead of being inserted inside the
experimental neural architecture or adding unlimited complexity.

## Limits and interpretation

- These are *engineering-scoped implementations* of HCLM control,
  calibration and governed action. They do not change the internal
  parameters of ChatGPT and are not a formal derivation of the
  differentiable HCLM entropy-force equations.
- The paired bootstrap in the governor is an exploratory sensitivity
  diagnostic, not confirmatory coverage for nonrandom, repeatedly
  inspected benchmark archives.
- The 1-shot control does not pretend to use a class-preserving
  leave-one-labeled-out cross-validation.
- UCR samples are not chronological process streams: no ARL0 guarantee.
- Affine and waveform modifications are synthetic and are assumed
  label-preserving for this experiment.
- Compact MultiRocket is not full-capacity MultiRocket; fair matched
  comparisons remain necessary.
- Human approval remains required before merging to `master`,
  deploying a method or asserting broad superiority.

## Reproducibility

- HCLM meta-controller:
  `MESA_SSL/hclm_search_governor.py`
- Cumulative failure memory:
  `MESA_SSL/hclm_failure_ledger_2026-11-23.json` (17 entries)
- Frozen research contract:
  `MESA_SSL/HCLM_FROZEN_AFFINE_EXPERIMENT_2026-11-23.json`
- Executed candidate policy:
  `MESA_SSL/mesa_biophysics_memory.py`
- All 480 records:
  `MESA_SSL/HCLM_AFFINE_FROZEN_RUN_2026-11-23.json`
- 240 model/dataset/budget means and SD:
  `MESA_SSL/HCLM_AFFINE_4DATASET_SUMMARY_2026-11-23.csv`
- Independent post-run risk gate:
  `MESA_SSL/hclm_sandbox_post_audit.py`
- Second-stage HCLM feedback workflow:
  `.github/workflows/hclm-feedback-next.yml`

**Current action (as determined by HCLM, not after-the-fact TEST ranking):
MATCHED_FULL_BASELINE_AUDIT. The current candidate has NOT been promoted.**
