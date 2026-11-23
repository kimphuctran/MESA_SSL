# MESA BioPhysics — Risk-Gated Independent Few-Shot Validation

**Date:** 23 November 2026  
**Repository:** https://github.com/kimphuctran/MESA_SSL  
**Experiment branch:** `mesa-biophysical-riskgate-2026`  
**Validated source revision:** `ebcad1f0c75b05681dc45d57a6d26e46877aaa95`  
**GitHub Actions:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45280823062  
**Execution:** 8/8 successful jobs; 4 independent archive tasks × 2 seeds × 3 label budgets × 19 methods = 456 recorded final evaluations.

## Fixed design and research question

This validation evaluates a risk-gated, metaplastic, complementary-memory
controller against raw/canonical/dual MultiRocket episodic readouts and
brain/physics inspired alternatives. Four archive tasks were predeclared
as evaluation tasks after development on BME, ECG200, MoteStrain,
ShapeletSim, and SonyAIBORobotSurface1:

- DistalPhalanxOutlineAgeGroup
- ProximalPhalanxOutlineCorrect
- RefrigerationDevices
- SonyAIBORobotSurface2

Seeds 5501 and 5502 use nested 1-, 3-, and 5-label-per-class support
subsets from the official TRAIN split. The slow cortical representation,
physical features, AR(2) innovation, sparse expansion and two SPC charts
are fitted from unlabeled TRAIN only. When three or five labels per
class exist, the controller screens experts using grouped
leave-one-labeled-episode-out predictions on TRAIN and predeclared
TRAIN-side nuisance augmentations; at 1-shot, that screen is undefined
and a fixed non-oracle fallback is used. Test labels evaluate the final
method only.

TEST perturbations (synthetic affine offset/trend and waveform jitter)
are **assumed to preserve original labels**. They do not establish
robustness to real biomedical, industrial, or adversarial shifts.

## All 1/3/5-shot results (Macro-F1, equal-dataset mean)

### 1-shot
| Method | Clean | Affine drift | Waveform jitter |
|---|---:|---:|---:|
| Raw MR episodic | 0.5511 | 0.5386 | 0.4148 |
| Canonical MR episodic | 0.5512 | 0.5515 | 0.4212 |
| Dual MR episodic | 0.5525 | 0.5475 | 0.4154 |
| Slow cortex episodic | 0.4903 | 0.4543 | 0.4778 |
| BioPhysics no homeostasis | 0.5346 | 0.5251 | 0.4905 |
| BioPhysics full | 0.5322 | 0.5237 | 0.4897 |
| BioPhysics safe TRAIN | 0.5525 | 0.5475 | 0.4154 |
| BioPhysics risk-gated | 0.5379 | 0.5240 | 0.5106 |
| Raw MR ridge | 0.5548 | 0.5422 | 0.4272 |
### 3-shot
| Method | Clean | Affine drift | Waveform jitter |
|---|---:|---:|---:|
| Raw MR episodic | 0.6324 | 0.6183 | 0.4647 |
| Canonical MR episodic | 0.6277 | 0.6268 | 0.4721 |
| Dual MR episodic | 0.6312 | 0.6348 | 0.4684 |
| Slow cortex episodic | 0.6000 | 0.4987 | 0.5925 |
| BioPhysics no homeostasis | 0.6395 | 0.6269 | 0.6070 |
| BioPhysics full | 0.6313 | 0.6260 | 0.6115 |
| BioPhysics safe TRAIN | 0.6399 | 0.6167 | 0.5749 |
| BioPhysics risk-gated | 0.6227 | 0.6073 | 0.5956 |
| Raw MR ridge | 0.6438 | 0.6130 | 0.5112 |
### 5-shot
| Method | Clean | Affine drift | Waveform jitter |
|---|---:|---:|---:|
| Raw MR episodic | 0.6636 | 0.6417 | 0.5017 |
| Canonical MR episodic | 0.6541 | 0.6520 | 0.4891 |
| Dual MR episodic | 0.6619 | 0.6590 | 0.4866 |
| Slow cortex episodic | 0.6499 | 0.5394 | 0.6393 |
| BioPhysics no homeostasis | 0.6627 | 0.6436 | 0.6365 |
| BioPhysics full | 0.6571 | 0.6389 | 0.6398 |
| BioPhysics safe TRAIN | 0.6613 | 0.6383 | 0.5802 |
| BioPhysics risk-gated | 0.6523 | 0.6320 | 0.6216 |
| Raw MR ridge | 0.6547 | 0.6031 | 0.4984 |

## Paired method differences vs dual-MR episodic reference

| Budget | Clean difference | Affine difference | Jitter difference |
|---|---:|---:|---:|
| 1-shot | -0.01458 | -0.02345 | 0.09521 |
| 3-shot | -0.00856 | -0.02749 | 0.12719 |
| 5-shot | -0.00955 | -0.02703 | 0.13491 |

At 5-shot, this amounts to a +0.13491
improvement under synthetic jitter, accompanied by
-0.00955 on clean data and
-0.02703 on affine perturbations.

## Five-shot task-level audit

| Dataset | Dual clean | Risk-gated clean | Dual affine | Risk-gated affine | Dual jitter | Risk-gated jitter |
|---|---:|---:|---:|---:|---:|---:|
| DistalPhalanxOutlineAgeGroup | 0.687 | 0.672 | 0.697 | 0.602 | 0.314 | 0.628 |
| ProximalPhalanxOutlineCorrect | 0.674 | 0.676 | 0.675 | 0.676 | 0.440 | 0.663 |
| RefrigerationDevices | 0.431 | 0.405 | 0.406 | 0.392 | 0.337 | 0.339 |
| SonyAIBORobotSurface2 | 0.856 | 0.856 | 0.858 | 0.858 | 0.856 | 0.856 |

## Negative evidence and interpretation

1. The risk-gated method improves synthetic waveform-jitter performance
   on average but does not outperform the dual-MR episodic reference
   simultaneously on clean and affine conditions.
2. The component-heavy fixed `BioPhysics full` variant has a higher
   **aggregate** five-shot jitter Macro-F1
   (0.63977) than
   `BioPhysics risk-gated`
   (0.62156).
   Consequently the revised decision rule is NOT an unconditional
   performance improvement over every candidate.
3. The slow cortical component by itself is sometimes a strong jitter
   expert but is less reliable under affine changes; this is why context
   recognition, rather than a single global mixture, was studied.
4. Across four tasks with only two seeds, no general statistical
   superiority or universal few-shot learning claim is supported.
   Per-dataset standard deviations and all negative results are in the
   complete experiment CSV and JSON.
5. Some individual tasks, notably RefrigerationDevices, show unstable
   decisions and limited absolute Macro-F1. The algorithm should not
   be judged only by the largest improvement achieved on one task.
6. Competing full-kernel MultiRocket, TS2Vec, TimeMAE, MOMENT and a
   proper episodic foundation-model comparison have not been executed
   under the identical four-task protocol. The observed findings are
   pre-submission development evidence, not SOTA validation.

## What to improve next

An empirical challenge is to preserve the clean/affine expert while
switching to the cortical/noise-robust memory *only* when the waveform
chart supplies reliable evidence. Hard per-case switches at a fixed
p-value of 0.10 can still misroute data, especially under small
calibration samples. Further ideas require new untouched datasets
and independent seeds rather than selecting settings based on the
four TEST sets already examined.

## Provenance and files

- `MESA_SSL/mesa_biophysics_memory.py` — full algorithm
- `MESA_SSL/RISK_GATED_BIOPHYSICS_DESIGN.md` — frozen tri-state policy
- `MESA_SSL/riskgate_biophysics_holdout_2026-11-23.json` —
  run-level evaluations and metadata
- `MESA_SSL/riskgate_biophysics_summary_2026-11-23.csv` —
  dataset, method and budget means/SD
- `.github/workflows/mesa-biophysics-riskgate-validation.yml` —
  validated workflow
- [Successful Actions run](https://github.com/kimphuctran/MESA_SSL/actions/runs/45280823062)

**Decision: promising synthetic-jitter robustness, but no broad superiority.
Maintain research branches independently of `master`.**
