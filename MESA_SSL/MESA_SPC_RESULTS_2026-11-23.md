# MESA-SPC | Six-dataset chart-controlled classification audit

**Research record:** 23 November 2026  
**Repository:** https://github.com/kimphuctran/MESA_SSL  
**Branch:** `mesa-spc-chart-experiment-2026`  
**Executed Actions run:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45191943870  
**Evaluated source commit:** `6d66ea9b46689a598ebe6ddf6dafa536d7017dde`

## Research question

Can a separately calibrated multivariate control chart activate an
affine-invariant representation only when mean/slope concepts leave their
TRAIN-reference distribution, reducing the loss of clean predictive
information while improving classification under the declared nuisance shift?

The treatment is a **chart-directed fallback**, not a new published SOTA
encoder. It is contrasted with both non-SSL and matched generic-SSL controls.

## Fixed protocol

Official UCR/UEA TRAIN/TEST splits on GunPoint, ItalyPowerDemand, BasicMotions,
Coffee, CBF, ECGFiveDays. Seeds 801–802, exactly five labeled TRAIN observations
per class, 15 epochs of unlabeled SSL pretraining. Raw and affine-canonical
MultiRocket use 336 kernels, kernel ridge classifiers use TRAIN-only
leave-one-labeled-out selection (ridge grid 0.01, 0.1, 1). Generic and
mechanism-conditioned MESA encoders have matched training resources.

The chart fits a shrinkage covariance (Ledoit-Wolf) on 60% of unlabeled
TRAIN, withholding the other 40% for split-conformal threshold calibration.
It observes the mean and linear slope across channels. Alarm p-values are
`(1 + #{calibration statistics >= test statistic}) / (n_cal + 1)`, with
predeclared alpha 0.10. The chart never sees TEST targets.
For every TEST example it independently selects either raw or affine-canonical
predictions. **TEST series order is not interpreted as chronological**;
this is a Phase-II individual-observation chart, not an EWMA or sequential
ARL experiment.

Shift is an additive constant offset plus linear trend at strength 1.5.
Class preservation is **assumed by the synthetic experiment**, not certified
from industrial/medical reality. The affine-canonical branch is invariant
up to floating point tolerance for this perturbation family.

## Methods

- Raw MultiRocket
- Canonical MultiRocket
- Canonical + generic SSL
- Canonical + MESA SSL
- SPC raw-to-canonical
- SPC raw-to-generic
- SPC raw-to-MESA
- TRAIN-selected static

## Observed results (Macro-F1 mean, two seeds)

| Dataset | Raw clean | Raw shifted | SPC→Canonical clean | SPC→Canonical shifted | SPC→MESA shifted |
|---|---:|---:|---:|---:|---:|
| GunPoint | 0.953 | 0.806 | 0.950 | 0.926 | 0.906 |
| ItalyPowerDemand | 0.933 | 0.924 | 0.928 | 0.932 | 0.926 |
| BasicMotions | 1.000 | 0.987 | 1.000 | 0.987 | 0.987 |
| Coffee | 0.982 | 0.982 | 0.982 | 0.982 | 0.982 |
| CBF | 0.968 | 0.930 | 0.966 | 0.962 | 0.965 |
| ECGFiveDays | 0.943 | 0.914 | 0.943 | 0.938 | 0.939 |

## Matched differences and alarm rates

| Dataset | SPC-C vs Raw shifted Δ | SPC-C vs Canonical shifted Δ | SPC-MESA vs SPC-generic shifted Δ | Clean flags | Shifted flags |
|---|---:|---:|---:|---:|---:|
| GunPoint | 0.1207 | 0.0000 | 0.0035 | 6.0% | 96.0% |
| ItalyPowerDemand | 0.0088 | -0.0005 | 0.0049 | 7.5% | 98.7% |
| BasicMotions | 0.0000 | 0.0000 | 0.0000 | 1.3% | 100.0% |
| Coffee | 0.0002 | 0.0000 | 0.0000 | 3.6% | 100.0% |
| CBF | 0.0320 | -0.0011 | 0.0056 | 5.4% | 89.7% |
| ECGFiveDays | 0.0232 | -0.0006 | 0.0006 | 2.3% | 90.4% |

All 96 individual method-by-dataset-by-seed records are archived in
`mesa_spc_results_all_2026-11-23.json`; sample standard deviations and all
eight methods appear in `mesa_spc_summary_2026-11-23.csv`.
All six GitHub Actions matrix jobs completed successfully.

## Interpretation

1. SPC→Canonical improves shifted Macro-F1 relative to Raw MultiRocket on
   five datasets and ties on BasicMotions. It never surpasses Canonical
   MultiRocket on the shifted tests, though it often preserves cleaner raw
   performance when no shift is detected.
2. SPC→MESA beats its *matched generic-SSL chart fallback* on some
   datasets (notably CBF and ECGFiveDays), but is inferior to
   SPC→Canonical elsewhere. Its added representation loss is not
   consistently justified by classification gain.
3. The clean and shifted alert rates reflect the specific synthetic
   intervention and TEST distribution. Split-conformal calibration guarantees
   a marginal alarm bound only if future in-control observations are
   exchangeable with calibration observations. It does not establish an
   in-control average run length (ARL0), detection delay, or physical
   invariance beyond the specified affine family.
4. The reported two-seed benchmark is an exploratory development audit.
   Small numerical differences are not statistically significant evidence
   of dominance. This design requires independent untouched archive tasks,
   more seeds, several shift types and a true sequential benchmark before
   any Neurocomputing performance claim.
5. Statistical control of learned representations is not new:
   Biegel et al., 2024, *Journal of Intelligent Manufacturing*,
   https://doi.org/10.1007/s10845-023-02156-7,
   already proposed self-supervised MSPC with a Hotelling T² chart.
   Any new contribution here must be conditional selection,
   split-calibrated chart governance, strong matched controls, and
   demonstrable predictive rather than merely monitoring benefits.

## Reproducible software and run provenance

- `MESA_SSL/mesa_spc_chart.py` implements chart calibration, matched
  generic and MESA encoder comparisons, chart fallbacks and metrics.
- `.github/workflows/mesa-spc-chart.yml` is the executed workflow.
- GitHub Actions run: https://github.com/kimphuctran/MESA_SSL/actions/runs/45191943870
- Experiment branch files: https://github.com/kimphuctran/MESA_SSL/tree/mesa-spc-chart-experiment-2026/MESA_SSL

**Status:** completed development experiment; not validated SOTA.
