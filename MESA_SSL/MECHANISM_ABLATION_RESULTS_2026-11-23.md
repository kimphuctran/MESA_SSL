# MESA Encoder | Counterfactual Mechanism-Loss Ablation

**Date:** 23 November 2026.
**Original HPO workflow:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45217129377
**Completed paired ablation:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45220095525
**Experiment branch:** `mesa-encoder-hpo-2026`
**Source:** `MESA_SSL/mesa_encoder_mechanism_ablation.py`
**Execution:** 8/8 dataset jobs successful; 64 executed evaluations.

## What is held fixed?

For every dataset/seed, we load the **MESA hyperparameters selected earlier
from labeled TRAIN only**. A pair of encoders starts from identical
weights, train examples, batch order, mask/noise/harmonic draws,
warmup, decay, regularization and representation dimension.
Both encoder training runs execute the same mechanism branch, but
one multiplies both the concept loss and the equivariance loss by
**exactly zero**. Thus, the perturbation isolates the presence of
the mechanism objective at the already-selected MESA configuration,
rather than comparing two independently tuned encoders. TEST labels
are never used in configuration selection.

The concept discrepancy is the root mean squared error between the
learned concept head and *analytical harmonic shape summaries* on
TRAIN. It is an interpretability/proxy metric, **not validation of
real physical or clinical causal mechanisms**.

## Latent-only classification results

| Dataset | Mechanism clean | Zero-loss clean | Mechanism jitter | Zero-loss jitter | Concept RMSE mechanism | Zero-loss RMSE |
|---|---:|---:|---:|---:|---:|---:|
| BasicMotions | 0.748 | 0.820 | 0.712 | 0.718 | 0.109 | 0.246 |
| CBF | 0.915 | 0.906 | 0.893 | 0.896 | 0.177 | 0.455 |
| ECGFiveDays | 0.912 | 0.766 | 0.813 | 0.757 | 0.117 | 1.107 |
| GunPoint | 0.701 | 0.708 | 0.683 | 0.684 | 0.096 | 0.450 |
| GunPointAgeSpan | 0.818 | 0.822 | 0.769 | 0.786 | 0.111 | 0.629 |
| ItalyPowerDemand | 0.934 | 0.900 | 0.874 | 0.837 | 0.110 | 0.835 |
| Meat | 0.616 | 0.626 | 0.501 | 0.451 | 0.092 | 0.435 |
| SyntheticControl | 0.576 | 0.587 | 0.558 | 0.560 | 0.091 | 0.249 |

## Equal-weight dataset means (two seeds each)

| Readout | Mechanism clean | Zero-loss clean | Mechanism jitter | Zero-loss jitter | Mechanism concept RMSE | Zero-loss concept RMSE |
|---|---:|---:|---:|---:|---:|---:|
| latent | 0.777 | 0.767 | 0.725 | 0.711 | 0.113 | 0.551 |
| latent+canonical | 0.870 | 0.872 | 0.645 | 0.647 | 0.113 | 0.551 |

Across the eight datasets, mechanism loss changes latent-only clean
Macro-F1 by 0.010639, jitter Macro-F1 by
0.014156, and reduces the mean concept error by
0.437821 relative to the zero-loss
counterfactual. The large concept-alignment improvement
is accompanied by
 consistent improvement in classification.

The original HPO-MESA latent-only TEST results were reproduced on
all eight datasets within 0.000003 Macro-F1 after retraining from the
saved configuration; this is an implementation fidelity check.

## Implications

- Mechanism loss improves alignment with **prespecified statistical
  concepts**. Whether this helps predictive generalization depends on
  how relevant those concepts are to class labels.
- A different learned representation dimension, changed augmentation,
  higher mask ratio or reduced training duration is **not sufficient**
  to attribute improvements to the mechanism objective.
- The independently optimized generic encoder is an additional strong
  control in the primary HPO report, but the paired zero-loss control
  is more specific about mechanism contribution.
- The tested synthetic affine offset/trend and jitter interventions
  are assumed label-preserving. With only two seeds and eight datasets
  these results remain developmental; no comprehensive SOTA claim
  is justified.
- Any stronger mechanism claim requires actual annotated process
  variables, causal interventions or domain-validated concept targets.

## Data and scripts

- Full paired run-level records:
  `MESA_SSL/encoder_mechanism_ablation_2026-11-23.json`.
- Compact summary with sample standard deviations:
  `MESA_SSL/encoder_mechanism_ablation_summary_2026-11-23.csv`.
- Selected MESA training hyperparameters:
  `MESA_SSL/encoder_hpo_selected_2026-11-23.json`.
- GitHub Actions workflow:
  `.github/workflows/mesa-encoder-mechanism-ablation.yml`.
- Full action artifacts include `results.json`,
  `requirements.txt` and checked script copies.
