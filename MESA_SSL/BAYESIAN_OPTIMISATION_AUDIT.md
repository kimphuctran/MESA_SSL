# MESA-SPC | Train-only Bayesian optimization audit

**Date:** 23 November 2026
**Run:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45194460384
**Code:** https://github.com/kimphuctran/MESA_SSL/blob/mesa-bayes-opt-2026/MESA_SSL/mesa_spc_bayesian.py
**Branch:** `mesa-bayes-opt-2026`
**Status:** all nine matrix jobs successful; 144 method evaluations (9 datasets × 2 seeds × 8 methods).

## Protocol

Multivariate TPE (Optuna), 64 trials in each of canonical, generic SSL and
mechanism-conditioned MESA SSL controller families per dataset and seed.
Each family tunes raw ridge, safe ridge, chart alarm alpha, gate temperature,
gate strength, and (if appropriate) latent weight. The objective uses only
five labelled TRAIN observations per class and exact leave-one-label-out
predictions on raw TRAIN and two predeclared TRAIN augmentations.
Representation training remains fixed at 15 epochs to isolate controller
hyperparameter gains. The *official TEST* is only used to calculate the final
score of a TRAIN-selected hyperparameter configuration.
All synthetic shifts are declared label-preserving assumptions.

## Official TEST Macro-F1 under synthetic affine shift

| Dataset | Raw shifted | Canonical shifted | BO Canonical shifted | BO Generic shifted | BO MESA shifted | BO-selected shifted |
|---|---:|---:|---:|---:|---:|---:|
| GunPoint | 0.785 | 0.903 | 0.903 | 0.903 | 0.903 | 0.903 |
| ItalyPowerDemand | 0.894 | 0.933 | 0.930 | 0.930 | 0.928 | 0.926 |
| BasicMotions | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Coffee | 0.925 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| CBF | 0.926 | 0.960 | 0.939 | 0.939 | 0.945 | 0.939 |
| ECGFiveDays | 0.817 | 0.916 | 0.909 | 0.917 | 0.907 | 0.908 |
| Trace | 0.953 | 1.000 | 0.994 | 0.994 | 0.994 | 0.994 |
| SyntheticControl | 0.885 | 0.630 | 0.908 | 0.913 | 0.901 | 0.901 |
| Lightning2 | 0.513 | 0.572 | 0.580 | 0.638 | 0.622 | 0.638 |

## Mean across nine equally weighted datasets

| Method | Mean clean Macro-F1 | Mean shifted Macro-F1 |
|---|---:|---:|
| Raw MultiRocket | 0.914 | 0.853 |
| Canonical MultiRocket | 0.878 | 0.879 |
| Canonical + generic SSL | 0.879 | 0.878 |
| Canonical + MESA SSL | 0.878 | 0.877 |
| BO-SPC canonical | 0.921 | 0.907 |
| BO-SPC generic | 0.923 | 0.915 |
| BO-SPC MESA | 0.921 | 0.911 |
| BO-SPC TRAIN-selected | 0.923 | 0.912 |

The mean is a descriptive macro-average across **dataset averages**, not a
claim of population generalization, nor a dataset-count-weighted metric.

## Important findings

- MESA TPE does **not** universally surpass MultiRocket or generic SSL. A
  large number of trials cannot correct a representational information loss.
- On SyntheticControl, raw signals preserve class-discriminating mean/trend
  that canonical projection can remove; static canonical falls from a raw
  clean Macro-F1 near 0.934 to 0.630 on two selected seeds. Generic Bayesian
  chart gating scores approximately 0.913 under imposed shifts, above raw
  shifted near 0.885 but still below raw clean accuracy.
- On Lightning2, BO generic improves shifted Macro-F1 over raw and canonical,
  but the two-seed estimate is unstable and should not be interpreted as
  significance.
- In CBF, the MESA Bayesian controller improves clean accuracy relative to
  canonical (and sometimes raw), yet retains lower shifted accuracy than the
  fixed canonical branch.
- The final `BO-SPC TRAIN-selected` policy chooses a family on TRAIN objective
  only; performance does not equal the post-hoc best on TEST. Reporting an
  oracle chosen by TEST scores as an actual method would be data leakage.
- Increasing the number of trials on a five-label/class objective risks
  overfitting the small tuning sample. The results suggest that robustness
  must also be learned *within the raw feature space* using train-grouped
  augmentation, rather than assuming that normalization is always safe.

## Links and provenance

- Complete 144 evaluation records:
  `MESA_SSL/bayes_optimization_results_9datasets_2026-11-23.json`.
- Dataset summaries with 2-seed SD:
  `MESA_SSL/bayes_optimization_summary_2026-11-23.csv`.
- Trial-level optimization logs remain in GitHub Actions artifacts; each job
  writes `results.json` and `trials.json`.
- Run: https://github.com/kimphuctran/MESA_SSL/actions/runs/45194460384

**Scope:** The published work is not yet a state-of-the-art claim; the planned
follow-on will investigate grouped-augmentation ridge and chart-constrained
fallback with additional untouched datasets.
