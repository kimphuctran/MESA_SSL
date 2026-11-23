# MESA-SSL | Bayesian augmentation and chart-controller audit

**Completed:** 23 November 2026.
**GitHub Actions:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45196162840
**Code:** https://github.com/kimphuctran/MESA_SSL/blob/mesa-augmented-bo-2026/MESA_SSL/mesa_bo_augmentation.py
**Branch:** `mesa-augmented-bo-2026` (independent from `master`).

## Scientific protocol

Official UCR/UEA TRAIN/TEST, two seeds (901–902), five labelled TRAIN cases per
class and 15 epochs of fixed unlabeled SSL pretraining. Bayesian optimization:
48 multivariate Optuna-TPE trials to tune group-augmentation strengths and
ridge penalty, then 48 trials each for three chart-controlled fallbacks
(canonical, generic SSL, mechanism MESA). Total 4,608 TPE trials across 12
datasets and two seeds, yielding 240 final model evaluations.

**Grouped cross-validation:** Each left-out labelled TRAIN case is removed
together with both augmented versions when training the ridge predictor.
An exact group-LOO identity is asserted against explicit group deletion
at numerical precision in CI. This prevents inflated tuning scores caused
by a label duplicate appearing in a training augmentation.

The chart uses shrinkage-covariance mean/slope monitoring, with split-
conformal calibration on held-out *unlabelled TRAIN*. TEST features and
labels do not determine hyperparameters or the final `TRAIN-selected`
controller; metrics are recorded only after selection. No ARL0 guarantees
or non-affine robustness claims follow from this experiment.

## Shifted Macro-F1 (two-seed mean)

| Dataset | Raw shift | Canonical shift | RawAug fixed shift | RawAug BO shift | BO-Aug generic shift | BO-Aug MESA shift | TRAIN-selected shift
---|---:|---:|---:|---:|---:|---:|---:
| GunPoint | 0.785 | 0.903 | 0.903 | 0.903 | 0.910 | 0.910 | 0.903 |
| ItalyPowerDemand | 0.894 | 0.933 | 0.906 | 0.903 | 0.909 | 0.908 | 0.907 |
| BasicMotions | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Coffee | 0.925 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| CBF | 0.926 | 0.960 | 0.967 | 0.961 | 0.966 | 0.961 | 0.961 |
| ECGFiveDays | 0.817 | 0.916 | 0.890 | 0.864 | 0.884 | 0.890 | 0.893 |
| Trace | 0.953 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SyntheticControl | 0.885 | 0.630 | 0.878 | 0.876 | 0.913 | 0.883 | 0.883 |
| Lightning2 | 0.513 | 0.572 | 0.611 | 0.565 | 0.623 | 0.590 | 0.623 |
| Meat | 0.561 | 0.917 | 0.841 | 0.857 | 0.899 | 0.925 | 0.917 |
| GunPointAgeSpan | 0.736 | 0.938 | 0.842 | 0.894 | 0.945 | 0.943 | 0.943 |
| DistalPhalanxOutlineCorrect | 0.649 | 0.705 | 0.685 | 0.693 | 0.687 | 0.677 | 0.677 |

## Aggregates


### all_12

| Method | Clean Macro-F1 | Shift Macro-F1 |
|---|---:|---:|
| Raw MultiRocket | 0.898 | 0.802 |
| Canonical MultiRocket | 0.871 | 0.873 |
| Canonical + generic SSL | 0.869 | 0.868 |
| Canonical + MESA SSL | 0.868 | 0.868 |
| RawAug-fixed | 0.899 | 0.877 |
| RawAug-BO | 0.895 | 0.876 |
| BO-Aug-SPC canonical | 0.897 | 0.893 |
| BO-Aug-SPC generic | 0.898 | 0.895 |
| BO-Aug-SPC MESA | 0.896 | 0.891 |
| BO-Aug TRAIN-selected | 0.897 | 0.892 |

### common_9

| Method | Clean Macro-F1 | Shift Macro-F1 |
|---|---:|---:|
| Raw MultiRocket | 0.914 | 0.853 |
| Canonical MultiRocket | 0.878 | 0.879 |
| Canonical + generic SSL | 0.879 | 0.878 |
| Canonical + MESA SSL | 0.878 | 0.877 |
| RawAug-fixed | 0.920 | 0.906 |
| RawAug-BO | 0.911 | 0.897 |
| BO-Aug-SPC canonical | 0.912 | 0.905 |
| BO-Aug-SPC generic | 0.914 | 0.912 |
| BO-Aug-SPC MESA | 0.912 | 0.905 |
| BO-Aug TRAIN-selected | 0.914 | 0.908 |

### new_3

| Method | Clean Macro-F1 | Shift Macro-F1 |
|---|---:|---:|
| Raw MultiRocket | 0.852 | 0.649 |
| Canonical MultiRocket | 0.852 | 0.853 |
| Canonical + generic SSL | 0.838 | 0.837 |
| Canonical + MESA SSL | 0.839 | 0.838 |
| RawAug-fixed | 0.838 | 0.789 |
| RawAug-BO | 0.845 | 0.814 |
| BO-Aug-SPC canonical | 0.853 | 0.855 |
| BO-Aug-SPC generic | 0.849 | 0.844 |
| BO-Aug-SPC MESA | 0.848 | 0.848 |
| BO-Aug TRAIN-selected | 0.847 | 0.846 |

## Interpretation

- Bayesian optimization and augmented feature ridge yield context-specific
  improvements but **cannot be called universally superior**.
- The clean/raw and shifted/canonical performance tradeoff remains. A
  mechanism encoder is not consistently selected by TRAIN-only validation.
- On Meat, BO-Aug-SPC MESA improves shifted Macro-F1 above canonical in this
  two-seed trial, but the TRAIN-selected policy chooses a different controller.
- On CBF, a *fixed* augmentation baseline rivals or outperforms more complex
  tuned controllers, a warning against high-dimensional tuning with 5 labels
  per class.
- On SyntheticControl, canonical projection eliminates discriminative signal
  variation (clean Macro-F1 0.630 versus raw 0.934). Optimizing a component
  cannot recover information removed before classification.
- Three additional datasets (Meat, GunPointAgeSpan,
  DistalPhalanxOutlineCorrect) evaluate the revised method with no
  parameter selection based on their TEST labels.
- Paired task-wise statistical inference with more seeds, genuine process
  streams, and full official TS2Vec/TimeMAE/MOMENT comparisons is required
  before preparing a superiority claim for Neurocomputing.

## Relation to first Bayesian iteration

The first 9-dataset TPE study
(https://github.com/kimphuctran/MESA_SSL/actions/runs/45194460384)
achieved a nine-task equal-dataset mean shifted Macro-F1 around 0.912
for its TRAIN-selected controller. The group-augmentation controller should
be compared **only on the same nine datasets and seeds**, not against the
12-dataset mean as those have different task compositions.

First Bayesian shifted mean = 0.912172; grouped-augmentation TRAIN-selected shifted mean on the nine shared tasks = 0.907792.

These are descriptive two-seed comparisons, not significance tests.

## Files

- `MESA_SSL/augmented_bayes_12dataset_2026-11-23.json`: 240 full evaluations
- `MESA_SSL/augmented_bayes_summary_2026-11-23.csv`: all 10 methods, clean and shifted means/SD
- `MESA_SSL/mesa_bo_augmentation.py`: source with grouped LOO verification
- Actions run has each dataset's `results.json`, `trials.json`,
  `requirements.txt` and fixed source files.

**Recommendation:** retain the first controller-only Bayesian experiment as
a candidate for stronger development comparison; do not merge this augmented
branch into the submitted code without independent large-scale validation.
