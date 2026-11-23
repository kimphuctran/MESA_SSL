# MESA-SSL: mechanism-aware kernel fusion and robustness selection

Research audit — 23 November 2026. This is an experimental development record, not a claim that MESA-SSL universally outperforms established self-supervised or time-series methods.

## Motivation and method

MESA-MK combines four explicitly identified views: (i) original time-series MultiRocket features, (ii) affine-canonical shape MultiRocket features, (iii) measured mean, slope and log-residual-RMS concepts, and (iv) a self-supervised multiresolution MESA encoder. The kernel is an additive, nonnegative sum of group kernels, allowing exact, inspectable decomposition of class scores by group.

The matched ablation trains the same encoder architecture **without** mechanism equivariance. MultiRocket is the existing method by Tan, Dempster, Bergmeir and Webb (2022), *Data Mining and Knowledge Discovery* 36:1623–1646, DOI [10.1007/s10618-022-00844-1](https://doi.org/10.1007/s10618-022-00844-1). MultiRocket feature extraction is not a new contribution of this paper.

A first model chooses the clean-data kernel using leave-one-out predictions on labelled TRAIN only. A second model chooses a robust kernel from leave-one-out scores on two synthetically shifted labelled TRAIN views, excluding the original instance from its own training reference in each fold. The unlabelled concept-distribution shift determines a per-sample interpolation between the two predictors. No TEST labels are used to select the mixture, classifiers, seed, augmentation, or regularisation. Synthetic additive offsets and linear trends are **assumed to preserve target labels** for this stress evaluation.

## Evaluation design

Datasets: GunPoint, ItalyPowerDemand, BasicMotions (development seeds 304–306); Coffee, CBF, ECGFiveDays (additional untouched dataset check, seeds 307–309). In all cases the official aeon UCR/UEA TRAIN and TEST splits were used. Five labelled TRAIN observations per class; SSL encoder pretraining uses only unlabelled TRAIN observations and 25 epochs. Other observation counts are in the machine-readable results.

| Dataset | Raw MultiRocket clean | Raw shifted | MESA robust clean | MESA robust shifted | Canonical MultiRocket shifted |
|---|---:|---:|---:|---:|---:|
| GunPoint | 0.969 | 0.821 | 0.969 | **0.960** | **0.960** |
| ItalyPowerDemand | 0.912 | 0.880 | **0.914** | 0.902 | **0.910** |
| BasicMotions | **1.000** | 0.983 | **1.000** | 0.983 | **0.983** |
| Coffee | **0.964** | 0.820 | **0.964** | 0.916 | **0.952** |
| CBF | **0.977** | 0.927 | 0.977 | **0.958** | 0.956 |
| ECGFiveDays | 0.901 | 0.827 | 0.902 | 0.878 | **0.912** |

All values are Macro-F1 means across **three independent seeds**, rounded to three decimals. Bolded values represent selected within-row comparisons, **not a global SOTA ranking**. The three new dataset checks were not used for designing the candidate selection policy. Results are conditional on the chosen 5-label protocol and affine perturbation family.

The shift-aware policy improves shifted Macro-F1 relative to raw MultiRocket on five of six datasets and ties on BasicMotions, but it **does not always exceed canonical MultiRocket**. A canonical-only representation achieves higher shifted Macro-F1 on ItalyPowerDemand, Coffee, and ECGFiveDays.

## Attribution to the MESA mechanism loss

The exact same inference policy applied to a generic SSL encoder with no mechanism objective is a matched control. The MESA mechanism encoder was selected by TRAIN-only criteria in only **one out of 18 dataset–seed cases**. Its observed incremental gain relative to the generic SSL control is therefore limited and dataset-dependent. The present experiments support an improvement in *governed robustness under declared nuisance transformations*, not a broad improvement in learned representation quality.

The additive class-score decomposition is exact by construction but does not itself establish that a statistical concept is a causal physical mechanism. To make a strong journal-level claim about MESA rather than the adapted MultiRocket control, independent concept references and intervention metadata will be needed alongside matched performance and explanation metrics.

## Reproducibility

- [Original multi-kernel experiment source](./mesa_multikernel.py)
- [Robust augmented TRAIN-LOO selection source](./mesa_robust_select.py)
- [90 machine-readable outcome records](./robust_multikernel_6dataset_3seed.json)
- [Compact result table](./robust_multikernel_summary.csv)
- [First multi-kernel study](https://github.com/kimphuctran/MESA_SSL/actions/runs/45169204995)
- [TRAIN-only robustness development](https://github.com/kimphuctran/MESA_SSL/actions/runs/45169860386)
- [Independent Coffee, CBF, ECGFiveDays checks](https://github.com/kimphuctran/MESA_SSL/actions/runs/45170164555)
- [Numerical contract tests](https://github.com/kimphuctran/MESA_SSL/actions/workflows/mesa-mk-tests.yml)

All code in this study is on branch `mesa-multikernel-experiment-2026`, leaving the main submitted-manuscript branch untouched until the method is scientifically validated.
