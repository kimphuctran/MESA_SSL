# Direct Bayesian Optimization of MESA-SSL Encoders
Research audit | 23 November 2026

**Repository:** https://github.com/kimphuctran/MESA_SSL
**Experiment branch:** `mesa-encoder-hpo-2026`
**Completed GitHub Actions run:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45217129377
**Evaluated source revision:** `4de2ad68cd2201530154e5a18a1b8706a544bd7f`
**Run integrity:** 16/16 jobs successful; 192 final TEST evaluations and 32 selected configurations.

## Objectives and experimental design

The parameter search directly optimizes self-supervised encoder training rather
than restricting optimization to the classifier. Optuna multivariate TPE uses
18 trials separately for mechanism-conditioned MESA and matched generic SSL per
dataset and seed. Eight official UCR/UEA datasets were examined with seeds
1521/1522, yielding 576 optimization trials; each final classifier uses exactly
five labeled TRAIN observations per class.

The joint search domain comprises learning rate (0.0002–0.003), representation
dimension (48/96/144), masking ratio (0.08–0.30), Gaussian input noise
(0.01–0.12), harmonic augmentation strength (0.04–0.35), total epochs
(8/12/16), warmup epochs (0–3), and constant/linear/cosine learning-rate
schedules. MESA additionally tunes concept regression weight (0.08–2)
and equivariance weight (0.06–1.6); generic SSL has both weights fixed to
zero. The core encoder width is fixed at 48 to isolate the requested choices.

The study objective uses exact leave-one-label-out TRAIN ridge predictions,
a margin tie breaker, and a prespecified TRAIN-only jitter view.
Ridge strength and fusion weights for evaluation are chosen on labeled TRAIN
only. The official TEST portion is never used to tune encoder parameters,
fusion, chart thresholds, or classifier settings.

The comparison includes raw and affine-canonical MultiRocket, fixed and
HPO generic/MESA SSL feature-only baselines, each fused with MultiRocket,
and a fixed train-calibrated Hotelling-type T² chart that switches from raw
to invariant representation. Three TEST modes are recorded separately:
clean, synthetic additive offset/trend, and synthetic jitter plus harmonic
interference. Synthetic perturbations are *assumed*, rather than validated,
to preserve true classification labels.

## Dataset results (Macro-F1 mean of two seeds)

| Dataset | Fixed MESA clean | HPO MESA clean | HPO Generic clean | Canonical clean | HPO MESA jitter | HPO Generic jitter |
|---|---:|---:|---:|---:|---:|---:|
| BasicMotions | 0.842 | 0.748 | 0.849 | 0.987 | 0.712 | 0.792 |
| CBF | 0.924 | 0.915 | 0.921 | 0.955 | 0.893 | 0.905 |
| ECGFiveDays | 0.904 | 0.912 | 0.865 | 0.929 | 0.813 | 0.830 |
| GunPoint | 0.693 | 0.701 | 0.719 | 0.774 | 0.683 | 0.635 |
| GunPointAgeSpan | 0.842 | 0.818 | 0.815 | 0.879 | 0.769 | 0.800 |
| ItalyPowerDemand | 0.932 | 0.934 | 0.952 | 0.952 | 0.874 | 0.889 |
| Meat | 0.729 | 0.616 | 0.613 | 0.891 | 0.501 | 0.420 |
| SyntheticControl | 0.581 | 0.576 | 0.583 | 0.632 | 0.558 | 0.564 |

## Equal-dataset-weight mean Macro-F1

| Method | Mean clean Macro-F1 | Mean affine Macro-F1 | Mean jitter Macro-F1 |
|---|---:|---:|---:|
| Raw MultiRocket | 0.920 | 0.832 | 0.570 |
| Canonical MultiRocket | 0.875 | 0.875 | 0.651 |
| Fixed generic SSL-only | 0.802 | 0.802 | 0.711 |
| Fixed MESA SSL-only | 0.806 | 0.806 | 0.705 |
| HPO generic SSL-only | 0.790 | 0.790 | 0.729 |
| HPO MESA SSL-only | 0.777 | 0.777 | 0.725 |
| Fixed generic + MultiRocket | 0.872 | 0.872 | 0.645 |
| Fixed MESA + MultiRocket | 0.876 | 0.876 | 0.660 |
| HPO generic + MultiRocket | 0.872 | 0.872 | 0.647 |
| HPO MESA + MultiRocket | 0.870 | 0.870 | 0.645 |
| SPC HPO generic | 0.911 | 0.871 | 0.609 |
| SPC HPO MESA | 0.912 | 0.869 | 0.610 |

## Paired-dataset audit of encoder optimization

- HPO MESA minus fixed MESA, clean Macro-F1: mean -0.02858, improvements on 3/8 datasets.
- HPO MESA minus fixed MESA, jitter Macro-F1: mean 0.02047, improvements on 4/8 datasets.
- HPO MESA minus independently HPO generic SSL, clean Macro-F1: mean -0.01216, improvements on 3/8 datasets.
- HPO MESA minus independently HPO generic SSL, jitter Macro-F1: mean -0.00424, improvements on 2/8 datasets.

The per-dataset results are descriptive. Two seeds are insufficient to
make a statistically rigorous superiority claim. Repeated optimization
over a small labeled set can overfit leave-one-out estimates. Some fused
models select a **zero MESA weight**, which is crucial for attributing
performance to pretrained features rather than existing MultiRocket.

## Selection diagnostics

MESA selected representation dimensions:
{"48":4,"96":6,"144":6}.
MESA selected epoch budgets: {"8":8,"12":5,"16":3}.
MESA selected schedules: {"cosine":3,"linear":10,"constant":3}.
Generic selected representation dimensions:
{"48":5,"96":3,"144":8}.
Generic selected schedules: {"constant":5,"cosine":5,"linear":6}.

The MESA-specific loss coefficients vary by task and seed and should
not be presented as a universal optimal parameter vector.
A follow-on paired experiment evaluates the same MESA-selected
hyperparameters with both mechanism loss weights set to zero and
identical stochastic augmentation streams; this separates mechanism
attribution from differences in optimizer-selected configurations.

## Reproducibility

- Python experiment: `MESA_SSL/mesa_encoder_hpo.py`.
- Executed workflow: `.github/workflows/mesa-encoder-hpo.yml`.
- All final evaluations: `MESA_SSL/encoder_hpo_results_2026-11-23.json`.
- Means and sample standard deviations: `MESA_SSL/encoder_hpo_summary_2026-11-23.csv`.
- Selected configurations with TRAIN objectives:
  `MESA_SSL/encoder_hpo_selected_2026-11-23.json` and
  `MESA_SSL/encoder_hpo_selected_2026-11-23.csv`.
- Full Optuna trial histories and version-locked environments:
  `encoder_trials.json` and `requirements.txt` inside each of 16
  GitHub Actions job artifacts.

**Scientific decision:** This is a completed development benchmark, not
evidence that the optimized MESA encoder universally surpasses
MultiRocket, TS2Vec, TimeMAE, MOMENT, or a matched generic encoder.
Do not merge the experimental branch into the main manuscript code
without broader independent benchmarking and mechanism ablation.
