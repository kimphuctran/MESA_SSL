# MESA-SSL shape-targeted mechanism loss: matched five-dataset study

Date 23 November 2026.
Source branch: `mesa-shape-mechanism-2026`
Official GitHub Actions run: https://github.com/kimphuctran/MESA_SSL/actions/runs/45270985220
Executed source revision: `6af79136c95d2bd8b707c8ff31e42e8186f3f539`

## Controlled structural change

Earlier gauge-centered encoders were often identical to a generic counterpart
when the classifier selected only the shape feature map. Their mechanism loss
mainly affected the raw stream and concept head, leaving no substantial
gradient through the shape stream. This experiment adds a **separately trained
shape-concept head**, fitted on exact low harmonic projections of the
affine-canonical residual, with a matched equivariance objective under a
predeclared harmonic intervention. The generic branch has the same
architecture and augmentation draws with mechanism loss weights zero.

This is not proof of causal mechanisms, but it provides direct gradient
pressure and testable concept alignment in *both* neural streams.
Both the previous raw concept targets and the new harmonic shape targets
are analytic signal summaries.

## Label protocol

Official UCR/UEA TRAIN/TEST splits, 5 labeled TRAIN signals per class,
12 epochs pretraining, seeds 2601 and 2602, and matched TRAIN-only
leave-one-label-out ridge selection. Development tasks GunPoint and
SyntheticControl; additional datasets Ham, Car and BirdChicken were
not used to construct or tune this variant. Report the individual
classification modes (clean, synthetic affine and synthetic jitter)
separately; they do not represent real-world causal interventions.
No TEST labels were used in training or hyperparameter decisions.

## Individual latent encoder results (mean of two seeds)

| Dataset | Generic SSL clean | MESA SSL clean | Generic SSL jitter | MESA SSL jitter |
|---|---:|---:|---:|---:|
| BirdChicken | 0.798 | 0.798 | 0.773 | 0.798 |
| Car | 0.555 | 0.555 | 0.561 | 0.571 |
| GunPoint | 0.683 | 0.750 | 0.625 | 0.706 |
| Ham | 0.413 | 0.509 | 0.392 | 0.515 |
| SyntheticControl | 0.952 | 0.928 | 0.942 | 0.920 |

## Group means (each dataset equally weighted)

| Study group | Generic clean | MESA clean | Generic jitter | MESA jitter |
|---|---:|---:|---:|---:|
| all | 0.680 | 0.708 | 0.659 | 0.702 |
| development | 0.817 | 0.839 | 0.783 | 0.813 |
| holdout | 0.589 | 0.621 | 0.575 | 0.628 |

## Ablation and concept fidelity

Average concept errors from unlabelled TRAIN:

- Raw concepts: Generic 0.35093 versus MESA 0.11737.
- Shape harmonic concepts: Generic 0.21116 versus MESA 0.07735.

MESA compared with matched Generic learned-only:
- BirdChicken: clean Δ 0.0000, jitter Δ 0.0247.
- Car: clean Δ 0.0001, jitter Δ 0.0094.
- GunPoint: clean Δ 0.0671, jitter Δ 0.0815.
- Ham: clean Δ 0.0959, jitter Δ 0.1234.
- SyntheticControl: clean Δ -0.0232, jitter Δ -0.0216.

## Cautions

Mechanism-related losses improve adherence to chosen signal concepts
across the measured datasets, and sometimes improve representation
classification. The gains do not automatically propagate through the
train-only fusion selector: on many tasks the classifier chooses to
use MultiRocket alone. This result supports attribution of a learned
representation difference but **does not establish global superiority**
over TS2Vec, TimeMAE, MOMENT or powerful time-series feature transforms.
Two seeds are exploratory, so no formal statistical significance is claimed.
Related seasonal/trend decompositions predate this study; the research
claim must rest on measured mechanism fidelity plus robust decision policy.

## Files

- `MESA_SSL/shape_mechanism_5datasets_2026-11-23.json`:
  all 90 run-level results and source provenance.
- `MESA_SSL/shape_mechanism_summary_2026-11-23.csv`:
  per-dataset means and sample SD.
- `MESA_SSL/mesa_dualpath_ssl.py`: evaluated neural architecture.
- `.github/workflows/mesa-shape-mechanism-validation.yml`: executed CI.
