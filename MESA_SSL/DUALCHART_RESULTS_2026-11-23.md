# MESA Dual-Chart SSL: Selective Nuisance Monitoring and Mechanism-Grounded Experts

**Date:** 23 November 2026  
**Project:** https://github.com/kimphuctran/MESA_SSL  
**Evaluation branch:** `mesa-dualchart-adaptive-2026`  
**Completed run:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45271961659  
**Evaluated source commit:** `cc4f48d3f654784434a2548890c1a5aa439cb516`  
**Audit status:** 14/14 successful GitHub Actions jobs, 154 final records.

## Abstract of the method

MESA Dual-Chart retains the raw time series, its affine-canonical shape, and
the exact per-channel mean/slope/residual-scale concepts. A multiscale
dual-tower CNN separately encodes raw and residual streams. Under the
mechanism-conditioned treatment, two concept-prediction heads are optimized:
the raw head learns mean/slope/scale under known affine interventions and
the shape head learns harmonic projections and their intervention responses.
A matched generic encoder follows the identical training protocol, with
mechanism objective weights zero.

Two TRAIN-calibrated multivariate control charts then monitor **different**
nuisance families: a shrinkage mean/slope chart (affine offsets/trends) and
a morphology chart based on differences, curvature, high-frequency power,
lag correlation and amplitude quantiles (waveform disturbances).
On alert, a deterministic two-stage policy switches among expert classifiers
selected using only labeled TRAIN and leave-one-label-out classification.
Roughness alarms take precedence over mean/slope alarms.

This is not a novel claim about simply splitting raw and trend components
(which earlier work has studied); the testable contribution is **conditional
nuisance identification, mechanism-targeted representations and governed
expert selection** under few labels.

## Data and selection protocol

Official UCR/UEA TRAIN/TEST partitions, 5 labeled TRAIN cases/class,
12 epochs of SSL pretraining, two seeds 2601 and 2602 per dataset,
same architecture and runtime budget for matched generic and MESA arms.
Dataset groupings were fixed before TEST evaluation:

- Development: Fish, GunPoint, SyntheticControl.
- New evaluation: Ham, Car, BirdChicken, Lightning7.

Synthetic interventions are additive offset/trend and Gaussian/harmonic
waveform jitter with fixed predeclared parameters. The original test labels
are reused under a **label-preserving assumption**, which is not a physical
or clinical ground truth. No TEST label chooses experts, chart thresholds,
fusion weights or ridge regularization. The finite UCR TEST cases are not a
chronological process: false alarm *rates on these cases* do not estimate
ARL0 or detection delay of a sequential chart.

## Macro-F1 for each dataset (mean of 2 seeds)

| Dataset | Raw clean | MESA dual-chart clean | Raw affine | MESA dual-chart affine | Raw jitter | MESA dual-chart jitter |
|---|---:|---:|---:|---:|---:|
| BirdChicken | 0.924 | 0.975 | 0.924 | 0.975 | 0.333 | 0.798 |
| Car | 0.859 | 0.858 | 0.777 | 0.858 | 0.089 | 0.571 |
| Fish | 0.842 | 0.846 | 0.775 | 0.846 | 0.072 | 0.472 |
| GunPoint | 0.912 | 0.906 | 0.823 | 0.875 | 0.605 | 0.631 |
| Ham | 0.423 | 0.438 | 0.433 | 0.438 | 0.482 | 0.496 |
| Lightning7 | 0.688 | 0.708 | 0.587 | 0.656 | 0.596 | 0.639 |
| SyntheticControl | 0.931 | 0.968 | 0.866 | 0.955 | 0.921 | 0.969 |

## Mean scores weighted equally by dataset

### development

| Method | Clean Macro-F1 | Affine Macro-F1 | Jitter Macro-F1 |
|---|---:|---:|---:|
| Raw MultiRocket | 0.895 | 0.821 | 0.533 |
| Canonical MultiRocket | 0.794 | 0.794 | 0.417 |
| Dual MultiRocket | 0.915 | 0.876 | 0.546 |
| Generic learned-only | 0.744 | 0.640 | 0.675 |
| MESA learned-only | 0.757 | 0.622 | 0.699 |
| Generic dual-path hybrid | 0.911 | 0.870 | 0.571 |
| MESA dual-path hybrid | 0.909 | 0.867 | 0.576 |
| MESA SPC hybrid | 0.898 | 0.874 | 0.536 |
| MESA dual-chart | 0.907 | 0.892 | 0.691 |
| Adaptive dual-chart | 0.908 | 0.892 | 0.682 |
| TRAIN-selected admissible | 0.909 | 0.870 | 0.571 |

### new_holdout

| Method | Clean Macro-F1 | Affine Macro-F1 | Jitter Macro-F1 |
|---|---:|---:|---:|
| Raw MultiRocket | 0.723 | 0.680 | 0.375 |
| Canonical MultiRocket | 0.719 | 0.717 | 0.382 |
| Dual MultiRocket | 0.735 | 0.710 | 0.383 |
| Generic learned-only | 0.597 | 0.573 | 0.588 |
| MESA learned-only | 0.620 | 0.605 | 0.617 |
| Generic dual-path hybrid | 0.725 | 0.721 | 0.382 |
| MESA dual-path hybrid | 0.725 | 0.719 | 0.382 |
| MESA SPC hybrid | 0.723 | 0.715 | 0.377 |
| MESA dual-chart | 0.745 | 0.732 | 0.626 |
| Adaptive dual-chart | 0.736 | 0.732 | 0.613 |
| TRAIN-selected admissible | 0.712 | 0.707 | 0.382 |

### all

| Method | Clean Macro-F1 | Affine Macro-F1 | Jitter Macro-F1 |
|---|---:|---:|---:|
| Raw MultiRocket | 0.797 | 0.741 | 0.443 |
| Canonical MultiRocket | 0.751 | 0.750 | 0.397 |
| Dual MultiRocket | 0.812 | 0.781 | 0.453 |
| Generic learned-only | 0.660 | 0.602 | 0.625 |
| MESA learned-only | 0.678 | 0.612 | 0.652 |
| Generic dual-path hybrid | 0.805 | 0.785 | 0.463 |
| MESA dual-path hybrid | 0.804 | 0.782 | 0.465 |
| MESA SPC hybrid | 0.798 | 0.783 | 0.445 |
| MESA dual-chart | 0.814 | 0.800 | 0.654 |
| Adaptive dual-chart | 0.810 | 0.801 | 0.643 |
| TRAIN-selected admissible | 0.797 | 0.777 | 0.463 |

## Independent group, effect size and attribution

Over four previously unused tasks, MESA Dual-Chart versus Raw MultiRocket:
clean Macro-F1 difference 0.021341, affine
difference 0.051486, waveform-jitter difference
0.250652. These are descriptive task means, not
statistically validated population improvements.

On Fish (development set), the morphology chart detects the synthetic
waveform disturbance and switches to mechanism-conditioned neural
embeddings, avoiding the collapse of raw/canonical MultiRocket features.
On Car and BirdChicken (new checks), the same structure yields considerable
jitter improvements. On Ham and Lightning7, benefits are smaller and
sometimes the selected expert remains an existing MultiRocket or hybrid
classifier. A policy may select a generic or non-neural expert; method names
must not be mistaken for causal attribution of every score to MESA itself.

Compared with *Adaptive dual-chart* (allowed to select generic experts),
the MESA-restricted policy achieves higher average jitter Macro-F1 on
this particular new group. The matched MESA learned-only encoder also
outperforms its generic counterpart on the grouped means, but not
every individual task or seed.

### Learning from failures

1. The original MESA canonicalization discarded mean/slope information
   which was class-relevant on SyntheticControl. The paired raw stream
   and exact physical-statistic skip preserve it algebraically.
2. Earlier mechanism loss had no direct gradient path to a shape-only
   decision branch. The added harmonic shape-concept objective improves
   feature differentiation and analytical concept prediction.
3. A single mean/slope SPC chart missed waveform perturbations on Fish
   and Car. The second morphology chart provides a distinct nuisance test.
4. Small labeled sets can choose the wrong expert (e.g., Ham);
   outcome tables show the actually TRAIN-selected policy, not the
   post-hoc best TEST member of the family.
5. Neural contributions may be selected with weight zero in a hybrid.
   Attribution to learned features requires reading each selected kernel
   weight from the raw records.

## Mathematical contracts and limits

Writing a sample as x(t)=μ+βτ(t)+σq(t), where q is the centered,
detrended and unit-residual-RMS series, yields exact recovery of x
from (q,μ,β,σ). Adding a constant a and linear trend bτ changes
only μ and β; q remains unchanged (up to floating-point error).
The mechanism-concept heads are learned approximations to
predeclared statistical summaries, **not verified real-world causal
mechanisms**.

For exchangeable in-control test examples, a split-conformal
p-value is (1 + number of calibration scores at least as large as
the test score)/(n_cal+1). The marginal false-alarm statement
does not extend automatically to dependent streams or dynamic
recalibration, nor does it guarantee Macro-F1 dominance.

## Comparison boundaries

- The MultiRocket extraction in this experiment is a **compact
  336-kernel configuration**. It is not a comparison to unrestricted
  default MultiRocket or Hydra+MultiRocket, nor a full SOTA audit.
- Existing SSL baselines TS2Vec, TimeMAE and MOMENT have been evaluated
  elsewhere in the MESA project, but were **not re-executed in this
  identical seven-task/two-seed protocol**.
- Four new tasks and two seeds remain limited; publication requires
  a larger untouched archive, more paired seeds, verified physical
  interventions, concept fidelity and execution-time analysis.
- Prior decomposition-aware SSL exists (e.g. ST-CL, IEEE Access 2025,
  DOI 10.1109/ACCESS.2025.3621317); the mere raw/shape split should
  not be claimed as unique.

## Reproducibility

- Full source:
  `MESA_SSL/mesa_dualpath_ssl.py`
- Full test records and TRAIN-selection metadata:
  `MESA_SSL/dualchart_7datasets_2026-11-23.json`
- All method/dataset means and sample standard deviations:
  `MESA_SSL/dualchart_summary_2026-11-23.csv`
- Control-chart alarms, thresholds and chosen experts:
  `MESA_SSL/dualchart_policy_audit_2026-11-23.csv`
- Workflow:
  `.github/workflows/mesa-dualchart-validation.yml`
- Each CI artifact contains versioned requirements, evaluated source,
  training curves and per-case metrics.

**Decision: this architecture is a promising robustness-focused
research candidate, NOT a verified universally superior
state-of-the-art self-supervised time-series method.**
