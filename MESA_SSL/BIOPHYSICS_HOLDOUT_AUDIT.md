# MESA-BioPhysics | Independent 1-, 3- and 5-shot verification

**Completed:** 23 November 2026.
**Repository:** https://github.com/kimphuctran/MESA_SSL
**Branch:** `mesa-biophysics-metaplastic-2026`
**Run:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45279440498
**Evaluated source revision:** `d798fd383bcce86529e62c0055ba687d7c356414`
**Jobs:** 10/10 successful; five datasets × two seeds × three
label budgets × 18 methods = **540 final evaluations**.

## Protocol

Previously unused UCR tasks in this project:
BME, ECG200, MoteStrain, ShapeletSim, SonyAIBORobotSurface1.
Five labeled TRAIN observations per class is the upper support size.
1-, 3- and 5-shot subsets are **nested** within each dataset/seed.
Unlabeled TRAIN is used for the slow cortical encoder, fixed physics
features, Gaussian standardization, heat-diffusion signatures,
AR(2) predictive code, sparse expansion, and chart calibration.

Every support memory contains only examples with the corresponding
labeled TRAIN IDs. No TEST labels/features are used to select
hyperparameters, experts or weights. Affine and waveform-corrupted TEST
signals reuse the original targets **only by a declared synthetic
class-preservation assumption**.

The conservative metaplastic selector uses grouped
leave-one-labeled-episode-out classification for 3/5-shot, excluding
each target support example from all class memories. At 1-shot,
class-preserving LOO is impossible, so it always selects the
predeclared Dual MultiRocket episodic reference. This avoids invalid
one-shot cross-validation.

## Equal-dataset mean Macro-F1

| Method | 1-shot clean | 3-shot clean | 5-shot clean | 5-shot affine | 5-shot jitter |
|---|---:|---:|---:|---:|---:|
| Raw MR episodic | 0.730 | 0.846 | 0.902 | 0.894 | 0.773 |
| Dual MR episodic | 0.737 | 0.854 | 0.914 | 0.917 | 0.788 |
| Slow cortex episodic | 0.629 | 0.708 | 0.745 | 0.657 | 0.743 |
| DG episodic | 0.670 | 0.794 | 0.835 | 0.832 | 0.707 |
| Physics episodic | 0.602 | 0.730 | 0.803 | 0.746 | 0.679 |
| Hopfield physics episodic | 0.602 | 0.733 | 0.800 | 0.746 | 0.669 |
| BioPhysics no physics | 0.729 | 0.816 | 0.884 | 0.874 | 0.821 |
| BioPhysics full | 0.701 | 0.810 | 0.868 | 0.843 | 0.744 |
| BioPhysics safe TRAIN | 0.737 | 0.827 | 0.877 | 0.862 | 0.838 |

## Five-shot dataset-level comparison

| Dataset | Dual clean | Safe clean | Dual affine | Safe affine | Dual jitter | Safe jitter |
|---|---:|---:|---:|---:|---:|---:|
| BME | 0.957 | 0.866 | 0.977 | 0.888 | 0.442 | 0.707 |
| ECG200 | 0.797 | 0.761 | 0.807 | 0.701 | 0.777 | 0.735 |
| MoteStrain | 0.876 | 0.820 | 0.863 | 0.784 | 0.793 | 0.814 |
| ShapeletSim | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SonyAIBORobotSurface1 | 0.938 | 0.939 | 0.938 | 0.938 | 0.930 | 0.932 |

## Findings and negative evidence

1. At five labels per class, MESA BioPhysics **safe TRAIN** achieves
   clean Macro-F1 0.877331, below Dual MR episodic
   0.913607 (difference -0.036276).
   Under synthetic jitter its mean Macro-F1 is 0.837721
   versus 0.788466 (difference 0.049255).
   The gain in jitter robustness therefore comes with a
   **substantial accuracy tradeoff** on clean and affine data.
2. Pure Hopfield/physics retrieval and full fixed-weight
   physics fusion often underperform simpler raw/dual models.
   More biology-inspired components do **not** imply improved
   performance without stable class-relevant representation.
3. The ablation without physics sometimes outperforms the full
   fixed mixture. This is empirical evidence against making
   universal conservation/physics claims in unrelated UCR datasets.
4. On ShapeletSim most relevant baselines approach 1.0 after
   three labels/class, so the task leaves little headroom.
   On BME and MoteStrain, support selection is especially unstable.
5. With 1-shot, the safe controller reverts to a simple and auditable
   dual-MR episodic policy. That is a **methodological safety
   property**, not an improvement beyond that baseline.
6. The physical contract and predicted signal error are precisely
   defined, but should be called **algorithmic priors** rather than
   proof of physical law or neural realism.
7. Only two seeds per dataset and the compact
   336-kernel MultiRocket baseline were used. No matched
   official TS2Vec/TimeMAE/MOMENT run is included here. There
   is no scientific basis for claiming universal SOTA dominance.

## Consequence for the next iteration

The fixed mixture is not a good final classifier. Its
unconditional physics/Hopfield weight can lower clean accuracy
where those signatures carry little class information.
A scientifically defensible follow-on would introduce a
*predeclared, nuisance-conditional* supervisor that falls
back to the raw/dual episodic model under normal conditions,
and calls specialized memories only for nuisance families
where they have train-only support. The current five-dataset
evaluation must then be regarded as developmental evidence
for that follow-on, and a distinct untouched test set is
necessary.

## Files and provenance

- Source: MESA_SSL/mesa_biophysics_memory.py
- Mathematical contract: MESA_SSL/BIOPHYSICS_ARCHITECTURE.md
- Full data: MESA_SSL/biophysics_holdout_2026-11-23.json
- Every method, dataset, budget, and seed summary with sample SD:
  MESA_SSL/biophysics_holdout_summary_2026-11-23.csv
- All selected experts:
  MESA_SSL/biophysics_selection_audit_2026-11-23.csv
- Full GitHub Actions:
  https://github.com/kimphuctran/MESA_SSL/actions/runs/45279440498

**Status: completed independent validation with mixed and often
negative findings. Not an established state-of-the-art model.**
