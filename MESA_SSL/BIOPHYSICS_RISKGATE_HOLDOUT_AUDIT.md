# MESA BioPhysics Risk-Gated | independent 1,3,5-shot audit

Date: 23 November 2026.
Repository: https://github.com/kimphuctran/MESA_SSL
Experimental branch: `mesa-biophysical-riskgate-2026`
Completed workflow: https://github.com/kimphuctran/MESA_SSL/actions/runs/45280823062
Evaluated source commit: `ebcad1f0c75b05681dc45d57a6d26e46877aaa95`

## Scientific question

Can a brain-inspired *complementary learning system* and physically
constrained signal representation improve one- or few-label time-series
classification without discarding class-relevant mean/trend and without
using official TEST labels to choose among its different memories?

The algorithm learns reusable features slowly on unlabeled TRAIN, stores
each labeled support example immediately, separates memories with k-WTA
sparsity, retrieves patterns with a two-step Hopfield-style update, extracts
stable leaky response, predictive innovations and Fourier heat-diffusion
signatures, and estimates affine and waveform nuisance via separately
train-calibrated control charts.

The **risk-gated** revision explicitly limits how specialized memories
are selected: at 1-shot it has predeclared clean/affine/waveform experts
and does not attempt invalid per-class LOO. At 3/5-shot it uses TRAIN
leave-one-labeled-episode-out comparisons, a predeclared complexity
penalty and improvement margin before changing experts. During
deployment a waveform chart alert takes priority over affine alarms.
No TEST label or TEST example enters threshold estimation or selection.

## Protocol integrity

- Official UCR/UEA TRAIN/TEST.
- Datasets evaluated for the first time with this revision:
  SonyAIBORobotSurface2, RefrigerationDevices,
  DistalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect.
- 2 seeds (5501,5502), nested 1/3/5 labeled TRAIN cases per class.
- SSL training: 8 epochs, common slow cortical backbone.
- 19 method/ablation arms, 8 successful independent jobs,
  456 final evaluations.
- Three reporting modes: clean, synthetic offset/trend, waveform
  Gaussian/harmonic jitter. Label preservation is an assumption of
  the synthetic evaluation, not a physical fact.
- No average run length or process detection delay is estimated.

## Equal-dataset mean Macro-F1

| Method | 1-shot clean | 1-shot jitter | 3-shot clean | 3-shot jitter | 5-shot clean | 5-shot affine | 5-shot jitter |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw MR episodic | 0.551 | 0.415 | 0.632 | 0.465 | 0.664 | 0.642 | 0.502 |
| Dual MR episodic | 0.553 | 0.415 | 0.631 | 0.468 | 0.662 | 0.659 | 0.487 |
| Slow cortex episodic | 0.490 | 0.478 | 0.600 | 0.592 | 0.650 | 0.539 | 0.639 |
| BioPhysics full | 0.532 | 0.490 | 0.631 | 0.611 | 0.657 | 0.639 | 0.640 |
| BioPhysics safe TRAIN | 0.553 | 0.415 | 0.640 | 0.575 | 0.661 | 0.638 | 0.580 |
| BioPhysics risk-gated | 0.538 | 0.511 | 0.623 | 0.596 | 0.652 | 0.632 | 0.622 |

## Five-shot dataset-level comparison

| Dataset | Dual clean | Risk clean | Dual affine | Risk affine | Dual jitter | Risk jitter |
|---|---:|---:|---:|---:|---:|---:|
| DistalPhalanxOutlineAgeGroup | 0.687 | 0.672 | 0.697 | 0.602 | 0.314 | 0.628 |
| ProximalPhalanxOutlineCorrect | 0.674 | 0.676 | 0.675 | 0.676 | 0.440 | 0.663 |
| RefrigerationDevices | 0.431 | 0.405 | 0.406 | 0.392 | 0.337 | 0.339 |
| SonyAIBORobotSurface2 | 0.856 | 0.856 | 0.858 | 0.858 | 0.856 | 0.856 |

## Outcome and scientifically important limitations

With 5 labels/class, MESA risk-gated compared with Dual MR episodic:

- Clean difference: -0.009551
- Synthetic affine difference: -0.027032
- Synthetic waveform jitter difference: 0.134912

The controller improves average performance under synthetic waveform
noise, but **does not dominate** in clean or affine conditions. On
some datasets the slower semantic memory, the Hopfield-style
episodic memory and explicit physics components have negative
contributions. The global ensemble cannot be attributed to a
particular biological module without ablation.

The model's physical claims are limited to exact affine decomposition
within numerical tolerance, contraction of the dimensionless heat
semigroup in L2, and stability bounds on the AR(2) predictor and
first-order leaky integrators. Biological inspiration is not
equivalent to biological fidelity.

The underlying MultiRocket is a compact 336-kernel configuration,
not exhaustive high-capacity MultiRocket. No official matched
TS2Vec, TimeMAE, MOMENT or published near-SOTA neuro-symbolic
baseline was rerun in this exact four-dataset trial. Results are
descriptive two-seed means without statistical inference.

The development record also shows that unconditional fusion can
reduce accuracy, and an expert selector based on five labels/class
may miss an expert that performs well on TEST. The complete
per-seed selection record is retained rather than reporting
a post-hoc oracle.

## Code and records

- MESA_SSL/mesa_biophysics_memory.py
- MESA_SSL/RISK_GATED_BIOPHYSICS_DESIGN.md
- MESA_SSL/BIOPHYSICS_ARCHITECTURE.md (from ancestor branch)
- MESA_SSL/riskgate_biophysics_holdout_2026-11-23.json
  contains 456 complete final records.
- MESA_SSL/riskgate_biophysics_summary_2026-11-23.csv
  contains per-method/dataset/budget mean, sample SD and sample count.
- MESA_SSL/riskgate_biophysics_policy_2026-11-23.csv
  records each selected expert and two chart alarm diagnostics.
- GitHub Actions: https://github.com/kimphuctran/MESA_SSL/actions/runs/45280823062

## Decision

Keep the risk-gated branch as a robustness-oriented exploratory
algorithm. Do not merge it to main or call it universally superior.
Retain the simpler dual-MR episodic method as the clean/affine
reference. Further work should use a larger untouched benchmark,
physically justified nuisance labels, clean-regret constraints
and many more paired seeds, rather than endlessly tuning on
previously seen official TEST datasets.
