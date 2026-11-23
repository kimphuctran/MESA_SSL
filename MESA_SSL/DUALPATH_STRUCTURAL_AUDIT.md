# MESA Dual-Path Structural SSL | Completed benchmark audit

Date: 23 November 2026.
Repository: https://github.com/kimphuctran/MESA_SSL
Branch: `mesa-dualpath-structural-2026`.
Complete GitHub Actions run: https://github.com/kimphuctran/MESA_SSL/actions/runs/45270021456
Evaluated source revision: `c40b0ddc7690ed3ae0b1ca07018af5cdea64ddda`.

## Architecture and separation of information

Two independent multiresolution CNNs encode (i) the original time series and
(ii) the exactly affine-canonical residual. An analytical channel-wise
concept path retains the mean, linear trend and log residual scale. This
addresses a concrete failure of the previous MESA architecture: forced
invariance could destroy label-discriminating mean/slope information.

Let the centered time grid be τ, μ=mean(x), β=(τ·x)/(τ·τ), and
r=x−μ−βτ, σ=sqrt(mean(r²)), q=r/σ. For nondegenerate residuals,
x=μ+βτ+σq exactly. For an offset/trend transformation
x'=x+a+bτ, we have μ'=μ+a, β'=β+b, σ'=σ, and q'=q.
Thus the tuple (q,μ,β,σ) is invertible and its q component is
affine-invariant under the declared nuisance family. This is an algebraic
result, not an assertion that affine shifts always preserve real-world labels.

The raw stream is trained with noise consistency and concept/equivariance
loss; the shape stream has its own spectral prediction objective.
The generic control shares the two-stream architecture, seeds, data and
training budget but has mechanism weights fixed to zero.

The benchmark includes raw MultiRocket, canonical MultiRocket,
TRAIN-selected dual MultiRocket, learned-only generic/MESA representations,
both hybrid kernels, the earlier mean/slope SPC gate, and a final
TRAIN-selected admissible expert. All kernel weights and ridge penalties
are optimized on five labelled TRAIN samples per class, not TEST.

## Official UCR/UEA TRAIN/TEST measurements

- Development tasks previously used to motivate the algorithm:
  GunPoint, ItalyPowerDemand, BasicMotions, CBF, SyntheticControl.
- Distinct new datasets checked after freezing the architecture:
  Wine, Fish, OSULeaf and Earthquakes.
- Two seeds: 2601–2602; 12 epochs; nine methods; all 18 jobs successful.
- TEST modes: original, synthetic offset/trend, synthetic waveform jitter.
  Both interventions are assumed to preserve class labels.
- 162 individual evaluations; the results are descriptive two-seed estimates.

| Dataset | Raw clean | DualMR clean | GenericHybrid clean | MESAHybrid clean | Selected clean | Raw jitter | Selected jitter |
|---|---:|---:|---:|---:|---:|---:|
| BasicMotions | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.948 | 1.000 |
| CBF | 0.988 | 0.988 | 0.981 | 0.981 | 0.981 | 0.987 | 0.980 |
| Earthquakes | 0.429 | 0.436 | 0.436 | 0.436 | 0.436 | 0.404 | 0.429 |
| Fish | 0.842 | 0.861 | 0.855 | 0.855 | 0.855 | 0.072 | 0.109 |
| GunPoint | 0.912 | 0.912 | 0.902 | 0.902 | 0.902 | 0.605 | 0.631 |
| ItalyPowerDemand | 0.955 | 0.953 | 0.953 | 0.953 | 0.953 | 0.949 | 0.947 |
| OSULeaf | 0.781 | 0.789 | 0.789 | 0.789 | 0.789 | 0.142 | 0.142 |
| SyntheticControl | 0.931 | 0.970 | 0.977 | 0.973 | 0.977 | 0.921 | 0.975 |
| Wine | 0.659 | 0.715 | 0.715 | 0.715 | 0.715 | 0.333 | 0.333 |

### Development group (five datasets)


| Method | Clean Macro-F1 | Affine Macro-F1 | Jitter Macro-F1 | Selected neural features / dataset-seed |
|---|---:|---:|---:|---:|
| Raw MultiRocket | 0.957 | 0.907 | 0.882 | 0/10 |
| Canonical MultiRocket | 0.885 | 0.885 | 0.768 | 0/10 |
| Dual MultiRocket | 0.965 | 0.936 | 0.901 | 0/10 |
| Generic learned-only | 0.887 | 0.749 | 0.870 | 10/10 |
| MESA learned-only | 0.887 | 0.743 | 0.869 | 10/10 |
| Generic dual-path hybrid | 0.963 | 0.932 | 0.907 | 3/10 |
| MESA dual-path hybrid | 0.962 | 0.930 | 0.908 | 3/10 |
| MESA SPC hybrid | 0.959 | 0.930 | 0.884 | 0/10 |
| TRAIN-selected admissible | 0.963 | 0.932 | 0.907 | 0/10 |


### New holdout group (four datasets)


| Method | Clean Macro-F1 | Affine Macro-F1 | Jitter Macro-F1 | Selected neural features / dataset-seed |
|---|---:|---:|---:|---:|
| Raw MultiRocket | 0.678 | 0.607 | 0.238 | 0/8 |
| Canonical MultiRocket | 0.703 | 0.701 | 0.248 | 0/8 |
| Dual MultiRocket | 0.700 | 0.699 | 0.244 | 0/8 |
| Generic learned-only | 0.509 | 0.498 | 0.431 | 8/8 |
| MESA learned-only | 0.499 | 0.473 | 0.418 | 8/8 |
| Generic dual-path hybrid | 0.699 | 0.703 | 0.254 | 1/8 |
| MESA dual-path hybrid | 0.699 | 0.703 | 0.254 | 1/8 |
| MESA SPC hybrid | 0.677 | 0.702 | 0.238 | 0/8 |
| TRAIN-selected admissible | 0.699 | 0.703 | 0.254 | 0/8 |


### All nine tasks


| Method | Clean Macro-F1 | Affine Macro-F1 | Jitter Macro-F1 | Selected neural features / dataset-seed |
|---|---:|---:|---:|---:|
| Raw MultiRocket | 0.833 | 0.774 | 0.596 | 0/18 |
| Canonical MultiRocket | 0.804 | 0.804 | 0.537 | 0/18 |
| Dual MultiRocket | 0.847 | 0.831 | 0.609 | 0/18 |
| Generic learned-only | 0.719 | 0.637 | 0.675 | 18/18 |
| MESA learned-only | 0.715 | 0.623 | 0.668 | 18/18 |
| Generic dual-path hybrid | 0.845 | 0.830 | 0.616 | 4/18 |
| MESA dual-path hybrid | 0.845 | 0.829 | 0.617 | 4/18 |
| MESA SPC hybrid | 0.834 | 0.829 | 0.597 | 0/18 |
| TRAIN-selected admissible | 0.845 | 0.830 | 0.616 | 0/18 |


## Failure attribution and research decision

1. On SyntheticControl, train-selected dual representations achieve clean
   Macro-F1 around 0.977 versus raw MultiRocket around 0.931, confirming
   that removing mean/slope permanently is unnecessary and sometimes harmful.
   But the matched *generic hybrid* is slightly stronger than the MESA
   mechanism hybrid. The improvement is not attributable uniquely to
   the mechanism loss.
2. On Fish and OSULeaf, a synthetic waveform disturbance degrades the
   MultiRocket features severely, while neural-only features often remain
   stronger. Mean/slope SPC cannot reliably diagnose this disturbance;
   a separate spectral/roughness chart is justified for follow-on study.
3. On untouched datasets the TRAIN-selected ensemble improves clean
   Macro-F1 over raw, but not consistently over canonical MultiRocket.
   Mean shifted performance increases relative to raw, yet method-selection
   and error attribution remain task-dependent.
4. On many runs the selected 'MESA hybrid' assigns *zero* weight to MESA
   features (e.g., it reduces to raw/canonical MultiRocket). Attribution
   to the new encoder must be conditioned on nonzero feature contribution.
5. Comparing a single TEST-selected best-of-all-models would leak ground
   truth. The report intentionally retains actual TRAIN-only selection
   instead of post-hoc oracle accuracy.
6. Given related work on seasonal/trend decomposition (e.g., ST-CL,
   DOI 10.1109/ACCESS.2025.3621317) and existing contrastive SSL
   (TS2Vec, Series2Vec), the mere existence of two streams does not
   establish novelty. A publishable claim needs independent evidence of
   mechanism-grounded selection, fidelity and robustness.

## Complete source and outcomes

- `MESA_SSL/mesa_dualpath_ssl.py`: mechanism encoder, matched generic
  baseline, rigorous TRAIN-only CV and nuisance policy.
- `MESA_SSL/dualpath_structural_9dataset_2026-11-23.json`:
  all 162 executed records and workflow provenance.
- `MESA_SSL/dualpath_structural_summary_2026-11-23.csv`:
  each method/dataset mean, sample SD and number of nonzero neural picks.
- Workflow: `.github/workflows/mesa-dualpath-benchmark.yml`.
- Each run uploads exact source, versioned environment and outputs.

**Status: completed developmental benchmark; insufficient evidence for
universal superiority or a claimed state of the art.**
