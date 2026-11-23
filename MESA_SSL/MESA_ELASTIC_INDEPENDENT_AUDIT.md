# MESA-ElasticSynapse: Mechanism-Preserving Elastic Episodic Memory

Date: 24 November 2026.
Research branch: `mesa-elastic-synaptic-memory-2026`.
Frozen validation source revision: `5706afb8510d53eca295ea38365bf2e251136f67`.
Independent benchmark: https://github.com/kimphuctran/MESA_SSL/actions/runs/45287302968 (8/8 jobs successful).
HCLM post-run gate: https://github.com/kimphuctran/MESA_SSL/actions/runs/45288124330 (successful).
Failed pretest *CI-only* attempt: https://github.com/kimphuctran/MESA_SSL/actions/runs/45287128261 (8 shallow-checkout hash-verification failures before any TEST data were evaluated).

## Why the model was restructured

The HCLM research controller had recorded repeated Transformer/Routing
failures in low-label settings: representations lost class-specific
mean/slope during forced canonicalization, few-label neural encoders
overfit, and mean/slope SPC missed waveform perturbations.
Rather than adding more attention heads, MESA-ElasticSynapse replaces
the core similarity mechanism with multiscale **dynamic time warping
episodic memory**, while retaining an analytically invertible raw/shape
decomposition and a separately trained self-supervised slow cortex.

The frozen design uses 96-point canonical time series, Gaussian
heat-like smoothing at widths 2.0 and 7.0, DTW with radius 12, and
a memory of 1/3/5 TRAIN-labeled cases per class. A mechanism-concept
memory additionally preserves original mean/slope/rms, which may be
discriminative on otherwise affine-sensitive tasks. Two TRAIN-calibrated
SPC monitors choose when episodic experts are allowed to override
a reliable dual-MultiRocket reference. Three-/five-shot expert
choices use leave-one-labeled-episode-out TRAIN validation and
a predeclared improvement margin; one-shot has no such valid
class-preserving LOO selection.

## Frozen official UCR/UEA heldout evaluation

Four previously unexamined tasks were predeclared: CricketX,
MedicalImages, Worms and Yoga. Matched seeds 3801/3802, 1/3/5 labels
per class; 8 epochs of unlabeled TRAIN SSL.

### 1-shot
| Model | Clean | Affine | Jitter |
|---|---:|---:|---:|
| Dual MR episodic | 0.366 | 0.358 | 0.218 |
| Canonical MR episodic | 0.370 | 0.370 | 0.223 |
| Cortex episodic | 0.332 | 0.291 | 0.328 |
| Elastic slow memory | 0.317 | 0.317 | 0.315 |
| Elastic shape-concept memory | 0.313 | 0.298 | 0.317 |
| Elastic HCLM-authorized router | 0.363 | 0.358 | 0.272 |

### 3-shot
| Model | Clean | Affine | Jitter |
|---|---:|---:|---:|
| Dual MR episodic | 0.465 | 0.453 | 0.237 |
| Canonical MR episodic | 0.456 | 0.456 | 0.242 |
| Cortex episodic | 0.396 | 0.336 | 0.387 |
| Elastic slow memory | 0.357 | 0.357 | 0.358 |
| Elastic shape-concept memory | 0.358 | 0.333 | 0.355 |
| Elastic HCLM-authorized router | 0.461 | 0.454 | 0.291 |

### 5-shot
| Model | Clean | Affine | Jitter |
|---|---:|---:|---:|
| Dual MR episodic | 0.487 | 0.479 | 0.262 |
| Canonical MR episodic | 0.488 | 0.488 | 0.277 |
| Cortex episodic | 0.417 | 0.347 | 0.421 |
| Elastic slow memory | 0.349 | 0.349 | 0.346 |
| Elastic shape-concept memory | 0.355 | 0.341 | 0.356 |
| Elastic HCLM-authorized router | 0.486 | 0.476 | 0.313 |

## Five-shot dataset-level effects

| Dataset | Baseline clean | ElasticRouter clean | Baseline jitter | ElasticRouter jitter |
|---|---:|---:|---:|---:|
| CricketX | 0.501 | 0.501 | 0.410 | 0.409 |
| MedicalImages | 0.487 | 0.483 | 0.095 | 0.095 |
| Worms | 0.419 | 0.419 | 0.224 | 0.239 |
| Yoga | 0.542 | 0.542 | 0.317 | 0.508 |

## Governed research decision

The predeclared HCLM audit compares **Elastic HCLM-authorized router**
against **Dual MR episodic (336 kernels)**, matched by dataset, seed
and exact support ID.

- 1-shot: Δclean -0.003149, Δaffine -0.000079, Δjitter 0.053498.
- 3-shot: Δclean -0.004052, Δaffine 0.000835, Δjitter 0.053565.
- 5-shot: Δclean -0.001016, Δaffine -0.002663, Δjitter 0.051219.

The HCLM frozen risk gate passes the **sandbox-level** criteria at each
shot budget, but does NOT permit promotion. The four tasks and two
seeds do not satisfy the evidence floor of eight new datasets and
five paired seeds. Strong full 10k MultiRocket, TS2Vec,
TimeMAE, MOMENT and Hydra comparisons are required before any
generalized superiority claim.

## Failure learning and negative evidence

- **MedicalImages:** compact MR representations collapse under synthetic
  waveform jitter, while the existing morphology chart almost never
  alarms (about 2.9%, similarly to clean). The heldout router therefore
  fails to switch to a potentially stronger cortical/elastic memory.
- **CricketX:** the elastic memory by itself generally underperforms
  simple MultiRocket and the routed model does not consistently improve
  clean/jitter Macro-F1.
- **Yoga:** a guarded switch can improve jitter, but clean and affine
  tradeoffs remain and should be reported per seed.
- **Training-set leakage guard:** synthetic copies of support cases
  are never used as independent labeled reference during leave-one-
  original-out selection.
- The static multivariate UCR TEST benchmark does not establish
  sequential SPC ARL0 or detection-delay guarantees.

These TEST sets are now development evidence. Any further change
motivated by their results must be evaluated on new, frozen tasks.

## Files

- Implementation: `MESA_SSL/mesa_elastic_synaptic.py`.
- Pretest contract:
  `MESA_SSL/HCLM_ELASTIC_FROZEN_PROTOCOL_2026-11-24.json`.
- CI-only source provenance fix:
  `MESA_SSL/HCLM_ELASTIC_CI_PROVENANCE_AMENDMENT_2026-11-24.json`.
- All 144 executed records, source and study context:
  `MESA_SSL/HCLM_ELASTIC_INDEPENDENT_RESULTS_2026-11-24.json`.
- HCLM independent post-run audit:
  `MESA_SSL/hclm_elastic_post_audit.py`.
- GitHub Actions source and dependencies attached to the individual jobs.

**Status: promising limited synthetic waveform robustness; no absolute
win or state-of-the-art result established.**
