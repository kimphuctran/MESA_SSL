# HCLM decision: full 10k MultiRocket defeats current ElasticSynapse clean model

Research date: 24 November 2026.
Repository: https://github.com/kimphuctran/MESA_SSL
**Full 10k baseline:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45287492773 (8/8 successful).
**MESA-ElasticSynapse:** https://github.com/kimphuctran/MESA_SSL/actions/runs/45287302968 (8/8 successful).
Frozen strong comparator contract: `MESA_SSL/HCLM_MATCHED_ELASTIC_FULLMR_CONTRACT_2026-11-24.json`.

## Full independent matched evaluation

Official UCR TRAIN/TEST splits for CricketX, MedicalImages, Worms,
Yoga; paired seeds 3801 and 3802; support budgets 1/3/5 per class.
The scripts verified exact support ID matching per dataset/seed/budget.
The full MultiRocket reference uses 10,000 kernels with
`max_dilations_per_kernel=32`, trained on *unlabeled TRAIN only*.

This is a deliberately stronger reference than the 336-kernel model.
Data and compute costs differ, so no claim of matched computational
cost is made. TEST labels were not used to select methods or tune
hyperparameters.

## HCLM-governed risk acceptance

Effect = ElasticSynapse − fixed predeclared Full MR raw ridge.

| Shot | Δ Clean vs 10k ridge | Δ Affine vs 10k ridge | Δ Jitter vs 10k ridge | HCLM gate |
|---|---:|---:|---:|---|
| 1 | -0.0142 | -0.0030 | 0.0311 | REJECT |
| 3 | -0.0429 | -0.0101 | 0.0066 | REJECT |
| 5 | -0.0504 | -0.0300 | 0.0196 | REJECT |

The risk envelope specifies clean noninferiority >= -0.01,
affine noninferiority >= -0.01 and jitter improvement >= +0.04.
None of the budgets passes. The **research HCLM governor**
therefore returns **REJECT PROMOTION / DESIGN PROTECTED BASELINE
STRUCTURE** rather than making another post-hoc hyperparameter
choice on the same TEST.

## Five-shot equal-dataset macro-F1 means

| Method | Clean | Affine | Waveform jitter |
|---|---:|---:|---:|
| Full MR raw episodic | 0.4943 | 0.4537 | 0.2560 |
| Full MR canonical episodic | 0.4948 | 0.4948 | 0.2780 |
| Full MR dual episodic | 0.4970 | 0.4792 | 0.2657 |
| Full MR raw ridge | 0.5367 | 0.5063 | 0.2932 |
| Dual MR episodic | 0.4873 | 0.4790 | 0.2616 |
| Elastic HCLM-authorized router | 0.4862 | 0.4763 | 0.3128 |

## Five-shot dataset-level clean and synthetic-jitter comparison

| Dataset | 10k ridge clean | Elastic clean | 10k ridge jitter | Elastic jitter |
|---|---:|---:|---:|---:|
| CricketX | 0.633 | 0.501 | 0.554 | 0.409 |
| MedicalImages | 0.468 | 0.483 | 0.164 | 0.095 |
| Worms | 0.507 | 0.419 | 0.137 | 0.239 |
| Yoga | 0.539 | 0.542 | 0.317 | 0.508 |

The specialist helps on noisy Yoga and some Worms cases,
but standalone elastic memory is not a good replacement for full
MultiRocket on clean CricketX or MedicalImages. Do not cite the
largest per-case win as an operational algorithm if no
TRAIN-only selector actually chooses that method.

## New research direction

Protect a full-strength classifier as the default and move biological
fast episodic memory, elastic matching and shape/physics invariance into
**optional specialist branches**. The HCLM research metacontroller
governs whether a new proposal is evaluated, promoted or rejected.
A *genuine split-calibrated* nuisance detector can bound an in-control
0/1 risk increase by its alert rate under exchangeability, but this
is **not** a Macro-F1 dominance theorem or an absolute guarantee.

Full design:
https://github.com/kimphuctran/MESA_SSL/blob/hclm-governed-research-search-2026/MESA_SSL/NEXT_STRUCTURAL_CANDIDATE_RISK_BOUNDED_MEMORY_2026-11-24.md

## Scientific caveats

- Four tasks, two seeds: far fewer than needed for a general claim.
- Synthetic perturbations are assumed label-preserving, not domain validated.
- Comparison to RocketPFN, Hydra, TS2Vec, TimeMAE, MOMENT still needed.
- Previously failed FaceFour and Adiac runs were rejected **before TEST**
  because class-specific TRAIN support was too small for 5-shot;
  their failures are retained and not counted as losses.
- These four evaluated TEST datasets now become development evidence.
  Do not claim they remain untouched after this audit.
- The HCLM outcome is a *research decision*, not a neural-network layer.
  Human approval remains necessary for a master merge.

**Final status of this iteration: no absolute win;
strong-baseline comparison REJECTS promotion of ElasticSynapse.**
