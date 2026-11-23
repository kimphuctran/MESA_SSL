# HCLM full-10k MultiRocket versus ElasticSynapse — matched evidence

Date: 24 November 2026.
Source records: exact same four official UCR datasets, seed 3801 and 3802, nested 1/3/5 labeled TRAIN cases per class.
Elastic study: https://github.com/kimphuctran/MESA_SSL/actions/runs/45287302968
10k MultiRocket study: https://github.com/kimphuctran/MESA_SSL/actions/runs/45287492773

## All method means (dataset-equal, two seeds)

| Labels/class | Method | Clean Macro-F1 | Affine drift | Waveform jitter |
|---|---|---:|---:|---:|
| 1-shot | Elastic HCLM-authorized router | 0.3631 | 0.3579 | 0.2715 |
| 1-shot | Full MR raw ridge | 0.3773 | 0.3608 | 0.2405 |
| 1-shot | Full MR dual episodic | 0.3715 | 0.3601 | 0.2139 |
| 1-shot | Dual MR episodic | 0.3663 | 0.3579 | 0.2180 |
| 1-shot | Elastic shape-concept memory | 0.3130 | 0.2979 | 0.3166 |
| 3-shot | Elastic HCLM-authorized router | 0.4610 | 0.4536 | 0.2907 |
| 3-shot | Full MR raw ridge | 0.5039 | 0.4637 | 0.2841 |
| 3-shot | Full MR dual episodic | 0.4594 | 0.4516 | 0.2325 |
| 3-shot | Dual MR episodic | 0.4650 | 0.4528 | 0.2372 |
| 3-shot | Elastic shape-concept memory | 0.3581 | 0.3328 | 0.3551 |
| 5-shot | Elastic HCLM-authorized router | 0.4862 | 0.4763 | 0.3128 |
| 5-shot | Full MR raw ridge | 0.5367 | 0.5063 | 0.2932 |
| 5-shot | Full MR dual episodic | 0.4970 | 0.4792 | 0.2657 |
| 5-shot | Dual MR episodic | 0.4873 | 0.4790 | 0.2616 |
| 5-shot | Elastic shape-concept memory | 0.3552 | 0.3410 | 0.3563 |

## Paired HCLM-Elastic router minus full 10k-MR raw ridge

| Shot | Clean Δ | Affine Δ | Jitter Δ |
|---|---:|---:|---:|
| 1 | -0.0142 | -0.0030 | +0.0311 |
| 3 | -0.0429 | -0.0101 | +0.0066 |
| 5 | -0.0504 | -0.0300 | +0.0196 |

## Five-shot by dataset

| Dataset | Clean Δ | Affine Δ | Jitter Δ |
|---|---:|---:|---:|
| CricketX | -0.1314 | -0.0981 | -0.1449 |
| MedicalImages | +0.0154 | +0.0886 | -0.0696 |
| Worms | -0.0885 | -0.0998 | +0.1018 |
| Yoga | +0.0028 | -0.0106 | +0.1911 |

## HCLM governed action

**NO-GO for performance promotion.** The prespecified scientific ambition requires clean and affine noninferiority to a strong baseline plus useful jitter gains. For the five-shot strong raw ridge comparison:
- Clean Δ=-0.0504: FAIL (threshold ≥ -0.01).
- Affine Δ=-0.0300: FAIL (threshold ≥ -0.01).
- Jitter Δ=0.0196: FAIL (desired gain ≥ +0.04).
- Four tasks and two seeds also do not meet eight-task/five-seed minimum even if risk thresholds had passed.

**Important:** matching supports were checked explicitly on all 24 dataset/seed/budget pairs. Official TEST labels were not used for model selection. The nominal 10,000-kernel MultiRocket control uses a fixed ridge=0.10 and the same support selection protocol. This is NOT a full survey of stronger SSL models such as TS2Vec, TimeMAE or MOMENT.

The previous 336-kernel compact reference was not strong enough to establish a state-of-the-art claim. On these four tasks, ElasticSynapse often provides useful synthetic-jitter robustness but the 10k ridge reference has clearly better clean/affine Macro-F1. The structural pivot thus fails the current HCLM promotion gate. Differences vary by dataset: the new model is harmed especially on CricketX and can show jitter benefits on Yoga and Worms. Dataset-level performance under these known perturbations must not be generalized to real physics without label-preservation validation.

## Ancillary development results (NOT independent confirmation)

The later representation-surprise variant was run as a smoke on already inspected Fish, Car and MedicalImages. On MedicalImages the new feature-space surprise monitor raised jitter Macro-F1 from 0.0981 (old router) to 0.1662; on Car it harmed jitter Macro-F1 from 0.5902 to 0.2278; Fish remained 0.4709. This variant is NOT eligible for promotion. It requires stronger detection calibration and new independent datasets before any conclusion.

## Exclusions and provenance

A separate full-kernel challenge on ArrowHead, SwedishLeaf and ScreenType produced 72 executed observations. FaceFour had fewer than five labeled TRAIN cases in at least one class and was excluded before TEST; a later Adiac supplement also failed the same eligibility condition. Neither was counted as a zero score or cherry-picked replacement.

Artifacts:
- This summary: MESA_SSL/HCLM_ELASTIC_VS_FULL10K_AUDIT_2026-11-24.md
- Frozen ElasticSynapse evidence: MESA_SSL/HCLM_ELASTIC_INDEPENDENT_RESULTS_2026-11-24.json on branch mesa-elastic-synaptic-memory-2026
- Strong 10k-MR evidence: MESA_SSL/HCLM_MR10K_ELASTIC_MATCHED_2026-11-24.json on branch hclm-fullbaseline-challenge-2026
- HCLM prior research search governor: MESA_SSL/hclm_search_governor.py

**The author may continue development in a new sandbox, but no state-of-the-art claim or master merge is justified.**
