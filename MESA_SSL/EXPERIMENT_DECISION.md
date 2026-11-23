# MESA-SSL mechanism, few-label adaptation and online governance

Research audit — 23 November 2026. This experiment branch is separate from `master`. All results below were recovered from successfully executed GitHub Actions. The reported values do not establish universal state-of-the-art superiority.

## 1. New self-supervised graph and prototype branches

The graph experiment keeps unlabeled UCR/UEA TRAIN observations as graph nodes and chooses diffusion or kernel-ridge candidates using only a small labeled subset of TRAIN. No TEST labels are used in model construction, validation, or selection. On GunPoint, ItalyPowerDemand and BasicMotions (3 new seeds, 5 labels/class), the selector chose the MultiRocket ridge baseline on all runs. Thus, diffusion did not yield incremental benefits under this fixed protocol.

The second method fine-tunes a mechanism-constrained MESA encoder after 20 epochs of unlabeled self-supervised learning, using 30 epochs of prototype discrimination with only five labeled TRAIN samples per class. Its matched generic encoder has the same architecture, optimizer and label budget, but no mechanism loss. The reference encoder's unlabeled features and concept outputs regularize fine-tuning.

### Development Macro-F1, seeds 510–512

| Method | GunPoint | ItalyPowerDemand | BasicMotions |
|---|---:|---:|---:|
| MultiRocket raw + ridge | 0.935 | 0.937 | **1.000** |
| Canonical MultiRocket + ridge | 0.915 | 0.946 | 0.983 |
| Generic SSL + few-shot ridge | 0.647 | 0.882 | 0.908 |
| MESA SSL + few-shot ridge | 0.666 | 0.945 | 0.931 |
| MESA SSL + predeclared fixed mixture | 0.875 | **0.949** | 0.983 |

### Independent UCR datasets, seeds 513–515

| Method | Beef | TwoLeadECG | Plane | Chinatown |
|---|---:|---:|---:|---:|
| MultiRocket raw + ridge | 0.764 | **0.994** | **1.000** | 0.956 |
| Canonical MultiRocket + ridge | **0.796** | 0.992 | **1.000** | 0.960 |
| Generic SSL + few-shot ridge | 0.418 | 0.964 | 0.437 | 0.934 |
| MESA SSL + few-shot ridge | 0.448 | 0.970 | 0.980 | 0.931 |
| MESA SSL + fixed mixture | 0.787 | 0.991 | **1.000** | 0.956 |

Macro-F1 means over three seeds, official UCR/UEA TRAIN/TEST split and exactly five labeled TRAIN samples per class. OliveOil was excluded *without changing the protocol* because one class has fewer than five samples in the official TRAIN split. The table is a scientific audit, not a selection of only favourable tasks. All run-level results are retained.

**Inference:** Mechanism-conditioned pretraining can improve the few-shot MESA latent versus its generic counterpart, particularly in ItalyPowerDemand and several holdout tasks, but the classifier still does not uniformly exceed an established MultiRocket feature map. Improving a baseline by adding it to a weighted ensemble is not sufficient attribution of new performance to MESA.

## 2. HCLM fixed-admissibility online expert aggregation

A separate, clearly marked **online prequential protocol** uses four fixed representation experts: raw MultiRocket, canonical MultiRocket, generic SSL and mechanism-conditioned MESA SSL. A human/organisation may screen inadmissible experts *before* a sequence begins. The remaining models produce strictly positive class probabilities. For observation t, the system:

1. Forms the weighted predictive mixture before observing its true label.
2. Issues a prediction.
3. Receives the actual label and only then updates expert weights.

Given K initially admissible experts, a uniform prior and fixed expert-probability sequences, the Bayesian mixture obeys the exact deterministic inequality

`L_mix(T) <= min_j L_expert_j(T) + log K`

for cumulative logarithmic loss. The proof follows from the likelihood identity

`exp(-L_mix(T)) = (1/K) * sum_j exp(-L_expert_j(T))`.

The comparison is to the **best fixed expert in hindsight**, not to an oracle that changes experts at each observation. It does **not** guarantee better Macro-F1, accuracy, or per-sample decisions, and does not hold unchanged for dynamic safety-screen updates.

### Prequential experimental verification

| Dataset | Mixture Macro-F1 | Raw-expert Macro-F1 | Mechanism-expert Macro-F1 | Mean log-loss regret | Maximum permitted |
|---|---:|---:|---:|---:|---:|
| GunPoint | 0.836 | 0.832 | 0.681 | 1.383 | 1.386 |
| ItalyPowerDemand | 0.912 | 0.907 | 0.902 | 1.210 | 1.386 |
| BasicMotions | 0.992 | **1.000** | 0.823 | 1.331 | 1.386 |

Three seeds (610–612) per dataset, initial five labeled TRAIN examples/class, 15 epochs of unlabeled expert pretraining. Labels of TEST examples are revealed **after** each prediction to the online aggregator. These numbers must not be pooled with the static inductive UCR benchmark table. All nine run-level results satisfy the log-loss guarantee.

## 3. Research decision

The current development work does **not** substantiate claims that MESA-SSL “always wins” on unknown tasks. No learning procedure can guarantee that property for arbitrary previously unobserved label-generating mechanisms without additional assumptions or ground-truth information. The meaningful advances retained are: (i) measured benefit from mechanism-constrained few-label adaptation relative to a matched generic SSL encoder on some tasks, (ii) exact accounting of the concept and expert-score contributions where defined, and (iii) an online cumulative-log-loss bound under a fixed admissible expert set.

For an eventual Neurocomputing submission, expand external comparisons to Hydra+MultiRocket, RocketPFN and a larger time-series archive; collect independent concept-reference and intervention measurements; and evaluate selected performance, explanation fidelity and robust adaptation without iterating on the final TEST splits. Any new empirical additions require rerunning all baselines under a fixed common protocol.

## Reproducibility

- [Self-supervised graph benchmark code](./mesa_graph_inductive.py)
- [Few-shot prototype adaptation code](./mesa_prototype_ssl.py)
- [Online HCLM expert aggregation code](./hclm_online_experts.py)
- [180 dev comparison observations](./graph_and_prototype_dev_results.json)
- [144 holdout scores and dataset exclusion](./prototype_holdout_results_4datasets.json)
- [9 online prequential evaluations](./hclm_online_expert_results.json)
- [Graph workflow](https://github.com/kimphuctran/MESA_SSL/actions/runs/45174073340)
- [Prototype development](https://github.com/kimphuctran/MESA_SSL/actions/runs/45174736789)
- [Prototype independent datasets](https://github.com/kimphuctran/MESA_SSL/actions/runs/45175125733)
- [Chinatown validation](https://github.com/kimphuctran/MESA_SSL/actions/runs/45175465507)
- [Online expert bound](https://github.com/kimphuctran/MESA_SSL/actions/runs/45176115247)

The existing `master` branch is unchanged.
