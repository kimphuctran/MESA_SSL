# MESA-RiskBounded Memory: structural next candidate after failed Transformer and elastic-metric dominance

**Research design only — NOT an executed winning model.**
Date: 24 November 2026.
Approved scope: HCLM-controlled sandbox.
Strong existing baseline to preserve: FULL 10,000-kernel MultiRocket,
not the compact 336-kernel exploratory reference.

## 1. Why the architecture must change again

The executed MESA-ElasticSynapse model has promising 1/3/5-shot
synthetic-jitter results but has NOT beaten all fully matched strong
classifiers on clean and shifted data. Direct standalone elastic memory
underperforms MultiRocket on several tasks, and an extra empirical
representation-space OOD monitor failed on Car even while improving
MedicalImages. The original central scientific objective remains:
**mechanism-grounded few-label representation, fast adaptation and
safe human-governed decisions under changing operation**.

The next architectural candidate is a *selective specialization system*
with an unusually strong protected default, rather than requiring a
single latent space to represent every class and every perturbation.

## 2. Proposed architecture — protected strong path

```text
sensor x -> exact raw/affine-canonical factorization -> mechanism descriptors
             |                                      |
             +--> FULL MultiRocket + TRAIN labels -> f_base
             |
             +--> cortical SSL + elastic/associative memory -> f_specialist
             |
             +--> independent split-calibrated OOD chart -> p_morph, p_repr
                                      |
                       HCLM-governed constrained switch
                                      |
                    if no validated nuisance evidence: f_base
                    if robust alert AND TRAIN memory advantage: f_specialist
                    if conflict/uncertainty: f_base or auditable deferral
```

Neural encoders, physics-inspired memories and prior domain knowledge
remain candidates; however, their involvement is conditional on
TRAIN-only evidence and does not force them into every prediction.
The role of HCLM remains the **research metacontroller** and
human-governed process; the production-time controller is separately
defined and testable.

## 3. Simple provable safeguard for the next experiment

Let `f0` be a frozen baseline classifier and `f1` a specialist.
Let `A(x)` be a nuisance-alert event calibrated from a separate
unlabeled TRAIN calibration split, and set

```text
f(x) = f1(x) if A(x) occurs, else f0(x)
```

For 0/1 classification loss in the true in-control population P0,

```text
R0(f) - R0(f0)
= E0[ 1_A(x) * (ell(f1(x),y)-ell(f0(x),y)) ]
<= P0(A).
```

This inequality follows because the pointwise excess 0/1 loss is
at most 1. If `A` is constructed from a genuine split-conformal
p-value and `P0(A)<=alpha` under exchangeability, then

```text
R0(f) <= R0(f0) + alpha.
```

With independent or dependent two-alert screens separately
superuniform at levels `alpha1`, `alpha2`, a union alert has
`P0(A)<=alpha1+alpha2` by Bonferroni/union bound.

**Scope restrictions:** this is an in-control 0/1 *risk* bound,
not an accuracy-improvement guarantee, not a Macro-F1 guarantee,
not a per-class guarantee, and not an ARL result. It requires
split-calibration validity. The earlier 2026 MESA empirical
representation-novelty chart used a feature transform fit on ALL
TRAIN including calibration examples; its p-values must NOT be
called exact split conformal without refitting the detector on
a disjoint reference split.

A conditional expert may improve robustness, but the theorem
does not force any positive gain on contaminated examples.
That must be measured.

## 4. Learning from observed negative transfers

- CNN/Transformer failure: use full-capacity MultiRocket as the
  protected representational foundation rather than expecting
  a short SSL training schedule to outperform it.
- Mean/slope information loss: keep exact raw/shape factorization and
  physical concept skip. Never force single-canonical representation.
- Small-label HPO overfit: learn specialist eligibility from
  leave-one-whole-episode-out TRAIN, never TEST.
- Morphology chart blind spots: build independent, genuinely
  split-calibrated spectral/feature distribution monitors.
- New monitor harmful on Car: do not tighten a proven chart's
  threshold solely to add another detector. Audit the union alarm,
  and freeze a family-wise error budget before evaluation.
- Biology/physics overcomplexity: activation is conditional. No fixed
  positive weight on Hopfield/DTW/physics experts.
- Full baseline validity: require the same support IDs, seeds,
  corruption functions and n_kernels=10000.
- Holdout reuse: Fish, Car, MedicalImages, CricketX, Worms and Yoga
  are now development evidence if their TEST results were inspected.
  Do not claim independent confirmation on these tasks again.

## 5. Predeclared acceptance rules before building another network

The next **new** official UCR/UEA study should include at least
eight genuinely previously unexamined tasks, five paired support seeds,
nested label budgets (1,3,5), verified TRAIN class counts, and
a separate full-strength TS2Vec/TimeMAE/MOMENT/RocketPFN comparison.
Record clean/affine/waveform Macro-F1, 0/1 risk, nuisance detection
false-alarm rate, inference energy/cost, and calibration validity.

The HCLM controller may authorize an experiment only after:
1. The **protected baseline** is specified, validated and frozen.
2. A specialist has at least a predeclared robust TRAIN benefit.
3. The clean risk tolerance alpha is frozen before TEST.
4. The TEST dataset has not influenced architecture, thresholds or
   evaluation protocol.
5. Every apparently winning result is accompanied by full
   dataset-level uncertainty and negative evidence.

Any missing condition results in REJECT or SANDBOX, not deployment,
master merge, publication SOTA claims, or 'absolute victory'.

## 6. Scientific prior art (avoid novelty inflation)

Dynamic time warping and elastic nearest-neighbor models are
established. Full MultiRocket, Hydra, RocketPFN, SelF-Rocket,
mechanism-aware shapelet methods and deformable/predictive learning
are relevant research competitors. A mixture of them is not,
by itself, a demonstrably novel AI law.

Potential publishable contribution: a **bounded-risk,
mechanism-auditable specialist memory and governed experiment
selection**, with an explicit measured link between nuisance
identification, few-label transfer and the rarity of harmful
model switching.

Examples of relevant prior work:
- https://link.springer.com/article/10.1007/s10618-026-01269-w
- https://www.nature.com/articles/s41598-025-25667-0
- https://www.sciencedirect.com/science/article/pii/S0925231226028018

**Decision as of 24 November 2026:** Design proposal only.
Do not claim a new model has beaten the full 10k baseline until
fresh tests under this frozen structure are executed.
