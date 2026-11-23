# HCLM as the Research-Search Controller (NOT as a MESA-SSL network layer)

**Date:** 23 November 2026  
**Controller:** `MESA_SSL/hclm_search_governor.py`  
**Source of authority:** Human-Centered Learning Mechanics (HCLM), Tran, K. P. (Springer Nature, in press, per author's citation), adapted as an *experiment control protocol*.

## Meaning of integration

The user's request is **not** to label a neural loss `HCLM` or add another
HCLM module inside MESA. It is to make the *scientific search process*
itself obey HCLM: control the search direction, calibrate uncertain
information, govern which action may be taken, and learn systematically
from feedback. This implementation does exactly that by executing
an observable, reproducible decision policy over actual GitHub Actions
experiment records.

The controller does not modify the weights or internals of the ChatGPT
model; it governs the reproducible actions that the assistant chooses to
run in the connected research repository.

## Closed-loop state and action

```text
Previous MESA-SSL experiments + failure ledger + human scope
          │
          ▼
OBSERVE: paired dataset/seed effects + failures + evidence provenance
          │
          ▼
CALIBRATE: epistemic uncertainty, bootstrap sensitivity and data scope
          │
          ▼
CONTROL: hard clean/affine/jitter noninferiority and fairness constraints
          │
          ├──── unacceptable / insufficient ──> REJECT promotion
          │                                    ▼
          │                         select next sandbox experiment
          │                                    ▼
          │                         freeze hypothesis, data, rules
          │                                    ▼
          │                         run permitted experiment and audit
          │                                    └───────┐
          └──── acceptable + complete ──> human review ┘
```

The state `s_t` contains task-wise risk estimates, evidence uncertainty,
the current set of previously inspected TEST datasets, experimental budget
and allowed human-approved actions. The output control action `u_t` is
one of `AUDIT`, `DESIGN_SANDBOX`, or `RUN_SANDBOX` until stronger
admissibility and explicit human approval permit promotion.

The principles reflect HCLM's separation of
**control, calibration and governed action**, and the broader
research formulation combining loss, entropy/information, constraints and
decision costs. The Python controller is a *discrete operational analogue*.
Its uncertainty proxy must not be confused with the gradient-based
entropy-force term `grad H` from HCLM's learning-mechanics theory.
It does not assert a new theorem for HCLM.

## Concrete constraints

1. **Evidence scope.** A dataset inspected in any earlier experiment is
   development evidence from that point on. The TEST metrics can teach the
   *researcher* which failure mode matters; they cannot be reused to
   certify the next model without controlling for adaptive selection.
2. **Matched comparisons.** Same official TRAIN/TEST, support IDs,
   seeds, label budget, and metrics; all differences are paired at
   dataset/seed level.
3. **Calibration.** A two-stage dataset/seed paired bootstrap is
   reported as a descriptive sensitivity interval, not as a valid
   population confidence guarantee for reused or nonrandom archives.
4. **Hard risk envelope.** Mean clean Macro-F1 drop no greater than
   0.01, mean affine drop no greater than 0.01, no individual-dataset
   clean/affine drop worse than 0.04; meaningful jitter gain above 0.02.
5. **Evidence quantity.** A broad claim requires at least 8 fresh tasks
   and 5 paired seeds per task.
6. **Baselines.** Before promoting a research candidate, compare against
   full MultiRocket, TS2Vec, TimeMAE and MOMENT under a fixed common
   protocol. Compact 336-kernel MultiRocket is not enough.
7. **Governed action.** The controller may perform sandbox experiments
   only. Master branch modification, deployment or SOTA claims remain
   prohibited. Human confirmation is required to permit promotion.
8. **1-shot validity.** Removing one of exactly one labeled example per
   class is not a valid class-preserving LOO procedure. Prespecify
   the 1-shot fallback; compare 3/5-shot by leaving a full labeled
   episode out.

## Failure memory is first-class data

The ledger `MESA_SSL/hclm_failure_ledger_2026-11-23.json` documents
16 verified or qualified recurring failure modes: class information
removed by canonicalization; test-vs-train tuning; concept alignment
without predictive gain; SPC blindness to waveform jitter; neural
features receiving zero kernel weight; context misrouting;
unconditional physics/Hopfield mixing; unstable sparse DG codes;
few-shot Transformer versus CNN negative transfer; invalid 1-shot LOO;
unjustified ARL assertions; inadequate baselines; reused holdouts;
two-seed claims; incorrect transfer of online logloss regret into
Macro-F1 claims; and physics-analogy overstatement.

This state is read on **every** HCLM controller invocation rather than
retold as a vague claim to have remembered failures.

## Executed HCLM decision (latest 1/3/5-shot riskgate audit)

Input archive:
`MESA_SSL/riskgate_biophysics_holdout_2026-11-23.json`, GitHub Actions
https://github.com/kimphuctran/MESA_SSL/actions/runs/45280823062.

The controller computes the paired effect of `BioPhysics risk-gated`
relative to `Dual MR episodic`, without substituting the post-hoc
best method as the actual decision. It tests hard risk/coverage limits
**before** choosing what to do next.

For this archive the action is `RUN_SANDBOX`, not `PROMOTION`.
The selected experiment is
`AFFINE_FALLBACK_CALIBRATION`, because mean affine performance
regresses while the waveform-jitter performance improves.

The next independent experiment is frozen at
`MESA_SSL/HCLM_FROZEN_AFFINE_EXPERIMENT_2026-11-23.json` in a
**separate research branch**, before any of its TEST evaluation:
Strawberry, ToeSegmentation1, ToeSegmentation2 and Computers,
seeds 5601/5602, 1/3/5-shot.
It tests a conservative baseline fallback with independent
0.05 p-value limits, selected experts requiring at least 0.04
TRAIN-only evidence of improvement, and safe reversion on
conflicting alarms.

The workflow must execute the HCLM governor and check its selected
sandbox action **before** any neural experiment code is run.

## Explicit novelty / reliability boundary

This protocol is a practical instantiation of HCLM-guided **research
governance**. It is neither a proof of HCLM's full mathematical theory
nor a self-modifying ChatGPT model. Do not state the scientific
controller has attained universal reliability because the CI workflow
passes. New evidence must pass the prespecified acceptance rules;
negative results remain in Git and the failure ledger.

## Executed or frozen artifacts

- Controller source: `MESA_SSL/hclm_search_governor.py`
- Persistent failure ledger: `MESA_SSL/hclm_failure_ledger_2026-11-23.json`
- HCLM decision workflow: `.github/workflows/hclm-governed-search.yml`
- Next frozen experiment: `MESA_SSL/HCLM_FROZEN_AFFINE_EXPERIMENT_2026-11-23.json`
- HCLM-sanctioned evaluation code:
  `MESA_SSL/mesa_biophysics_memory.py` on the sandbox branch
- Research benchmark workflow:
  `.github/workflows/hclm-affine-sandbox.yml` on the sandbox branch.
