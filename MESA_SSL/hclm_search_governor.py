#!/usr/bin/env python3
"""HCLM Research Governor — an operational controller of *research actions*.

This is deliberately NOT another HCLM component inside MESA-SSL.
It drives the experiment planning and advancement process:
    OBSERVE -> CALIBRATE -> ADMISSIBILITY -> ACT -> AUDIT/LEARN.
The governor reads the actual archived completed experiments and an explicit
human-scoped contract, measures paired adverse effects, checks uncertainty
and missing evidence, and emits the next *permitted experiment*.

No TEST labels are used to tune an evaluated classifier. Archived TEST
metrics *can* diagnose research failures and motivate NEW predeclared
hypotheses; those archive datasets thereby become development data and
cannot serve again as an untouched confirmation. Claims must be evaluated
on fresh data after the candidate and its acceptance rules are frozen.

Entropy here means uncertainty of a discrete research decision. It is an
operational control diagnostic, not a proof of the gradient-based entropy
force of the original HCLM learning mechanics. A future formal optimizer
could use the full differentiable free-energy formulation.

Inspired by Tran, K. P., Human-Centered Learning Mechanics:
Control, Calibration and Governed Action for Industrial AI Decision Systems
(Springer Nature, in press; user-provided citation status).

Copyright (c) research authors. Reproduce branch/commit/run provenance.
"""
import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean

METRICS=("clean_macro_f1","affine_macro_f1","jitter_macro_f1")
REFERENCE="Dual MR episodic"
CANDIDATE="BioPhysics risk-gated"
COMPARATORS=("Raw MR episodic","Canonical MR episodic",
             "BioPhysics full","BioPhysics safe TRAIN","Raw MR ridge")

@dataclass(frozen=True)
class Constraints:
    min_independent_datasets:int=8
    min_paired_seeds_per_dataset:int=5
    clean_noninferiority_margin:float=.01
    affine_noninferiority_margin:float=.01
    jitter_minimum_mean_gain:float=.02
    max_single_dataset_clean_drop:float=.04
    max_single_dataset_affine_drop:float=.04
    ci_level:float=.95
    number_of_bootstraps:int=2000
    min_quantile_jitter_gain:float=.0
    required_full_baselines:tuple=(
        "MultiRocket full","TS2Vec","TimeMAE","MOMENT")
    permitted_actions:tuple=("AUDIT","DESIGN_SANDBOX","RUN_SANDBOX")
    forbidden_actions:tuple=("MERGE_MASTER","DEPLOY","CLAIM_SOTA")


def load_json(p):
    with open(p,encoding="utf-8") as f:return json.load(f)


def available_targets(archive):
    r=archive.get("records",[])
    if not r:raise ValueError("Missing archived records")
    protocol=archive.get("protocol",{})
    if protocol.get("train_only_expert_selection") is not True:
        raise ValueError("Archive does not assert TRAIN-only expert selection")
    if protocol.get("heldout_after_development") is not True:
        raise ValueError("Archive lacks predeclared validation role")
    jobs=archive.get("jobs",[])
    if not jobs or any(z.get("conclusion")!="success" for z in jobs):
        raise ValueError("A source workflow job is missing or unsuccessful")
    for rec in r:
        if rec.get("status")!="EXECUTED":
            raise ValueError("Incomplete or simulated metric record")
        for metric in METRICS:
            v=rec.get(metric)
            if not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1:
                raise ValueError(f"Invalid {metric}")
        labels=rec.get("label_ids")
        if not isinstance(labels,list) or not labels or len(labels)!=len(set(labels)):
            raise ValueError("Missing/duplicated labeled support provenance")
        if len(labels)!=rec.get("n_labeled"):
            raise ValueError("Labeled support count inconsistent with record")
    return r


def paired_archive(records,budget=5,reference=REFERENCE,candidate=CANDIDATE):
    """Never pool different seed- or dataset-specific labels."""
    by={}
    methods=set()
    for r in records:
        if r["budget"]!=budget:continue
        key=(r["dataset"],r["seed"])
        by.setdefault(key,{})
        if r["method"] in by[key]:
            raise ValueError("Duplicate method / dataset / seed / budget")
        by[key][r["method"]]=r
        methods.add(r["method"])
    if not by:raise ValueError("No budget observations")
    out={}
    for (dataset,seed),items in by.items():
        if reference not in items or candidate not in items:
            raise ValueError("Missing matched comparator")
        a,b=items[reference],items[candidate]
        if a["label_ids"]!=b["label_ids"]:
            raise ValueError("Reference and candidate use different TRAIN labels")
        if a["n_labeled"]!=b["n_labeled"]:
            raise ValueError("Mismatched label budgets")
        if a["test_cases"]!=b["test_cases"]:
            raise ValueError("Mismatched official TEST size")
        deltas={k:float(b[k]-a[k]) for k in METRICS}
        out.setdefault(dataset,[]).append(dict(seed=seed,delta=deltas,
            reference={k:float(a[k]) for k in METRICS},
            candidate={k:float(b[k]) for k in METRICS},
            labels=list(a["label_ids"])))
    return out,methods


def cluster_bootstrap(matched,level=.95,replicates=2000,seed=20261123):
    """Exploratory paired cluster bootstrap: dataset first, seed second.

    Because the 4 validation tasks have already been inspected and are not
    a random superpopulation sample, these are descriptive sensitivity
    intervals, NOT nominal frequentist coverage guarantees.
    """
    rng=random.Random(seed)
    tasks=sorted(matched)
    observations={}
    for k in METRICS:
        bytask=[mean(v["delta"][k] for v in matched[d]) for d in tasks]
        observations[k]=mean(bytask)
    sampled={k:[] for k in METRICS}
    for _ in range(replicates):
        # Resample tasks, then independently resample their paired seeds.
        take=[rng.choice(tasks) for _ in tasks]
        # The same resampled paired observations are used for ALL metrics;
        # independent resampling per metric would destroy the joint
        # clean/affine/jitter risk dependence.
        paired_draws=[
            [rng.choice(matched[d]) for _ in matched[d]] for d in take
        ]
        for k in METRICS:
            value=mean(mean(obs["delta"][k] for obs in sub)
                       for sub in paired_draws)
            sampled[k].append(value)
    def pct(a,q):
        a=sorted(a)
        t=(len(a)-1)*q
        lo=math.floor(t);hi=math.ceil(t)
        return a[lo]+(t-lo)*(a[hi]-a[lo])
    result={}
    for k in METRICS:
        vals=sampled[k]
        result[k]=dict(mean=observations[k],
           lower=pct(vals,(1-level)/2),
           upper=pct(vals,1-(1-level)/2),
           positive_fraction=sum(z>0 for z in vals)/len(vals),
           negative_fraction=sum(z<0 for z in vals)/len(vals),
           interval_type="exploratory_hierarchical_bootstrap_not_confirmatory")
    return result


def dataset_summary(matched):
    out={}
    for d in sorted(matched):
        rows=matched[d]
        out[d]=dict(
            n_seed=len(rows),
            seeds=sorted(x["seed"] for x in rows),
            deltas={k:mean(x["delta"][k] for x in rows) for k in METRICS},
            ref={k:mean(x["reference"][k] for x in rows) for k in METRICS},
            candidate={k:mean(x["candidate"][k] for x in rows) for k in METRICS})
    return out


def controlled_admissibility(summary,interval,methods,limits):
    blockers=[]
    observations={
      "available_independent_tasks":len(summary),
      "min_seeds_per_task":min(x["n_seed"] for x in summary.values()),
      "available_full_baselines":sorted(set(limits.required_full_baselines)
                                         .intersection(methods)),
      "required_full_baselines":list(limits.required_full_baselines),
      "mean_effect":{k:interval[k]["mean"] for k in METRICS}
    }
    if len(summary)<limits.min_independent_datasets:
        blockers.append("insufficient untouched independent datasets")
    if min(x["n_seed"] for x in summary.values())<limits.min_paired_seeds_per_dataset:
        blockers.append("insufficient independent paired seeds")
    if interval["clean_macro_f1"]["mean"]< -limits.clean_noninferiority_margin:
        blockers.append("mean clean Macro-F1 non-inferiority failed")
    if interval["affine_macro_f1"]["mean"]< -limits.affine_noninferiority_margin:
        blockers.append("mean affine Macro-F1 non-inferiority failed")
    if interval["jitter_macro_f1"]["mean"]<limits.jitter_minimum_mean_gain:
        blockers.append("minimum meaningful jitter effect not met")
    if interval["jitter_macro_f1"]["lower"]<=limits.min_quantile_jitter_gain:
        blockers.append("exploratory jitter gain interval crosses zero")
    if any(x["deltas"]["clean_macro_f1"] <
            -limits.max_single_dataset_clean_drop for x in summary.values()):
        blockers.append("at least one dataset exceeds allowed clean degradation")
    if any(x["deltas"]["affine_macro_f1"] <
            -limits.max_single_dataset_affine_drop for x in summary.values()):
        blockers.append("at least one dataset exceeds allowed affine degradation")
    missing=sorted(set(limits.required_full_baselines)-set(methods))
    if missing:blockers.append("full-strength matched baselines missing: "+", ".join(missing))
    return dict(admissible_for_promotion=not blockers,
                blockers=blockers,observations=observations,
                governance_scope="RESEARCH_SANDBOX_ONLY")


def binary_entropy(p):
    p=min(max(float(p),1e-12),1-1e-12)
    return -(p*math.log(p)+(1-p)*math.log(1-p))


def plan_next_action(interval,governance,ledger,limits):
    """HCLM control policy: choose a *research* action, not a model result.

    All options are predeclared. The selected target is based on observed
    failure severity, epistemic uncertainty, existing held-out evidence and
    experiment cost. The inspected tasks become DEVELOPMENT tasks. No
    claim/promotion action is available without passing the gate.
    """
    deficit=max(0.,-interval["affine_macro_f1"]["mean"]-
                  limits.affine_noninferiority_margin)
    clean_def=max(0.,-interval["clean_macro_f1"]["mean"]-
                  limits.clean_noninferiority_margin)
    p_jitter=interval["jitter_macro_f1"]["positive_fraction"]
    uncertainty=binary_entropy(p_jitter)/math.log(2.)
    # Score is a transparent scheduling heuristic, not a calibrated VOI.
    # An uncertainty term is an operational analogue of HCLM entropy;
    # no claim of theta-entropy-gradient identities is made.
    options=[
       dict(id="AFFINE_FALLBACK_CALIBRATION",
            hypothesis="Train-only affine gating and uncertainty-safe fallback",
            score=3.0+35*deficit+2*clean_def+.25*uncertainty,
            cost_units=2,scope="SANDBOX"),
       dict(id="MATCHED_FULL_BASELINE_AUDIT",
            hypothesis="Same-seed full-kernel MultiRocket and official SSL baselines",
            score=2.6+0.35*len(limits.required_full_baselines),
            cost_units=3,scope="SANDBOX"),
       dict(id="EXPERT_UNCERTAINTY_AUDIT",
            hypothesis="Audit paired uncertainty, chart failure and 1-shot limits",
            score=2.4+.35*uncertainty,
            cost_units=2,scope="SANDBOX"),
       dict(id="NEW_NEURAL_ARCHITECTURE",
            hypothesis="Build another network only after controlled baselines",
            score=.25,cost_units=7,scope="SANDBOX"),
    ]
    completed={name for item in ledger for name in
               item.get("completed_experiment_ids",[])}
    for x in options:
        x["already_executed"]=x["id"] in completed
        x["priority_per_cost"]=(x["score"]/x["cost_units"]
             if not x["already_executed"] else None)
        x["historical_test_reuse_for_confirmatory_evidence"]=False
    # The HCLM feedback ledger stops repeated tuning of the same completed
    # experiment after its TEST has been inspected.
    available=[x for x in options if not x["already_executed"]]
    if not available:raise ValueError("No unused admissible research actions")
    ordered=sorted(available,key=lambda x:(-x["priority_per_cost"],x["id"]))
    nxt=ordered[0]
    return dict(
        action=("PROMOTION_PERMITTED" if governance["admissible_for_promotion"]
                else "RUN_SANDBOX"),
        next_experiment=nxt,
        alternatives=ordered[1:],
        hclm_state=dict(
            observed_failure_affine=interval["affine_macro_f1"]["mean"],
            observed_failure_clean=interval["clean_macro_f1"]["mean"],
            observed_jitter_gain=interval["jitter_macro_f1"]["mean"],
            decision_entropy_proxy=uncertainty,
            feedback_integrated_from_failures=len(ledger),
            blocked_completed_experiments=sorted(completed)),
        governed_controls=dict(
            allow_only=list(limits.permitted_actions),
            forbidden=list(limits.forbidden_actions),
            requires_human_approval_for_master_merge=True,
            requires_new_frozen_test_set_for_any_promotion=True,
            no_test_oracle_selection=True),
        rule="No promotion while any evidence or risk gate is violated")


def controller(archive,ledger,limits=Constraints(),budget=5):
    records=available_targets(archive)
    matched,methods=paired_archive(records,budget)
    summary=dataset_summary(matched)
    interval=cluster_bootstrap(matched,level=limits.ci_level,
                replicates=limits.number_of_bootstraps)
    gate=controlled_admissibility(summary,interval,methods,limits)
    decision=plan_next_action(interval,gate,ledger,limits)
    tested=set(x["dataset"] for x in records)
    previous=set()
    for issue in ledger:
        previous.update(issue.get("datasets",[]))
    return dict(
        controller="HCLM Research Governor 1.0",
        hclm_role="metacontroller of scientific search, NOT neural architecture",
        source_provenance=dict(
            archived_run_id=archive.get("run"),
            head_commit=archive.get("source_commit"),
            validation_datasets=sorted(tested),
            prior_development_datasets=sorted(previous),
            evaluation_budget=budget),
        constraints=asdict(limits),
        calibrated_effects=interval,
        paired_dataset_review=summary,
        admissibility=gate,
        governed_decision=decision,
        failure_ledger=ledger,
        evidence_reuse_policy=dict(
            all_examined_datasets_now_development=True,
            future_confirmatory_test_must_be_frozen_and_new=True,
            no_retroactive_oracle_or_multiple_test_selection=True,
            intervention_labels_are_unverified_assumptions=True))




def integrate_completed_sandbox_feedback(hclm_audit,completed_archive):
    """Reconcile a sandbox TEST event into the HCLM RESEARCH state.

    TEST data is now consumed and can diagnose a failure, but never choose
    revised parameters for its own validation. The old model and new model
    are compared on exactly matched TRAIN support, dataset and seed IDs.
    """
    events=completed_archive.get("rows",completed_archive.get("records"))
    if not events:raise ValueError("Missing completed feedback records")
    b=5
    matched,methods=paired_archive(events,b,
                reference="BioPhysics risk-gated",
                candidate="BioPhysics HCLM-guarded")
    effect=cluster_bootstrap(matched,replicates=2000,seed=20261124)
    summaries=dataset_summary(matched)
    deltas={k:effect[k]["mean"] for k in METRICS}
    used=set(x["dataset"] for x in events)
    original=set(hclm_audit["source_provenance"]["validation_datasets"])
    hclm_audit["feedback_from_executed_sandbox"]=dict(
      run_id=completed_archive.get("run"),
      affected_budget=b,
      comparison="HCLM-guarded minus incumbent BioPhysics risk-gated",
      paired_delta=deltas,
      descriptive_uncertainty=effect,
      per_dataset=summaries,
      new_datasets_now_development=sorted(used),
      no_confirmatory_reuse=True,
      incumbent_improvement_on_all_three=False if any(v<0 for v in deltas.values()) else True)
    hclm_audit["source_provenance"]["all_examined_validation_datasets"]=sorted(original|used)
    hclm_audit["governed_decision"]["additional_feedback"]=(
      "The previous sandbox was executed: baseline risk limits passed, "
      "but the new candidate slightly underperformed the incumbent on "
      "matched clean/affine/jitter metrics. Do not repeat that same "
      "experiment or tune on the four consumed TEST datasets.")
    return hclm_audit


def numerical_tests():
    # The paired test must refuse a fake improvement with different supports.
    r={"dataset":"Toy","seed":1,"budget":3,"status":"EXECUTED",
       "n_labeled":6,"test_cases":12,"label_ids":[1,2,3,4,5,6],
       "clean_macro_f1":.8,"affine_macro_f1":.7,"jitter_macro_f1":.5}
    a=dict(r,method=REFERENCE)
    b=dict(r,method=CANDIDATE,clean_macro_f1=.79,
           affine_macro_f1=.67,jitter_macro_f1=.65)
    paired,_=paired_archive([a,b],3)
    assert abs(paired["Toy"][0]["delta"]["jitter_macro_f1"]-.15)<1e-12
    effects=cluster_bootstrap(paired,replicates=60)
    assert abs(effects["clean_macro_f1"]["mean"]+.01)<1e-10
    wrong=dict(b,label_ids=[8,2,3,4,5,6])
    try:paired_archive([a,wrong],3)
    except ValueError:pass
    else:raise AssertionError("Mismatched label supports were accepted")
    dummy=dict(r,method=CANDIDATE,clean_macro_f1=1.5)
    a2=dict(r,method=REFERENCE)
    fake=dict(records=[a2,dummy],protocol=dict(
        train_only_expert_selection=True,
        heldout_after_development=True),jobs=[dict(conclusion="success")])
    try:available_targets(fake)
    except ValueError:pass
    else:raise AssertionError("Unbounded fake score accepted")
    assert binary_entropy(.5)>binary_entropy(.99)
    return True


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",default="MESA_SSL/riskgate_biophysics_holdout_2026-11-23.json")
    ap.add_argument("--ledger",default="MESA_SSL/hclm_failure_ledger_2026-11-23.json")
    ap.add_argument("--out",default="results/hclm_search")
    ap.add_argument("--budget",type=int,default=5)
    ap.add_argument("--test-only",action="store_true")
    ap.add_argument("--feedback-archive",default=None,
          help="Previously completed sandbox TEST records, now consumed as development feedback")
    args=ap.parse_args()
    numerical_tests()
    if not args.test_only:
        archive=load_json(args.archive)
        ledger=load_json(args.ledger)
        audit=controller(archive,ledger,budget=args.budget)
        if args.feedback_archive:
            feedback=load_json(args.feedback_archive)
            audit=integrate_completed_sandbox_feedback(audit,feedback)
        output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
        content=json.dumps(audit,indent=2,allow_nan=False)
        (output/"governance_decision.json").write_text(content,encoding="utf-8")
        # Stable immutable hash for the decision contract.
        sha=hashlib.sha256(content.encode()).hexdigest()
        compact=dict(budget=args.budget,
           action=audit["governed_decision"]["action"],
           next_experiment=audit["governed_decision"]["next_experiment"],
           failures=len(audit["failure_ledger"]),
           gate=audit["admissibility"],
           observed=audit["calibrated_effects"],
           audit_sha256=sha)
        print("HCLM_DECISION_JSON="+json.dumps(compact,allow_nan=False),flush=True)
