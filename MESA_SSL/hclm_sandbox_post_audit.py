#!/usr/bin/env python3
"""Post-experiment HCLM control, driven ONLY by PRE-FROZEN contract thresholds.

The TEST labels are now readable because the experiment has completed.
They may evaluate the frozen hypothesis but CANNOT be reused to tune
its decision policy or make a new method 'best' by TEST selection.
The script compares paired dataset/seed supports and emits admissibility
feedback to the HCLM research loop, without merging/deploying anything.
"""
import argparse
import json
import math
from pathlib import Path

from MESA_SSL.hclm_search_governor import (
    METRICS, paired_archive, dataset_summary, cluster_bootstrap,
)

BASE="Dual MR episodic"
NEW="BioPhysics HCLM-guarded"
OLD="BioPhysics risk-gated"


def audit(all_records,contract,seed=20261123):
    expected_ds=set(contract["protocol"]["datasets"])
    expected_seeds=set(contract["protocol"]["seeds"])
    budgets=contract["protocol"]["labeled_train_support_per_class"]
    frozen=contract["risk_envelope"]
    received_ds=set(x["dataset"] for x in all_records)
    if received_ds!=expected_ds:
        raise ValueError("Unexpected or missing dataset: "+str(received_ds))
    results={}
    for b in budgets:
        r=[x for x in all_records if x.get("budget")==b]
        ids=set((x["dataset"],x["seed"]) for x in r)
        expected={(d,s) for d in expected_ds for s in expected_seeds}
        if ids!=expected:
            raise ValueError("Incomplete dataset x seed x budget evaluation")
        for x in r:
            if x.get("status")!="EXECUTED":raise ValueError("Nonexecuted record")
            if x.get("seed") not in expected_seeds:raise ValueError("Unknown seed")
            if x.get("label_ids") is None or len(x["label_ids"])!=x["n_labeled"]:
                raise ValueError("Bad support provenance")
            for k in METRICS:
                val=x.get(k)
                if not isinstance(val,(float,int)) or not math.isfinite(val):
                    raise ValueError("Non-finite or absent metric")
        matched,methods=paired_archive(r,b,BASE,NEW)
        if OLD not in methods:
            raise ValueError("Missing frozen old policy comparator")
        summaries=dataset_summary(matched)
        intervals=cluster_bootstrap(matched,seed=seed+b,replicates=2000)
        mean_d={k:intervals[k]["mean"] for k in METRICS}
        reject_reasons=[]
        if mean_d["clean_macro_f1"]<frozen["clean_mean_noninferiority_delta_at_least"]:
            reject_reasons.append("clean non-inferiority margin violated")
        if mean_d["affine_macro_f1"]<frozen["affine_mean_noninferiority_delta_at_least"]:
            reject_reasons.append("affine non-inferiority margin violated")
        if mean_d["jitter_macro_f1"]<frozen["jitter_mean_improvement_at_least"]:
            reject_reasons.append("jitter improvement less than threshold")
        if min(x["deltas"]["clean_macro_f1"] for x in summaries.values())<\
                frozen["any_dataset_clean_delta_at_least"]:
            reject_reasons.append("per-dataset clean harm exceeds limit")
        if min(x["deltas"]["affine_macro_f1"] for x in summaries.values())<\
                frozen["any_dataset_affine_delta_at_least"]:
            reject_reasons.append("per-dataset affine harm exceeds limit")
        old_matched,_=paired_archive(r,b,OLD,NEW)
        old_effect={
            k:sum(sum(x["delta"][k] for x in obs)/len(obs)
                  for obs in old_matched.values())/len(old_matched)
            for k in METRICS}
        caution="4 datasets x 2 seeds does not pass 8 datasets x 5 seed promotion evidence floor"
        results[b]=dict(
            budget=b,
            effect_vs_dual_mr=mean_d,
            effect_vs_previous_riskgate=old_effect,
            descriptive_bootstrap=intervals,
            task_level_effects=summaries,
            fixed_thresholds_passed=not reject_reasons,
            fixed_threshold_failures=reject_reasons,
            allowed_action=("RESEARCH_SANDBOX_CANDIDATE" if not reject_reasons
                            else "REJECT_OR_REVISE_IN_SANDBOX"),
            promotion="FORBIDDEN",
            promotion_blockers=[caution,
                "Full MultiRocket / TS2Vec / TimeMAE / MOMENT not matched",
                "No physical ground truth that synthetic disturbances preserve labels"])
    return dict(
       experiment_contract_id=contract["contract_id"],
       source_of_authority="HCLM research-governed action",not_neural_module=True,
       locked_before_test=contract["scientifically_frozen_before_results"],
       results=results,
       human_required_before_promotion=True,
       all_new_test_sets_now_development=True,
       no_posthoc_optimization_from_these_test_labels=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--results",required=True)
    p.add_argument("--contract",default="MESA_SSL/HCLM_FROZEN_AFFINE_EXPERIMENT_2026-11-23.json")
    p.add_argument("--output",default="results/hclm_postrun_decision.json")
    args=p.parse_args()
    with open(args.results,encoding="utf-8") as f:rows=json.load(f)
    with open(args.contract,encoding="utf-8") as f:contract=json.load(f)
    result=audit(rows,contract)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
    for b,d in result["results"].items():
        print("HCLM_POSTRUN_JSON="+json.dumps(dict(budget=b,action=d["allowed_action"],
            effects=d["effect_vs_dual_mr"],issues=d["fixed_threshold_failures"],
            promotion=d["promotion"]),allow_nan=False),flush=True)
