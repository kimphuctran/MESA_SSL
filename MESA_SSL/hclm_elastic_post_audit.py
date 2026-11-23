#!/usr/bin/env python3
"""External HCLM metacontroller audits frozen ElasticSynapse research evidence.

HCLM here governs scientific decisions. It is not a neural-network layer.
All thresholds were frozen before TEST. The archive's TEST labels were used
only to report existing candidate metrics, never to change its policy.
"""
import argparse,json,math
from pathlib import Path
from MESA_SSL.hclm_search_governor import (
    METRICS,paired_archive,dataset_summary,cluster_bootstrap)

def evaluate(archive,contract):
    r=archive["records"]
    fixed=contract["data_protocol"]
    label=contract["performance_admissibility"]
    if set(x["dataset"] for x in r)!=set(fixed["independent_check_datasets"]):
        raise ValueError("Unexpected TEST dataset roster")
    if set(x["seed"] for x in r)!=set(fixed["seeds"]):
        raise ValueError("Unexpected seed roster")
    if set(x["budget"] for x in r)!=set(fixed["budgets"]):
        raise ValueError("Unexpected shot budget")
    nmethods=len(fixed["method_set"])
    if len(r)!=len(fixed["independent_check_datasets"])*len(fixed["seeds"])*len(fixed["budgets"])*nmethods:
        raise ValueError("Incomplete run matrix")
    results={}
    for b in fixed["budgets"]:
        matched,methods=paired_archive(r,b,
              "Dual MR episodic","Elastic HCLM-authorized router")
        byds=dataset_summary(matched)
        ci=cluster_bootstrap(matched,replicates=1000,seed=2050+b)
        d={k:ci[k]["mean"] for k in METRICS}
        fail=[]
        if d["clean_macro_f1"]<label["mean_clean_delta_vs_dual_at_least"]:
            fail.append("clean noninferiority")
        if d["affine_macro_f1"]<label["mean_affine_delta_vs_dual_at_least"]:
            fail.append("affine noninferiority")
        if d["jitter_macro_f1"]<label["mean_jitter_delta_vs_dual_at_least"]:
            fail.append("jitter improvement threshold")
        if min(x["deltas"]["clean_macro_f1"] for x in byds.values())<\
                label["individual_dataset_clean_delta_at_least"]:
            fail.append("one dataset has unacceptable clean damage")
        if min(x["deltas"]["affine_macro_f1"] for x in byds.values())<\
                label["individual_dataset_affine_delta_at_least"]:
            fail.append("one dataset has unacceptable affine damage")
        results[b]=dict(deltas=d,descriptive_bootstrap=ci,
            by_dataset=byds,frozen_sandbox_conditions_passed=not fail,
            conditions_failed=fail,
            promoted=False,
            next_action="KEEP_SANDBOX_AND_RUN_STRONG_COMPARATORS",
            note="Even if exploratory thresholds pass: 4 tasks 2 seeds and unmatched strong baselines cannot establish dominance")
    return dict(controller="HCLM research metacontroller",
        research_action_scope="sandbox only; no TEST-selected model changes",
        contract_id=contract["contract_id"],original_source=contract["source_revision_before_test"],
        archive_run=archive["run"],n_records=len(r),
        results=results,already_inspected_data_now_development=True,
        no_guarantee_of_universal_superiority=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--archive",default="MESA_SSL/HCLM_ELASTIC_INDEPENDENT_RESULTS_2026-11-24.json")
    p.add_argument("--contract",default="MESA_SSL/HCLM_ELASTIC_FROZEN_PROTOCOL_2026-11-24.json")
    p.add_argument("--out",default="results/hclm_elastic_decision.json")
    args=p.parse_args()
    result=evaluate(json.load(open(args.archive)),
                    json.load(open(args.contract)))
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    for b,z in result["results"].items():
        print("HCLM_ELASTIC_JSON="+json.dumps(dict(budget=b,effects=z["deltas"],
               risks=z["conditions_failed"],pass_sandbox=z["frozen_sandbox_conditions_passed"],
               promoted=z["promoted"]),allow_nan=False),flush=True)
