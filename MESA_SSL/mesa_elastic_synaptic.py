#!/usr/bin/env python3
"""MESA-ElasticSynapse: Mechanism-preserving elastic episodic memory.

Architecture changes the *decision primitive*, not merely the NN topology.
  1. Slow cortical SSL encodes raw and shape using unlabeled TRAIN.
  2. Exact affine decomposition retains mean/slope/scale + canonical shape.
  3. Heat-regularized multiscale shape memories align short segments with
     Sakoe-Chiba constrained dynamic time warping (DTW).
  4. Fast hippocampal imprinting stores 1/3/5 labeled TRAIN cases per class;
     no parameter gradient or extra labels when a new class arrives.
  5. TRAIN leave-one-episode-out selection permits elastic memory only if
     it improves a specified nuisance regime without unacceptable loss.
  6. Separately calibrated waveform and affine charts route cases.

HCLM is used separately as the METACONTROLLER OF RESEARCH; this classifier
does not pretend to be the HCLM algorithm. The supervised train-only policy
is a predeclared experiment suggested by the HCLM failure ledger.

Existing DTW, scattering, shapelet, CLS and mechanism features have prior art;
the novelty hypothesis is their testable governed combination.

No TEST data is used for fitting any feature, similarity metric, nuisance
chart, classifier or hyperparameters. TEST labels only score a frozen model.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score,accuracy_score
from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import affine_gauge
from MESA_SSL.mesa_neuromem import (
    independent_budget_ids,build_representations,retrieve,softmax_class)
METHODS=("Dual MR episodic","Canonical MR episodic",
         "Cortex episodic","Elastic slow memory",
         "Elastic shape-concept memory","Elastic HCLM-authorized router")
CANONICAL_SMOOTHING=(2.0,7.0)
DOWNSAMPLE=96
RADIUS=12
EVIDENCE_MARGIN=.04
MAX_CLEAN_DEGRADATION=.04
CHART_ALPHA=.05

def downsample(x,steps=96):
    x=np.asarray(x,dtype=np.float64)
    t=np.linspace(0.,1.,x.shape[1]);g=np.linspace(0.,1.,steps)
    out=np.empty((len(x),steps,x.shape[2]),dtype=np.float64)
    for i in range(len(x)):
        for c in range(x.shape[2]):
            out[i,:,c]=np.interp(g,t,x[i,:,c])
    return out

@njit(cache=True)
def dtw_distance(a,b,band):
    """Hard DTW with fixed Sakoe-Chiba window and positive path cost."""
    n=a.shape[0]
    c=a.shape[1]
    old=np.empty(n+1,np.float64)
    new=np.empty(n+1,np.float64)
    old[:]=np.inf
    old[0]=0.
    for i in range(1,n+1):
        new[:]=np.inf
        lo=max(1,i-band);hi=min(n,i+band)
        for j in range(lo,hi+1):
            cost=0.
            for k in range(c):
                q=a[i-1,k]-b[j-1,k]
                cost+=q*q
            cost=cost/c
            best=old[j-1]
            if old[j]<best:best=old[j]
            if new[j-1]<best:best=new[j-1]
            new[j]=cost+best
        hold=old
        old=new
        new=hold
    return old[n]/n

@njit(cache=True)
def dist_matrix(x,ref,band):
    z=np.empty((len(x),len(ref)),dtype=np.float64)
    for i in range(len(x)):
        for j in range(len(ref)):
            z[i,j]=dtw_distance(x[i],ref[j],band)
    return z

def shape_views(samples):
    q=[affine_gauge(x)[0] for x in samples]
    ref=[downsample(x) for x in q]
    return ref,q

def cosine_cortex(cortex):
    v=cortex["views"]
    return tuple(np.column_stack([v["cortRaw"][j],
                                  v["cortShape"][j]])
                 for j in range(6))

def build_elastic(cortex,max_ids):
    full,q=shape_views(cortex["samples"])
    scales=[[],[]]
    for arr in full:
        for i,sigma in enumerate(CANONICAL_SMOOTHING):
            sm=gaussian_filter1d(arr,sigma,axis=1,mode="reflect")
            scales[i].append(np.ascontiguousarray(sm,dtype=np.float64))
    # Multiplicative normalization uses ONLY random pairs of unlabeled TRAIN,
    # not distances or targets in any TEST set.
    rng=np.random.default_rng(28441)
    n=len(full[0]);k=min(48,max(4,n//2))
    aa=rng.integers(0,n,size=k);bb=rng.integers(0,n,size=k)
    scales_norm=[]
    for i in range(2):
        v=np.array([dtw_distance(scales[i][0][a],scales[i][0][b],RADIUS)
                    for a,b in zip(aa,bb)])
        scales_norm.append(max(float(np.median(v[v>1.e-10])) if np.any(v>1.e-10)
                               else 1.,1.e-6))
    # Normalize the two physical temporal scales separately.
    matrices=[]
    for mode in range(6):
        matrices.append(.45*dist_matrix(scales[0][mode],
                    scales[0][0][max_ids],RADIUS)/scales_norm[0]
                  +.55*dist_matrix(scales[1][mode],
                    scales[1][0][max_ids],RADIUS)/scales_norm[1])
    # Exact analytical concepts are retained for a context-dependent metric.
    c=[affine_gauge(x)[1] for x in cortex["samples"]]
    mean=c[0].mean(axis=0)
    std=np.maximum(c[0].std(axis=0),.1)
    concept=[np.clip((z-mean)/std,-5,5) for z in c]
    concept_ref=concept[0][max_ids]
    for mode in range(6):
        diff=concept[mode][:,None,:]-concept_ref[None,:,:]
        dist=np.sum(diff*diff,axis=-1)/max(1,concept_ref.shape[1])
        # Pure canonical DTW and concept-aware DTW remain independent
        # experts, allowing an affine alarm to turn concept sensitivity off.
        if mode==0:physics=[]
        physics.append(matrices[mode]+.25*dist)
    return matrices,physics,dict(train_only_scales=scales_norm,
                                 k_reference=len(max_ids),steps=DOWNSAMPLE,
                                 radius=RADIUS)

def distance_class_scores(D,support_cols,support_labels,classes):
    out=[]
    for k in classes:
        p=D[:,support_cols[support_labels==k]]
        # 0.7 best episodic match + 0.3 prototype distance,
        # independent of the number of class memories.
        out.append(-(.7*p.min(axis=1)+.3*p.mean(axis=1)))
    return np.column_stack(out)

def fixed_matrix_probs(scores):
    return softmax_class(scores,temperature=7.)

def basic_synthetic_tests():
    rng=np.random.default_rng(725)
    x=rng.normal(size=(6,31,2))
    z=downsample(x)
    assert z.shape==(6,DOWNSAMPLE,2)
    assert dtw_distance(z[0],z[0],RADIUS)<1e-12
    assert dtw_distance(z[0],z[1],RADIUS)>0
    m=dist_matrix(z,z[:2],RADIUS)
    assert m.shape==(6,2) and (m>=0).all()
    y=np.repeat(np.arange(3),5)
    ids=independent_budget_ids(y,3701,(1,3,5))
    assert len(ids[3])==9
    scores=distance_class_scores(rng.random((7,15)),
                                 ids[3],y[ids[3]],np.unique(y))
    assert scores.shape==(7,3)
    return True

def select_train_only(v,dist,physics,y,support,columns,classes,
                      charts,budget):
    """Choose nuisance-specific memories using grouped TRAIN LOO.

    The selected reference excludes the labeled original episode, its
    projected physical features, and all training augmentations.
    No class-preserving LOO is possible with 1-shot: keep fixed defaults.
    """
    chosen={"clean":"Dual","affine":"Canonical",
            "jitter":"Cortex"}
    metric={}
    if budget==1:
        return chosen,dict(reason="Fixed predeclared 1-shot policies; LOO invalid")
    truth=y[support]
    scores={name:[[] for _ in range(3)] for name in
            ("Dual","Canonical","Cortex","Elastic","ElasticConcept")}
    for i,origin in enumerate(support):
        remaining=np.delete(support,i)
        remain_col=np.array([columns[int(a)] for a in remaining])
        remain_y=y[remaining]
        for regime in range(3):
            case=regime
            pred_raw=retrieve(v["rawMR"],remaining,y,classes,
                            [v["rawMR"][case][origin:origin+1]])[0]
            pred_shape=retrieve(v["shapeMR"],remaining,y,classes,
                            [v["shapeMR"][case][origin:origin+1]])[0]
            pred_cortex=retrieve(v["cortex"],remaining,y,classes,
                            [v["cortex"][case][origin:origin+1]])[0]
            shape=distance_class_scores(
                dist[case][origin:origin+1],
                remain_col,remain_y,classes)
            physical=distance_class_scores(
                physics[case][origin:origin+1],
                remain_col,remain_y,classes)
            for name,sc in (("Dual",.5*(pred_raw+pred_shape)),
                ("Canonical",pred_shape),("Cortex",pred_cortex),
                ("Elastic",shape),("ElasticConcept",physical)):
                scores[name][regime].append(int(classes[sc.argmax(1)[0]]))
    for mode in range(3):
        names=list(scores)
        f1={name:float(f1_score(truth,scores[name][mode],
                               average="macro",zero_division=0))
            for name in names}
        metric[["clean","affine","jitter"][mode]]=f1
    ref=metric["jitter"]["Dual"]
    eligible=("Cortex","Elastic","ElasticConcept")
    best=max(eligible,key=lambda z:(metric["jitter"][z],-eligible.index(z)))
    # Prefer no new pathway if its jitter benefit is small or its
    # clean-support behavior is markedly worse than the baseline.
    if (metric["jitter"][best]>=ref+EVIDENCE_MARGIN and
        metric["clean"][best]>=metric["clean"]["Dual"]-MAX_CLEAN_DEGRADATION):
        chosen["jitter"]=best
    if metric["affine"]["Canonical"]<metric["affine"]["Dual"]+.04:
        chosen["affine"]="Dual"
    return chosen,dict(train_loo_macro_f1=metric,
        selected_experts=chosen,minimum_jitter_gain=EVIDENCE_MARGIN,
        clean_support_tolerance=MAX_CLEAN_DEGRADATION,
        selection_uses_test_labels=False)

def run(args):
    basic_synthetic_tests()
    train,test,y,yt,_=load_data(args.dataset)
    classes=np.unique(y)
    budgets=tuple(sorted(set(args.budgets)))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in args.seeds:
        since=time.monotonic()
        supports=independent_budget_ids(y,seed,budgets)
        max_ids=supports[max(budgets)]
        cortex=build_representations(train,test,seed,args.epochs,
                                    args.hidden,args.dim)
        v=cortex["views"]
        v["cortex"]=cosine_cortex(cortex)
        d,dp,fitmeta=build_elastic(cortex,max_ids)
        pos={int(v):i for i,v in enumerate(max_ids)}
        for budget in budgets:
            support=supports[budget]
            cols=np.array([pos[int(i)] for i in support])
            labels=y[support]
            lookup={}
            for mode in (3,4,5):
                base_raw=retrieve(v["rawMR"],support,y,classes,
                                  [v["rawMR"][mode]])[0]
                base_can=retrieve(v["shapeMR"],support,y,classes,
                                  [v["shapeMR"][mode]])[0]
                base_cort=retrieve(v["cortex"],support,y,classes,
                                  [v["cortex"][mode]])[0]
                lookup[mode]=dict(Dual=.5*(base_raw+base_can),
                                  Canonical=base_can,Cortex=base_cort,
                                  Elastic=distance_class_scores(
                                      d[mode],cols,labels,classes),
                                  ElasticConcept=distance_class_scores(
                                      dp[mode],cols,labels,classes))
            experts,assessment=select_train_only(v,d,dp,y,support,
                pos,classes,cortex["charts"][:3],budget)
            if set(experts.values())-set(lookup[3]):
                raise AssertionError("Unregistered expert")
            predicted={name:[] for name in METHODS}
            meta=[]
            for mode in (3,4,5):
                choices=lookup[mode]
                pa,pm=cortex["charts"][mode]
                # A conservative router that loses nothing when the
                # nuisance detectors are inconclusive or conflict.
                alarm_a=pa<=CHART_ALPHA
                alarm_m=pm<=CHART_ALPHA
                p=softmax_class(choices[experts["clean"]])
                a=softmax_class(choices[experts["affine"]])
                w=softmax_class(choices[experts["jitter"]])
                routed=np.where((alarm_m & ~alarm_a)[:,None],w,
                   np.where((alarm_a & ~alarm_m)[:,None],a,p))
                methods={
                  "Dual MR episodic":choices["Dual"],
                  "Canonical MR episodic":choices["Canonical"],
                  "Cortex episodic":choices["Cortex"],
                  "Elastic slow memory":choices["Elastic"],
                  "Elastic shape-concept memory":choices["ElasticConcept"],
                  "Elastic HCLM-authorized router":routed}
                for name in METHODS:predicted[name].append(
                    classes[np.argmax(methods[name],axis=1)])
                meta.append(dict(affine_alert=float(np.mean(alarm_a)),
                    waveform_alert=float(np.mean(alarm_m)),
                    conflict_rate=float(np.mean(alarm_a&alarm_m))))
            for method in METHODS:
                pred=predicted[method]
                rec=dict(dataset=args.dataset,seed=seed,budget=budget,
                    method=method,status="EXECUTED",
                    clean_macro_f1=float(f1_score(yt,pred[0],
                                  average="macro",zero_division=0)),
                    affine_macro_f1=float(f1_score(yt,pred[1],
                                  average="macro",zero_division=0)),
                    jitter_macro_f1=float(f1_score(yt,pred[2],
                                  average="macro",zero_division=0)),
                    accuracy=float(accuracy_score(yt,pred[0])),
                    label_ids=[int(k) for k in support],n_labeled=len(support),
                    train_cases=len(train),test_cases=len(test),
                    ssl_epochs=args.epochs,elastic_fit=fitmeta,
                    elastic_assessment=assessment if method==METHODS[-1] else None,
                    applied_route=experts if method==METHODS[-1] else None,
                    alerts=meta if method==METHODS[-1] else None,
                    runtime_sec=round(time.monotonic()-since,2),
                    assumed_label_preserving_synthetic_corruption=True)
                rows.append(rec)
                print("RESULT_JSON="+json.dumps(rec,allow_nan=False),flush=True)
            (out/"results.json").write_text(json.dumps(rows,indent=2,allow_nan=False))
    assert len(rows)==len(args.seeds)*len(budgets)*len(METHODS)
    return rows

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--dataset",required=True)
    a.add_argument("--seeds",type=int,nargs="+",default=[3701])
    a.add_argument("--budgets",type=int,nargs="+",default=[1,3,5])
    a.add_argument("--epochs",type=int,default=8)
    a.add_argument("--hidden",type=int,default=40)
    a.add_argument("--dim",type=int,default=64)
    a.add_argument("--out",default="results/elastic_memory")
    run(a.parse_args())
