#!/usr/bin/env python3
"""MESA-NeuroMem: complementary cortical/episodic few-shot time-series AI.

Neuroscience-inspired, not a literal biological simulator:
  EC sensory cortex: exact raw/canonical decomposition + fixed MultiRocket
      plus slow self-supervised multiscale CNN trained on unlabeled TRAIN only.
  DG pattern separation: frozen random expansion + per-case sparse k-WTA,
      calibrated using unlabelled TRAIN; no class labels fit the codebook.
  CA3 episodic memory: one-step Hebbian exemplar imprinting of 1/3/5
      labeled TRAIN examples/class, with optional replay of predeclared
      label-preserving TRAIN perturbations (never TEST).
  CA1 comparator/context: sample-specific agreement/confidence and separate
      TRAIN-calibrated affine/morphological control-chart nuisance gates.
  Decision: fixed, auditable mixture of semantic and episodic class scores.

Scientific scope:
- CLS and sparse associative memory are established prior art; the claimed
  *experimental design* is its application to this particular decomposed
  time-series representation and calibrated nuisance-aware retrieval.
- Each labeled budget uses the same predeclared parameters, with nested
  labeled subsets; no LOO search possible at 1-shot.
- No TEST labels or TEST examples are used in any pretraining, normalization,
  calibration, reference, memory imprinting or policy selection.
- Synthetic shifts are experimental label-preserving assumptions, not
  guarantees of medical/industrial validity.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import affine_gauge, generate_affine_shift
from MESA_SSL.mesa_multikernel import feature_scale
from MESA_SSL.mesa_robust_select import extract
from MESA_SSL.mesa_spc_chart import chart_train, chart_predict
from MESA_SSL.mesa_dualpath_ssl import (
    train_ssl, extract_nn, signal_jitter, morphology_stats,
    calibrate_morphology, morphology_p,
)

METHODS=(
    "Raw MR episodic", "Canonical MR episodic", "Dual MR episodic",
    "Slow cortical episodic", "Sparse DG episodic",
    "NeuroMem no DG", "NeuroMem no cortex",
    "NeuroMem fixed-context", "NeuroMem + episodic replay",
    "Raw MR ridge", "Canonical MR ridge"
)
DEFAULT_BUDGETS=(1,3,5)


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1.e-12)


def independent_budget_ids(y,seed,budgets):
    """Nested, fixed per-class support prefixes, nothing chosen from TEST."""
    rng=np.random.default_rng(seed*971+1251)
    labels=np.unique(y)
    max_budget=max(budgets)
    pool={}
    for lab in labels:
        ids=rng.permutation(np.flatnonzero(y==lab))
        if len(ids)<max_budget:
            raise ValueError(f"Class {lab}: fewer TRAIN cases than {max_budget}")
        pool[int(lab)]=ids[:max_budget]
    chosen={}
    for b in budgets:
        ix=np.concatenate([pool[int(lab)][:b] for lab in labels])
        if not np.array_equal(np.unique(y[ix]),labels):
            raise AssertionError("Missing TRAIN class")
        chosen[b]=ix
    for a in budgets:
        for b in budgets:
            if a<b and not set(chosen[a]).issubset(set(chosen[b])):
                raise AssertionError("Budget support not nested")
    return chosen


class DentateSparseExpansion:
    """Label-blind random projection with local k-winner competition."""
    def __init__(self,expand=384,active_fraction=.09,seed=0):
        self.expand=int(expand)
        self.active_fraction=float(active_fraction)
        self.seed=int(seed)

    def fit(self, unlabeled_train):
        a=np.asarray(unlabeled_train,dtype=np.float64)
        self.scaler=StandardScaler().fit(a)
        raw=self.scaler.transform(a)
        rng=np.random.default_rng(self.seed+31)
        D=raw.shape[1]
        self.P=rng.normal(size=(D,self.expand)).astype(np.float64)/np.sqrt(max(1,D))
        # Center by TRAIN-only projected mean. Maintain a sparse code
        # without learning any class-specific projection.
        self.center=(raw@self.P).mean(axis=0)
        self.k=max(1,int(math.ceil(self.expand*self.active_fraction)))
        self.width=float(np.sqrt(np.mean((raw@self.P-self.center)**2)))+1.e-8
        return self

    def transform(self, values):
        x=self.scaler.transform(np.asarray(values,dtype=np.float64))
        h=(x@self.P-self.center)/self.width
        order=np.argpartition(h,-self.k,axis=1)[:,-self.k:]
        mask=np.zeros(h.shape,dtype=bool)
        np.put_along_axis(mask,order,True,axis=1)
        # Continuous kWTA, non-negative, and L2 for similarity retrieval.
        x=np.where(mask,np.maximum(h,0.),0.)
        return unit(x)


def build_representations(train,test,seed,epochs=12,hidden=40,dim=64):
    """Slow, label-free cortical statistics plus fast episodic sensory views."""
    start=time.monotonic()
    train_shift=generate_affine_shift(train,seed+441,1.20)
    train_jitter=signal_jitter(train,seed+881,noise=.085,harmonic=.045)
    test_shift=generate_affine_shift(test,seed+531,1.20)
    test_jitter=signal_jitter(test,seed+981,noise=.085,harmonic=.045)
    samples=[train,train_shift,train_jitter,test,test_shift,test_jitter]
    shapes=[affine_gauge(a)[0] for a in samples]
    concept=[affine_gauge(a)[1] for a in samples]

    net,losses=train_ssl(train,seed+9000,mechanism=True,
                          epochs=epochs,hidden=hidden,dim=dim)
    feats=[extract_nn(net,a) for a in samples]
    # No scaling or SSL fitting on any TEST sample.
    view={
       "rawMR":extract(train, samples[1:],seed),
       "shapeMR":extract(shapes[0],shapes[1:],seed),
       "cortRaw":feature_scale(*(z["raw"] for z in feats)),
       "cortShape":feature_scale(*(z["shape"] for z in feats)),
       "concepts":feature_scale(*concept),
    }
    # DG receives robust cortical features and shape kernels. There is no
    # arbitrary choice of test-specific model or TEST-dependent normalization.
    dg_inputs=[]
    for i in range(6):
        dg_inputs.append(np.hstack((
            .25*view["shapeMR"][i],
            .75*view["cortRaw"][i],
            .75*view["cortShape"][i],
            .06*view["concepts"][i])))
    dg=DentateSparseExpansion(seed=seed+11000).fit(dg_inputs[0])
    view["dg"]=tuple(dg.transform(z) for z in dg_inputs)
    elapsed=time.monotonic()-start

    affine_chart=chart_train(concept[0],seed,alpha=.10)
    wave_chart=calibrate_morphology(morphology_stats(train),seed)
    # Raw/shape test deformations are known only to the evaluator. No labels.
    chart_probs=[]
    for x,c in zip(samples,concept):
        pa=chart_predict(affine_chart,c)[2]
        pm=morphology_p(wave_chart,morphology_stats(x))
        chart_probs.append((pa,pm))
    return dict(views=view,samples=samples,charts=chart_probs,
                loss_curve=losses,
                concept_rmse=float(np.sqrt(np.mean(
                    (feats[0]["concept_pred"]-concept[0])**2))),
                runtime_seconds=elapsed,
                n_train=len(train),n_test=len(test))


def retrieve(feat,indices,y,labels,queries,replay=False,
             temp=7.,prototype=.38):
    """Exemplar imprinting: class-wise max+prototype similarity.

    No SGD or new slow-cortical gradient is needed when labels arrive.
    Replaying TRAIN-side augmented labeled episodes is a form of fast
    imprinting. It consumes only already-held labeled TRAIN cases.
    """
    keys=[unit(feat[0][indices])]
    known=[y[indices]]
    if replay:
        keys +=[unit(feat[1][indices]), unit(feat[2][indices])]
        known +=[y[indices],y[indices]]
    memory=unit(np.vstack(keys))
    ys=np.concatenate(known)
    scores=[]
    for q in queries:
        sim=unit(q)@memory.T
        class_scores=[]
        for c in labels:
            belonging=(ys==c)
            pool=sim[:,belonging]
            # max: episodic pattern completion; mean: slower semantic
            # class prototype. The latter is natural even at 1-shot.
            best=np.max(pool,axis=1)
            typical=np.mean(pool,axis=1)
            class_scores.append((1-prototype)*best+prototype*typical)
        scores.append(np.stack(class_scores,axis=1))
    return scores


def ridge_memory(feat,indices,y,labels,queries):
    """Tuned-free 1/3/5-shot ridge baseline, same TRAIN labels."""
    Y=(y[indices,None]==labels[None,:]).astype(float)
    X=unit(feat[0][indices])
    K=X@X.T
    alpha=np.linalg.solve(K+.10*np.eye(len(indices)),Y)
    return [unit(z)@X.T@alpha for z in queries]


def softmax_class(s,temperature=7.):
    x=temperature*(s-s.max(axis=1,keepdims=True))
    z=np.exp(np.clip(x,-35.,0.))
    return z/np.maximum(z.sum(axis=1,keepdims=True),1.e-12)


def controlled_weights(pa,pm,use_context=True,omit_dg=False,omit_cortex=False):
    """Predeclared transparent source mixture, label-blind across TEST."""
    n=len(pa)
    if not use_context:
        pri=np.tile(np.array([.42,.22,.23,.13]),(n,1))
    else:
        # Soft conditional transitions use separate nuisance families.
        # Hard thresholding in very small calibration sets is avoided.
        affine=np.clip((.15-pa)/.15,0,1)
        waveform=np.clip((.18-pm)/.18,0,1)
        clean=np.array([.42,.25,.20,.13])
        affine_p=np.array([.08,.48,.26,.18])
        wave_p=np.array([.035,.035,.59,.34])
        pri=(1-waveform[:,None])*(
            (1-affine[:,None])*clean[None,:]+affine[:,None]*affine_p[None,:]
            )+waveform[:,None]*wave_p[None,:]
    if omit_dg:pri[:,3]=0.
    if omit_cortex:pri[:,2]=0.
    pri=pri / np.maximum(pri.sum(axis=1,keepdims=True),1.e-12)
    return pri


def neuromem_decide(scores,chart,contexts=True,omit_dg=False,omit_cortex=False):
    # scores: [rawMR, shapeMR, cortex, DG], each N x #classes
    p_aff,p_morph=chart
    weights=controlled_weights(p_aff,p_morph,contexts,omit_dg,omit_cortex)
    logits=np.stack([softmax_class(z) for z in scores],axis=1)
    ensemble=np.einsum("nk,nkc->nc",weights,logits)
    # CA1 disagreement score: probability entropy, not calibrated posterior.
    margin=np.sort(ensemble,axis=1)[:,-1]-np.sort(ensemble,axis=1)[:,-2]
    disagreement=np.mean(np.var(logits,axis=1),axis=1)
    return ensemble,dict(avg_weights=weights.mean(0).tolist(),
         uncalibrated_margin_mean=float(margin.mean()),
         expert_disagreement_mean=float(disagreement.mean()),
         affine_alarm_rate=float(np.mean(p_aff<=.10)),
         morphology_alarm_rate=float(np.mean(p_morph<=.10)))


def test_scientific_contract():
    rng=np.random.default_rng(501)
    x=rng.normal(size=(28,32))
    sparse=DentateSparseExpansion(expand=96,active_fraction=.12,seed=2).fit(x)
    z=sparse.transform(x)
    assert z.shape==(28,96)
    assert np.isfinite(z).all()
    assert (z>=0).all()
    assert np.max(np.count_nonzero(z,axis=1))<=12
    y=np.repeat(np.arange(4),7)
    ids=independent_budget_ids(y,41,(1,3,5))
    assert all(len(ids[k])==4*k for k in (1,3,5))
    v=(unit(x),unit(x+.1),unit(x+.2),unit(x),unit(x),unit(x))
    classes=np.unique(y)
    pred=retrieve(v,ids[1],y,classes,[v[3]])
    assert pred[0].shape==(28,4)
    weights=controlled_weights(np.array([1.,0.,1.]),
                               np.array([1.,1.,0.]))
    assert np.allclose(weights.sum(axis=1),1)
    assert weights[1,1]>weights[0,1]  # affine -> canonical MR
    assert weights[2,2]>weights[0,2]  # waveform -> cortex
    return True


def run(args):
    test_scientific_contract()
    budgets=tuple(sorted(set(args.budgets)))
    if any(b<1 for b in budgets):raise ValueError("budget must be positive")
    train,test,y,yt,_=load_data(args.dataset)
    classes=np.unique(y)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    total=[];training=[]
    for seed in args.seeds:
        begin=time.monotonic()
        support=independent_budget_ids(y,seed,budgets)
        cortex=build_representations(train,test,seed,args.epochs,
                                     args.hidden,args.dim)
        v=cortex["views"]
        # Brain's long-term prior is shared across budgets and uses only
        # unlabelled TRAIN. The support examples are the only label source.
        # Build a robust cortical family by concatenating raw and canonical
        # learned representations; normalize in their already train-fitted space.
        merged=[]
        for j in range(6):
            merged.append(unit(np.hstack((v["cortRaw"][j],
                                         v["cortShape"][j]))))
        neural=tuple(merged)
        groups={
            "Raw MR episodic":v["rawMR"],
            "Canonical MR episodic":v["shapeMR"],
            "Slow cortical episodic":neural,
            "Sparse DG episodic":v["dg"],
        }
        for budget in budgets:
            ids=support[budget]
            scores={}
            for name,feat in groups.items():
                scores[name]=retrieve(feat,ids,y,classes,feat[3:],
                                       replay=False)
            # Dual MR ensemble is not trained/tuned on TEST.
            scores["Dual MR episodic"]=[
                .5*scores["Raw MR episodic"][j]+
                .5*scores["Canonical MR episodic"][j] for j in range(3)
            ]
            # Fixed ridge does not tune lambda from few labels.
            for key,feat in (("Raw MR ridge",v["rawMR"]),
                             ("Canonical MR ridge",v["shapeMR"])):
                scores[key]=ridge_memory(feat,ids,y,classes,feat[3:])
            base=[scores["Raw MR episodic"],scores["Canonical MR episodic"],
                  scores["Slow cortical episodic"],scores["Sparse DG episodic"]]
            for name,kind in [
                 ("NeuroMem no DG","no_dg"),
                 ("NeuroMem no cortex","no_cortex"),
                 ("NeuroMem fixed-context","fixed"),
                 ("NeuroMem + episodic replay","replay")]:
                if kind=="replay":
                    augmented=[]
                    for f in (v["rawMR"],v["shapeMR"],neural,v["dg"]):
                        augmented.append(retrieve(f,ids,y,classes,f[3:],
                                                   replay=True))
                else:augmented=base
                scores[name]=[]
                diagnostics=[]
                for j in range(3):
                    s,diag=neuromem_decide(
                        [x[j] for x in augmented],cortex["charts"][3+j],
                        contexts=(kind!="fixed"),
                        omit_dg=(kind=="no_dg"),omit_cortex=(kind=="no_cortex"))
                    scores[name].append(s)
                    diagnostics.append(diag)
            for method in METHODS:
                outputs=scores[method]
                predictions=[classes[np.argmax(q,axis=1)] for q in outputs]
                result=dict(dataset=args.dataset,seed=seed,budget=budget,
                     method=method,status="EXECUTED",
                     clean_macro_f1=float(f1_score(yt,predictions[0],
                                             average="macro",zero_division=0)),
                     affine_macro_f1=float(f1_score(yt,predictions[1],
                                             average="macro",zero_division=0)),
                     jitter_macro_f1=float(f1_score(yt,predictions[2],
                                             average="macro",zero_division=0)),
                     accuracy=float(accuracy_score(yt,predictions[0])),
                     train_cases=len(train),test_cases=len(test),
                     labeled_support_ids=[int(j) for j in ids],
                     n_labeled=int(len(ids)),cortex_epochs=args.epochs,
                     cortex_train_concept_rmse=cortex["concept_rmse"],
                     computational_sec=float(round(time.monotonic()-begin,3)),
                     policy_diagnostics=diagnostics if method.startswith("NeuroMem") else None,
                     synthetic_perturbations_assumed_label_preserving=True)
                total.append(result)
                print("RESULT_JSON="+json.dumps(result,allow_nan=False),flush=True)
            (out/"results.json").write_text(json.dumps(total,indent=2))
        training.append(dict(dataset=args.dataset,seed=seed,epochs=args.epochs,
             cortex_loss_curve=cortex["loss_curve"],
             cortex_unlabeled_fit_sec=cortex["runtime_seconds"]))
        (out/"training.json").write_text(json.dumps(training,indent=2))
    if len(total)!=len(args.seeds)*len(budgets)*len(METHODS):
        raise AssertionError("Incomplete METHOD × BUDGET × SEED output")
    return total


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[4201])
    p.add_argument("--budgets",nargs="+",type=int,default=list(DEFAULT_BUDGETS))
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--hidden",type=int,default=40)
    p.add_argument("--dim",type=int,default=64)
    p.add_argument("--out",default="results/mesa_neuromem")
    run(p.parse_args())
