#!/usr/bin/env python3
"""Frozen full-capacity MultiRocket few-shot comparison.

HCLM-frozen contract: HCLM_10K_BASELINE_FROZEN_PROTOCOL_2026-11-24.json.
The compact 336-kernel comparison is inadequate for any SOTA claim.
This evaluates 10,000 kernels and a fully predeclared readout on exactly
the same official UCR TRAIN/TEST splits and 1/3/5 support selection method.
Fit the transformations and all scaling on UNLABELED TRAIN only. TEST
labels are read only for final evaluation and never adjust parameters.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from aeon.transformations.collection.convolution_based import MultiRocket
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score
from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import affine_gauge,generate_affine_shift
from MESA_SSL.mesa_dualpath_ssl import signal_jitter
from MESA_SSL.mesa_neuromem import unit,independent_budget_ids

METHODS=("Full MR raw episodic","Full MR canonical episodic",
         "Full MR dual episodic","Full MR raw ridge")

def to_aeon(x):
    return np.ascontiguousarray(np.asarray(x,dtype=np.float32).transpose(0,2,1))

def train_transform(x,seed,kernels):
    mr=MultiRocket(n_kernels=kernels,max_dilations_per_kernel=32,
                   n_jobs=1,random_state=seed)
    train=np.asarray(mr.fit_transform(to_aeon(x)),dtype=np.float32)
    sc=StandardScaler().fit(train)
    z=np.clip(sc.transform(train),-8.,8.)
    norm=float(np.sqrt(max(np.mean(np.sum(z.astype(np.float64)**2,axis=1)),
                           1e-12)))
    return mr,sc,norm

def transform_blocks(model,scaler,norm,x,batch=128):
    for i in range(0,len(x),batch):
        arr=np.asarray(model.transform(to_aeon(x[i:i+batch])),
                       dtype=np.float32)
        yield unit(np.clip(scaler.transform(arr),-8.,8.)/norm)

def memory_scores(ref,y_support,classes,query):
    S=unit(query)@unit(ref).T
    out=[]
    for lab in classes:
        p=S[:,y_support==lab]
        out.append(.62*p.max(1)+.38*p.mean(1))
    return np.stack(out,axis=1)

def ridge_scores(ref,y_support,classes,query):
    r=unit(ref);q=unit(query)
    Y=(y_support[:,None]==classes[None,:]).astype(float)
    inv=np.linalg.solve(r@r.T+.1*np.eye(len(r)),Y)
    return q@r.T@inv

def numerical_tests():
    rng=np.random.default_rng(4100)
    y=np.repeat(np.arange(3),5)
    a=unit(rng.normal(size=(15,10)))
    b=unit(rng.normal(size=(7,10)))
    ids=independent_budget_ids(y,5801,(1,3,5))
    assert all(len(ids[k])==3*k for k in (1,3,5))
    assert memory_scores(a,y,np.unique(y),b).shape==(7,3)
    assert ridge_scores(a,y,np.unique(y),b).shape==(7,3)
    assert set(ids[1]).issubset(set(ids[3]))
    return True

def run(args):
    numerical_tests()
    train,test,y,yt,_=load_data(args.dataset)
    classes=np.unique(y)
    budgets=sorted(set(args.budgets))
    supports=independent_budget_ids(y,args.seed,tuple(budgets))
    max_ids=supports[budgets[-1]]
    shape_train=affine_gauge(train)[0]
    perturbed=[
        test,
        generate_affine_shift(test,args.seed+531,1.20),
        signal_jitter(test,args.seed+981,noise=.085,harmonic=.045)]
    canonical=[affine_gauge(a)[0] for a in perturbed]
    since=time.monotonic()
    maps={}
    for name,X in (("raw",train),("canonical",shape_train)):
        model,sc,norm=train_transform(X,args.seed,args.kernels)
        ref=np.asarray(model.transform(to_aeon(X[max_ids])),
                       dtype=np.float32)
        ref=unit(np.clip(sc.transform(ref),-8.,8.)/norm)
        maps[name]=(model,sc,norm,ref)
        print("FIT_JSON="+json.dumps(dict(dataset=args.dataset,seed=args.seed,
             stream=name,features=int(ref.shape[1]),fit_seconds=round(
             time.monotonic()-since,2))),flush=True)
    loc={int(ix):j for j,ix in enumerate(max_ids)}
    support_vectors={name:{
        b:ref[[loc[int(ix)] for ix in supports[b]]]
        for b in budgets}
        for name,(*_,ref) in maps.items()}
    predictions={b:{m:[[] for _ in range(3)] for m in METHODS}
                 for b in budgets}
    for mode in range(3):
        raw_iter=transform_blocks(*maps["raw"][:3],perturbed[mode],
                                  batch=args.batch)
        shape_iter=transform_blocks(*maps["canonical"][:3],canonical[mode],
                                    batch=args.batch)
        n=0
        for r,s in zip(raw_iter,shape_iter):
            assert len(r)==len(s)
            n+=len(r)
            for b in budgets:
                yl=y[supports[b]]
                raw=memory_scores(support_vectors["raw"][b],yl,classes,r)
                can=memory_scores(support_vectors["canonical"][b],yl,classes,s)
                dual=.5*(raw+can)
                ridge=ridge_scores(support_vectors["raw"][b],yl,classes,r)
                for name,z in zip(METHODS,(raw,can,dual,ridge)):
                    predictions[b][name][mode].append(z)
        if n!=len(test):raise AssertionError("Incomplete TEST set")
        print("MODE_JSON="+json.dumps(dict(dataset=args.dataset,seed=args.seed,
             mode=mode,n=n)),flush=True)
    rows=[]
    for b in budgets:
        for method in METHODS:
            scores=[np.vstack(x) for x in predictions[b][method]]
            choices=[classes[z.argmax(1)] for z in scores]
            rec=dict(dataset=args.dataset,seed=args.seed,budget=b,
                method=method,status="EXECUTED",
                clean_macro_f1=float(f1_score(yt,choices[0],average="macro",
                                               zero_division=0)),
                affine_macro_f1=float(f1_score(yt,choices[1],average="macro",
                                                zero_division=0)),
                jitter_macro_f1=float(f1_score(yt,choices[2],average="macro",
                                                zero_division=0)),
                accuracy=float(accuracy_score(yt,choices[0])),
                label_ids=[int(v) for v in supports[b]],
                n_labeled=len(supports[b]),train_cases=len(train),
                test_cases=len(test),kernel_count=args.kernels,
                max_dilations_per_kernel=32,
                feature_fit_uses_test=False,
                runtime_seconds=round(time.monotonic()-since,2))
            rows.append(rec)
            print("RESULT_JSON="+json.dumps(rec,allow_nan=False),flush=True)
    assert len(rows)==len(budgets)*len(METHODS)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(rows,indent=2,
                                                   allow_nan=False))
    return rows

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--budgets",nargs="+",type=int,default=[1,3,5])
    p.add_argument("--kernels",type=int,default=10000)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--out",default="results/mr_full")
    run(p.parse_args())
