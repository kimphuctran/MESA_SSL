#!/usr/bin/env python3
"""MESA-BioPhysics: symmetry-preserving cortex + fast energy-based episodic memory.

Biological inspirations: complementary learning systems, hippocampal fast
pattern completion, sparse dentate pattern separation, predictive coding,
homeostatic activity normalization and one-step associative imprinting.

Physics inspirations: exact affine group decomposition, Fourier heat diffusion
semigroup with non-increasing energy, contractive linear predictive dynamics,
stable leaky integration and a train-calibrated energy/surprise monitor.

The algorithm is NOT a literal brain simulator or a uniquely first use
of any ingredient. All choices are predeclared; no TEST labels/features
affect fitting, expert selection or nuisance calibration. This is a research
hypothesis subject to clean/shifted 1/3/5-shot comparison and ablations.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score

from MESA_SSL.mesa_gauge_ssl import affine_gauge
from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_multikernel import feature_scale
from MESA_SSL.mesa_neuromem import (
    unit,independent_budget_ids,DentateSparseExpansion,
    build_representations,retrieve,ridge_memory,neuromem_decide,
    softmax_class
)

METHODS=(
    "Raw MR episodic",
    "Canonical MR episodic",
    "Dual MR episodic",
    "Slow cortex episodic",
    "DG episodic",
    "Physics episodic",
    "Predictive innovation episodic",
    "Hopfield physics episodic",
    "NeuroMem prior",
    "BioPhysics no homeostasis",
    "BioPhysics no cortex",
    "BioPhysics no physics",
    "BioPhysics no predictive",
    "BioPhysics no Hopfield",
    "BioPhysics no context",
    "BioPhysics full",
    "BioPhysics safe TRAIN",
    "Raw MR ridge",
)
BUDGETS=(1,3,5)
MODEL_WEIGHT_KEYS=("raw","shape","cortex","dg","physics","predictive")
CONTEXT_WEIGHTS={
 "clean":np.array([.21,.13,.18,.12,.24,.12]),
 "affine":np.array([.04,.24,.18,.12,.28,.14]),
 "wave":np.array([.02,.03,.24,.14,.39,.18]),
 "surprise":np.array([.08,.06,.20,.11,.30,.25]),
}


def normed_spectrum(x):
    """One-sided Parseval-type energy density, used for stable features."""
    x=np.asarray(x,dtype=np.float64)
    u=np.fft.rfft(x,axis=1,norm="ortho")
    p=(u.real*u.real+u.imag*u.imag)
    # Double internal frequency bins to account for negative frequencies.
    if x.shape[1]%2==0 and p.shape[1]>2:
        p[:,1:-1,:]*=2.
    elif p.shape[1]>1:
        p[:,1:,:]*=2.
    return p


def heat_diffusion(x,seconds):
    """Fourier heat semigroup exp(-t*k^2), contractive in L2."""
    x=np.asarray(x,dtype=np.float64)
    n=x.shape[1]
    f=np.fft.rfftfreq(n,d=1.)
    lamb=(f / max(f.max(),1e-9))**2
    F0=np.fft.rfft(x,axis=1,norm="ortho")
    return np.fft.irfft(
        F0*np.exp(-float(seconds)*lamb)[None,:,None],
        n=n,axis=1,norm="ortho")


def leaky_resonator(x,rate):
    """A stable leaky-integrator surrogate for neural time constants."""
    x=np.asarray(x,dtype=np.float64)
    h=np.zeros((len(x),x.shape[2]),dtype=np.float64)
    cumulative=np.zeros_like(h)
    energy=np.zeros_like(h)
    maximum=np.zeros_like(h)
    for t in range(x.shape[1]):
        h=(1.-rate)*h+rate*x[:,t,:]
        cumulative+=h
        energy+=h*h
        maximum=np.maximum(maximum,np.abs(h))
    T=x.shape[1]
    return np.concatenate([h,cumulative/T,np.sqrt(energy/T),maximum],axis=1)


class PredictiveCoding:
    """Train-fitted contractive AR(2) generative prior, label-blind.

    Its innovation carries violations of the TRAIN dynamics. This is a
    low-complexity predictive-coding analogue, not a neural implementation
    of Rao-Ballard cortical feedback circuitry.
    """
    def fit(self,train_canonical):
        q=np.asarray(train_canonical,dtype=np.float64)
        if q.shape[1]<4:raise ValueError("At least 4 time points required")
        a=np.stack((q[:,1:-1,:],q[:,:-2,:]),axis=-1).reshape(-1,2)
        b=q[:,2:,:].reshape(-1)
        coef=np.linalg.solve(a.T@a+0.10*np.eye(2),a.T@b)
        # Strict contraction bound; stable one-step predictive response.
        norm=np.sum(np.abs(coef))
        if norm>.95:coef=coef*(.95/norm)
        self.coef=coef
        return self
    def innovation(self,q):
        q=np.asarray(q,dtype=np.float64)
        estimate=self.coef[0]*q[:,1:-1,:]+self.coef[1]*q[:,:-2,:]
        return q[:,2:,:]-estimate
    def features(self,q):
        er=self.innovation(q)
        delta=np.diff(er,axis=1)
        return np.concatenate((
            er.mean(axis=1),
            np.sqrt((er*er).mean(axis=1)),
            np.log1p((er*er).mean(axis=1)),
            np.mean(np.abs(delta),axis=1),
            np.quantile(np.abs(er),.90,axis=1),
            np.maximum(np.max(er,axis=1),0.),
            np.maximum(-np.min(er,axis=1),0.)
        ),axis=1)
    def surprise(self,q):
        e=self.innovation(q)
        return np.mean(e*e,axis=(1,2))


def physical_signature(x):
    """Fixed physics-inspired multi-scale observables, never label fitted."""
    canonical,concept=affine_gauge(x)
    q=np.asarray(canonical,dtype=np.float64)
    c=np.asarray(concept,dtype=np.float64)
    parts=[.20*c]
    spectral=normed_spectrum(q)
    # Eight broad spectral bands, preserved as sums of energy density.
    bands=[]
    for idx in np.array_split(np.arange(spectral.shape[1]),8):
        if len(idx):
            bands.append(np.log1p(spectral[:,idx,:].sum(axis=1)))
        else:bands.append(np.zeros((len(q),q.shape[2])))
    parts.extend(bands)
    # Heat diffusion energies are monotonically nonincreasing in time.
    for step in (0.,.20,1.,4.):
        sm=heat_diffusion(q,step) if step else q
        grad=np.diff(sm,axis=1)
        parts.append(np.log1p((sm*sm).mean(axis=1)))
        parts.append(np.log1p((grad*grad).mean(axis=1)))
        parts.extend([seg.mean(axis=1) for seg in
                      np.array_split(sm,4,axis=1)])
    for rate in (.035,.14,.45):
        parts.append(leaky_resonator(q,rate))
    return np.hstack(parts)


def freeze_features(cortex,seed):
    """Build slow cortical and physics representations on TRAIN only."""
    v=dict(cortex["views"])
    samples=cortex["samples"]
    canonical=[affine_gauge(x)[0] for x in samples]
    physical=[physical_signature(x) for x in samples]
    ar=PredictiveCoding().fit(canonical[0])
    surprise=[ar.surprise(x) for x in canonical]
    innovation=[ar.features(x) for x in canonical]
    v["physics"]=feature_scale(*physical)
    v["predictive"]=feature_scale(*innovation)
    # Keep cortex raw and canonical together: mean/slope remain available.
    v["cortex"]=tuple(unit(np.hstack((v["cortRaw"][j],
            v["cortShape"][j]))) for j in range(6))
    composite=[np.hstack((.35*v["shapeMR"][i],
              .65*v["cortex"][i],.60*v["physics"][i],
              .18*v["predictive"][i])) for i in range(6)]
    dentate=DentateSparseExpansion(expand=384,
           active_fraction=.09,seed=seed+8702).fit(composite[0])
    v["homeo_dg"]=tuple(dentate.transform(a) for a in composite)
    # Matched ablation: same fixed expansion but no k-WTA homeostasis.
    a=dentate.scaler
    raw_dense=[a.transform(z)@dentate.P-dentate.center for z in composite]
    v["dense_dg"]=tuple(unit(z) for z in raw_dense)
    return v,surprise,ar.coef


def conformal_univariate(ref,queries,seed):
    """Split train reference: p-values rely on exchangeability only."""
    a=np.asarray(ref,dtype=np.float64)
    rng=np.random.default_rng(seed+15555)
    ids=rng.permutation(len(a))
    k=max(4,int(.60*len(a)))
    k=min(k,len(a)-4)
    calibration=np.sort(a[ids[k:]])
    return [
        (1+(calibration[None,:]>=np.asarray(q)[:,None]).sum(1))/
        (len(calibration)+1) for q in queries]


def hopfield_scores(features, support, labels, classes, queries,
                    steps=2,rate=.35,beta=8.0):
    """Retrieval by iterative bounded modern-Hopfield-style updates.

    Memory records are exactly the labeled TRAIN episodes. We do not
    impose class labels onto the pretraining corpus.
    """
    memory=unit(features[0][support])
    ys=labels[support]
    scores=[]
    for query in queries:
        q=unit(query)
        for _ in range(steps):
            s=beta*q@memory.T
            ex=np.exp(np.clip(s-s.max(axis=1,keepdims=True),-60,0))
            alpha=ex/np.maximum(ex.sum(axis=1,keepdims=True),1.e-12)
            attractor=alpha@memory
            q=unit((1-rate)*q+rate*attractor)
        affinity=q@memory.T
        class_score=[]
        for c in classes:
            sim=affinity[:,ys==c]
            scaled=beta*sim
            peak=scaled.max(axis=1,keepdims=True)
            lme=(peak[:,0]+np.log(np.mean(np.exp(scaled-peak),
                                         axis=1)))/beta
            class_score.append(.70*lme+.30*np.mean(sim,axis=1))
        scores.append(np.stack(class_score,axis=1))
    return scores


def entropy_disagreement(scored):
    """Quantify disagreement without accessing correctness labels."""
    ps=np.stack([softmax_class(s) for s in scored],axis=1)
    prob=ps.mean(axis=1)
    h=-np.sum(prob*np.log(np.maximum(prob,1.e-9)),axis=1)
    return h,ps


def prior_weights(chart,psurprise,has_context=True,
                  without_cortex=False,without_physics=False,
                  without_prediction=False,without_sparse=False):
    pa,pm=chart
    n=len(pa)
    if has_context:
        aff=np.clip((.15-pa)/.15,0,1)
        mor=np.clip((.18-pm)/.18,0,1)
        surpr=np.clip((.15-psurprise)/.15,0,1)
        weights=((1-aff[:,None])*CONTEXT_WEIGHTS["clean"]+
                 aff[:,None]*CONTEXT_WEIGHTS["affine"])
        weights=(1-mor[:,None])*weights+mor[:,None]*CONTEXT_WEIGHTS["wave"]
        weights=(1-.40*surpr[:,None])*weights+(.40*surpr[:,None])*CONTEXT_WEIGHTS["surprise"]
    else:
        weights=np.tile(CONTEXT_WEIGHTS["clean"],(n,1))
    if without_cortex:weights[:,2]=0.
    if without_sparse:weights[:,3]=0.
    if without_physics:weights[:,4]=0.
    if without_prediction:weights[:,5]=0.
    return weights/np.maximum(weights.sum(1,keepdims=True),1.e-10)


def decision(scored,chart,p_surprise,**ablation):
    weights=prior_weights(chart,p_surprise,**ablation)
    _,probs=entropy_disagreement(scored)
    value=np.einsum("nk,nkc->nc",weights,probs)
    return value,dict(weights=weights.mean(axis=0).tolist(),
             chart_affine_rate=float(np.mean(chart[0]<=.10)),
             chart_morph_rate=float(np.mean(chart[1]<=.10)),
             predictive_surprise_rate=float(np.mean(p_surprise<=.10)))




def choose_meta_controller(views,support,y,classes,train_charts,p_surprise,budget):
    """Episodic metaplastic controller with *honest* grouped support exclusion.

    When budget >=3, every labeled TRAIN episode is withheld in turn from
    each memory, along with its synthetic views. The model is evaluated on
    clean and both predeclared perturbed views of that same held-out TRAIN
    observation, using the original label for those synthetic perturbations.
    Neither TEST examples nor TEST labels affect expert selection.

    At 1-shot leave-one-label-out is undefined because it removes the only
    exemplar of a class. We use an explicitly conservative dual-MR fallback,
    rather than fitting a spurious hyperparameter to a single support.
    """
    default="Dual MR episodic"
    if budget<3:
        return default,{"selected":default,"reason":"No valid 1-shot LOO per class"}
    choices=("Raw MR episodic","Canonical MR episodic","Dual MR episodic",
             "Slow cortex episodic","NeuroMem prior",
             "BioPhysics no physics","BioPhysics no Hopfield",
             "BioPhysics full")
    n=len(support)
    cross={k:[np.zeros((n,len(classes)),dtype=float) for _ in range(3)]
           for k in choices}
    reference_keys=("rawMR","shapeMR","cortex","homeo_dg",
                    "physics","predictive")
    for row,support_case in enumerate(support):
        remaining=np.delete(support,row)
        # Each withheld TRAIN observation is a fresh query in every mode.
        p={}
        for key in reference_keys:
            feats=views[key]
            queries=[feats[mode][support_case:support_case+1]
                     for mode in (0,1,2)]
            out=retrieve(feats,remaining,y,classes,queries,replay=False)
            p[key]=[q[0] for q in out]
        hp=hopfield_scores(views["physics"],remaining,y,classes,
             [views["physics"][mode][support_case:support_case+1]
              for mode in (0,1,2)])
        for mode in range(3):
            S=[p[key][mode][None,:] for key in reference_keys]
            S[4]=hp[mode]
            # The chart is trained on unlabeled TRAIN. Conditional p-values
            # for this label are taken from the clean/augmented TRAIN inputs.
            cond=(train_charts[mode][0][support_case:support_case+1],
                  train_charts[mode][1][support_case:support_case+1])
            surprise=p_surprise[mode][support_case:support_case+1]
            full=decision(S,cond,surprise)[0][0]
            no_physics=decision(S,cond,surprise,
                               without_physics=True)[0][0]
            no_hopfield_S=list(S)
            no_hopfield_S[4]=p["physics"][mode][None,:]
            no_hop=decision(no_hopfield_S,cond,surprise)[0][0]
            prior=decision(S,cond,surprise,
                         without_physics=True,
                         without_prediction=True)[0][0]
            simplex={
              "Raw MR episodic":softmax_class(p["rawMR"][mode][None,:])[0],
              "Canonical MR episodic":softmax_class(p["shapeMR"][mode][None,:])[0],
              "Dual MR episodic":softmax_class(
                  .5*(p["rawMR"][mode][None,:]+
                      p["shapeMR"][mode][None,:]))[0],
              "Slow cortex episodic":softmax_class(
                  p["cortex"][mode][None,:])[0],
              "NeuroMem prior":prior,
              "BioPhysics no physics":no_physics,
              "BioPhysics no Hopfield":no_hop,
              "BioPhysics full":full
            }
            for k in choices:cross[k][mode][row]=simplex[k]
    score={}
    truth=y[support]
    for model in choices:
        y_pred=[classes[np.argmax(p,axis=1)] for p in cross[model]]
        f=[float(f1_score(truth,z,average="macro",zero_division=0))
           for z in y_pred]
        # Prefer small, replayable memory if paired TRAIN evidence is close.
        penalty=(0.012 if model in ("BioPhysics full",
                                   "NeuroMem prior",
                                   "BioPhysics no physics",
                                   "BioPhysics no Hopfield") else 0.)
        objective=.55*f[0]+.25*f[1]+.20*f[2]-penalty
        score[model]=dict(objective=objective,cv_macro_f1=f)
    winner=max(choices,key=lambda k:(score[k]["objective"],
                   -choices.index(k)))
    baseline=score[default]["objective"]
    # Bounded metaplastic switch threshold protects 5-label reference data
    # against falsely concluding a complex brain-inspired module wins.
    if winner!=default and score[winner]["objective"]<baseline+.025:
        winner=default
    return winner,dict(selected=winner,baseline_cv=baseline,
                       scored_candidates=score,leakage_guard=True)


def numerical_contracts():
    rng=np.random.default_rng(2004)
    x=rng.normal(size=(10,80,3)).astype("float32")
    f=physical_signature(x)
    assert f.shape[0]==len(x) and np.isfinite(f).all()
    q=affine_gauge(x)[0]
    energies=[np.mean(heat_diffusion(q,t)**2) for t in (0.,.2,1.,4.)]
    if not np.all(np.diff(energies)<=1.e-9):
        raise AssertionError(f"Heat semigroup must contract L2: {energies}")
    predict=PredictiveCoding().fit(q)
    assert np.sum(np.abs(predict.coef))<=.9500001
    dg=DentateSparseExpansion(expand=128,active_fraction=.08,seed=5)
    dg.fit(f);codes=dg.transform(f)
    assert np.count_nonzero(codes,axis=1).max()<=11
    labels=np.repeat(np.arange(2),5)
    support=independent_budget_ids(labels,123,(1,3,5))
    out=hopfield_scores((unit(f),),support[1],labels,
                        np.array([0,1]),[unit(f)])
    assert out[0].shape==(len(x),2)
    p=conformal_univariate(np.arange(20),[np.arange(4)],4)[0]
    assert np.all((p>0)&(p<=1))
    return True


def run(args):
    numerical_contracts()
    budgets=tuple(sorted(set(args.budgets)))
    train,test,y,yt,_=load_data(args.dataset)
    classes=np.unique(y)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];meta=[]
    for seed in args.seeds:
        t0=time.monotonic()
        choices=independent_budget_ids(y,seed,budgets)
        slow=build_representations(train,test,seed,epochs=args.epochs,
                                   hidden=args.hidden,dim=args.dim)
        views,surprises,ar_coef=freeze_features(slow,seed)
        surprise_p=conformal_univariate(surprises[0],surprises,seed)
        for budget in budgets:
            idx=choices[budget]
            sources=("rawMR","shapeMR","cortex","homeo_dg",
                     "physics","predictive")
            fast={k:retrieve(views[k],idx,y,classes,views[k][3:],
                             replay=False) for k in sources}
            hopfield=hopfield_scores(views["physics"],idx,y,classes,
                                     views["physics"][3:])
            dense=retrieve(views["dense_dg"],idx,y,classes,
                            views["dense_dg"][3:])
            ref=retrieve(views["rawMR"],idx,y,classes,
                           views["rawMR"][3:])
            ridge=ridge_memory(views["rawMR"],idx,y,classes,
                              views["rawMR"][3:])
            dual_scores=[.5*(fast["rawMR"][j]+fast["shapeMR"][j]) for j in range(3)]
            predictions={"Raw MR episodic":ref,
              "Canonical MR episodic":fast["shapeMR"],
              "Dual MR episodic":dual_scores,
              "Slow cortex episodic":fast["cortex"],
              "DG episodic":fast["homeo_dg"],
              "Physics episodic":fast["physics"],
              "Predictive innovation episodic":fast["predictive"],
              "Hopfield physics episodic":hopfield,
              "Raw MR ridge":ridge}
            per_method={}
            for name in METHODS:
                if name in predictions or name=="BioPhysics safe TRAIN":continue
                s=[];metadata=[]
                for mode in range(3):
                    branch=[
                        fast["rawMR"][mode],fast["shapeMR"][mode],
                        fast["cortex"][mode],fast["homeo_dg"][mode],
                        hopfield[mode],fast["predictive"][mode]]
                    context=slow["charts"][mode+3]
                    surprise=surprise_p[mode+3]
                    setting={}
                    if name=="BioPhysics no homeostasis":
                        branch[3]=dense[mode]
                    elif name=="BioPhysics no cortex":
                        setting["without_cortex"]=True
                    elif name=="BioPhysics no physics":
                        setting["without_physics"]=True
                    elif name=="BioPhysics no predictive":
                        setting["without_prediction"]=True
                    elif name=="BioPhysics no Hopfield":
                        branch[4]=fast["physics"][mode]
                    elif name=="BioPhysics no context":
                        setting["has_context"]=False
                    elif name=="NeuroMem prior":
                        setting["without_physics"]=True
                        setting["without_prediction"]=True
                        branch[4]=fast["physics"][mode]
                    elif name!="BioPhysics full":
                        raise ValueError(f"Unrecognized model {name}")
                    yhat,diag=decision(branch,context,surprise,**setting)
                    s.append(yhat);metadata.append(diag)
                predictions[name]=s
                per_method[name]=metadata
            chosen,training_assessment=choose_meta_controller(
                views,idx,y,classes,slow["charts"][:3],surprise_p[:3],
                budget)
            predictions["BioPhysics safe TRAIN"]=predictions[chosen]
            per_method["BioPhysics safe TRAIN"]=[dict(
                selected_expert=chosen,train_objectives=training_assessment,
                selection_uses_test_labels=False)]*3
            for method in METHODS:
                scores=predictions[method]
                pred=[classes[np.argmax(s,axis=1)] for s in scores]
                if len(pred)!=3:raise AssertionError(method)
                rec=dict(dataset=args.dataset,seed=seed,budget=budget,
                    method=method,status="EXECUTED",
                    clean_macro_f1=float(f1_score(yt,pred[0],average="macro",
                                              zero_division=0)),
                    affine_macro_f1=float(f1_score(yt,pred[1],average="macro",
                                              zero_division=0)),
                    jitter_macro_f1=float(f1_score(yt,pred[2],average="macro",
                                              zero_division=0)),
                    accuracy=float(accuracy_score(yt,pred[0])),
                    train_cases=len(train),test_cases=len(test),n_labeled=len(idx),
                    ar2_coefficients=[float(q) for q in ar_coef],
                    seconds=round(time.monotonic()-t0,3),
                    label_ids=[int(i) for i in idx],
                    policy_diagnostics=per_method.get(method,None),
                    synthetic_label_preservation_assumed=True)
                rows.append(rec)
                print("RESULT_JSON="+json.dumps(rec,allow_nan=False),flush=True)
            (out/"results.json").write_text(json.dumps(rows,indent=2,
                                                       allow_nan=False))
        meta.append(dict(dataset=args.dataset,seed=seed,
              cortical_loss=slow["loss_curve"],
              cortical_fit_seconds=slow["runtime_seconds"],
              predictive_coefficients=ar_coef.tolist()))
        (out/"training.json").write_text(json.dumps(meta,indent=2,
                                                     allow_nan=False))
    if len(rows)!=len(args.seeds)*len(budgets)*len(METHODS):
        raise AssertionError("Missing complete benchmark grid")
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[5301])
    p.add_argument("--budgets",type=int,nargs="+",default=[1,3,5])
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--hidden",type=int,default=40)
    p.add_argument("--dim",type=int,default=64)
    p.add_argument("--out",default="results/mesa_biophysics")
    run(p.parse_args())
