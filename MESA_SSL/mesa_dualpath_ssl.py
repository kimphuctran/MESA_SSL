#!/usr/bin/env python3
"""MESA-DP: dual-path selective-invariance time-series SSL research prototype.

Unlike earlier gauge-only MESA encoders, the raw stream is *never discarded*.
A second independently learned stream encodes an affine-canonical residual.
An exact analytical concept skip carries mean, linear slope and residual scale,
while a separately trained head predicts these concepts for attribution.

The experiment evaluates both mechanism and zero-mechanism-loss networks,
paired seeds, MultiRocket and raw/canonical multi-kernel controls. All kernel
weights/regularizers and drift policies are selected using only five labeled
TRAIN instances/class and TRAIN-only synthetic perturbations.

TEST is not accessed for hyperparameter/model selection, including architecture
choice. Its labels are used only for final benchmark reporting. The development
tasks are not independent unseen tasks due to prior research iterations.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import affine_gauge, generate_affine_shift
from MESA_SSL.mesa_robust_select import extract
from MESA_SSL.mesa_multikernel import feature_scale
from MESA_SSL.mesa_spc_chart import chart_train, chart_predict
from mesa_mr_experiment import indices_for_seed

torch.set_num_threads(2)
RIDGES = (.01, .1, 1.)
DEV_DATASETS = ("GunPoint","ItalyPowerDemand","BasicMotions",
                "CBF","SyntheticControl")
METHODS = ("Raw MultiRocket", "Canonical MultiRocket", "Dual MultiRocket",
           "Generic learned-only", "MESA learned-only",
           "Generic dual-path hybrid", "MESA dual-path hybrid",
           "MESA SPC hybrid", "MESA dual-chart", "Adaptive dual-chart",
           "TRAIN-selected admissible")


def torch_decompose(x, eps=1e-6):
    """Exact orthogonal offset/trend residual plus retained concepts."""
    t = torch.linspace(-1., 1., x.size(1), device=x.device, dtype=x.dtype)
    mu = x.mean(dim=1, keepdim=True)
    beta = ((x*t[None,:,None]).sum(dim=1, keepdim=True) /
            (t*t).sum())
    residual = x-mu-t[None,:,None]*beta
    rms = residual.square().mean(dim=1,keepdim=True).sqrt().clamp_min(eps)
    canonical = residual/rms
    concepts = torch.cat((mu[:,0,:],beta[:,0,:],
                          torch.log(rms[:,0,:])),dim=-1)
    return canonical, concepts


class ScaleConvTower(nn.Module):
    """Compact multi-resolution temporal CNN with separate receptive fields."""
    def __init__(self, channels, hidden=40, dim=64):
        super().__init__()
        self.stem=nn.Conv1d(channels,hidden,7,padding=3)
        self.paths=nn.ModuleList(
            [nn.Sequential(nn.Conv1d(hidden,hidden,k,padding=k//2,
                                     groups=hidden),
                           nn.Conv1d(hidden,hidden,1))
             for k in (3,7,15)])
        self.norms=nn.ModuleList([nn.GroupNorm(4,hidden) for _ in range(3)])
        self.out=nn.Sequential(nn.Linear(hidden*6,dim),
                               nn.LayerNorm(dim),nn.GELU())
    def forward(self,x):
        h=F.gelu(self.stem(x.transpose(1,2)))
        for path,norm in zip(self.paths,self.norms):
            h=F.gelu(norm(h+.35*path(h)))
        # Global, local-peak and four-bin temporal summaries retained.
        pooled=torch.cat((h.mean(-1),h.amax(-1),
                          F.adaptive_avg_pool1d(h,4).flatten(1)),dim=-1)
        return self.out(pooled)


class MESASelectiveEncoder(nn.Module):
    def __init__(self, channels, hidden=40, dim=64):
        super().__init__()
        self.raw=ScaleConvTower(channels,hidden,dim)
        self.shape=ScaleConvTower(channels,hidden,dim)
        self.concept_pred=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),
                                        nn.Linear(dim,3*channels))
        self.freq_pred=nn.Linear(dim,8*channels)
        self.shape_concept_pred=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),
                                             nn.Linear(dim,2*channels))
    def forward(self,x):
        canonical,concept=torch_decompose(x)
        raw=self.raw(x)
        shape=self.shape(canonical)
        return {"raw":raw,"shape":shape,"concept":concept,
                "concept_pred":self.concept_pred(raw),
                "shape_concept_pred":self.shape_concept_pred(shape),
                "freq_pred":self.freq_pred(shape)}


def shape_harmonic_targets(x):
    """Two bounded harmonics of the canonical residual, not causal labels."""
    q,_=torch_decompose(x)
    tau=torch.linspace(0.,2*math.pi,q.shape[1],
                       device=q.device,dtype=q.dtype)
    a=(q*torch.cos(tau)[None,:,None]).mean(1)
    b=(q*torch.sin(tau)[None,:,None]).mean(1)
    return torch.cat([a,b],dim=-1)


def band_spectrum(x):
    q,_=torch_decompose(x)
    f=torch.fft.rfft(q,dim=1).abs().transpose(1,2)
    return torch.log1p(F.adaptive_avg_pool1d(f,8).flatten(1))


def train_ssl(train, seed, mechanism=True, epochs=12, lr=.00075,
              hidden=40, dim=64, lambda_concept=.30,
              lambda_equiv=.24, lambda_shape=.20,
              lambda_shape_equiv=.16, mask=.12, noise=.05):
    """Paired controls use identical initialization and RNG consumption."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device="cpu"
    train_tensor=torch.as_tensor(train,dtype=torch.float32,device=device)
    net=MESASelectiveEncoder(train_tensor.shape[-1],hidden,dim).to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.0001)
    history=[]
    time_axis=torch.linspace(-1,1,train_tensor.shape[1],dtype=torch.float32)
    for ep in range(epochs):
        schedule = .15+.85*.5*(1+math.cos(math.pi*ep/max(1,epochs-1)))
        for group in optim.param_groups:group["lr"]=lr*schedule
        epoch_losses=[]
        for idx in torch.randperm(len(train_tensor)).split(64):
            base=train_tensor[idx]
            x_jitter=base+noise*torch.randn_like(base)
            mask_tensor=torch.rand_like(base)<mask
            augmented=torch.where(mask_tensor,torch.zeros_like(x_jitter),
                                  x_jitter)
            # The raw stream receives the original signal; the shape stream
            # is an exact affine projection. Neither loses mean/slope globally.
            clean=net(base)
            view=net(augmented)
            invariance=(1-F.cosine_similarity(
                clean["shape"],view["shape"],dim=-1).mean()
                +.5*(1-F.cosine_similarity(
                clean["raw"],view["raw"],dim=-1).mean()))
            variance=(
                F.relu(.45-clean["shape"].std(0,unbiased=False)).mean()
                +F.relu(.45-clean["raw"].std(0,unbiased=False)).mean())
            spectral=F.smooth_l1_loss(clean["freq_pred"],band_spectrum(base))
            # Consume identical affine perturbation draws in both arms.
            offset=.35*torch.randn((len(base),1,base.shape[2]))
            trend=.35*torch.randn((len(base),1,base.shape[2]))
            shifted=base+offset+time_axis[None,:,None]*trend
            observed=net(shifted)
            # The original mechanism objective left the shape tower
            # unaffected because affine canonicalization was exact.
            # A separate bounded harmonic intervention now acts on shape.
            tt=torch.linspace(0,2*math.pi,base.size(1),
                              device=base.device,dtype=base.dtype)
            harmonic=base+.30*torch.randn((len(base),1,base.shape[2]))*(
                torch.sin(tt)[None,:,None])
            shape_modified=net(harmonic)
            target_shape=shape_harmonic_targets(base)
            delta_shape_target=shape_harmonic_targets(harmonic)-target_shape
            shape_alignment=F.smooth_l1_loss(
                clean["shape_concept_pred"],target_shape)
            shape_equiv=F.smooth_l1_loss(
                shape_modified["shape_concept_pred"]-
                clean["shape_concept_pred"],delta_shape_target)
            expected_delta=observed["concept"]-clean["concept"]
            alignment=F.smooth_l1_loss(
                clean["concept_pred"],clean["concept"])
            equivariance=F.smooth_l1_loss(
                observed["concept_pred"]-clean["concept_pred"],
                expected_delta)
            loss=.30*invariance+.08*variance+.32*spectral
            w=float(mechanism)
            loss=loss+w*(lambda_concept*alignment+
                         lambda_equiv*equivariance+
                         lambda_shape*shape_alignment+
                         lambda_shape_equiv*shape_equiv)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite SSL loss")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),2.5)
            optim.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(epoch_losses)))
    return net.eval(),history


@torch.no_grad()
def extract_nn(model, x):
    model.eval()
    outputs={"raw":[],"shape":[],"concept_pred":[],
             "shape_concept_pred":[]}
    for ids in np.array_split(np.arange(len(x)),max(1,int(np.ceil(len(x)/64)))):
        out=model(torch.tensor(x[ids],dtype=torch.float32))
        for key in outputs:outputs[key].append(out[key].cpu().numpy())
    return {key:np.concatenate(val,axis=0) for key,val in outputs.items()}


def signal_jitter(x,seed,noise=.085,harmonic=.045):
    rng=np.random.default_rng(int(seed)*1891+297)
    t=np.linspace(0,4*np.pi,x.shape[1],dtype=np.float32)
    phase=rng.uniform(-np.pi,np.pi,size=(len(x),1,x.shape[-1]))
    amp=rng.normal(0,harmonic,size=(len(x),1,x.shape[-1]))
    sinusoid=(amp*np.sin(t[None,:,None]+phase)).astype("float32")
    return (x+noise*rng.standard_normal(x.shape).astype("float32")
            +sinusoid).astype("float32")


def prepare_inputs(train, test, seed):
    """No TEST labels or TEST-derived fitting statistics used."""
    augment_train=generate_affine_shift(train,seed+775,1.30)
    jitter_train=signal_jitter(train,seed+991)
    shift_test=generate_affine_shift(test,seed+803,1.30)
    jitter_test=signal_jitter(test,seed+1001)
    queries=[train,augment_train,jitter_train,test,shift_test,jitter_test]
    return queries


def features(train, test, seed, models):
    queries=prepare_inputs(train,test,seed)
    canonical=[affine_gauge(x)[0] for x in queries]
    concepts=[affine_gauge(x)[1] for x in queries]
    toquery=lambda xs:tuple(xs[j] for j in range(1,6))
    views={
        "rawMR":extract(train,toquery(queries),seed),
        "shapeMR":extract(canonical[0],toquery(canonical),seed),
        "stats":feature_scale(*concepts),
    }
    for suffix,net in models.items():
        arrays=[extract_nn(net,x) for x in queries]
        for part in ("raw","shape"):
            views[suffix+"_"+part]=feature_scale(
                *(arr[part] for arr in arrays))
    return views,queries,concepts


def build_kernels(views,ids):
    """Cached train, train-probe and TEST cross-kernels, all TRAIN-scaled."""
    K={}
    for key,blocks in views.items():
        ref=blocks[0][ids]
        K[key]={
            "train":ref@ref.T,
            "probe":[blocks[j][ids]@ref.T for j in (1,2)],
            "test":[blocks[j]@ref.T for j in (3,4,5)]}
    return K


def candidates(family):
    if family=="Raw MultiRocket":
        return [{"rawMR":1.}]
    if family=="Canonical MultiRocket":
        return [{"shapeMR":1.}]
    if family=="Dual MultiRocket":
        return [{"rawMR":1.},{"shapeMR":1.},
                {"rawMR":1.,"shapeMR":.5},
                {"rawMR":.5,"shapeMR":1.},
                {"rawMR":1.,"shapeMR":1.},
                {"rawMR":1.,"shapeMR":1.,"stats":.05}]
    if family.startswith("Generic"):k="generic"
    elif family.startswith("MESA"):k="mesa"
    else:raise ValueError(family)
    r=k+"_raw";s=k+"_shape"
    if "learned-only" in family:
        return [{r:1.},{s:1.},{r:1.,s:1.},
                {r:1.,s:1.,"stats":.10},{s:1.,"stats":.10}]
    if "hybrid" in family:
        baseline=candidates("Dual MultiRocket")
        return baseline+[
            {"rawMR":1.,"shapeMR":1.,r:.1},
            {"rawMR":1.,"shapeMR":1.,s:.1},
            {"rawMR":1.,"shapeMR":1.,r:.3,s:.3},
            {"rawMR":1.,"shapeMR":1.,r:.5,s:.5,"stats":.08},
            {"rawMR":1.,r:.4,"stats":.08},
            {"shapeMR":1.,s:.4},
        ]
    raise ValueError(family)


def fit_candidates(K,idx,y,classes,family):
    """Leakage-safe ridge TRAIN-LOO under clean, affine and jitter probes."""
    Y=(y[idx,None]==classes[None,:]).astype(float)
    labels=y[idx]
    options=[]
    for config in candidates(family):
        gram=sum(w*K[k]["train"] for k,w in config.items())
        for ridge in RIDGES:
            inv=np.linalg.solve(gram+ridge*np.eye(len(idx)),np.eye(len(idx)))
            coef=inv@Y
            diagonal=np.maximum(np.diag(inv),1e-10)
            original_loo=Y-coef/diagonal[:,None]
            loo=[original_loo]
            for mode in (0,1):
                cross=sum(w*K[k]["probe"][mode] for k,w in config.items())
                val=cross@coef-(np.diag(cross@inv)/diagonal)[:,None]*coef
                loo.append(val)
            f1=[float(f1_score(labels,classes[np.argmax(a,axis=1)],
                               average="macro",zero_division=0)) for a in loo]
            # Selection must not inspect TEST scores, even from earlier runs.
            # Complexity tie penalty reduces small-label model hopping.
            obj=.40*f1[0]+.38*f1[1]+.22*f1[2]-.008*(len(config)-1)
            options.append(dict(objective=float(obj),weights=config,
                                ridge=ridge,coef=coef,cv_f1=f1,loo=loo))
    choice=sorted(options,key=lambda c:(
        -c["objective"],len(c["weights"]),c["ridge"],
        tuple(sorted(c["weights"].items()))))[0]
    return choice


def predict(choice,K,view="test"):
    terms={}
    for key,w in choice["weights"].items():
        terms[key]=[w*K[key][view][j]@choice["coef"]
                    for j in range(3)]
    return [sum(terms[key][j] for key in terms) for j in range(3)]


def chart_policy(ctrain, cprobes, ctests, choices, K, idx, y, classes, seed):
    """TRAIN-only choose fixed raw/dual/MESA or chart-switched raw→MESA."""
    chart=chart_train(ctrain,seed,alpha=.10)
    Y=y[idx]
    raw,hybrid=choices["Raw MultiRocket"],choices["MESA dual-path hybrid"]
    raw_test,pred_test=predict(raw,K),predict(hybrid,K)
    raw_train,hybrid_train=raw["loo"],hybrid["loo"]
    alarm_train=[chart_predict(chart,x)[2][idx] for x in [ctrain]+cprobes]
    alarm_test=[chart_predict(chart,x)[2] for x in ctests]
    objectives=[]
    for threshold in (.05,.10,.20):
        for strength in (0.,.5,1.):
            out=[]
            for j in range(3):
                gate=strength*(alarm_train[j]<=threshold)
                out.append((1-gate[:,None])*raw_train[j]+
                           gate[:,None]*hybrid_train[j])
            f=[float(f1_score(Y,classes[np.argmax(x,axis=1)],
                              average="macro",zero_division=0)) for x in out]
            cv=.40*f[0]+.38*f[1]+.22*f[2]-.005*(strength>0)
            objectives.append((cv,-strength,-threshold,threshold,strength,f))
    _,__,___,threshold,strength,cv=sorted(objectives,reverse=True)[0]
    pred=[]
    for j in range(3):
        gate=strength*(alarm_test[j]<=threshold)
        pred.append((1-gate[:,None])*raw_test[j]+
                    gate[:,None]*pred_test[j])
    return pred,dict(alpha=threshold,strength=strength,cv_f1=cv,
                     reference_n_cal=int(chart["n_cal"]),
                     test_alarm_rate=[float(np.mean(q<=threshold))
                                      for q in alarm_test])




def morphology_stats(x):
    """Label-blind waveform roughness and spectral-shape signatures.

    This chart supplements the mean/slope chart; it does not assume that
    waveform jitter is equivalent to an affine offset or trend.
    """
    q,_=affine_gauge(x)
    q=q.astype(np.float64)
    d=np.diff(q,axis=1)
    d2=np.diff(d,axis=1) if q.shape[1]>3 else d
    freq=np.fft.rfft(q,axis=1)
    power=(freq.real**2+freq.imag**2)
    cut=max(1,int(math.ceil(.28*(power.shape[1]-1))))
    high=power[:,cut:,:].sum(1)
    total=np.maximum(power.sum(1),1e-8)
    rough=np.log1p(np.mean(d*d,axis=1))
    curvature=np.log1p(np.mean(d2*d2,axis=1))
    highratio=high/total
    lag=(q[:,1:,:]*q[:,:-1,:]).mean(1)
    extreme=np.quantile(np.abs(q),.95,axis=1)
    return np.column_stack([rough,curvature,highratio,lag,extreme])


def calibrate_morphology(train_features,seed):
    """Split-conformal individual-case shrinkage chart; TRAIN only."""
    arr=np.asarray(train_features,dtype=np.float64)
    order=np.random.default_rng(seed+220551).permutation(len(arr))
    cutoff=min(len(arr)-4,max(8,round(.6*len(arr))))
    if cutoff<5:raise ValueError("Insufficient TRAIN for split chart")
    idx0,idx1=order[:cutoff],order[cutoff:]
    scaler=StandardScaler().fit(arr[idx0])
    base=scaler.transform(arr[idx0])
    covariance=LedoitWolf().fit(base)
    precision=covariance.precision_
    center=covariance.location_
    def score(values):
        z=scaler.transform(np.asarray(values,dtype=np.float64))-center
        return np.einsum("ij,jk,ik->i",z,precision,z)
    cal=np.sort(score(arr[idx1]))
    return dict(center=center,precision=precision,scaler=scaler,
                calibration=cal,fit_n=len(idx0),cal_n=len(idx1),score=score)


def morphology_p(chart,features):
    vals=chart["score"](features)
    reference=chart["calibration"]
    p=(1+(reference[None,:]>=vals[:,None]).sum(1))/(len(reference)+1)
    return p


def two_chart_policy(ctrain,cprobes,ctests,queries,choices,K,idx,y,
                     classes,seed,restricted=False):
    """TRAIN-only structure selection plus independent nuisance controllers.

    Calibration and expert/model choice are made using TRAIN examples.
    The TEST modes enter only through feature extraction after selection.
    """
    morphological=calibrate_morphology(morphology_stats(queries[0]),seed)
    morphology_probs=[
        morphology_p(morphological,morphology_stats(q))
        for q in queries]
    affine=chart_train(ctrain,seed,alpha=.10)
    affine_probs=[chart_predict(affine,c)[2] for c in
                  [ctrain]+list(cprobes)+list(ctests)]
    choices_allowed=[
        "Raw MultiRocket","Canonical MultiRocket","Dual MultiRocket",
        "MESA learned-only","MESA dual-path hybrid"]
    if not restricted:
        choices_allowed+=["Generic learned-only","Generic dual-path hybrid"]
    def pick(mode):
        # Under affine shift only a bounded, predeclared invariant/hybrid
        # class is admissible: pure learned embeddings were too unstable
        # when chosen from five labels/class in development tests.
        allowed=(choices_allowed if mode==2 else
                 [n for n in choices_allowed if "learned-only" not in n])
        return sorted(allowed,key=lambda name:(
             -choices[name]["cv_f1"][mode]+.007*(
                len(choices[name]["weights"])-1),
             len(choices[name]["weights"]),name))[0]
    names=dict(clean=pick(0),affine=pick(1),jitter=pick(2))
    train_predictions={name:choices[name]["loo"] for name in choices_allowed}
    test_predictions={name:predict(choices[name],K) for name in choices_allowed}
    labeled=y[idx]
    chart_candidates=[]
    for a_cut in (.05,.10,.20):
        for m_cut in (.10,.20,.30):
            loo=[]
            for mode in range(3):
                # Altered TRAIN sample is never included as a labeled
                # predictor in any selected expert's exact LOO score.
                pm=morphology_probs[mode][idx]
                pa=affine_probs[mode][idx]
                base=train_predictions[names["clean"]][mode]
                after_affine=train_predictions[names["affine"]][mode]
                after_jitter=train_predictions[names["jitter"]][mode]
                # Roughness has precedence: an affine-only chart does not
                # detect a high-frequency waveform disturbance.
                mixed=np.where((pm<=m_cut)[:,None],after_jitter,
                     np.where((pa<=a_cut)[:,None],after_affine,base))
                loo.append(mixed)
            f=[float(f1_score(labeled,classes[np.argmax(x,axis=1)],
                              average="macro",zero_division=0)) for x in loo]
            objective=.40*f[0]+.37*f[1]+.23*f[2]-.003*(
                names["clean"]!=names["affine"])-.003*(
                names["clean"]!=names["jitter"])
            chart_candidates.append((objective,-m_cut,-a_cut,a_cut,m_cut,f))
    winner=sorted(chart_candidates,reverse=True)[0]
    _,__,___,a_cut,m_cut,train_f=winner
    preds=[];alarm_rates=[]
    for mode in range(3):
        index=mode+3
        pm=morphology_probs[index]
        pa=affine_probs[index]
        preds.append(np.where((pm<=m_cut)[:,None],
            test_predictions[names["jitter"]][mode],
            np.where((pa<=a_cut)[:,None],
                test_predictions[names["affine"]][mode],
                test_predictions[names["clean"]][mode])))
        alarm_rates.append(dict(morphology=float(np.mean(pm<=m_cut)),
                                affine=float(np.mean(pa<=a_cut))))
    return preds,dict(train_policy_f1=train_f,train_score=float(winner[0]),
        expert_names=names,morphology_threshold=m_cut,
        affine_threshold=a_cut,alarm_rates=alarm_rates,
        morphology_train_reference=morphological["fit_n"],
        morphology_calibration=morphological["cal_n"])


def test_scientific_contract():
    """Synthetic numeric checks for the two information paths."""
    rng=np.random.default_rng(881)
    data=rng.normal(size=(10,97,3)).astype("float32")
    shift=generate_affine_shift(data,11,1.1)
    q,c=affine_gauge(data)
    qs,cs=affine_gauge(shift)
    if np.max(np.abs(q-qs))>8e-5:
        raise AssertionError("Canonical nuisance invariance violated")
    if np.mean(np.abs(c-cs))<.001:
        raise AssertionError("Concept skip lost affine shift information")
    x=torch.tensor(data)
    qt,ct=torch_decompose(x)
    np.testing.assert_allclose(qt.detach().numpy(),q,rtol=1e-4,atol=2e-5)
    np.testing.assert_allclose(ct.detach().numpy(),c,rtol=2e-4,atol=2e-5)
    for channels in (1,3):
        m=MESASelectiveEncoder(channels,hidden=24,dim=32)
        z=m(torch.randn(4,83,channels))
        for field in ("raw","shape"):
            assert z[field].shape==(4,32)
        assert z["concept_pred"].shape==(4,3*channels)
        assert z["shape_concept_pred"].shape==(4,2*channels)
    return True


def evaluate(args):
    test_scientific_contract()
    train,test,y,yt,_=load_data(args.dataset)
    if min(np.bincount(y))<args.budget:
        raise ValueError("Insufficient labelled TRAIN cases in a class")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    records=[];training=[]
    for seed in args.seeds:
        t0=time.monotonic()
        indices=indices_for_seed(y,seed,budget=args.budget)
        classes=np.unique(y)
        models={}
        for mechanism,key in ((False,"generic"),(True,"mesa")):
            model,curve=train_ssl(train,seed+9000,mechanism,
                                  epochs=args.epochs,hidden=args.hidden,
                                  dim=args.dim)
            models[key]=model
            training.append(dict(dataset=args.dataset,seed=seed,
                     model=key,losses=curve))
        views,queries,concepts=features(train,test,seed,models)
        concept_rmse={};shape_concept_rmse={}
        shape_target=shape_harmonic_targets(
            torch.as_tensor(train,dtype=torch.float32)).detach().numpy()
        for model_name,net in models.items():
            representation=extract_nn(net,train)
            predicted=representation["concept_pred"]
            concept_rmse[model_name]=float(np.sqrt(np.mean(
                (predicted-concepts[0])**2)))
            shape_concept_rmse[model_name]=float(np.sqrt(np.mean(
                (representation["shape_concept_pred"]-shape_target)**2)))
        kernels=build_kernels(views,indices)
        choices={};predictions={}
        for family in METHODS[:7]:
            choice=fit_candidates(kernels,indices,y,classes,family)
            choices[family]=choice
            predictions[family]=predict(choice,kernels)
        predictions["MESA SPC hybrid"],policy=chart_policy(
            concepts[0],concepts[1:3],concepts[3:],
            choices,kernels,indices,y,classes,seed)
        predictions["MESA dual-chart"],meta_chart_mesa=two_chart_policy(
            concepts[0],concepts[1:3],concepts[3:],
            queries,choices,kernels,indices,y,classes,seed,restricted=True)
        predictions["Adaptive dual-chart"],meta_chart_full=two_chart_policy(
            concepts[0],concepts[1:3],concepts[3:],
            queries,choices,kernels,indices,y,classes,seed,restricted=False)
        # TRAIN-only select among bounded few-label expert candidates.
        selected=min(("Raw MultiRocket","Canonical MultiRocket",
                      "Dual MultiRocket","Generic dual-path hybrid",
                      "MESA dual-path hybrid"),
                     key=lambda k:(-choices[k]["objective"],
                                   len(choices[k]["weights"]),k))
        predictions["TRAIN-selected admissible"]=predictions[selected]
        for name in METHODS:
            if name not in predictions:raise AssertionError(name)
            p=predictions[name]
            actual=[classes[np.argmax(s,axis=1)] for s in p]
            record=dict(dataset=args.dataset,seed=seed,method=name,
              clean_macro_f1=float(f1_score(yt,actual[0],average="macro")),
              affine_macro_f1=float(f1_score(yt,actual[1],average="macro")),
              jitter_macro_f1=float(f1_score(yt,actual[2],average="macro")),
              accuracy=float(accuracy_score(yt,actual[0])),
              n_train=len(train),n_test=len(test),n_labeled=len(indices),
              train_cv_objective=(choices[name]["objective"]
                if name in choices else None),
              train_cv_f1=(choices[name]["cv_f1"]
                if name in choices else None),
              weights=(choices[name]["weights"]
                if name in choices else None),
              selected=selected if name=="TRAIN-selected admissible" else None,
              chart_policy=(
                  policy if name=="MESA SPC hybrid" else
                  meta_chart_mesa if name=="MESA dual-chart" else
                  meta_chart_full if name=="Adaptive dual-chart" else None),
              concept_rmse_train=concept_rmse,
              shape_concept_rmse_train=shape_concept_rmse,
              runtime_seconds=round(time.monotonic()-t0,2),
              status="EXECUTED",
              synthetic_shift_assumption="label-preserving offset/trend and jitter")
            records.append(record)
            print("RESULT_JSON="+json.dumps(record,allow_nan=False),flush=True)
        (out/"results.json").write_text(json.dumps(records,indent=2))
        (out/"training.json").write_text(json.dumps(training,indent=2))
    if len(records)!=len(args.seeds)*len(METHODS):
        raise AssertionError("Missing output records")
    return records


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--seeds",type=int,nargs="+",default=[2601,2602])
    ap.add_argument("--epochs",type=int,default=12)
    ap.add_argument("--hidden",type=int,default=40)
    ap.add_argument("--dim",type=int,default=64)
    ap.add_argument("--budget",type=int,default=5)
    ap.add_argument("--out",default="results/mesa_dualpath")
    a=ap.parse_args()
    evaluate(a)
