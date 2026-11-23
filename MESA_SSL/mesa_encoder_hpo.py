#!/usr/bin/env python3
"""Direct Bayesian search of the MESA time-series encoder (TRAIN only).

Jointly tunes learning rate, representation dimension, concept/equivariance
weights, masking, input augmentation, and warmup/decay/epoch schedule.
The same number of optimization trials, labelled observations and
classification readout are used for mechanism-aware and generic SSL.

No official TEST label or TEST feature is consulted in any trial or
hyperparameter choice. After selection, classifiers are fit on exactly
5 labelled TRAIN examples per class, and TEST is evaluated once.
The spectral and affine shifts used here are *assumed* label-preserving
experimental stressors; they are not real-world clinical guarantees.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score

from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import (
    affine_gauge, generate_affine_shift, harmonic_basis,
    analytic_shape_concepts, project_shape_torch
)
from MESA_SSL.mesa_multikernel import feature_scale
from MESA_SSL.mesa_robust_select import extract
from MESA_SSL.mesa_spc_chart import chart_train, chart_predict
from mesa_mr_experiment import MESAEncoder, train_encoder, encode, indices_for_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(2)
RIDGES=(.01,.1,1.)
FUSE_WEIGHTS=(0., .10, .30, 1.)
METHODS=(
 "Raw MultiRocket","Canonical MultiRocket",
 "Fixed generic SSL-only","Fixed MESA SSL-only",
 "HPO generic SSL-only","HPO MESA SSL-only",
 "Fixed generic + MultiRocket","Fixed MESA + MultiRocket",
 "HPO generic + MultiRocket","HPO MESA + MultiRocket",
 "SPC HPO generic","SPC HPO MESA"
)


def deterministic_jitter(x, seed, noise=.12, harmonic=.08):
    """Fixed train-side nuisance views, with shape disturbances and no labels."""
    rng=np.random.default_rng(int(seed)*719+19)
    basis=harmonic_basis(x.shape[1])
    perturb=(rng.normal(size=x.shape).astype("float32")*noise)
    coeff=rng.normal(size=(len(x),2,x.shape[2])).astype("float32")
    perturb+=(harmonic*np.einsum("tk,nkc->ntc",basis,coeff)).astype("float32")
    return affine_gauge((x+perturb).astype("float32"))[0]


def train_parametric_encoder(train, seed, params, mechanism):
    """Independent encoder training with a common base architecture."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    data=torch.as_tensor(train,dtype=torch.float32)
    net=MESAEncoder(data.shape[-1],width=48,
                    representation_dim=int(params["representation_dim"]))
    optimizer=torch.optim.AdamW(net.parameters(),lr=float(params["lr"]),
                                weight_decay=1.e-4)
    basis=torch.as_tensor(harmonic_basis(data.shape[1]))
    epochs=int(params["epochs"])
    warm=int(params["warmup_epochs"])
    full_lr=float(params["lr"])
    decay=params["schedule"]
    losses=[]
    for ep in range(epochs):
        if warm and ep<warm:
            frac=(ep+1)/warm
        else:
            progress=(ep-warm)/max(1,epochs-1-warm)
            progress=max(0.,min(1.,progress))
            if decay=="cosine":
                frac=.12+.88*.5*(1+math.cos(math.pi*progress))
            elif decay=="linear":
                frac=1-.88*progress
            elif decay=="constant":
                frac=1.
            else:
                raise ValueError(f"Invalid scheduler {decay}")
        for group in optimizer.param_groups:group["lr"]=full_lr*frac
        batch_losses=[]
        for ids in torch.randperm(len(data)).split(64):
            x=data[ids]
            mask=torch.rand_like(x)<float(params["mask_ratio"])
            noisy=x+float(params["input_noise"])*torch.randn_like(x)
            corrupted=torch.where(mask,torch.zeros_like(noisy),noisy)
            z,c,reconstructed=net(corrupted)
            zn,_,_=net(x+.5*float(params["input_noise"])*torch.randn_like(x))
            reconstruction=(
                F.smooth_l1_loss(reconstructed[mask],x[mask])
                if mask.any() else F.smooth_l1_loss(reconstructed,x))
            contrast=1-F.cosine_similarity(z,zn,dim=-1).mean()
            variance=F.relu(.5-z.std(dim=0,unbiased=False)).mean()
            loss=reconstruction+.12*contrast+.02*variance
            if mechanism:
                target=analytic_shape_concepts(x,basis).flatten(1)
                a=torch.randn((len(x),1,x.shape[2]),dtype=x.dtype)
                b=torch.randn((len(x),1,x.shape[2]),dtype=x.dtype)
                amp=float(params["harmonic_amplitude"])
                perturbed=x+amp*(basis[None,:,0:1]*a+basis[None,:,1:2]*b)
                perturbed=project_shape_torch(perturbed)
                _,clean_c,_=net(x)
                _,changed_c,_=net(perturbed)
                changed_target=analytic_shape_concepts(perturbed,basis).flatten(1)
                if clean_c.shape!=target.shape:
                    raise RuntimeError(
                        f"Concept head shape mismatch {clean_c.shape} vs {target.shape}")
                ramp=min(1.,(ep+1)/max(1,warm+1))
                loss=loss+ramp*(
                    float(params["lambda_concept"])*F.smooth_l1_loss(clean_c,target)
                   +float(params["lambda_equiv"])*F.smooth_l1_loss(
                       changed_c-clean_c,changed_target-target)
                   +.15*float(params["lambda_concept"])*
                       F.smooth_l1_loss(changed_c,changed_target))
            if not torch.isfinite(loss):raise FloatingPointError("SSL loss diverged")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))
    return net.eval(),losses


def encoder_views(net, train, jitter, test_views=None):
    """Train-fitted feature scaling avoids any test statistics at fitting."""
    vals=[encode(net, train),encode(net, jitter)]
    if test_views:
        vals.extend(encode(net, z) for z in test_views)
    return feature_scale(*vals)


def linear_readout(train, queries, indices, y, classes, lam, second=None, w=0.):
    """Exact ridge leave-one-out; no TEST targets enter the readout."""
    a=train[indices]
    Y=(y[indices,None]==classes[None,:]).astype(float)
    gram=a@a.T
    if second is not None and w:
        b=second[0][indices]
        gram=gram+w*b@b.T
    inv=np.linalg.solve(gram+lam*np.eye(len(indices)),np.eye(len(indices)))
    alpha=inv@Y
    original_loo=Y-alpha/np.diag(inv)[:,None]
    output=[]
    for j,x in enumerate(queries):
        cross=x@a.T
        if second is not None and w:
            cross=cross+w*second[1][j]@second[0][indices].T
        p=cross@alpha
        output.append(p)
    return original_loo,output,inv,alpha


def trial_scores(train_feat, jitter_feat, indices, y, classes):
    """Exact group-held-out evaluation of five labels/class (not TEST)."""
    Y=(y[indices,None]==classes[None,:]).astype(float)
    A=train_feat[indices]
    B=jitter_feat[indices]
    gram=A@A.T
    cross=B@A.T
    best=None
    for lam in RIDGES:
        inv=np.linalg.solve(gram+lam*np.eye(len(indices)),np.eye(len(indices)))
        alpha=inv@Y
        diagonal=np.maximum(np.diag(inv),1e-10)
        clean=Y-alpha/diagonal[:,None]
        # The augmented prediction excludes the original label of this signal.
        disturbed=cross@alpha-(
             np.diag(cross@inv)/diagonal)[:,None]*alpha
        obj,metrics=objective_from_predictions(
            [clean,disturbed],y[indices],classes)
        choice=(obj,-lam,lam,metrics)
        if best is None or choice[:2]>best[:2]:best=choice
    return best[0],best[2],best[3]


def objective_from_predictions(probes, target, classes):
    vals=[]
    class_idx=np.searchsorted(classes,target)
    for p in probes:
        pred=classes[np.argmax(p,axis=1)]
        f1=float(f1_score(target,pred,average="macro",zero_division=0))
        logits=4*(p-p.max(axis=1,keepdims=True))
        z=np.exp(np.clip(logits,-40.,0.))
        z/=np.maximum(z.sum(axis=1,keepdims=True),1e-12)
        mse=float(np.mean(np.sum((z-np.eye(len(classes))[class_idx])**2,axis=1)))
        vals.append((f1,max(0.,1.-.5*mse)))
    # Classification first, smooth score only to resolve small-label ties.
    return .65*vals[0][0]+.20*vals[0][1]+.10*vals[1][0]+.05*vals[1][1],vals


def choose_features(train, queries, idx, y, classes, secondary=None):
    """Choose ridge and (optional) feature fusion using labelled TRAIN only."""
    keys=[(0.,lam) for lam in RIDGES] if secondary is None else [
        (w,lam) for w in FUSE_WEIGHTS for lam in RIDGES]
    Y=(y[idx,None]==classes[None,:]).astype(float)
    A=train[idx]
    candidates=[]
    for w,lam in keys:
        K=A@A.T
        if w>0:
            B=secondary[0][idx]
            K=K+w*(B@B.T)
        inv=np.linalg.solve(K+lam*np.eye(len(idx)),np.eye(len(idx)))
        alpha=inv@Y
        pred=Y-alpha/np.diag(inv)[:,None]
        train_f1=float(f1_score(y[idx],classes[np.argmax(pred,axis=1)],
                               average="macro",zero_division=0))
        # Small predeclared complexity penalty: avoid gratuitous fusion.
        candidates.append((train_f1-.003*(w>0),train_f1,-(w>0),
                           -w,-lam,w,lam,alpha))
    _,f1,__,___,____,w,lam,alpha=sorted(
        candidates,key=lambda z:z[:5],reverse=True)[0]
    pred=[]
    for j,x in enumerate(queries):
        cross=x@A.T
        if secondary is not None and w:
            cross+=w*secondary[1][j]@secondary[0][idx].T
        pred.append(cross@alpha)
    return pred,dict(weight=float(w),ridge=float(lam),train_loo_macro_f1=float(f1))


def test_readout_identity():
    """Compare held-out augmented ridge formula to explicit case deletion."""
    rng=np.random.default_rng(80)
    x=rng.normal(size=(9,7))
    q=x+rng.normal(size=x.shape)*.3
    idx=np.arange(len(x))
    y=np.array([0,1,2]*3)
    classes=np.arange(3)
    obj,lam,metrics=trial_scores(x,q,idx,y,classes)
    for i in range(len(x)):
        others=np.array([j for j in idx if j!=i])
        Y=np.eye(3)[y[others]]
        A=x[others]
        direct=q[i]@A.T@np.linalg.solve(A@A.T+lam*np.eye(len(A)),Y)
        Yfull=np.eye(3)[y]
        inv=np.linalg.solve(x@x.T+lam*np.eye(len(x)),np.eye(len(x)))
        alpha=inv@Yfull
        cross=q@x.T
        corrected=(cross@alpha-
                   (np.diag(cross@inv)/np.diag(inv))[:,None]*alpha)[i]
        np.testing.assert_allclose(corrected,direct,rtol=1e-8,atol=1e-8)
    return True


def propose(trial, mechanism, smoke):
    max_budget=5 if smoke else 16
    valid_epochs=(3,4,5) if smoke else (8,12,16)
    p=dict(
      lr=trial.suggest_float("lr",2.e-4,3.e-3,log=True),
      representation_dim=trial.suggest_categorical(
          "representation_dim",[48,96,144]),
      mask_ratio=trial.suggest_float("mask_ratio",.08,.30),
      input_noise=trial.suggest_float("input_noise",.01,.12,log=True),
      harmonic_amplitude=trial.suggest_float("harmonic_amplitude",.04,.35),
      epochs=trial.suggest_categorical("epochs",list(valid_epochs)),
      warmup_epochs=trial.suggest_int("warmup_epochs",0,3),
      schedule=trial.suggest_categorical("schedule",
                                        ["cosine","linear","constant"])
    )
    if mechanism:
        p["lambda_concept"]=trial.suggest_float(
            "lambda_concept",.08,2.0,log=True)
        p["lambda_equiv"]=trial.suggest_float(
            "lambda_equiv",.06,1.6,log=True)
    else:
        p["lambda_concept"]=0.
        p["lambda_equiv"]=0.
    return p


def optimize(train, jitter, labels, idx, classes, seed, mechanism, trials, smoke):
    study=optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed+ (1000 if mechanism else 2000),
            multivariate=True,n_startup_trials=min(8,trials),
            warn_independent_sampling=False))
    def evaluate_trial(trial):
        p=propose(trial,mechanism,smoke)
        t0=time.monotonic()
        try:
            model,losses=train_parametric_encoder(
                train,seed+44000,p,mechanism)
            features=encoder_views(model,train,jitter)
            score,lam,metrics=trial_scores(
                features[0],features[1],idx,labels,classes)
            trial.set_user_attr("train_loo_f1_clean",metrics[0][0])
            trial.set_user_attr("train_loo_f1_jitter",metrics[1][0])
            trial.set_user_attr("best_readout_ridge",lam)
            trial.set_user_attr("final_ssl_loss",losses[-1])
            trial.set_user_attr("elapsed_seconds",round(time.monotonic()-t0,3))
            return float(score)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:
            trial.set_user_attr("error",str(e))
            return -1.
    study.optimize(evaluate_trial,n_trials=trials,show_progress_bar=False,n_jobs=1)
    if study.best_value<=-1.:raise RuntimeError("No admissible encoder trial")
    p=propose_for_best(study.best_trial.params,mechanism)
    model,losses=train_parametric_encoder(train,seed+44000,p,mechanism)
    return dict(model=model,params=p,
         selected_train_objective=float(study.best_value),
         best_trial_number=study.best_trial.number,
         trials=len(study.trials),loss_curve=losses,
         trial_history=[
             dict(number=t.number,objective=t.value,params=t.params,
                  diagnostics=t.user_attrs)
             for t in study.trials])


def propose_for_best(values,mechanism):
    p=dict(values)
    if not mechanism:
        p["lambda_concept"]=0.
        p["lambda_equiv"]=0.
    return p


def concept_error(model,train):
    model.eval()
    basis=torch.as_tensor(harmonic_basis(train.shape[1]))
    with torch.no_grad():
        pred=[]
        for batch in np.array_split(train,max(1,math.ceil(len(train)/64))):
            _,c,_=model(torch.as_tensor(batch,dtype=torch.float32))
            pred.append(c.detach().numpy())
        target=analytic_shape_concepts(
            torch.as_tensor(train,dtype=torch.float32),basis).flatten(1).numpy()
        out=np.concatenate(pred)
        if out.shape!=target.shape:raise ValueError("Concept head dimensions differ")
        return float(np.sqrt(np.mean((out-target)**2)))


def run(args):
    test_readout_identity()
    train,test,y,yt,_=load_data(args.dataset)
    train_shape,c_train=affine_gauge(train)
    test_shape,c_test=affine_gauge(test)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];optim_details=[]
    for seed in args.seeds:
        tick=time.monotonic()
        np.random.seed(seed);torch.manual_seed(seed)
        indices=indices_for_seed(y,seed,budget=args.budget)
        classes=np.unique(y)
        jitter_train=deterministic_jitter(train_shape,seed+175)
        # Only TRAIN features and TRAIN labels enter Optuna.
        models={}
        for mechanism,name in ((False,"generic"),(True,"MESA")):
            models["Fixed "+name],fix_loss=train_encoder(
                train_shape,seed, args.fixed_epochs,mechanism=mechanism)
            fit=optimize(train_shape,jitter_train,y,indices,classes,
                         seed,mechanism,args.trials,args.smoke)
            models["HPO "+name]=fit["model"]
            optim_details.append(dict(dataset=args.dataset,seed=seed,
                 encoder_family=name,params=fit["params"],
                 selected_train_objective=fit["selected_train_objective"],
                 best_trial_number=fit["best_trial_number"],
                 full_trial_history=fit["trial_history"],
                 selected_training_loss=fit["loss_curve"]))
            print("OPT_SUMMARY_JSON="+json.dumps({
                "dataset":args.dataset,"seed":seed,"encoder":name,
                "selected_train_objective":fit["selected_train_objective"],
                "best_trial":fit["best_trial_number"],
                "hyperparameters":fit["params"]},allow_nan=False),flush=True)

        test_shift=generate_affine_shift(test,seed,1.5)
        test_jitter=deterministic_jitter(test,seed+450,noise=.12,harmonic=.08)
        shape_queries=[affine_gauge(z)[0]
                       for z in (test,test_shift,test_jitter)]
        raw_queries=[test,test_shift,test_jitter]
        raw=extract(train,raw_queries,seed)
        shape=extract(train_shape,shape_queries,seed)
        features={name:encoder_views(net,train_shape,jitter_train,
                                    shape_queries)
                  for name,net in models.items()}
        preds={};meta={}
        preds["Raw MultiRocket"],meta["Raw MultiRocket"]=choose_features(
            raw[0],raw[1:],indices,y,classes)
        preds["Canonical MultiRocket"],meta["Canonical MultiRocket"]=choose_features(
            shape[0],shape[1:],indices,y,classes)
        for prefix in ("Fixed","HPO"):
            for name in ("generic","MESA"):
                key=prefix+" "+name
                z=features[key]
                single=key+" SSL-only"
                fused=key+" + MultiRocket"
                preds[single],meta[single]=choose_features(
                    z[0],z[2:],indices,y,classes)
                # z[1] is train-side validation jitter; z[2:] are TEST modes.
                preds[fused],meta[fused]=choose_features(
                    shape[0],shape[1:],indices,y,classes,
                    secondary=(z[0],z[2:]))
        chart=chart_train(c_train,seed,alpha=.10)
        chart_alarm=[]
        for x in (test,test_shift,test_jitter):
            _,c=affine_gauge(x)
            chart_alarm.append(chart_predict(chart,c)[0])
        for name in ("generic","MESA"):
            p=["SPC HPO "+name]
            safe=preds["HPO "+name+" + MultiRocket"]
            preds[p[0]]=[
                np.where(chart_alarm[j][:,None],safe[j],
                         preds["Raw MultiRocket"][j])
                for j in range(3)]
            meta[p[0]]=dict(chart_alpha=.10,
                             alpha_fitted_only_train=True,
                             switched_rate=[
                              float(np.mean(chart_alarm[j])) for j in range(3)])
        for method in METHODS:
            pp=preds[method]
            yi=[classes[np.argmax(q,axis=1)] for q in pp]
            rec=dict(dataset=args.dataset,seed=seed,method=method,status="EXECUTED",
             macro_f1=float(f1_score(yt,yi[0],average="macro",zero_division=0)),
             affine_macro_f1=float(f1_score(yt,yi[1],average="macro",zero_division=0)),
             jitter_macro_f1=float(f1_score(yt,yi[2],average="macro",zero_division=0)),
             accuracy=float(accuracy_score(yt,yi[0])),
             labelled_train=int(len(indices)),n_train=len(train),n_test=len(test),
             chart_alarm_clean=float(np.mean(chart_alarm[0])),
             chart_alarm_affine=float(np.mean(chart_alarm[1])),
             chart_alarm_jitter=float(np.mean(chart_alarm[2])),
             chart="TRAIN-only shrinkage Hotelling-type T2",
             test_affine="Synthetic offset+trend assumed label-preserving",
             test_jitter="Synthetic moderate jitter+harmonic interference assumed label-preserving",
             readout_selection=meta[method],elapsed_seed_sec=round(
                 time.monotonic()-tick,2))
            rows.append(rec)
            print("RESULT_JSON="+json.dumps(rec,allow_nan=False),flush=True)
        for d in optim_details:
            if d["dataset"]==args.dataset and d["seed"]==seed:
                d["mechanism_concept_rmse"]=concept_error(
                    models["HPO "+d["encoder_family"]],train_shape)
        (out/"results.json").write_text(json.dumps(rows,indent=2,allow_nan=False))
        (out/"encoder_trials.json").write_text(
            json.dumps(optim_details,indent=2,allow_nan=False))
        print("SEED_DONE="+str(seed)+" elapsed="+str(round(time.monotonic()-tick,1)),
              flush=True)
    if len(rows)!=len(args.seeds)*len(METHODS):
        raise AssertionError(f"Expected {len(args.seeds)*len(METHODS)}, got {len(rows)}")
    return rows


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=[901])
    ap.add_argument("--budget",type=int,default=5)
    ap.add_argument("--trials",type=int,default=18)
    ap.add_argument("--fixed-epochs",type=int,default=15)
    ap.add_argument("--out",default="results/mesa_encoder_hpo")
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    if a.trials<2:ap.error("Need >=2 trials")
    run(a)
