#!/usr/bin/env python3
"""Grouped-augmentation Bayesian ridge + chart-governed MESA-SSL.

Raw augmented ridge is trained on three feature views of each *same* labelled
TRAIN signal. Every tuning prediction removes *all three* appearances of
that signal from the training reference, preventing augmentation leakage.
Only TRAIN labels enter TPE optimization or family selection. The two
synthetic shifts are defined independently of TEST. Reports do not imply
ARL guarantees or universal performance superiority.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna

from MESA_SSL.mesa_spc_bayesian import (
    build_cache, solve, fixed_model, evaluate, f1_objective, mix
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
FAMILIES = ("canonical","generic","mechanism")
METHODS = (
    "Raw MultiRocket", "RawAug-fixed", "RawAug-BO",
    "Canonical MultiRocket", "Canonical + generic SSL",
    "Canonical + MESA SSL", "BO-Aug-SPC canonical",
    "BO-Aug-SPC generic", "BO-Aug-SPC MESA",
    "BO-Aug TRAIN-selected"
)


def raw_augmented(cache, w1, w2, ridge):
    """Weighted augmented feature ridge with exact group leave-one-out.

    Group i consists of feature references (original_i, offset_i, trend_i)
    associated with *one* labelled TRAIN item. All are deleted together
    for every optimization fold. This is NOT ordinary LOOCV on duplicated
    augmentations, which would leak the label through other copies.
    """
    ref = cache["raw_views"]
    ids = cache["labeled"]
    n = len(ids)
    X = np.vstack([ref[j][ids] for j in (0,1,2)])
    sample_weights = np.concatenate([
        np.ones(n), np.full(n, np.sqrt(max(w1,0.))),
        np.full(n, np.sqrt(max(w2,0.)))])
    Z = X * sample_weights[:,None]
    Y = np.tile(cache["Y"],(3,1))*sample_weights[:,None]
    C = Z@Z.T + float(ridge)*np.eye(3*n)
    inv = np.linalg.solve(C,np.eye(3*n))
    coefficient = inv@Y

    # Separate withheld example's labels from all its augmented copies.
    group_index = [np.array([i,n+i,2*n+i]) for i in range(n)]
    los=[]
    for m in (0,1,2):
        q = ref[m][ids]
        cross = q @ Z.T
        fit = cross @ coefficient
        influence = cross @ inv
        for i,I in enumerate(group_index):
            leave_group = np.linalg.solve(inv[np.ix_(I,I)],
                                          coefficient[I])
            fit[i] -= influence[i,I] @ leave_group
        los.append(fit)
    tests=[ref[m] @ Z.T @ coefficient for m in (3,4)]
    if not all(np.isfinite(z).all() for z in los+tests):
        raise FloatingPointError("Nonfinite augmented-ridge predictions")
    return los, tests


def objective_from_scores(cache, score_list, base_clean=None):
    f = [f1_objective(q, cache) for q in score_list]
    agg = (.40,.30,.30)
    objective = (.80*sum(w*pair[0] for w,pair in zip(agg,f))
                +.20*sum(w*pair[1] for w,pair in zip(agg,f)))
    if base_clean is not None:
        objective -= .20*max(0., base_clean-f[0][0]-.01)
    return float(objective),f


def make_study(seed):
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed, multivariate=True, n_startup_trials=12,
            warn_independent_sampling=False)
    )


def train_rawaug(cache, trials, seed):
    raw_preds, raw_ridge = fixed_model(cache, "raw")
    raw_loo,_=solve(cache,{"raw":1.},raw_ridge)
    raw_clean=f1_objective(raw_loo[0],cache)[0]
    study=make_study(seed+1000)
    def func(trial):
        w1=trial.suggest_float("aug_medium_weight",0.,2.5)
        w2=trial.suggest_float("aug_strong_weight",0.,2.5)
        ridge=trial.suggest_float("ridge",.005,5.,log=True)
        try:
            loo,_=raw_augmented(cache,w1,w2,ridge)
            value,_=objective_from_scores(cache,loo,raw_clean)
            return value
        except (FloatingPointError,ValueError,np.linalg.LinAlgError):
            return -1.
    study.optimize(func,n_trials=trials,show_progress_bar=False)
    best=study.best_trial.params
    scores,pred=raw_augmented(cache,best["aug_medium_weight"],
                             best["aug_strong_weight"],best["ridge"])
    objective,_=objective_from_scores(cache,scores,raw_clean)
    return dict(params=best,train_objective=objective,loo=scores,
                predictions=pred,raw_clean_train=raw_clean,
                history=[dict(trial=t.number,value=t.value,params=t.params)
                         for t in study.trials])


def choose_safe(cache, family, params):
    if family=="canonical":
        weights={"shape":1.}
    else:
        weights={"shape":1.,
                 {"generic":"generic","mechanism":"mechanism"}[family]:
                     params["latent_weight"]}
    return solve(cache,weights,params["safe_ridge"])


def evaluate_controller(cache, raw_aug, params, family, need_pred=False):
    safe_loo,safe_test=choose_safe(cache,family,params)
    pred_loo=[
        mix(raw_aug["loo"][j], safe_loo[j], cache["p"][j],
            params["chart_alpha"],params["transition_tau"],
            params["gate_strength"]) for j in range(3)]
    value,f=objective_from_scores(cache,pred_loo,
                                  raw_aug["raw_clean_train"])
    if family=="mechanism" or family=="generic":
        value-=.0015*params["latent_weight"]
    if not need_pred:return float(value)
    test_predictions=[
        mix(raw_aug["predictions"][j],safe_test[j],cache["ptest"][j],
            params["chart_alpha"],params["transition_tau"],
            params["gate_strength"]) for j in range(2)]
    return value,f,test_predictions


def train_controller(cache, raw_aug, family, trials, seed):
    study=make_study(seed+{"canonical":2000,"generic":3000,
                           "mechanism":4000}[family])
    def func(trial):
        params={
          "safe_ridge":trial.suggest_float("safe_ridge",.005,5.,log=True),
          "chart_alpha":trial.suggest_categorical("chart_alpha",
                                                   [.05,.10,.15,.20,.30,.40]),
          "transition_tau":trial.suggest_float("transition_tau",.004,.15,
                                                log=True),
          "gate_strength":trial.suggest_float("gate_strength",0.,1.)
        }
        params["latent_weight"]= (
            trial.suggest_float("latent_weight",0.,3.)
            if family!="canonical" else 0.)
        try:
            return evaluate_controller(cache,raw_aug,params,family)
        except (FloatingPointError,ValueError,np.linalg.LinAlgError):
            return -1.
    study.optimize(func,n_trials=trials,show_progress_bar=False)
    params=dict(study.best_trial.params)
    if family=="canonical":params["latent_weight"]=0.
    obj,loo,pred=evaluate_controller(cache,raw_aug,params,
                                     family,need_pred=True)
    return dict(params=params,objective=float(obj),loo=loo,predictions=pred,
                history=[dict(trial=t.number,value=t.value,params=t.params)
                         for t in study.trials])


def test_group_leave_one_out():
    """Deterministic tiny numerical test against explicit group deletion."""
    r=np.random.default_rng(10205)
    n,d,k=6,11,3
    f0=r.normal(size=(n,d))
    fs=(f0,f0+r.normal(size=(n,d))*.31,
        f0+r.normal(size=(n,d))*.56)
    classes=np.array([0,1,2,0,1,2])
    Y=np.eye(k)[classes]
    cache={"raw_views":fs+(
        r.normal(size=(4,d)),r.normal(size=(4,d))),
        "labeled":np.arange(n),"Y":Y}
    augmented, _=raw_augmented(cache,.7,1.4,.12)
    X=np.vstack(fs)
    wt=np.r_[np.ones(n),np.sqrt(.7)*np.ones(n),
             np.sqrt(1.4)*np.ones(n)]
    Z=wt[:,None]*X
    rep=np.tile(Y,(3,1))*wt[:,None]
    for i in range(n):
        removed={i,i+n,i+2*n}
        keep=np.array([j for j in range(3*n) if j not in removed])
        beta=np.linalg.solve(Z[keep]@Z[keep].T+.12*np.eye(len(keep)),
                             rep[keep])
        for j in (0,1,2):
            exact=fs[j][i]@Z[keep].T@beta
            np.testing.assert_allclose(augmented[j][i],exact,atol=1e-9,
                                       rtol=1e-8)
    return True


def run(args):
    test_group_leave_one_out()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    records=[];details=[]
    for seed in args.seeds:
        started=time.monotonic()
        cache=build_cache(args.dataset,seed,args.epochs,args.budget)
        rows=[]
        for family,name in (
            ("raw","Raw MultiRocket"),
            ("canonical","Canonical MultiRocket"),
            ("generic","Canonical + generic SSL"),
            ("mechanism","Canonical + MESA SSL")):
            pred,ridge=fixed_model(cache,family)
            rows.append(evaluate(cache,name,pred,
                                 optim=dict(type="fixed LOOCV",
                                            ridge=ridge)))
        fixed_loo,fixed_test=raw_augmented(cache,1.,1.,.1)
        rows.append(evaluate(cache,"RawAug-fixed",fixed_test,
                             optim=dict(type="fixed group-LOO augmentation",
                                        medium=1.,strong=1.,ridge=.1)))
        a=train_rawaug(cache,args.trials,seed)
        rows.append(evaluate(cache,"RawAug-BO",a["predictions"],
                             optim=dict(type="TPE + group-LOO",
                                        params=a["params"],
                                        train_objective=a["train_objective"],
                                        trials=args.trials)))
        controllers={}
        for family,name in (
            ("canonical","BO-Aug-SPC canonical"),
            ("generic","BO-Aug-SPC generic"),
            ("mechanism","BO-Aug-SPC MESA")):
            candidate=train_controller(cache,a,family,args.trials,seed)
            controllers[family]=candidate
            rows.append(evaluate(cache,name,candidate["predictions"],
                 optim=dict(type="TPE chart+group-LOO",
                            params=candidate["params"],
                            train_objective=candidate["objective"],
                            trials=args.trials)))
        eligible=[("raw",a["train_objective"])]
        eligible.extend((f,q["objective"]) for f,q in controllers.items())
        winner=max(eligible,key=lambda x:(x[1],
                    -{"raw":0,"canonical":1,"generic":2,"mechanism":3}[x[0]]))[0]
        winning_pred=a["predictions"] if winner=="raw" else controllers[winner]["predictions"]
        rows.append(evaluate(cache,"BO-Aug TRAIN-selected",
            winning_pred,
            optim=dict(type="TRAIN-only selection",winner=winner,
                       objectives=dict(eligible),
                       rawaug=a["params"],
                       controller=(controllers[winner]["params"]
                                   if winner!="raw" else None))))
        if len(rows)!=len(METHODS):raise AssertionError("Incomplete methods")
        for row in rows:
            row["runtime_seconds_seed"]=round(time.monotonic()-started,2)
            print("RESULT_JSON="+json.dumps(row,allow_nan=False),flush=True)
        records.extend(rows)
        details.append(dict(dataset=args.dataset,seed=seed,
            rawaug=a["params"], rawaug_trials=a["history"],
            controller_trials={k:v["history"] for k,v in controllers.items()},
            selection=winner,score=dict(eligible)))
        (out/"results.json").write_text(json.dumps(records,indent=2,allow_nan=False))
        (out/"trials.json").write_text(json.dumps(details,indent=2,allow_nan=False))
    if len(records)!=len(args.seeds)*len(METHODS):raise AssertionError("Missing records")
    return records


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[901,902])
    p.add_argument("--epochs",type=int,default=15)
    p.add_argument("--budget",type=int,default=5)
    p.add_argument("--trials",type=int,default=48)
    p.add_argument("--out",default="results/mesa_group_bo")
    args=p.parse_args()
    if args.trials<13:p.error("At least 13 TPE trials required")
    run(args)
