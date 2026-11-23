#!/usr/bin/env python3
"""Train-only Bayesian optimization for MESA-SPC classification.

Optuna TPE tunes: raw and canonical/MESA ridge regularization, mechanism
view strength, chart alpha, gate softness and gate strength. The objective
uses the *same labeled TRAIN examples* in leave-one-label-out form, both
unperturbed and with two predeclared synthetic affine transformations.
Not a single TEST target is used in fitting, tuning or controller selection.

Official TEST results are reported only after each trial-independent
selection. Dataset TEST observations are not chronological.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import accuracy_score, f1_score

from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import affine_gauge, generate_affine_shift
from MESA_SSL.mesa_multikernel import feature_scale
from MESA_SSL.mesa_robust_select import extract
from MESA_SSL.mesa_spc_chart import chart_train, chart_predict
from mesa_mr_experiment import train_encoder, encode, indices_for_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)
FAMILIES = ("canonical", "generic", "mechanism")
METHODS = ("Raw MultiRocket", "Canonical MultiRocket",
           "Canonical + generic SSL", "Canonical + MESA SSL",
           "BO-SPC canonical", "BO-SPC generic", "BO-SPC MESA",
           "BO-SPC TRAIN-selected")


def build_cache(dataset, seed, epochs, budget):
    train, test, y, yt, _ = load_data(dataset)
    train_shape, c_train = affine_gauge(train)
    test_shape, c_test = affine_gauge(test)
    shifted = generate_affine_shift(test, seed, 1.5)
    shifted_shape, c_shift = affine_gauge(shifted)
    if np.max(np.abs(shifted_shape-test_shape)) > 2e-4:
        raise AssertionError("Unsupported gauge perturbation")

    # Augmentation seeds and strengths are fixed *before* optimization.
    augmentations = [
        generate_affine_shift(train, seed + 11700 + j, strength)
        for j, strength in enumerate((.65, 1.50))
    ]
    a_shape = [affine_gauge(q)[0] for q in augmentations]
    c_aug = [affine_gauge(q)[1] for q in augmentations]
    inp_raw = augmentations + [test, shifted]
    inp_shape = a_shape + [test_shape, shifted_shape]
    views = {
        "raw": extract(train, inp_raw, seed),
        "shape": extract(train_shape, inp_shape, seed)
    }
    for mechanism, name in ((False, "generic"), (True, "mechanism")):
        net, _ = train_encoder(train_shape, seed, epochs, mechanism=mechanism)
        vals = [encode(net, t) for t in [train_shape] + inp_shape]
        views[name] = feature_scale(*vals)

    labeled = indices_for_seed(y, seed, budget=budget)
    classes = np.unique(y)
    Y = (y[labeled, None] == classes[None, :]).astype(float)
    target_idx = np.searchsorted(classes, y[labeled])
    chart = chart_train(c_train, seed, alpha=.10)
    # The chart is label-free. Its future-sample calibration statement
    # requires exchangeability; no sequential ARL claim is made.
    p_train = chart_predict(chart, c_train)[2][labeled]
    p_aug = [chart_predict(chart, q)[2][labeled] for q in c_aug]
    p_test = [chart_predict(chart, q)[2] for q in (c_test, c_shift)]
    chart_rates = [
        float(np.mean(chart_predict(chart, x)[0]))
        for x in (c_test, c_shift)
    ]
    kernels = {}
    for key, xs in views.items():
        ref = xs[0][labeled]
        kernels[key] = {
            "train": ref @ ref.T,
            "aug": [xs[j][labeled] @ ref.T for j in (1, 2)],
            "test": [xs[j] @ ref.T for j in (3, 4)],
        }
    return dict(dataset=dataset, seed=seed, y=y, yt=yt, classes=classes,
                labeled=labeled, target_idx=target_idx, Y=Y,
                k=kernels, raw_views=views["raw"], p=[p_train] + p_aug, ptest=p_test,
                chart_rates=chart_rates, chart_n_cal=len(chart["calibration"]),
                chart_n_fit=chart["n_fit"], n_train=len(train), n_test=len(test))


def solve(cache, weights, ridge):
    blocks = cache["k"]
    n = len(cache["labeled"])
    gram = sum(w * blocks[key]["train"] for key, w in weights.items())
    inv = np.linalg.solve(gram+ridge*np.eye(n), np.eye(n))
    alpha = inv @ cache["Y"]
    diagonal = np.maximum(np.diag(inv), 1e-10)
    # Exact ridge leave-one-label-out for the original TRAIN observations.
    clean_loo = cache["Y"] - alpha / diagonal[:, None]
    shifted_loo = []
    for j in range(2):
        cross = sum(w * blocks[key]["aug"][j] for key, w in weights.items())
        score = cross @ alpha
        # Remove the same TRAIN label from its own augmented prediction.
        adjustment = (np.diag(cross@inv)/diagonal)[:, None]*alpha
        shifted_loo.append(score-adjustment)
    predictions = [
        sum(w * blocks[key]["test"][j] for key, w in weights.items())@alpha
        for j in range(2)
    ]
    return [clean_loo] + shifted_loo, predictions


def soft_gate(p, alpha, tau, intensity):
    z = np.clip((alpha-np.asarray(p))/max(tau, 1e-6), -40., 40.)
    return float(intensity)/(1.0+np.exp(-z))


def mix(raw, safe, p, alpha, tau, intensity):
    g = soft_gate(p, alpha, tau, intensity)
    return raw*(1-g[:, None]) + safe*g[:, None]


def f1_objective(scores, cache):
    t = cache["target_idx"]
    truth = cache["y"][cache["labeled"]]
    labels = cache["classes"]
    macro = float(f1_score(truth, labels[np.argmax(scores, axis=1)],
                           average="macro", zero_division=0))
    # Proper probability calibration is *not* claimed: this smooth
    # margin proxy only resolves ties in discrete small-sample F1.
    x = scores - np.max(scores, axis=1, keepdims=True)
    x = np.clip(4*x, -30., 0.)
    prob = np.exp(x); prob /= prob.sum(axis=1, keepdims=True)
    brier = float(np.mean(np.sum(
        (prob - np.eye(len(labels))[t])**2, axis=1)))
    return macro, 1.-.5*brier


def train_objective(cache, params, family, return_scores=False):
    raw_l, raw_test = solve(cache, {"raw":1.}, params["raw_ridge"])
    w = params["latent_weight"]
    if family == "canonical":
        weights = {"shape":1.}
    elif family == "generic":
        weights = {"shape":1., "generic":w}
    else:
        weights = {"shape":1., "mechanism":w}
    safe_l, safe_test = solve(cache, weights, params["safe_ridge"])
    mixtures = [mix(raw_l[j], safe_l[j], cache["p"][j],
                    params["chart_alpha"], params["transition_tau"],
                    params["gate_strength"]) for j in range(3)]
    f = [f1_objective(q,cache) for q in mixtures]
    raw_clean = f1_objective(raw_l[0],cache)[0]
    weights_obj = (.40,.30,.30)
    score = (
        .80*sum(ww * a for ww,(a,b) in zip(weights_obj,f))
        +.20*sum(ww * b for ww,(a,b) in zip(weights_obj,f))
        -.25*max(0.,raw_clean-f[0][0]-.01)
    )
    if family != "canonical":
        score -= .002*min(w, 2.)
    if return_scores:
        return score, f, [
            mix(raw_test[j],safe_test[j],cache["ptest"][j],
                params["chart_alpha"],params["transition_tau"],
                params["gate_strength"]) for j in range(2)
        ]
    return score


def suggest(trial, family):
    params = {
        "raw_ridge": trial.suggest_float("raw_ridge", .003, 5., log=True),
        "safe_ridge": trial.suggest_float("safe_ridge", .003, 5., log=True),
        "chart_alpha": trial.suggest_categorical("chart_alpha",
                                                  [.05,.10,.15,.20,.30,.40]),
        "transition_tau": trial.suggest_float("transition_tau", .004, .15,
                                               log=True),
        "gate_strength": trial.suggest_float("gate_strength", 0., 1.)
    }
    params["latent_weight"] = (
        trial.suggest_float("latent_weight", 0., 3.) if family!="canonical" else 0.
    )
    return params


def tune(cache, family, trials, seed):
    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=12,
                                          multivariate=True,
                                          warn_independent_sampling=False))
    def function(trial):
        try:
            params = suggest(trial,family)
            return float(train_objective(cache, params, family))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return -1.
    study.optimize(function, n_trials=trials, show_progress_bar=False,
                   n_jobs=1, gc_after_trial=False)
    best = dict(study.best_trial.params)
    if family=="canonical":
        best["latent_weight"] = 0.
    score,loo,tests = train_objective(cache,best,family,return_scores=True)
    return dict(params=best, score=float(score), loo=loo, test=tests,
                n_trials=len(study.trials),
                best_trial=study.best_trial.number,
                best_objective=float(study.best_value),
                history=[dict(number=t.number,value=t.value,
                              params=t.params) for t in study.trials])


def fixed_model(cache, family):
    # Fair strong non-BO comparators, with ridge from TRAIN leave-one-label-out.
    target_key = {"raw":"raw", "canonical":"shape", "generic":"generic",
                  "mechanism":"mechanism"}[family]
    weights = {"raw":1.} if family=="raw" else {"shape":1.}
    if family in ("generic","mechanism"):
        weights[target_key] = .3
    choices=[]
    for reg in (.01,.1,1.,10.):
        lo,pred = solve(cache,weights,reg)
        f1, smooth=f1_objective(lo[0],cache)
        choices.append((f1, smooth, -reg, pred,reg))
    winner = sorted(choices,key=lambda q:(-q[0],-q[1],-q[2]))[0]
    return winner[3],float(winner[4])


def evaluate(cache, name, predictions, optim=None):
    yt = cache["yt"]
    cls = cache["classes"]
    c = cls[np.argmax(predictions[0],axis=1)]
    s = cls[np.argmax(predictions[1],axis=1)]
    return dict(dataset=cache["dataset"],seed=cache["seed"],
                method=name, status="EXECUTED",
                macro_f1=float(f1_score(yt,c,average="macro")),
                shifted_macro_f1=float(f1_score(yt,s,average="macro")),
                accuracy=float(accuracy_score(yt,c)),
                clean_chart_flag_rate=cache["chart_rates"][0],
                shifted_chart_flag_rate=cache["chart_rates"][1],
                chart_reference_fit=cache["chart_n_fit"],
                chart_calibration=cache["chart_n_cal"],
                labeled_train=len(cache["labeled"]),
                n_train=cache["n_train"],n_test=cache["n_test"],
                tuning_provenance="TRAIN-only leave-one-label-out with two TRAIN affine augmentations",
                optimization=optim)


def run(args):
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    optimization=[]
    for seed in args.seeds:
        started=time.monotonic()
        cache=build_cache(args.dataset, seed, args.epochs, args.budget)
        row_for_seed=[]
        fstats={}
        for family,name in (("raw","Raw MultiRocket"),
                            ("canonical","Canonical MultiRocket"),
                            ("generic","Canonical + generic SSL"),
                            ("mechanism","Canonical + MESA SSL")):
            pred, reg=fixed_model(cache,family)
            row_for_seed.append(evaluate(cache,name,pred,
                                         optim=dict(type="TRAIN-LOO ridge",
                                                    ridge=reg)))
        for family,name in (("canonical","BO-SPC canonical"),
                            ("generic","BO-SPC generic"),
                            ("mechanism","BO-SPC MESA")):
            optim = tune(cache,family,args.trials,
                         seed+{ "canonical":2200,"generic":3300,"mechanism":4400}[family])
            fstats[family]=optim
            row_for_seed.append(evaluate(cache,name,optim["test"],
                optim=dict(type="Optuna multivariate TPE",
                            params=optim["params"],train_objective=optim["score"],
                            trials=optim["n_trials"],
                            best_trial=optim["best_trial"])))
        # Select one controller using only its TRAIN-only objective.
        champion = sorted(FAMILIES,key=lambda f:(-fstats[f]["score"],
                         {"canonical":0,"generic":1,"mechanism":2}[f]))[0]
        row_for_seed.append(evaluate(cache,"BO-SPC TRAIN-selected",
            fstats[champion]["test"],
            optim=dict(type="TRAIN-only candidate selection",
                       family=champion,train_objective=fstats[champion]["score"],
                       candidates={f:fs["score"] for f,fs in fstats.items()})))
        for row in row_for_seed:
            row["runtime_seconds_seed"] = round(time.monotonic()-started,2)
            print("RESULT_JSON="+json.dumps(row, allow_nan=False),flush=True)
        records+=row_for_seed
        optimization.append(dict(dataset=args.dataset,seed=seed,
           n_trials_per_family=args.trials,
           details={f:dict(params=info["params"],
                           train_objective=info["score"],
                           best_trial=info["best_trial"],
                           history=info["history"]) for f,info in fstats.items()}))
        (out/"results.json").write_text(json.dumps(records,indent=2,allow_nan=False))
        (out/"trials.json").write_text(json.dumps(optimization,indent=2,
                                                 allow_nan=False))
    expected = len(args.seeds)*len(METHODS)
    if len(records)!=expected:
        raise RuntimeError(f"Missing evaluations: {len(records)} != {expected}")
    return records


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--seeds",nargs="+",type=int,default=[901,902])
    parser.add_argument("--epochs",type=int,default=15)
    parser.add_argument("--budget",type=int,default=5)
    parser.add_argument("--trials",type=int,default=64)
    parser.add_argument("--out",default="results/mesa_bayes")
    a=parser.parse_args()
    if a.trials<13:parser.error("at least 13 trials needed")
    run(a)
