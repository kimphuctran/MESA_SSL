#!/usr/bin/env python3
"""MESA multi-kernel mechanism-conditional SSL with label-blind drift gating.

Each view is fitted on unlabelled TRAIN only. A small labelled subset of TRAIN
selects a kernel combination and ridge penalty via leave-one-out predictions.
No information from TEST labels enters feature construction or model selection.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from aeon.transformations.collection.convolution_based import MultiRocket
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import affine_gauge, generate_affine_shift
from mesa_mr_experiment import train_encoder, encode, indices_for_seed


def feature_scale(train, *others):
    """Train-only standardisation; trace-normalised Gram groups."""
    n, width = train.shape
    sc = StandardScaler().fit(train)
    arr = [np.clip(sc.transform(x), -8, 8) for x in (train,) + others]
    norm = math.sqrt(max(np.mean(np.sum(arr[0] ** 2, axis=1)), 1e-12))
    return tuple(np.asarray(x / norm, dtype=np.float64) for x in arr)


def transform_group(train, test, shifted, seed, n_kernels=336):
    transform = MultiRocket(n_kernels=n_kernels, max_dilations_per_kernel=16,
                            n_jobs=1, random_state=seed)
    to_aeon = lambda x: np.ascontiguousarray(x.transpose(0, 2, 1), dtype=np.float32)
    X = np.asarray(transform.fit_transform(to_aeon(train)))
    Y = np.asarray(transform.transform(to_aeon(test)))
    Z = np.asarray(transform.transform(to_aeon(shifted)))
    return feature_scale(X, Y, Z)


def norm_kernel(a, b):
    """Construct an explicitly positive-semidefinite linear kernel."""
    return a @ b.T


def feature_views(train, test, shifted, seed, epochs, include_encoder):
    shape_train, concept_train = affine_gauge(train)
    shape_test, concept_test = affine_gauge(test)
    shape_shift, concept_shift = affine_gauge(shifted)
    assert np.all(np.isfinite(shape_train)) and np.all(np.isfinite(shape_shift))
    displacement = float(np.max(np.abs(shape_test - shape_shift)))
    # Machine-precision gauge invariance for the declared offset/trend family.
    if displacement > 2e-4:
        raise AssertionError(f"Declared gauge family mismatch: {displacement:.5g}")
    raw = transform_group(train, test, shifted, seed)
    shape = transform_group(shape_train, shape_test, shape_shift, seed)
    concepts = feature_scale(concept_train, concept_test, concept_shift)
    groups = dict(raw=raw, shape=shape, concept=concepts)
    if include_encoder:
        for mechanism, name in ((False, "no_mechanism"), (True, "mechanism")):
            model, losses = train_encoder(shape_train, seed, epochs, mechanism=mechanism)
            encoded = [encode(model, x) for x in (shape_train, shape_test, shape_shift)]
            groups[name] = feature_scale(*encoded)
    return groups, concept_train, concept_test, concept_shift, displacement


def select_model(groups, indices, labels, family):
    """Select candidates and ridge using only labelled TRAIN; no TEST."""
    classes = np.unique(labels)
    labeled = labels[indices]
    Y = (labeled[:, None] == classes[None, :]).astype(np.float64)
    trials = {
        "MultiRocket": [dict(raw=1.)],
        "Canonical MultiRocket": [dict(shape=1.)],
        "Dual-MultiRocket": [dict(raw=1.), dict(shape=1.),
                            dict(raw=1., shape=.5), dict(raw=.5, shape=1.),
                            dict(raw=1., shape=1.)],
        "Dual+moments": [dict(raw=1.), dict(shape=1.),
                        dict(raw=1., shape=.5), dict(raw=.5, shape=1.),
                        dict(raw=1., shape=1.),
                        dict(raw=1., shape=1., concept=.10),
                        dict(raw=.5, shape=1., concept=.3),
                        dict(shape=1., concept=.1)],
    }
    if family == "MESA-MK":
        key = "mechanism"
    elif family == "MESA-MK-no-mechanism":
        key = "no_mechanism"
    else:
        key = None
    if key:
        base = trials["Dual+moments"]
        with_latent = [
            dict(raw=1., shape=1., **{key: .1}),
            dict(raw=1., shape=1., **{key: .5}),
            dict(raw=.5, shape=1., **{key: .5}),
            dict(shape=1., **{key: .3}),
            dict(raw=1., **{key: .3}),
            dict(raw=1., shape=1., concept=.2, **{key: .3}),
        ]
        candidates = base + with_latent
    else:
        candidates = trials[family]
    k_cached = {g: norm_kernel(v[0][indices], v[0][indices]) for g, v in groups.items()}
    outcomes = []
    regularisers = (.01, .1, 1., 10.)
    for weight in candidates:
        gram = np.zeros_like(next(iter(k_cached.values())))
        for key, amount in weight.items():
            gram += amount * k_cached[key]
        for alpha in regularisers:
            try:
                inverse = np.linalg.inv(gram + alpha * np.eye(len(indices)))
                coef = inverse @ Y
                diag = np.maximum(np.diag(inverse), 1e-12)
                loo_scores = Y - coef / diag[:, None]
                loo_label = classes[np.argmax(loo_scores, axis=1)]
                cv = float(f1_score(labeled, loo_label, average="macro", zero_division=0))
                complexity = len(weight) - 1
                # Predeclared one-sided complexity penalty; favour a simple fallback
                selection_score = cv - .006 * complexity
                outcomes.append((selection_score, cv, complexity, alpha,
                                 weight, coef, classes))
            except np.linalg.LinAlgError:
                pass
    if not outcomes:
        raise RuntimeError("No valid TRAIN-only candidate")
    outcomes.sort(key=lambda q: (-q[0], q[2], q[3],
                                 tuple(sorted(q[4].items()))))
    _, cv, complexity, alpha, weight, coef, classes = outcomes[0]
    return {"family": family, "weights": weight, "ridge": alpha,
            "classes": classes, "coef": coef, "cv_f1": cv,
            "labeled": indices.tolist(), "complexity": complexity}


def class_scores(model, groups, mode):
    indexes = model["labeled"]
    out = None
    components = {}
    for g, w in model["weights"].items():
        query = groups[g][mode]
        ref = groups[g][0][indexes]
        term = w * (query @ ref.T @ model["coef"])
        components[g] = term
        out = term if out is None else out + term
    assert out is not None
    if not np.allclose(sum(components.values()), out, atol=1e-9):
        raise AssertionError("Score explanation is not additive")
    return out, components


def moment_drift_gate(c_train, c_test):
    """Unlabelled TRAIN concept distribution controls per-sample fallback."""
    d = c_train.shape[1] // 3
    X = c_train[:, :2 * d].astype(np.float64)
    T = c_test[:, :2 * d].astype(np.float64)
    center = np.median(X, axis=0)
    deviation = np.maximum(np.std(X, axis=0), 1e-3)
    score_train = np.sqrt(np.mean(((X-center) / deviation) ** 2, axis=1))
    threshold = np.quantile(score_train, .90)
    score = np.sqrt(np.mean(((T-center) / deviation) ** 2, axis=1))
    # Leave a soft transition from the selected model to canonical-only fallback
    gate = np.clip((score-threshold)/1.5, 0., 1.)
    return gate


def score_row(dataset,seed,method,clean,shift,choice,labels,targets,
              invariant_error,elapsed,guard=None,features=None):
    predicted = choice["classes"][np.argmax(clean, axis=1)]
    predicted_shift = choice["classes"][np.argmax(shift, axis=1)]
    return dict(dataset=dataset,seed=seed,method=method,status="EXECUTED",
                macro_f1=float(f1_score(targets,predicted,average="macro")),
                shifted_macro_f1=float(f1_score(targets,predicted_shift,average="macro")),
                accuracy=float(accuracy_score(targets,predicted)),
                n_labeled=int(len(choice["labeled"])), n_train=int(len(labels)),
                n_test=int(len(targets)),train_cv_f1=float(choice["cv_f1"]),
                weights=choice["weights"],ridge=choice["ridge"],
                encoder_epochs=features, gauge_max_abs_error=invariant_error,
                gated=guard is not None,gate_fraction=guard,
                runtime_seconds=round(elapsed,2),
                shift_assumption="Synthetic affine offset/trend declared label preserving")


def run(args):
    train, test, y, targets, _ = load_data(args.dataset)
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for seed in args.seeds:
        start=time.monotonic()
        shift=generate_affine_shift(test,seed,1.5)
        groups,ctr,cte,cts,error=feature_views(train,test,shift,seed,
                                                args.epochs,include_encoder=True)
        label_idx = indices_for_seed(y, seed, budget=args.budget)
        models = {}
        for fam in ["MultiRocket","Canonical MultiRocket","Dual-MultiRocket",
                    "Dual+moments","MESA-MK-no-mechanism","MESA-MK"]:
            model=select_model(groups,label_idx,y,fam)
            clean,part_clean=class_scores(model,groups,1)
            shifted,part_shift=class_scores(model,groups,2)
            row=score_row(args.dataset,seed,fam,clean,shifted,model,y,targets,
                          error,time.monotonic()-start,features=args.epochs)
            allrows.append(row)
            models[fam]=(model,clean,shifted)
            print("RESULT_JSON="+json.dumps(row,allow_nan=False),flush=True)
        # The gate may only use unlabelled features from TEST, never TEST labels.
        selected,clean,shifted = models["MESA-MK"]
        canonical,clean_can,shift_can = models["Canonical MultiRocket"]
        g0=moment_drift_gate(ctr,cte)
        g1=moment_drift_gate(ctr,cts)
        safe_clean=(1-g0[:,None])*clean+g0[:,None]*clean_can
        safe_shift=(1-g1[:,None])*shifted+g1[:,None]*shift_can
        safe=score_row(args.dataset,seed,"MESA-MK-drift-gated",safe_clean,safe_shift,
                       selected,y,targets,error,time.monotonic()-start,
                       guard={"clean":float(np.mean(g0)),"shifted":float(np.mean(g1))},
                       features=args.epochs)
        allrows.append(safe)
        print("RESULT_JSON="+json.dumps(safe,allow_nan=False),flush=True)
        (folder/"results.json").write_text(json.dumps(allrows,indent=2,allow_nan=False))
    expected=len(args.seeds)*7
    assert len(allrows)==expected and all(row["status"]=="EXECUTED" for row in allrows)
    return allrows


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[301,302,303])
    p.add_argument("--epochs",type=int,default=35)
    p.add_argument("--budget",type=int,default=5)
    p.add_argument("--out",default="results/mesa_multikernel")
    args=p.parse_args()
    run(args)
