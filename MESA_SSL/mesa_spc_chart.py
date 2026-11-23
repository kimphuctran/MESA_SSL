#!/usr/bin/env python3
"""MESA-SPC: train-calibrated Hotelling-type chart controlling SSL fallback.

The Phase-I chart fits on one *unlabelled TRAIN* partition and calibrates
on a disjoint TRAIN partition. Phase-II observations receive split-conformal
p-values of shrinkage Mahalanobis/Hotelling-type scores. The chart monitors
per-series mean and linear-slope concepts, NOT TEST labels. These individual
time-series examples are not assumed to arrive as a chronological stream.

No hyperparameter, ridge model, chart limit, mixture or expert is selected
with TEST labels. This is an exploratory *classification* trial, not an ARL
study; nuisance transforms are synthetically assumed label-preserving.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import f1_score, accuracy_score

from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import affine_gauge, generate_affine_shift
from MESA_SSL.mesa_robust_select import extract
from MESA_SSL.mesa_multikernel import feature_scale
from mesa_mr_experiment import train_encoder, encode, indices_for_seed


def chart_train(concepts, seed, alpha=0.10):
    """Train/calibration split + robust covariance shrinkage + conformal limit.

    With exchangeable in-control calibration and future examples, the
    miscoverage probability is at most alpha. No iid/ARL guarantee is made for
    UCR sample order or unknown deployment distributions.
    """
    c = np.asarray(concepts, dtype=np.float64)
    k = c.shape[1] // 3
    x = c[:, : 2 * k]
    ids = np.random.default_rng(seed + 917711).permutation(len(x))
    n_fit = max(8, int(round(0.60 * len(x))))
    n_fit = min(n_fit, len(x) - 4)
    if n_fit <= 0:
        raise ValueError("Insufficient reference observations")
    fit, cal = x[ids[:n_fit]], x[ids[n_fit:]]
    model = LedoitWolf().fit(fit)
    center = model.location_
    precision = model.precision_
    def t2(values):
        delta = np.asarray(values, dtype=np.float64) - center
        return np.einsum('np,pq,nq->n', delta, precision, delta)
    calibration = np.sort(t2(cal))
    return dict(center=center, precision=precision, calibration=calibration,
                n_fit=len(fit), n_cal=len(cal), alpha=float(alpha),
                upper_control_limit=float(calibration[min(len(calibration)-1,
                                   max(0, math.ceil((len(calibration)+1)*(1-alpha))-1))])
                                   if (len(calibration)+1)*alpha >= 1 else math.inf)


def chart_predict(chart, concepts):
    c = np.asarray(concepts, dtype=np.float64)
    channels = c.shape[1] // 3
    delta = c[:, : 2 * channels] - chart['center']
    t2 = np.einsum('np,pq,nq->n', delta, chart['precision'], delta)
    reference = chart['calibration']
    # Split-conformal upper-tail p-value, conservative for ties.
    p = (1 + np.sum(reference[None, :] >= t2[:, None], axis=1)) / (len(reference) + 1)
    flags = p <= chart['alpha']
    # Mean and slope channel contributions add to T2, though individual
    # covariance-adjusted components need not be nonnegative.
    weighted = delta @ chart['precision']
    contribution = delta * weighted
    if not np.allclose(contribution.sum(axis=1), t2, atol=1e-7, rtol=1e-7):
        raise AssertionError("Chart score decomposition failed")
    return flags, t2, p, contribution


def fit_kernel(views, ids, y, weights_list):
    labels = np.asarray(y)
    classes = np.unique(labels)
    Y = (labels[ids, None] == classes[None, :]).astype(float)
    candidates = []
    for weights in weights_list:
        A = [np.sqrt(w) * views[k][0][ids] for k,w in weights.items()]
        Z = np.column_stack(A)
        gram = Z @ Z.T
        for regularizer in (.01, .10, 1.0):
            inv = np.linalg.inv(gram + regularizer*np.eye(len(ids)))
            coef = inv @ Y
            diag = np.maximum(np.diag(inv), 1e-10)
            cv = Y - coef/diag[:, None]
            f1 = float(f1_score(labels[ids], classes[np.argmax(cv,axis=1)],
                                average='macro', zero_division=0))
            objective = f1 - 0.006*(len(weights)-1)
            candidates.append((objective, f1, weights, regularizer, coef))
    # Never select a method with a TEST observation or TEST score.
    objective, cv, weights, regularizer, coef = sorted(
        candidates, key=lambda a: (-a[0],len(a[2]),a[3],
                                   tuple(sorted(a[2].items()))))[0]
    return dict(weights=weights, ridge=regularizer, coef=coef,
                classes=classes, ids=ids, cv=cv, train_objective=objective)


def kernel_score(views, model, mode):
    out = None
    for key, weight in model['weights'].items():
        val = weight * (views[key][mode] @
                        views[key][0][model['ids']].T @ model['coef'])
        out = val if out is None else out + val
    return out


def run(args):
    train, test, labels, targets, _ = load_data(args.dataset)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    rows=[]
    # TEST is never treated as Phase-I chart reference.
    train_shape, c_train = affine_gauge(train)
    test_shape, c_test = affine_gauge(test)
    for seed in args.seeds:
        start = time.monotonic()
        shifted = generate_affine_shift(test, seed, 1.5)
        shifted_shape, c_shift = affine_gauge(shifted)
        gauge_error = float(np.max(np.abs(test_shape-shifted_shape)))
        if gauge_error > 2e-4:
            raise AssertionError(f"Affine invariance failed: {gauge_error}")

        chart = chart_train(c_train, seed, alpha=args.alpha)
        clean_alarm, clean_t2, clean_p, _ = chart_predict(chart, c_test)
        shift_alarm, shift_t2, shift_p, _ = chart_predict(chart, c_shift)
        ids = indices_for_seed(labels, seed, budget=args.budget)
        views = {}
        views['raw'] = extract(train, [test, shifted], seed)
        views['shape'] = extract(train_shape,[test_shape,shifted_shape],seed)
        for mechanism, key in ((False, 'generic'),(True, 'mechanism')):
            encoder, _ = train_encoder(train_shape, seed, args.epochs,
                                        mechanism=mechanism)
            feats = [encode(encoder, x) for x in (train_shape, test_shape,
                                                 shifted_shape)]
            views[key] = feature_scale(*feats)

        families = {
            'Raw MultiRocket': [{'raw':1.}],
            'Canonical MultiRocket': [{'shape':1.}],
            'Canonical + generic SSL':
                [{'shape':1.,'generic':w} for w in (.1,.3,1.)],
            'Canonical + MESA SSL':
                [{'shape':1.,'mechanism':w} for w in (.1,.3,1.)],
        }
        models = {name:fit_kernel(views,ids,labels,weights)
                  for name,weights in families.items()}
        clean_scores = {name:kernel_score(views,model,1)
                        for name,model in models.items()}
        shift_scores = {name:kernel_score(views,model,2)
                        for name,model in models.items()}
        outputs = dict()
        for name in families:
            outputs[name] = (clean_scores[name],shift_scores[name],
                             models[name]['cv'])
        for kind, target in (
                ('SPC raw-to-canonical','Canonical MultiRocket'),
                ('SPC raw-to-generic','Canonical + generic SSL'),
                ('SPC raw-to-MESA','Canonical + MESA SSL')):
            outputs[kind] = (
                np.where(clean_alarm[:,None],clean_scores[target],
                         clean_scores['Raw MultiRocket']),
                np.where(shift_alarm[:,None],shift_scores[target],
                         shift_scores['Raw MultiRocket']),
                models[target]['cv'])
        best_static = sorted(
            models, key=lambda k:(-models[k]['train_objective'],k))[0]
        outputs['TRAIN-selected static'] = (
            clean_scores[best_static],shift_scores[best_static],
            models[best_static]['cv'])

        for name,(s0,s1,loo_f1) in outputs.items():
            yc = models['Raw MultiRocket']['classes'][np.argmax(s0,axis=1)]
            ys = models['Raw MultiRocket']['classes'][np.argmax(s1,axis=1)]
            row = dict(status='EXECUTED',dataset=args.dataset,seed=seed,
                       method=name,macro_f1=float(f1_score(targets,yc,average='macro')),
                       shifted_macro_f1=float(f1_score(targets,ys,average='macro')),
                       accuracy=float(accuracy_score(targets,yc)),
                       chart='split-calibrated shrinkage Hotelling-type T2',
                       chart_n_fit=int(chart['n_fit']),
                       chart_n_cal=int(chart['n_cal']),
                       chart_alpha=float(args.alpha),
                       chart_ucl=float(chart['upper_control_limit']),
                       clean_flag_rate=float(np.mean(clean_alarm)),
                       shifted_flag_rate=float(np.mean(shift_alarm)),
                       clean_mean_t2=float(np.mean(clean_t2)),
                       shifted_mean_t2=float(np.mean(shift_t2)),
                       clean_mean_p=float(np.mean(clean_p)),
                       shifted_mean_p=float(np.mean(shift_p)),
                       train_loo_f1=float(loo_f1),
                       n_labeled=int(len(ids)),n_train=int(len(train)),
                       n_test=int(len(test)),pretrain_epochs=int(args.epochs),
                       canonical_gauge_error=gauge_error,
                       nuisance_assumption='synthetic affine label-preserving',
                       best_static_train_choice=best_static,
                       runtime_seconds=round(time.monotonic()-start,2))
            rows.append(row)
            print('RESULT_JSON='+json.dumps(row,allow_nan=False),flush=True)
        (output/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    expected = len(args.seeds)*len(outputs)
    if len(rows)!=expected:
        raise AssertionError(f'Incomplete evaluation: {len(rows)} vs {expected}')
    print('RESULT_COUNT='+str(len(rows)),flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[801,802])
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--budget',type=int,default=5)
    p.add_argument('--alpha',type=float,default=.10)
    p.add_argument('--out',default='results/mesa_spc')
    a=p.parse_args()
    if not (0<a.alpha<1):p.error('alpha must be between 0 and 1')
    run(a)
