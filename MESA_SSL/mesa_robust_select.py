#!/usr/bin/env python3
"""Mechanism-validated train-only selection of robust SSL representations.

For a specified affine nuisance family, candidate kernels are scored by
leave-one-out predictions of synthetically perturbed, labelled TRAIN samples.
Each left-out observation is also excluded from the kernel training reference.
No TEST label is used to choose models, augmentations, thresholds or weights.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
from aeon.transformations.collection.convolution_based import MultiRocket
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from MESA_SSL.mesa_multikernel import feature_scale, moment_drift_gate
from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import affine_gauge, generate_affine_shift
from mesa_mr_experiment import train_encoder,encode,indices_for_seed


def extract(train, tests, seed):
    mr=MultiRocket(n_kernels=336,max_dilations_per_kernel=16,
                   n_jobs=1,random_state=seed)
    to3d=lambda x: np.ascontiguousarray(x.transpose(0,2,1),dtype=np.float32)
    base=np.asarray(mr.fit_transform(to3d(train)))
    extras=[np.asarray(mr.transform(to3d(x))) for x in tests]
    return feature_scale(base,*extras)


def candidates(method):
    base=[{'raw':1.},{'shape':1.},
          {'raw':1.,'shape':.5},{'raw':.5,'shape':1.},
          {'raw':1.,'shape':1.},
          {'raw':1.,'shape':1.,'concept':.1},
          {'shape':1.,'concept':.15}]
    if method=='MultiRocket-augmented':return [dict(raw=1.)]
    if method=='Canonical-augmented':return [dict(shape=1.)]
    if method=='Robust kernel fusion':return base
    k={'MESA-robust-gated':'mechanism','Generic-SSL-robust-gated':'generic'}[method]
    return base + [{'raw':1.,'shape':1.,k:w} for w in (.1,.5)] + [
        {'raw':1.,'shape':1.,'concept':.15,k:.3},
        {'shape':1.,k:.3},
        {'raw':1.,k:.3}]


def matrices(views,ids,weights):
    K=np.zeros((len(ids),len(ids)),dtype=np.float64)
    aug=[np.zeros_like(K),np.zeros_like(K)]
    for key,w in weights.items():
        f=views[key][0][ids]
        K+=w*(f@f.T)
        for j in range(2):
            x=views[key][j+1][ids]
            aug[j]+=w*(x@f.T)
    return K,aug


def aug_leave_one_out(K,Kq,Y,reg):
    inverse=np.linalg.inv(K+reg*np.eye(len(K)))
    alpha=inverse@Y
    base=Kq@alpha
    cross_diag=np.diag(Kq@inverse)
    return base-(cross_diag/np.diag(inverse))[:,None]*alpha,alpha


def validate(views,indices,labels,method,criterion):
    classes=np.unique(labels[indices])
    yy=labels[indices]
    Y=(yy[:,None]==classes[None,:]).astype(np.float64)
    options=[]
    for weights in candidates(method):
        gram,aug=matrices(views,indices,weights)
        for ridge in (.01,.1,1.,10.):
            try:
                clean,coef=aug_leave_one_out(gram,gram,Y,ridge)
                scores=[clean] if criterion=='clean' else [
                    aug_leave_one_out(gram,q,Y,ridge)[0] for q in aug]
                cv=np.mean([f1_score(yy,classes[np.argmax(s,axis=1)],
                                     average='macro',zero_division=0) for s in scores])
                complexity=(len(weights)-1)
                adjusted=cv-.006*complexity
                options.append((adjusted,float(cv),complexity,ridge,weights,coef))
            except np.linalg.LinAlgError:
                continue
    if not options:raise RuntimeError("No admissible policy")
    score,cv,complexity,ridge,weights,coef=sorted(options,
                key=lambda z:(-z[0],z[2],z[3],tuple(sorted(z[4].items()))))[0]
    return dict(cv=cv,criterion=criterion,classes=classes,weights=weights,
                ridge=ridge,coef=coef,indices=indices,score=score)


def predict(views,choice,index):
    out=None
    contributions={}
    idx=choice['indices']
    for name,w in choice['weights'].items():
        a=views[name][index]
        ref=views[name][0][idx]
        component=w*(a@ref.T@choice['coef'])
        contributions[name]=component
        out=component if out is None else out+component
    if not np.allclose(out,sum(contributions.values()),atol=1e-9):
        raise AssertionError("Non-additive explanation")
    return out


def run(args):
    train,test,y,targets,_=load_data(args.dataset)
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in args.seeds:
        ts=time.monotonic()
        train_aug=[generate_affine_shift(train,seed+3000+i, v)
                   for i,v in enumerate((.75,1.5))]
        test_aug=generate_affine_shift(test,seed,1.5)
        train_shape,c_train=affine_gauge(train)
        test_shape,c_test=affine_gauge(test)
        shift_shape,c_shift=affine_gauge(test_aug)
        aug_shape=[affine_gauge(x)[0] for x in train_aug]
        drift_err=float(np.max(np.abs(test_shape-shift_shape)))
        if drift_err>2e-4: raise AssertionError(f"Gauge residual error: {drift_err}")
        # Multirocket sees every sample only through unlabeled TRAIN fitting.
        views={}
        raw=extract(train,train_aug+[test,test_aug],seed)
        shape=extract(train_shape,aug_shape+[test_shape,shift_shape],seed)
        moment=feature_scale(c_train,*[affine_gauge(x)[1] for x in train_aug],
                             c_test,c_shift)
        views['raw']=raw
        views['shape']=shape
        views['concept']=moment
        for mechanism,key in ((False,'generic'),(True,'mechanism')):
            net,_=train_encoder(train_shape,seed,args.epochs,mechanism=mechanism)
            encoded=[encode(net,x) for x in [train_shape]+aug_shape+[test_shape,shift_shape]]
            views[key]=feature_scale(*encoded)
        label_ids=indices_for_seed(y,seed,budget=args.budget)
        drift=moment_drift_gate(c_train,c_test)
        drift_shift=moment_drift_gate(c_train,c_shift)
        for method in ('MultiRocket-augmented','Canonical-augmented',
                       'Robust kernel fusion','Generic-SSL-robust-gated','MESA-robust-gated'):
            clean=validate(views,label_ids,y,method,'clean')
            robust=validate(views,label_ids,y,method,'robust')
            score_clean=predict(views,clean,3)
            score_shift=predict(views,clean,4)
            score_robust=predict(views,robust,3)
            score_robust_shift=predict(views,robust,4)
            blended=(1-drift[:,None])*score_clean+drift[:,None]*score_robust
            blended_shift=(1-drift_shift[:,None])*score_shift+drift_shift[:,None]*score_robust_shift
            pred=clean['classes'][blended.argmax(1)]
            pred_shift=clean['classes'][blended_shift.argmax(1)]
            rec=dict(dataset=args.dataset,seed=seed,method=method,status='EXECUTED',
                     macro_f1=float(f1_score(targets,pred,average='macro')),
                     shifted_macro_f1=float(f1_score(targets,pred_shift,average='macro')),
                     accuracy=float(accuracy_score(targets,pred)),n_labeled=len(label_ids),
                     train_count=len(train),test_count=len(test),encoder_epochs=args.epochs,
                     clean_weights=clean['weights'],robust_weights=robust['weights'],
                     clean_cv=clean['cv'],robust_cv=robust['cv'],
                     clean_gate_mean=float(np.mean(drift)),shift_gate_mean=float(np.mean(drift_shift)),
                     gauge_error=drift_err,runtime_seconds=round(time.monotonic()-ts,2),
                     shift_assumption="Declared affine nuisance treated as label preserving")
            rows.append(rec)
            print("RESULT_JSON="+json.dumps(rec,allow_nan=False),flush=True)
        (folder/"results.json").write_text(json.dumps(rows,indent=2,allow_nan=False))
    assert len(rows)==len(args.seeds)*5 and all(r['status']=='EXECUTED' for r in rows)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[304,305,306])
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--budget',type=int,default=5)
    p.add_argument('--out',default='results/mesa_robust')
    a=p.parse_args()
    run(a)
