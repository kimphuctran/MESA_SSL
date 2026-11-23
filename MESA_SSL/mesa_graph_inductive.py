#!/usr/bin/env python3
"""Train-only graph propagation with mechanism-conditioned SSL representations.

The graph uses only unlabelled TRAIN features. Labels are limited to a fixed
five-per-class subset of TRAIN. Hyperparameters and mixture weights are
selected by leave-one-labelled-out predictions. TEST labels never select models.
The exact influence of every labelled reference is available analytically.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from MESA_SSL.mesa_multikernel import feature_views
from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import generate_affine_shift
from mesa_mr_experiment import indices_for_seed


def sqdist(a,b):
    aa=np.sum(a*a,axis=1,keepdims=True)
    bb=np.sum(b*b,axis=1,keepdims=True).T
    return np.maximum(aa+bb-2*a@b.T,0.)


def graph_embedding(groups,key,which):
    if key in ('raw','shape','mechanism','no_mechanism'):
        return groups[key][which]
    if key=='raw_shape':
        return np.column_stack([groups['raw'][which],groups['shape'][which]])
    if key=='shape_mechanism':
        return np.column_stack([groups['shape'][which],.5*groups['mechanism'][which]])
    if key=='shape_generic':
        return np.column_stack([groups['shape'][which],.5*groups['no_mechanism'][which]])
    raise ValueError(key)


def graph_components(train,test,shift,scale,knn,alpha):
    """Return a diffusion operator and out-of-sample normalized affinities."""
    n=len(train)
    dist=sqdist(train,train)
    positive=dist[np.triu_indices(n,1)]
    positive=positive[positive>1e-9]
    reference=float(np.median(positive)) if len(positive) else 1.
    width=max(scale*reference,1e-8)
    W=np.exp(-dist/width)
    np.fill_diagonal(W,0.)
    if knn and knn<n-1:
        index=np.argsort(-W,axis=1)[:,knn:]
        W[np.arange(n)[:,None],index]=0.
        W=np.maximum(W,W.T)
    degree=np.maximum(np.sum(W,axis=1),1e-12)
    S=W/np.sqrt(degree[:,None]*degree[None,:])
    G=np.linalg.solve(np.eye(n)-alpha*S,np.eye(n))
    other=[]
    for x in (test,shift):
        distance=sqdist(x,train)
        K=np.exp(-distance/width)
        if knn and knn<n:
            subset=np.argsort(-K,axis=1)[:,knn:]
            K[np.arange(len(x))[:,None],subset]=0.
        K=K/np.maximum(K.sum(axis=1,keepdims=True),1e-12)
        other.append(K@G)
    return G,other


def label_propagation_scores(op,query,indices,y,classes):
    n=op.shape[0]
    Y=np.zeros((n,len(classes)),dtype=float)
    Y[indices]=(y[indices,None]==classes[None,:]).astype(float)
    propagated=op@Y
    # Exact leave-one-label-out; left-out feature remains as an unlabelled node.
    loo=propagated[indices]-op[indices,indices,None]*Y[indices]
    predictions=[t@Y for t in query]
    # No-label reference explanation: each class score is a weighted sum of
    # the explicitly labelled TRAIN examples only.
    return loo,predictions


def ridge_scores(X,others,indices,y,classes,ridge):
    labelled=X[indices]
    Y=(y[indices,None]==classes[None,:]).astype(float)
    K=labelled@labelled.T
    inv=np.linalg.inv(K+ridge*np.eye(len(indices)))
    coef=inv@Y
    loo=Y-coef/np.diag(inv)[:,None]
    query=[other@labelled.T@coef for other in others]
    return loo,query


def scaling(scores):
    centered=scores-scores.mean(axis=1,keepdims=True)
    return float(max(np.sqrt(np.mean(centered**2)),1e-4))


def select(kind,groups,indices,y,classes,graph_cache,ridge_cache):
    """Optimise only TRAIN labels; every candidate follows same selection rule."""
    options=[]
    def add(name,loo,others,complexity,mechanism=False):
        denom=scaling(loo)
        loo_norm=loo/denom
        future=[p/denom for p in others]
        f1=float(f1_score(y[indices],classes[np.argmax(loo_norm,axis=1)],average='macro',zero_division=0))
        # Fixed complexity penalty discourages selecting models on small LOO gains
        objective=f1-.008*complexity
        options.append((objective,f1,complexity,name,mechanism,loo_norm,future))
    if kind in ('raw_ridge','shape_ridge'):
        selected=[kind.split('_')[0]]
        graphkeys=[]
    elif kind=='graph_raw':
        selected=['raw']
        graphkeys=['raw']
    elif kind=='graph_shape':
        selected=['shape']
        graphkeys=['shape']
    elif kind=='graph_generic':
        selected=['raw','shape']
        graphkeys=['raw','shape','no_mechanism','shape_generic']
    elif kind=='graph_mechanism':
        selected=['raw','shape']
        graphkeys=['raw','shape','mechanism','shape_mechanism']
    elif kind=='hybrid_generic':
        selected=['raw','shape']
        graphkeys=['raw','shape','no_mechanism','shape_generic']
    elif kind=='hybrid_mechanism':
        selected=['raw','shape']
        graphkeys=['raw','shape','mechanism','shape_mechanism']
    else:
        raise ValueError(kind)
    for name in selected:
        for ridge in (.01,.1,1.,10.):
            loo,other=ridge_cache[(name,ridge)]
            add(f"ridge/{name}/{ridge}",loo,other,0,name=='mechanism')
    if graphkeys:
        for k in graphkeys:
            for scale in (.25,1.,4.):
                for nn in (5,15):
                    loo,other=graph_cache[(k,scale,nn)]
                    complexity=1+(k in ['mechanism','shape_mechanism','no_mechanism','shape_generic'])
                    add(f"graph/{k}/{scale}/{nn}",loo,other,complexity,k in ('mechanism','shape_mechanism'))
        if kind in ('hybrid_generic','hybrid_mechanism'):
            key='shape_mechanism' if kind=='hybrid_mechanism' else 'shape_generic'
            for scale in (.25,1.,4.):
                for nn in (5,15):
                    gloo,gother=graph_cache[(key,scale,nn)]
                    rloo,rother=ridge_cache[('raw',.1)]
                    gd=scaling(gloo);rd=scaling(rloo)
                    for share in (.25,.5,.75):
                        loo=(1-share)*rloo/rd+share*gloo/gd
                        other=[(1-share)*a/rd+share*b/gd for a,b in zip(rother,gother)]
                        add(f"mix/{key}/{scale}/{nn}/{share}",loo,other,2,kind=='hybrid_mechanism')
    if not options:raise RuntimeError("No candidates")
    options.sort(key=lambda t:(-t[0],t[2],t[3]))
    winner=options[0]
    return dict(objective=winner[0],cv_f1=winner[1],name=winner[3],
                uses_mechanism=winner[4],clean=winner[6][0],shift=winner[6][1],
                eligible_candidates=len(options),kind=kind)


def run(args):
    train,test,y,target,_=load_data(args.dataset)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    kinds=('raw_ridge','shape_ridge','graph_raw','graph_shape','graph_generic',
           'graph_mechanism','hybrid_generic','hybrid_mechanism')
    for seed in args.seeds:
        st=time.monotonic()
        shifted=generate_affine_shift(test,seed,1.2)
        groups,concept_train,concept_test,concept_shift,error=feature_views(
            train,test,shifted,seed,args.epochs,True)
        idx=indices_for_seed(y,seed,budget=args.budget)
        classes=np.unique(y)
        ridge_cache={}
        for key in ('raw','shape'):
            for ridge in (.01,.1,1.,10.):
                ridge_cache[(key,ridge)]=ridge_scores(groups[key][0],
                     [groups[key][1],groups[key][2]],idx,y,classes,ridge)
        graph_cache={}
        for key in ('raw','shape','mechanism','no_mechanism',
                    'shape_mechanism','shape_generic'):
            inp=[graph_embedding(groups,key,j) for j in range(3)]
            for scale in (.25,1.,4.):
                for nn in (5,15):
                    G,q=graph_components(inp[0],inp[1],inp[2],scale,nn,.9)
                    graph_cache[(key,scale,nn)]=label_propagation_scores(G,q,idx,y,classes)
        for kind in kinds:
            result=select(kind,groups,idx,y,classes,graph_cache,ridge_cache)
            pred=classes[np.argmax(result['clean'],axis=1)]
            shifted_pred=classes[np.argmax(result['shift'],axis=1)]
            row=dict(dataset=args.dataset,seed=seed,method=kind,status='EXECUTED',
                    macro_f1=float(f1_score(target,pred,average='macro')),
                    shift_macro_f1=float(f1_score(target,shifted_pred,average='macro')),
                    accuracy=float(accuracy_score(target,pred)),
                    n_train=len(y),n_labeled=len(idx),n_test=len(target),
                    epochs=args.epochs,choice=result['name'],cv_f1=result['cv_f1'],
                    uses_mechanism=result['uses_mechanism'],n_candidates=result['eligible_candidates'],
                    shift_family='synthetic label-preserving affine offsets and trends',
                    geometry_invariance_error=error,
                    time_seconds=round(time.monotonic()-st,2))
            print('RESULT_JSON='+json.dumps(row,allow_nan=False),flush=True)
            rows.append(row)
        (out/'results.json').write_text(json.dumps(rows,indent=2))
    if len(rows)!=len(args.seeds)*len(kinds):
        raise AssertionError('Incomplete evaluation')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[410,411,412])
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--budget',type=int,default=5)
    p.add_argument('--out',default='results/mesa_graph')
    run(p.parse_args())
