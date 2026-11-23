#!/usr/bin/env python3
"""HCLM-governed online aggregation of fixed, verified representation experts.

On each sample: predict first, receive the true class afterwards, and only then
update mixture weights. For positive class probabilities and a fixed admissible
expert set, the Bayesian mixture has cumulative log-loss no more than log(K)
above the best fixed admissible expert. This does not guarantee higher F1 or
per-sample accuracy than every expert, and it is a separate ONLINE protocol.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss,accuracy_score,f1_score

from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import affine_gauge
from MESA_SSL.mesa_multikernel import transform_group
from mesa_mr_experiment import train_encoder,indices_for_seed
from MESA_SSL.mesa_prototype_ssl import neural_features


def normalize_probs(probs,epsilon=1e-7):
    p=np.asarray(probs,dtype=np.float64)
    if p.ndim!=2 or not np.isfinite(p).all() or np.min(p)<0:
        raise ValueError("Invalid expert probabilities")
    p=np.maximum(p,epsilon)
    return p/p.sum(axis=1,keepdims=True)


class FixedAdmissibilityBayesianAggregator:
    """Exact Bayesian predictive mixture; static HCLM permission screen."""
    def __init__(self,expert_names,permitted=None,epsilon=1e-7):
        self.names=list(expert_names)
        allowed=np.ones(len(self.names),dtype=bool) if permitted is None else np.asarray(permitted,dtype=bool)
        if len(allowed)!=len(self.names) or not allowed.any():
            raise ValueError("No admissible experts")
        self.active=np.where(allowed)[0]
        self.epsilon=epsilon
        self.logw=np.full(len(self.active),-math.log(len(self.active)))
        self.cumloss=np.zeros(len(self.active))
        self.total_mix_loss=0.
        self.n=0

    def predict(self,full_experts):
        full=np.asarray(full_experts,dtype=float)
        if full.ndim!=2 or full.shape[0]!=len(self.names):
            raise ValueError("Expert probability matrix must have shape (K,C)")
        p=normalize_probs(full[self.active],self.epsilon)
        w=np.exp(self.logw)
        prediction=np.sum(w[:,None]*p,axis=0)
        return prediction,p

    def update(self,after_predict,probs,truth):
        if not 0<=truth<len(after_predict):
            raise ValueError("Unknown class")
        # Truth is revealed only after prediction was issued.
        mixture_likelihood=max(float(after_predict[truth]),self.epsilon)
        expert_likelihoods=np.maximum(probs[:,truth],self.epsilon)
        self.total_mix_loss-=math.log(mixture_likelihood)
        self.cumloss-=np.log(expert_likelihoods)
        updated=self.logw+np.log(expert_likelihoods)
        maximum=updated.max()
        updated-=maximum+math.log(np.exp(updated-maximum).sum())
        self.logw=updated
        self.n+=1

    def audit(self):
        regret=self.total_mix_loss-float(self.cumloss.min())
        bound=math.log(len(self.active))
        if regret>bound+1e-8:
            raise AssertionError(f"Log-loss regret bound failed: {regret}>{bound}")
        return dict(cumulative_mixture_logloss=self.total_mix_loss,
                    cumulative_best_fixed_logloss=float(self.cumloss.min()),
                    regret=regret,guaranteed_max_regret=bound,
                    active_expert_names=[self.names[j] for j in self.active],
                    posterior_weights=np.exp(self.logw).tolist(),n=self.n)


def expert_probs(train_features,test_features,y,idx):
    scaler=StandardScaler().fit(train_features)
    a=scaler.transform(train_features[idx])
    b=scaler.transform(test_features)
    model=LogisticRegression(C=1.,max_iter=2500,solver="lbfgs").fit(a,y[idx])
    pred=model.predict_proba(b)
    classes=np.unique(y)
    result=np.full((len(b),len(classes)),1e-7)
    for k,c in enumerate(model.classes_):
        result[:,np.where(classes==c)[0][0]]=pred[:,k]
    return normalize_probs(result)


def get_features(train,test,seed,epochs):
    shape_train,_=affine_gauge(train)
    shape_test,_=affine_gauge(test)
    raw=transform_group(train,test,test,seed,n_kernels=336)
    canonical=transform_group(shape_train,shape_test,shape_test,seed,n_kernels=336)
    reps={'Raw MultiRocket':(raw[0],raw[1]),
          'Canonical MultiRocket':(canonical[0],canonical[1])}
    for mechanism,name in ((False,'Generic SSL'),(True,'MESA concept SSL')):
        net,_=train_encoder(shape_train,seed,epochs,mechanism=mechanism)
        features_train=neural_features(net,shape_train)
        features_test=neural_features(net,shape_test)
        reps[name]=(np.column_stack(features_train),np.column_stack(features_test))
    return reps


def evaluate(dataset,seed,epochs,budget,output):
    train,test,y,truth,_=load_data(dataset)
    idx=indices_for_seed(y,seed,budget=budget)
    classes=np.unique(y)
    reps=get_features(train,test,seed,epochs)
    names=list(reps)
    probs=np.stack([expert_probs(a,b,y,idx) for a,b in reps.values()],axis=1)
    online=FixedAdmissibilityBayesianAggregator(names)
    guesses=[]
    for t in range(len(truth)):
        issued,active=online.predict(probs[t])
        guesses.append(int(issued.argmax()))
        online.update(issued,active,int(truth[t]))
    result=online.audit()
    result.update(dataset=dataset,seed=seed,protocol="ONLINE PREQUENTIAL: each TEST label revealed only after that prediction",
                  n_labeled_initial=len(idx),num_experts=len(names),
                  mixture_macro_f1=float(f1_score(truth,guesses,average='macro')),
                  mixture_accuracy=float(accuracy_score(truth,guesses)))
    for i,name in enumerate(names):
        preds=probs[:,i,:].argmax(axis=1)
        result[name+'_macro_f1']=float(f1_score(truth,preds,average='macro'))
        result[name+'_logloss']=float(-np.sum(np.log(np.maximum(probs[np.arange(len(truth)),i,truth],1e-7))))
    print('RESULT_JSON='+json.dumps(result,allow_nan=False),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    (output/f'{dataset}_seed{seed}.json').write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def numerical_test(seed=17):
    rng=np.random.default_rng(seed)
    expert_probs=rng.uniform(.005,1.,size=(170,4,3))
    expert_probs=expert_probs/expert_probs.sum(axis=2,keepdims=True)
    y=rng.integers(0,3,size=170)
    for allowed in [None,[1,0,1,0]]:
        names=['a','b','c','d']
        hedge=FixedAdmissibilityBayesianAggregator(names,allowed)
        for t in range(len(y)):
            prob,active=hedge.predict(expert_probs[t])
            hedge.update(prob,active,int(y[t]))
        audit=hedge.audit()
        assert audit["regret"]<=audit["guaranteed_max_regret"]+1e-8
    return True


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='GunPoint')
    p.add_argument('--seeds',type=int,nargs='+',default=[610,611,612])
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--budget',type=int,default=5)
    p.add_argument('--out',default='results/hclm_online')
    p.add_argument('--test-only',action='store_true')
    args=p.parse_args()
    numerical_test()
    if not args.test_only:
        path=Path(args.out)
        for seed in args.seeds:evaluate(args.dataset,seed,args.epochs,args.budget,path)
