#!/usr/bin/env python3
"""Few-label prototype adaptation after unlabelled, mechanism-conditioned SSL.

The same augmentation, optimiser, training budgets and classifier are used for
the mechanism and generic SSL encoders. Only a fixed labelled TRAIN subset is
available to prototype adaptation; all other training observations remain
unlabelled. Every reported TEST score is evaluated once after training.
"""
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import affine_gauge, generate_affine_shift
from MESA_SSL.mesa_multikernel import transform_group, feature_scale, select_model
from mesa_mr_experiment import train_encoder, indices_for_seed


def neural_features(model,x):
    out=[]
    model.eval()
    with torch.no_grad():
        for q in np.array_split(x,max(1,int(np.ceil(len(x)/64)))):
            z,c,recon=model(torch.tensor(q,dtype=torch.float32))
            out.append((z.detach().cpu().numpy(),c.detach().cpu().numpy()))
    return tuple(np.concatenate([q[i] for q in out],axis=0) for i in range(2))


def label_order(labels,indices):
    classes=np.unique(labels)
    Y=np.searchsorted(classes,labels[indices])
    return classes,Y


def train_prototypes(x,lab,indices,seed,epochs,has_mechanism):
    torch.manual_seed(seed)
    np.random.seed(seed)
    before,_=train_encoder(x,seed,epochs,mechanism=has_mechanism)
    after=copy.deepcopy(before)
    ref=copy.deepcopy(before).eval()
    tensor=torch.tensor(x,dtype=torch.float32)
    labels,used=label_order(lab,indices)
    yi=torch.tensor(used,dtype=torch.long)
    optimizer=torch.optim.AdamW(after.parameters(),lr=.0005,weight_decay=.0007)
    source=torch.tensor(indices,dtype=torch.long)
    concepts_before=neural_features(before,x)[1]
    for ep in range(30):
        after.train()
        sample=tensor[source]
        augmented=sample+.075*torch.randn_like(sample)
        z,c,_=after(augmented)
        z_clean,c_clean,_=after(sample)
        z=F.normalize(z,dim=-1)
        z_clean=F.normalize(z_clean,dim=-1)
        proto=[]
        for k in range(len(labels)):
            mask=(yi==k)
            count=mask.sum().clamp_min(1)
            proto.append(z_clean[mask].mean(dim=0))
        proto=torch.stack(proto,dim=0)
        # Exclude self from same-class prototype, removing self-match advantage.
        distances=[]
        for j in range(len(sample)):
            current=proto.clone()
            own=int(yi[j])
            mask=(yi==own)
            current[own]=(z_clean[mask].sum(dim=0)-z_clean[j])/(mask.sum()-1)
            current=F.normalize(current,dim=1)
            distances.append(z[j]@current.T)
        logits=12*torch.stack(distances)
        supervised=F.cross_entropy(logits,yi)
        # Retain unlabelled representation geometry and concept semantics.
        uidx=torch.randperm(len(tensor))[:min(64,len(tensor))]
        with torch.no_grad():
            orig_z,orig_c,_=ref(tensor[uidx])
        adapt_z,adapt_c,_=after(tensor[uidx])
        stability=F.smooth_l1_loss(adapt_z,orig_z)+.5*F.smooth_l1_loss(adapt_c,orig_c)
        loss=supervised+.16*stability+.03*F.smooth_l1_loss(c,c_clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(after.parameters(),2.)
        optimizer.step()
    after.eval()
    concepts_after=neural_features(after,x)[1]
    concept_change=float(np.sqrt(np.mean((concepts_after-concepts_before)**2)))
    return before,after,concept_change,labels


def prototype_predict(model,train,test,indices,y,classes):
    ztr,ctr=neural_features(model,train)
    zte,cte=neural_features(model,test)
    ztr=ztr/np.maximum(np.linalg.norm(ztr,axis=1,keepdims=True),1e-12)
    zte=zte/np.maximum(np.linalg.norm(zte,axis=1,keepdims=True),1e-12)
    prototypes=[]
    for cls in classes:
        mu=ztr[indices[y[indices]==cls]].mean(0)
        prototypes.append(mu/max(np.linalg.norm(mu),1e-12))
    proto=np.stack(prototypes)
    return zte@proto.T,ztr@proto.T,ztr,zte


def ridge_readout(ztr,zte,indices,y,classes,lam=.1):
    # Unsupervised SSL features only; ridge uses labelled TRAIN subset.
    scaler=feature_scale(ztr,zte)
    a,b=scaler
    A=a[indices]
    Y=(y[indices,None]==classes[None,:]).astype(float)
    coef=np.linalg.solve(A@A.T+lam*np.eye(len(A)),Y)
    return b@A.T@coef


def run(args):
    train,test,y,yt,_=load_data(args.dataset)
    testshift=generate_affine_shift(test,args.seeds[0],1.2)
    train_shape,_=affine_gauge(train)
    test_shape,_=affine_gauge(test)
    test_shift_shape,_=affine_gauge(testshift)
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in args.seeds:
        start=time.monotonic()
        idx=indices_for_seed(y,seed,budget=args.budget)
        classes=np.unique(y)
        raw=transform_group(train,test,testshift,seed,n_kernels=336)
        shape=transform_group(train_shape,test_shape,test_shift_shape,seed,n_kernels=336)
        base={"raw":raw,"shape":shape}
        predictions={}
        for kind in ["raw","shape"]:
            xtr,xtest,xshift=base[kind]
            scores=[]
            for query in (xtest,xshift):
                Y=(y[idx,None]==classes[None,:]).astype(float)
                A=xtr[idx]
                coef=np.linalg.solve(A@A.T+.1*np.eye(len(idx)),Y)
                scores.append(query@A.T@coef)
            predictions["MultiRocket-"+kind]=tuple(scores)
        for mechanism in [False,True]:
            kind="MESA" if mechanism else "Generic"
            before,model,change,_=train_prototypes(train_shape,y,idx,seed,args.epochs,mechanism)
            for state,net in [("SSL-only",before),("FewShot",model)]:
                clean,_,a,b=prototype_predict(net,train_shape,test_shape,idx,y,classes)
                shifted,_,__,___=prototype_predict(net,train_shape,test_shift_shape,idx,y,classes)
                predictions[kind+"-"+state+"-Prototype"]=(clean,shifted)
                ztr=neural_features(net,train_shape)[0]
                zte=neural_features(net,test_shape)[0]
                zsh=neural_features(net,test_shift_shape)[0]
                predictions[kind+"-"+state+"-Ridge"]=(ridge_readout(ztr,zte,idx,y,classes),
                                                    ridge_readout(ztr,zsh,idx,y,classes))
            # A fixed-weight combination avoids choosing the mixture by TEST.
            mix=[]
            for j,query in enumerate((test_shape,test_shift_shape)):
                zr=neural_features(model,query)[0]
                zz=neural_features(model,train_shape)[0]
                zscore=ridge_readout(zz,zr,idx,y,classes)
                mr=predictions["MultiRocket-shape"][j]
                ps=predictions[kind+"-FewShot-Prototype"][j]
                # Predeclared non-validated weights, not an oracle.
                mix.append(.70*mr+.20*zscore+.10*(ps/12))
            predictions[kind+"-FewShot-FixedMix"]=tuple(mix)
        for method,(c,s) in predictions.items():
            yc=classes[np.argmax(c,axis=1)]
            ys=classes[np.argmax(s,axis=1)]
            row=dict(dataset=args.dataset,seed=seed,method=method,status="EXECUTED",
                     macro_f1=float(f1_score(yt,yc,average="macro")),
                     shifted_macro_f1=float(f1_score(yt,ys,average="macro")),
                     accuracy=float(accuracy_score(yt,yc)),n_labeled=len(idx),
                     epochs_pretraining=args.epochs,
                     additional_labeled_adaptation_epochs=30 if "FewShot" in method else 0,
                     n_train=len(train),n_test=len(test),
                     shift_assumption="Declared synthetic affine nuisance is label-preserving",
                     elapsed_seconds=round(time.monotonic()-start,2))
            print("RESULT_JSON="+json.dumps(row,allow_nan=False),flush=True)
            rows.append(row)
        (folder/"results.json").write_text(json.dumps(rows,indent=2))
    if len(rows)!=len(args.seeds)*len(predictions):raise AssertionError("Missing records")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[510,511,512])
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--budget",type=int,default=5)
    p.add_argument("--out",default="results/mesa_proto")
    run(p.parse_args())
