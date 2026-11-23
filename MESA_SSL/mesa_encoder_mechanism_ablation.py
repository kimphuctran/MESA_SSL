#!/usr/bin/env python3
"""Paired mechanism-loss ablation at the TRAIN-selected MESA hyperparameters.

Given the TRAIN-only Optuna-selected MESA parameters, train two encoders with
identical seeds, architecture, batch order, masks, augmentation random draws
and learning-rate schedule. Both execute the mechanism branch, but the control
sets BOTH mechanism loss weights to exactly zero. Thus differences in training
gradients arise from the mechanism objective, not a different architecture
or a different base stochastic training path.

This is an ablation conditioned on MESA-selected hyperparameters, not a fair
replacement for independently optimized generic SSL (already included in
the main benchmark). TEST targets are never used to select hyperparameters.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from MESA_SSL.mesa_canonical import load_data
from MESA_SSL.mesa_gauge_ssl import affine_gauge, generate_affine_shift
from MESA_SSL.mesa_multikernel import feature_scale
from MESA_SSL.mesa_robust_select import extract
from MESA_SSL.mesa_encoder_hpo import (
    deterministic_jitter, train_parametric_encoder,
    encoder_views, choose_features, concept_error
)
from mesa_mr_experiment import indices_for_seed


def lookup_configs(path,dataset,seed):
    source=json.loads(Path(path).read_text())
    matching=[
        x for x in source["selected_hyperparameters"]
        if x["dataset"]==dataset and x["seed"]==seed
            and x["encoder_family"]=="MESA"
    ]
    if len(matching)!=1:raise ValueError(
        f"Need exactly one TRAIN-selected MESA config: {dataset}/{seed}")
    return matching[0]["params"]


def run(args):
    train,test,y,yt,_=load_data(args.dataset)
    train_shape,_=affine_gauge(train)
    fixed_jitter=deterministic_jitter(train_shape,args.seeds[0]+175)
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for seed in args.seeds:
        st=time.monotonic()
        params=lookup_configs(args.config,args.dataset,seed)
        selected=copy.deepcopy(params)
        no_mechanism=copy.deepcopy(params)
        no_mechanism["lambda_concept"]=0.
        no_mechanism["lambda_equiv"]=0.
        fixed_jitter=deterministic_jitter(train_shape,seed+175)
        shift=generate_affine_shift(test,seed,1.5)
        jitter=deterministic_jitter(test,seed+450,noise=.12,harmonic=.08)
        test_views=[affine_gauge(x)[0] for x in (test,shift,jitter)]
        idx=indices_for_seed(y,seed,budget=args.budget)
        classes=np.unique(y)
        canonical=extract(train_shape,test_views,seed)
        conditions=(("MESA-mechanism-loss",selected),
                    ("Same-config-zero-loss",no_mechanism))
        for condition,p in conditions:
            # Both cases deliberately enter the mechanism branch and consume
            # identical random streams; the control loss weights are zero.
            net,losses=train_parametric_encoder(
                train_shape,seed+44000,p,mechanism=True)
            feats=encoder_views(net,train_shape,fixed_jitter,test_views)
            latent,latent_meta=choose_features(
                feats[0],feats[2:],idx,y,classes)
            fused,fused_meta=choose_features(
                canonical[0],canonical[1:],idx,y,classes,
                secondary=(feats[0],feats[2:]))
            for readout,logits,meta in (
                 ("latent",latent,latent_meta),
                 ("latent+canonical",fused,fused_meta)):
                predictions=[classes[np.argmax(x,axis=1)] for x in logits]
                row=dict(dataset=args.dataset,seed=seed,method=condition,
                    readout=readout,status="EXECUTED",
                    clean_macro_f1=float(f1_score(yt,predictions[0],
                                               average="macro",zero_division=0)),
                    affine_macro_f1=float(f1_score(yt,predictions[1],
                                               average="macro",zero_division=0)),
                    jitter_macro_f1=float(f1_score(yt,predictions[2],
                                               average="macro",zero_division=0)),
                    accuracy=float(accuracy_score(yt,predictions[0])),
                    concept_rmse=concept_error(net,train_shape),
                    selected_train_hyperparameters=selected,
                    used_mechanism_weights={
                        "lambda_concept":p["lambda_concept"],
                        "lambda_equiv":p["lambda_equiv"]},
                    train_loo_readout=meta,
                    final_training_loss=losses[-1],
                    n_train=len(y),labelled_train=len(idx),n_test=len(yt),
                    seconds=round(time.monotonic()-st,3))
                allrows.append(row)
                print("ABLATION_JSON="+json.dumps(row,allow_nan=False),
                      flush=True)
        (output/"results.json").write_text(
            json.dumps(allrows,indent=2,allow_nan=False))
    if len(allrows)!=4*len(args.seeds):
        raise AssertionError("Missing paired ablation results")
    return allrows


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[1521,1522])
    p.add_argument("--config",default="MESA_SSL/encoder_hpo_selected_2026-11-23.json")
    p.add_argument("--budget",type=int,default=5)
    p.add_argument("--out",default="results/encoder_mechanism_ablation")
    run(p.parse_args())
