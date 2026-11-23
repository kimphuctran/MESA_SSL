# MESA-RouterFormer: concept-routed dual-stream attention for time series

## Status and novelty boundary

A prototype neural network implemented for the MESA-SSL research programme.
Its **architectural hypothesis** is that explicit separation of raw and affine-
canonical information, with analytically conditioned local/global attention
routing, may improve few-label representation under heterogeneous shifts.

Patch tokenization, compressed/global attention and Transformer encoders are
**established ideas**. The combination here must not be described as the
first Transformer for time series or claimed state of the art without
comprehensive prior-art review and matched experiments.

Reference architecture:
- PatchTST: https://huggingface.co/docs/transformers/en/model_doc/patchtst
- iTransformer: https://github.com/thuml/iTransformer
- Sensorformer (global patch compression): https://arxiv.org/abs/2501.03284

## Mathematical structure

For a univariate channel of a multivariate sequence `x(t)`, define a centered
time axis `tau`, series mean `mu`, fitted linear slope `beta` and residual
scale `sigma`:

```text
mu    = mean_t(x)
beta  = sum_t[tau_t * x_t] / sum_t[tau_t^2]
r_t   = x_t - mu - beta*tau_t
sigma = sqrt(mean_t(r_t^2))
q_t   = r_t / max(sigma, eps)
```

The mapping `x -> (q,mu,beta,sigma)` retains all components needed to
reconstruct a non-degenerate series, while `q` is invariant under
`x' = x + constant + linear_slope*tau`. Retaining the raw stream prevents
information loss when class labels depend on mean or slope.

Two **independently parameterized** tokenizers produce 8, 16 and 24 temporal
summary tokens each from raw `x` and normalized shape `q`. Each token
contains a window average, RMS and mean absolute first difference, plus
fixed sinusoidal positional and trainable scale encoding.

For each stream `s`, define `X_s in R^(B*N*D)` and an analytic concept
embedding `c_s in R^D`. For the raw stream, `c_raw` is based on
`(mu,beta,log sigma)`. For the shape stream, `c_shape` is based only on
two harmonic projections of `q`; notably, it **never sees raw mean/slope**,
so the shape's affine invariance is preserved through routing.

Each block has:

1. A local neighborhood of radius `r`, with masked softmax attention
   over `2r+1` tokens (no dense `N*N` attention matrix).
2. `K` global learnable memory-slot queries attending to `N` tokens,
   followed by token-to-slot attention. Each slot is shifted by `c_s`.
3. A per-head concept gate `g_h = sigmoid(MLP(c_s)_h)`, merging local
   and slot responses:
   `H = (1-g)*Local(X) + g*SlotMemory(X,c_s)`.
4. Residual projections, LayerNorm and feed-forward MLP.

With fixed `r` and `K`, the attention contribution scales as
`O(B * heads * N * head_dim * ((2r+1)+K))` instead of quadratic dense
self-attention `O(B * heads * N^2 * head_dim)`. Overall model cost and
comparison to vanilla attention depend on the number of tokens and MLPs;
measure actual latency rather than asserting an automatic speedup.

An attention-weighted global token pool plus average pool gives independent
`z_raw` and `z_shape` representations. Analytical concept heads predict
raw mean/slope/scale and harmonic shape projections. A frequency head
predicts the canonical residual's log band spectrum.

## Self-supervised objectives

All variants share masked/noisy-view consistency, raw/shape variance
regularization, affine shape consistency and frequency prediction.

The mechanism variant adds:
- raw concept-head alignment with analytical (mean,slope,log_rms);
- raw concept equivariance under affine offset and trend;
- shape harmonic concept-head alignment;
- shape concept equivariance under a declared harmonic intervention.

The **matched generic RouterFormer control** has the same two-stream
architecture, initialization, schedule, data and optimizer but zero
mechanism loss. A **dense Transformer control** has the same tokenizer
and input decomposition, with ordinary dense multihead self-attention
instead of the concept-conditioned local/slot router. It is not exactly
parameter-matched; the experiment records parameter counts.

For a separate legacy baseline, the earlier dual-CNN MESA-DP model can
also be trained under the same sample and epoch budget.

## Evaluation protocol and safeguards

- Official UCR/UEA TRAIN/TEST partitions.
- Predeclared 5 labeled TRAIN examples/class; unlabeled TRAIN for SSL.
- TRAIN-only ridge regularization/feature-fusion selection by exact
  leave-one-labeled-out (LOO). If the held-out observation has a synthetic
  augmented copy, its own label is algebraically removed.
- Evaluate separately clean, synthetic affine shift and waveform jitter.
- Explicitly compare RouterFormer learned-only, generic learned-only,
  vanilla dense Transformer, CNN MESA-DP, raw/canonical MultiRocket,
  kernel fusion and two-chart selection. Do not attribute a fusion score
  to the new network if its weight is zero.
- Preserve random seeds and source revisions, per-run results, TRAIN
  validation objectives, architecture parameter counts, and wall times.
- Never tune anything with official TEST labels.
- Compare against official TS2Vec, TimeMAE and MOMENT protocols before
  drawing Neurocomputing-level superiority claims; this prototype alone
  does not establish universal performance.

## Code and invocation

```bash
python -m MESA_SSL.mesa_routerformer \
  --dataset GunPoint --seeds 3601 3602 --epochs 10 \
  --dim 64 --depth 2 --cnn-baseline \
  --out results/routerformer/GunPoint
```

Source: [mesa_routerformer.py](./mesa_routerformer.py).

The experimental branch remains separate from repository `master`.
