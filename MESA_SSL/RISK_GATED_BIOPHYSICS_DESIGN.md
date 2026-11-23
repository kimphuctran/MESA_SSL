# MESA-BioPhysics: nuisance-conditional metaplastic memory gate

Research record: 23 November 2026.  
Source: MESA_SSL/mesa_biophysics_memory.py  
Validation workflow: .github/workflows/mesa-biophysics-riskgate-validation.yml

## Why this revision was necessary

The earlier fixed biophysical memory mixture used cortex, dentate sparse
coding, physics signatures and associative Hopfield retrieval on every
classification, even when the corresponding signal priors carried little
class information. Five-label UCR measurements on BME, ECG200, MoteStrain,
ShapeletSim and SonyAIBORobotSurface1 showed that the fixed mixture and
one global TRAIN-selected expert could improve noise robustness at the
cost of meaningful clean and affine classification loss.

An *additional predeclared task-state gate* separates clean, affine and
waveform-corrupted operating contexts. It should not be confused with a
human brain mechanism. It is a transparent engineering decision policy.

## Fixed policy

Input features and scores arise from a shared, label-blind cortical/physics
representation and fast 1/3/5-shot support memories. Unlabelled TRAIN
calibrates a shrinkage mean/slope chart and a morphology/spectral-shape chart.

For each labeled budget b:
- At 1-shot: clean source = Dual-MR episodic; affine fallback =
  Canonical-MR episodic; waveform fallback = Slow-cortex episodic.
  A class-preserving LOO estimate is not possible with one labeled
  observation per class.
- At 3/5-shot: for each support example i, remove the *entire labeled
  episode i*, including any augmented copies, from the memory reference.
  Predict the held-out TRAIN case and its TRAIN affine/jitter perturbations.
  Obtain clean, affine and waveform LOO Macro-F1 for each candidate.
- Clean candidates: Raw MR, Canonical MR and Dual MR episodic.
- Affine candidates: Canonical MR, Dual MR, NeuroMem prior and no-physics
  biophysical memory.
- Waveform candidates: Slow cortex, no-physics BioPhysics, NeuroMem
  prior, no-Hopfield BioPhysics and Dual MR.
- Penalize complex choices by 0.012 and require an improvement margin
  of 0.020 over the Dual MR reference, using TRAIN only.
- On an individual TEST series, use morphology alert (p<=0.10) first;
  else affine alert (p<=0.10); else clean chosen expert.
  This rule never sees TEST labels or evaluation metrics.

Different sources are converted to normalized class pseudo-scores before
switched comparison. These scores are NOT guaranteed calibrated
probabilities; their correct use is as decision scores.

## Biological and physical components

The slow cortex is trained on unlabeled TRAIN, the hippocampal store is
imprinted with labeled examples, DG projects features into a sparse
fixed high-dimensional code, a modern-Hopfield-style retrieval iterates
two attractor updates, AR(2) innovations detect prediction errors,
Fourier heat diffusion supplies a contractive multiscale physical prior,
and leaky integrators provide stable temporal response features.
The entire source family is ablated, rather than assuming every
component must be used.

The affine decomposition is exact within floating-point tolerance and
retains shape as well as raw mean/slope/scale. No unverified conservation
law is imposed on UCR time series.

## Scientific evidence boundaries

This method is inspired by prior complementary learning systems,
associative memory, predictive coding and homeostasis. These ingredients
are established in the literature; novelty must be demonstrated at the
level of the combined architecture, justified decision policy and
independent ablation results.

The prior BME/ECG200/MoteStrain/ShapeletSim/SonyAIBORobotSurface1 evaluation
was used to motivate the revision and is now developmental.
The separately fixed validation matrix is:
SonyAIBORobotSurface2, RefrigerationDevices,
DistalPhalanxOutlineAgeGroup and ProximalPhalanxOutlineCorrect,
with seeds 5501/5502, b=1/3/5 and 8 epochs.

No SOTA or clinical effectiveness claim follows without a
larger independent archive, more seeds, exact official TS2Vec/TimeMAE/MOMENT
and full MultiRocket comparisons, and domain-valid physical changes.
