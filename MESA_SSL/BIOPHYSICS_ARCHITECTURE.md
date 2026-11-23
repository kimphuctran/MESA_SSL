# MESA-BioPhysics: Complementary Memory and Physics-Constrained Few-Shot Learning

**Research status:** experiment, not validated state of the art. 23 November 2026.

## Biological design

1. **Complementary learning systems.** A slowly trained cortex (unlabeled TRAIN) supplies reusable embeddings; a fast hippocampal memory imprints labeled samples without gradient fine-tuning. Biological inspiration is the well-established complementary learning systems framework: https://doi.org/10.1111/j.1551-6709.2011.01214.x
2. **Dentate gyrus.** Class-blind random expansion followed by 9% sparse k-winner positive competition. Homeostatic control through TRAIN-only scaling; matched dense-projection ablation.
3. **CA3.** Stored support patterns are retrieved by cosine similarity, a class-wise energy-like log-mean-exp score, and two Hopfield-style iterations. The implementation does NOT assert a monotone Hopfield energy descent theorem. Related work: https://mlanthology.org/iclr/2021/ramsauer2021iclr-hopfield/
4. **Predictive coding.** A contractive TRAIN-fitted AR(2) model measures innovation errors. This is a simplified generative-prediction analogy, not a reconstruction of cortical feedback. Inspiration: Rao and Ballard 1999, https://doi.org/10.1038/4580
5. **Metaplastic expert governance.** The fast memory can change its chosen readout without retraining cortex. With 3 or 5 labels/class, expert selection requires leaving a whole labeled TRAIN episode and its augmentations out. If no candidate beats a simpler dual-MultiRocket reference by a predeclared margin, preserve the reference. With 1-shot, LOO would remove a class, so use the fixed dual-MultiRocket fallback.

## Physics design

6. **Group factorization.** For centered time axis tau, mu=mean(x), beta=<tau,x>/<tau,tau>, r=x-mu-beta*tau, sigma=sqrt(mean(r^2)), q=r/max(sigma,eps). The tuple (q,mu,beta,sigma) allows reconstruction; q is invariant to declared additive offset/linear trend. This is not an assertion that all real-world disturbances preserve labels.
7. **Heat diffusion semigroup.** Fourier multiplier exp(-t*(f/fmax)^2) at t=0,0.2,1,4. In each case Parseval and the modulus <=1 ensure L2 energy does not increase. The PDE filter is a signal-processing prior, not a claim that every sensor signal literally obeys heat diffusion.
8. **Stable leaky dynamics.** h_t=(1-alpha)*h_(t-1)+alpha*q_t, with alpha=0.035,0.14,0.45. For bounded inputs each first-order recursion is stable.
9. **Predictive surprise.** The AR(2) innovation e_t=q_t-a1*q_(t-1)-a2*q_(t-2), with |a1|+|a2|<=0.95. Its train-calibrated surprise adds a separate nuisance detector.
10. **Spectral statistics.** Eight bands of energy computed with one-sided Fourier weights and train-only scaling. Physics-guided equivariance prior art: https://doi.org/10.1073/pnas.2415656122

## Exact evaluation contract

- Official UCR TRAIN/TEST, three modes: clean, synthetic additive affine and synthetic Gaussian/harmonic waveform noise.
- Nested 1-, 3-, and 5-shot support IDs for every method. Synthetic interventions are ASSUMED class-preserving.
- Unlabeled TRAIN only for SSL, covariance fitting, predictor estimation, sparse projection scaling and conformal chart calibration.
- Absolutely no TEST labels or TEST features are used to choose parameters, source weights or experts.
- Mechanism-specific ablations: without sparse homeostasis, without cortical features, without physics, without predictive innovation, without Hopfield retrieval, without chart gating, plus fixed/learned episodic references.
- With 1-shot, TRAIN leave-one-out model selection is invalid; use explicit fixed fallback rather than an incorrectly optimistic validation score.
- Compare on clean and shifted data simultaneously; report failures and side effects as well as improvements.
- A large external benchmark and official TS2Vec, TimeMAE, MOMENT and full MultiRocket are required before any broad SOTA claim.

## Repository

- MESA_SSL/mesa_neuromem.py: original slow-cortex and hippocampal sparse memory experiment.
- MESA_SSL/mesa_biophysics_memory.py: heat diffusion, leaky stability, predictor, Hopfield retrieval, metaplastic controller.
- MESA_SSL/BIOPHYSICS_ARCHITECTURE.md: this technical design.
- Separate branches preserve the fixed mixture and safe controller to avoid post-hoc TEST-driven replacement.

## Novelty caution

CLS, modern Hopfield memory, homeostatic plasticity, predictor errors and equivariant learning all have significant prior art. The project contributes a proposed task-specific integration and validated train-only selection protocol; originality and superiority require comparison with the current literature, not merely a biological metaphor.