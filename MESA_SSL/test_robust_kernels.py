"""Numerical contract tests for MESA robust kernel selection."""
import unittest
import numpy as np

from MESA_SSL.mesa_gauge_ssl import affine_gauge
from MESA_SSL.mesa_multikernel import class_scores,select_model,moment_drift_gate
from MESA_SSL.mesa_robust_select import aug_leave_one_out


class MechanismKernelTests(unittest.TestCase):
    def test_affine_invariance(self):
        rng=np.random.default_rng(8)
        x=rng.normal(size=(12,64,3)).astype("float32")
        t=np.linspace(-1,1,64,dtype="float32")
        a=rng.normal(size=(12,1,3)).astype("float32")
        b=rng.normal(size=(12,1,3)).astype("float32")
        p=x+a+t[None,:,None]*b
        rx,concept_x=affine_gauge(x)
        rp,concept_p=affine_gauge(p)
        self.assertLess(float(np.max(np.abs(rx-rp))),2e-5)
        self.assertLess(float(np.max(np.abs(concept_p[:,:3]-concept_x[:,:3]-a[:,0,:]))),2e-5)

    def test_augmented_leave_one_out_exact(self):
        rng=np.random.default_rng(7)
        z=rng.normal(size=(8,4));q=rng.normal(size=(8,4))
        K=z@z.T;Kq=q@z.T
        Y=rng.normal(size=(8,2));reg=.15
        analytic,alpha=aug_leave_one_out(K,Kq,Y,reg)
        brute=[]
        for i in range(8):
            ids=np.array([j for j in range(8) if j!=i])
            inv=np.linalg.solve(K[np.ix_(ids,ids)]+reg*np.eye(7),Y[ids])
            brute.append(Kq[i,ids]@inv)
        self.assertLess(float(np.max(np.abs(analytic-np.asarray(brute)))),1e-9)

    def test_exact_score_decomposition(self):
        rng=np.random.default_rng(12)
        groups={}
        for key in ("raw","shape","concept","mechanism","no_mechanism"):
            base=rng.normal(size=(20,7))
            clean=rng.normal(size=(3,7))
            shift=rng.normal(size=(3,7))
            groups[key]=(base,clean,shift)
        labels=np.array([0]*10+[1]*10)
        ids=np.r_[0:5,10:15]
        chosen=select_model(groups,ids,labels,"MESA-MK")
        score,components=class_scores(chosen,groups,1)
        self.assertLess(float(np.max(np.abs(score-sum(components.values())))),1e-9)

    def test_unlabelled_shift_detector(self):
        rng=np.random.default_rng(2)
        # Mean, slope and log residual RMS of two mock channels
        c=rng.normal(scale=.1,size=(100,6))
        x=c[:10].copy();shift=x.copy();shift[:,0]+=2.
        gate_normal=moment_drift_gate(c,x)
        gate_shift=moment_drift_gate(c,shift)
        self.assertGreater(float(gate_shift.mean()),float(gate_normal.mean())+.5)


if __name__=="__main__":
    unittest.main(verbosity=2)
