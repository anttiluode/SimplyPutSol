import copy
import unittest
import numpy as np
from simplyputsol import Cable, World, frames, make, probe, evaluate, experiment

class MechanismTests(unittest.TestCase):
    def test_history_is_hidden_from_final_observation(self):
        p=np.array([0.2,1.])
        a,b=frames(p,0,-1),frames(p,0,1)
        np.testing.assert_array_equal(a[-1],b[-1])
        self.assertNotEqual(a[1,8],b[1,8])

    def test_dynamic_cable_distinguishes_history(self):
        p=np.array([0.2,1.])
        a,b=frames(p,0,1),frames(p,0,-1)
        m=make("cable",20)
        baseline=make("no_state",20)
        self.assertGreater(np.linalg.norm(m.forward(a)[0]-m.forward(b)[0]),1e-8)
        np.testing.assert_allclose(baseline.forward(a)[0],baseline.forward(b)[0])

    def test_local_phase_tangent(self):
        m=make("cable",11)
        xs=frames(np.array([0.3,1.7]),2,1)
        y,(h,dw,dc)=m.forward(xs)
        k,j=3,8
        eps=1e-6
        plus=copy.deepcopy(m)
        minus=copy.deepcopy(m)
        plus.omega[k]+=eps
        minus.omega[k]-=eps
        actual=(plus.forward(xs)[0]-minus.forward(xs)[0])/(2*eps)
        np.testing.assert_allclose(actual,m.B[:,k]*dw[k],atol=1e-8)
        plus.C[k,j]+=eps
        minus.C[k,j]-=eps
        actual=(plus.forward(xs)[0]-minus.forward(xs)[0])/(2*eps)
        np.testing.assert_allclose(actual,m.B[:,k]*dc[k,j],atol=1e-8)

    def test_pair_updates_cable_without_teacher(self):
        m=make("cable",18)
        before=m.B.copy()
        probe(m,World(),np.array([0.2,1.0]),0,1,np.random.default_rng(16))
        self.assertGreater(np.linalg.norm(m.B-before),0)

    def test_evaluation_does_not_change_parameters(self):
        m=make("cable",5)
        old=(m.B.copy(),m.C.copy(),m.omega.copy())
        evaluate(m,91,postures=3)
        for a,b in zip(old,(m.B,m.C,m.omega)):
            np.testing.assert_array_equal(a,b)

    def test_smoke_all_controllers(self):
        for name in ("cable","phase0","frozen","no_state","rnn","route"):
            record=experiment(name,0,episodes=5,evaluation=3)
            self.assertEqual(len(record["phases"]),5)
            self.assertGreater(record["parameters"],0)
            self.assertTrue(np.isfinite(record["phases"][-1]["tasks"][0]["distance"]))

if __name__=="__main__":
    unittest.main()
