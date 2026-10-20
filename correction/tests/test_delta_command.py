import unittest
import numpy as np
from correction.src.optimization.delta_command import solve_endpoint, held_value, bernstein_bounds


class DeltaCommandTests(unittest.TestCase):
    def test_entire_curve_bounds_and_replan_continuity(self):
        state=(np.zeros(2),np.zeros(2),np.zeros(2))
        curve,audit=solve_endpoint(state,.5,np.array([1.,-1.]),[],[],
            np.full(2,-.2),np.full(2,.2),np.ones(2),np.full(2,4.))
        self.assertTrue(audit['solved'])
        sample=np.linspace(0,.5,1001)
        self.assertLessEqual(np.max(abs(curve(sample))),.20001)
        self.assertLessEqual(np.max(abs(curve(sample,1))),1.00001)
        self.assertLessEqual(np.max(abs(curve(sample,2))),4.00001)
        continued=tuple(curve(.1,d) for d in range(3))
        following,audit=solve_endpoint(continued,.5,np.zeros(2),[],[],
            np.full(2,-.2),np.full(2,.2),np.ones(2),np.full(2,4.))
        self.assertTrue(audit['solved'])
        for d in range(3):np.testing.assert_allclose(following(0,d),continued[d],atol=1e-12)
        np.testing.assert_array_equal(held_value(curve,.6,1),np.zeros(2))

    def test_infeasible_initial_velocity_is_reported_not_clipped(self):
        state=(np.zeros(1),np.array([2.]),np.zeros(1))
        curve,audit=solve_endpoint(state,.5,np.zeros(1),[],[],
            np.array([-1.]),np.array([1.]),np.ones(1),np.ones(1))
        self.assertIsNone(curve)
        self.assertFalse(audit['solved'])

    def test_observation_target_does_not_overwrite_command(self):
        state=(np.array([.1]),np.array([.2]),np.array([.3]))
        curve,audit=solve_endpoint(state,.8,np.array([-.5]),[],[],
            np.array([-1.]),np.array([1.]),np.ones(1),np.full(1,3.))
        self.assertTrue(audit['solved'])
        for d in range(3):np.testing.assert_allclose(curve(0,d),state[d],atol=1e-12)
