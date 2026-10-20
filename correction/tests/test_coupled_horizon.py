"""Mechanics and identifiability tests; passing these is not a G1 success claim."""
import unittest

import casadi as ca
import numpy as np

from correction.src.dynamics.symbolic_mechanics import quat_mul, quat_rotation, rotation_quat
from correction.src.optimization.coupled_horizon import HorizonSettings, solve_horizon, sample_solution
from correction.src.preprocessing.controlled_horizon import controlled_carry
from correction.src.preprocessing.horizon_reference import ContactPatch


class MechanicsTests(unittest.TestCase):
    def test_quaternion_convention_matches_scipy(self):
        from scipy.spatial.transform import Rotation
        a, b = Rotation.from_rotvec([.2, -.1, .3]), Rotation.from_rotvec([-.1, .4, .2])
        expression = quat_rotation(quat_mul(ca.DM(a.as_quat(scalar_first=True)), ca.DM(b.as_quat(scalar_first=True))))
        np.testing.assert_allclose(np.asarray(ca.Function('rotation', [], [expression])()['o0']), (a*b).as_matrix(), atol=1e-12)
        zero = ca.MX.sym('z', 3)
        fun = ca.Function('small', [zero], [rotation_quat(zero), ca.jacobian(rotation_quat(zero), zero)])
        value, jac = fun(np.zeros(3))
        np.testing.assert_allclose(np.asarray(value).ravel(), [1, 0, 0, 0])
        self.assertTrue(np.isfinite(np.asarray(jac)).all())


class CoupledHorizonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref, cls.robot, cls.truth = controlled_carry(knots=11)
        cls.result = solve_horizon(cls.ref, cls.robot, HorizonSettings(max_seconds=30., max_iterations=350))

    def test_noisy_poses_yield_integrated_feasible_motion(self):
        r, ref = self.result, self.ref
        self.assertTrue(r['success'], r['stages'])
        self.assertLess(r['max_constraint_violation'], 1e-5)
        sample = sample_solution(ref, r, self.truth['times'])
        original_error = np.sqrt(np.mean((ref.observed_positions-self.truth['positions'])**2))
        corrected_error = np.sqrt(np.mean((sample['position']-self.truth['positions'])**2))
        self.assertLess(corrected_error, original_error)
        self.assertLess(np.linalg.norm(r['position'][-1]-ref.goal_position), ref.goal_position_tolerance+1e-6)
        np.testing.assert_allclose(r['velocity'][0], 0, atol=1e-6)
        np.testing.assert_allclose(r['angular_momentum'][0], 0, atol=1e-6)
        np.testing.assert_allclose(r['q'][0], ref.initial_q, atol=1e-6)
        self.assertLessEqual(np.max(abs(r['torque'])/self.robot.caps), .85+1e-5)

    def test_same_force_connects_robot_contact_and_object_momentum(self):
        r, ref = self.result, self.ref
        f = r['forces'].reshape(len(ref.times), 2, 3)
        acceleration = f.sum(1)/ref.mass+[0, 0, -9.81]
        np.testing.assert_allclose(np.diff(r['velocity'], axis=0),
                                  np.diff(ref.times)[:, None]*(acceleration[1:]+acceleration[:-1])/2, atol=2e-6)
        for k in range(len(ref.times)):
            rotation = np.asarray(ca.Function('r', [], [quat_rotation(ca.DM(r['wxyz'][k]))])()['o0'])
            for i, c in enumerate(ref.contacts):
                pad = r['q'][k, 3*i:3*i+3]+self.robot.radius*rotation@c.inward
                np.testing.assert_allclose(pad, r['position'][k]+rotation@r['anchors'][i], atol=2e-6)
        expected = self.robot.mass*(r['a']-np.tile([0, 0, -9.81], 2))+r['forces']
        np.testing.assert_allclose(r['torque'], expected, atol=1e-8)

    def test_inactive_or_top_only_contacts_cannot_lift(self):
        for kwargs in ({'contact': False}, {'top_only': True}):
            ref, robot, _ = controlled_carry(knots=7, **kwargs)
            result = solve_horizon(ref, robot, HorizonSettings(max_iterations=100, max_seconds=12., lexicographic=False))
            self.assertFalse(result['success'], 'Cannot create fictitious support or adhesive forces')

    def test_regularity_cannot_purchase_arbitrary_observation_error(self):
        result = self.result
        self.assertEqual(len(result['stages']), 2)
        if result['regularity_stage_succeeded']:
            self.assertLessEqual(result['data_cost'], result['data_fit_optimum']+result['settings']['data_allowance']+1e-5)

    def test_sticking_support_cannot_hide_motion_between_knots(self):
        ref, robot, _ = controlled_carry(knots=5, noise=0, outlier=0)
        ref.observed_positions[:] = ref.initial_position
        ref.goal_position = ref.initial_position.copy()
        ref.joint_reference[:] = ref.initial_q
        for sx in (-1, 1):
            for sy in (-1, 1):
                point = ref.half_size*np.array([sx, sy, -1])
                ref.contacts.append(ContactPatch(f'support{sx}{sy}', point, np.array([0., 0., 1.]),
                    np.array([[1., 0.], [0., 1.], [0., 0.]]), np.ones(5, bool), radius=0.,
                    support_z=ref.initial_position[2]-ref.half_size[2]))
        result = solve_horizon(ref, robot, HorizonSettings(max_seconds=20., max_iterations=200))
        self.assertTrue(result['success'], result['stages'])
        sample = sample_solution(ref, result, np.linspace(0, 2, 101))
        np.testing.assert_allclose(sample['position'], np.tile(ref.initial_position, (101, 1)), atol=1e-6)
        np.testing.assert_allclose(sample['velocity'], 0, atol=1e-6)

    def test_uncertainty_is_explicit_and_invalid_confidence_is_rejected(self):
        ref, robot, _ = controlled_carry(knots=5)
        ref.position_sigma = np.full((len(ref.observation_times), 3), .01)
        ref.rotation_sigma = np.array([.05, .10, .20])
        ref.validate(robot)
        ref.position_sigma[3, 0] = 0.
        with self.assertRaisesRegex(ValueError, 'position_sigma'):
            ref.validate(robot)

    def test_candidates_cannot_win_by_changing_initial_state(self):
        from correction.src.optimization.horizon_candidates import solve_candidates
        ref, robot, _ = controlled_carry(knots=5)
        other, other_robot, _ = controlled_carry(knots=5)
        other.initial_velocity[2] = 1.
        with self.assertRaisesRegex(ValueError, 'initial_velocity'):
            solve_candidates([('first', ref, robot), ('cheating', other, other_robot)],
                             HorizonSettings(max_iterations=100), seconds=30.)

    def test_contacts_cannot_supply_force_from_outside_object_geometry(self):
        ref, robot, _ = controlled_carry(knots=5)
        ref.contacts[0].center[1] -= .01
        with self.assertRaisesRegex(ValueError, 'physical face'):
            ref.validate(robot)

    def test_dense_audit_detects_unmodelled_approach_penetration(self):
        from correction.src.dynamics.horizon_audit import audit_horizon
        ref, robot, _ = controlled_carry(knots=11)
        for contact in ref.contacts:
            contact.active[:] = False
        bad = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self.result.items()}
        # Place both inactive pads inside the object while preserving a plausible
        # object path. A solver success flag must not override this geometry.
        bad['q'] = np.tile(bad['position'], (1, 2))
        bad['v'] = np.tile(bad['velocity'], (1, 2))
        bad['forces'][:] = 0
        audit = audit_horizon(ref, robot, bad)
        self.assertLess(audit['min_inactive_box_clearance_m'], -.02)
        self.assertIsNone(audit['min_active_normal_separation_m'])
        self.assertFalse(audit['accepted_nominal'])


if __name__ == '__main__':
    unittest.main()
