"""Independent execution and noisy-observation tests for the reduced benchmark."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from correction.src.preprocessing.controlled_horizon import controlled_carry
from correction.src.optimization.coupled_horizon import HorizonSettings, solve_horizon, sample_solution
from correction.src.dynamics.horizon_replay import replay_pads
from correction.src.dynamics.horizon_audit import audit_horizon


class HorizonReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref, cls.robot, cls.truth = controlled_carry(knots=21)
        cls.result = solve_horizon(cls.ref, cls.robot, HorizonSettings(max_seconds=30.))

    def test_execution_not_just_optimizer_status(self):
        self.assertTrue(self.result['success'], self.result['stages'])
        audit = audit_horizon(self.ref, self.robot, self.result)
        self.assertTrue(audit['accepted_nominal'], audit)
        nominal = replay_pads(self.ref, self.robot, self.result)
        baseline = replay_pads(self.ref, self.robot, {'q': self.ref.joint_reference}, force_guidance=False)
        for key in ('object_positions', 'object_wxyz', 'q', 'v'):
            np.testing.assert_array_equal(nominal[key][0], baseline[key][0])
        for record in (nominal, replay_pads(self.ref, self.robot, self.result, dt=.0005),
                       replay_pads(self.ref, self.robot, self.result, mass_scale=1.1),
                       replay_pads(self.ref, self.robot, self.result, friction=.5)):
            expected = np.column_stack([np.interp(record['times'], self.truth['times'], self.truth['positions'][:, j]) for j in range(3)])
            self.assertLess(np.max(np.linalg.norm(record['object_positions']-expected, axis=1)), .01)
            self.assertLess(Rotation.from_quat(record['object_wxyz'], scalar_first=True).magnitude().max(), np.deg2rad(5))
            self.assertEqual(record['max_object_applied_force'], 0.)
            self.assertLess(record['max_penetration_m'], .001)
            self.assertLessEqual(record['max_requested_actuator_fraction'], 1.)
        self.assertGreater(np.linalg.norm(baseline['object_positions'][-1]-self.ref.goal_position), .1)

    def test_observation_outlier_and_missing_interval_do_not_create_impulse_targets(self):
        ref, robot, truth = controlled_carry(knots=11, outlier=.10)
        # Keep a strong isolated outlier and omit another stretch entirely.
        keep = ~((ref.observation_times > 1.2) & (ref.observation_times < 1.5))
        ref.observation_times = ref.observation_times[keep]
        ref.observed_positions = ref.observed_positions[keep]
        ref.observed_wxyz = ref.observed_wxyz[keep]
        result = solve_horizon(ref, robot, HorizonSettings(max_seconds=30.))
        self.assertTrue(result['success'], result['stages'])
        sample = sample_solution(ref, result, truth['times'])
        self.assertLess(np.max(np.linalg.norm(sample['position']-truth['positions'], axis=1)), .015)
        self.assertLess(np.max(abs(result['forces'])), 20.)

    def test_weights_have_explicit_tradeoff_without_changing_task(self):
        ref, robot, truth = controlled_carry(knots=11)
        outcomes = []
        for weight in (.5, 2.):
            cfg = HorizonSettings(max_seconds=30., jerk_weight=.05*weight, force_weight=.03/weight)
            result = solve_horizon(ref, robot, cfg)
            self.assertTrue(result['success'], result['stages'])
            self.assertLessEqual(result['data_cost'], result['data_fit_optimum']+cfg.data_allowance+1e-5)
            self.assertLess(np.linalg.norm(result['position'][-1]-ref.goal_position), ref.goal_position_tolerance+1e-5)
            outcomes.append(result)
        self.assertLess(np.max(np.linalg.norm(outcomes[0]['position']-outcomes[1]['position'], axis=1)), .015)

    def test_rotational_observation_noise_is_not_an_angular_acceleration_target(self):
        ref, robot, truth = controlled_carry(knots=9, noise=0., outlier=0.)
        rng = np.random.default_rng(31)
        errors = rng.normal(0, np.deg2rad(2.), (len(ref.observation_times), 3))
        errors[len(errors)//2] += [0, np.deg2rad(20), 0]
        errors[0] = errors[-1] = 0
        ref.observed_wxyz = Rotation.from_rotvec(errors).as_quat(scalar_first=True)
        result = solve_horizon(ref, robot, HorizonSettings(max_seconds=30.))
        self.assertTrue(result['success'], result['stages'])
        sample = sample_solution(ref, result, truth['times'])
        estimated = Rotation.from_quat(sample['wxyz'], scalar_first=True).magnitude()
        self.assertLess(np.sqrt(np.mean(estimated**2)), np.sqrt(np.mean(np.sum(errors**2, axis=1))))
        self.assertLess(estimated.max(), np.deg2rad(6))

    def test_position_preload_and_normal_force_feedback_realize_a_grasp(self):
        for options in (dict(force_guidance=False, normal_preload_n=6.),
                        dict(force_guidance=True, normal_feedback_gain=.25)):
            record = replay_pads(self.ref, self.robot, self.result, **options)
            target = np.column_stack([np.interp(record['times'], self.truth['times'], self.truth['positions'][:, j]) for j in range(3)])
            self.assertLess(np.linalg.norm(record['object_positions']-target, axis=1).max(), .01)
            self.assertLess(record['max_penetration_m'], .001)
            self.assertLessEqual(record['max_requested_actuator_fraction'], 1.)
            self.assertEqual(record['max_object_applied_force'], 0.)
            np.testing.assert_array_equal(record['q'][0], self.ref.initial_q)
            np.testing.assert_array_equal(record['object_positions'][0], self.ref.initial_position)

    def test_controller_aware_plan_exports_and_executes_equivalent_position_targets(self):
        from correction.src.pass_traj.controller_reference import controller_reference
        config = HorizonSettings(max_seconds=30., servo_kp=(1200.,)*self.robot.n,
                                 servo_max_offset=.01, servo_offset_rate=.5)
        plan = solve_horizon(self.ref, self.robot, config)
        self.assertTrue(plan['success'], plan['stages'])
        audit = audit_horizon(self.ref, self.robot, plan)
        self.assertTrue(audit['accepted_nominal'], audit)
        exported = controller_reference(self.ref, self.robot, plan, kp=np.full(self.robot.n, 1200.), max_offset=.01)
        self.assertFalse(exported['ready_for_training'])
        np.testing.assert_allclose(1200*(exported['servo_q']-exported['physical_q'])+exported['feedforward_torque'],
                                   exported['desired_total_torque'], atol=1e-10)
        torque = replay_pads(self.ref, self.robot, plan)
        position = replay_pads(self.ref, self.robot, plan, contact_interface='position')
        np.testing.assert_allclose(position['object_positions'], torque['object_positions'], atol=1e-8)
        self.assertLess(position['max_conversion_residual'], 1e-10)
        expected = np.column_stack([np.interp(position['times'], self.truth['times'], self.truth['positions'][:, j]) for j in range(3)])
        self.assertLess(np.linalg.norm(position['object_positions']-expected, axis=1).max(), .01)
        self.assertEqual(position['max_object_applied_force'], 0.)
        # Dense acceptance must reject an incompatible command contract even if
        # the underlying physical trajectory is feasible.
        bad = dict(plan, settings=dict(plan['settings'], servo_max_offset=1e-9))
        self.assertFalse(audit_horizon(self.ref, self.robot, bad)['accepted_nominal'])

    def test_force_feedback_cannot_support_an_object_across_a_large_gap(self):
        from copy import deepcopy
        ref = deepcopy(self.ref)
        offset = np.tile([.5, 0., 0.], self.robot.count)
        ref.initial_q += offset
        command = dict(self.result, q=self.result['q']+offset)
        without = replay_pads(ref, self.robot, command, force_guidance=False)
        feedback = replay_pads(ref, self.robot, command, force_guidance=True, normal_feedback_gain=.25)
        np.testing.assert_allclose(feedback['object_positions'], without['object_positions'], atol=1e-12)
        np.testing.assert_allclose(feedback['q'], without['q'], atol=1e-12)
        self.assertEqual(feedback['max_object_applied_force'], 0.)


if __name__ == '__main__': unittest.main()
