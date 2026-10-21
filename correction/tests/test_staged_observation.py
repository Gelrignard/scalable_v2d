"""Force-stage contract tests; physical improvement requires a full replay."""
import unittest
import mujoco
import numpy as np

from correction.src.optimization.observed_force_update import force_update
from correction.src.dynamics.contact_controller import G1ContactController
from correction.src.dynamics.staged_observation import StagedObservationController, StagedObservationConfig
from correction.tests.test_guarded_residual import fixture


def problem(**updates):
    args = dict(points=np.array([[0., 0., 0.]]), normals=np.array([[0., 0., 1.]]),
        touching=np.array([True]), com=np.zeros(3), jacobians=np.eye(3)[None],
        nominal_force=np.array([[0., 0., 2.]]), nominal_torque=np.zeros(3), caps=np.full(3, 10.),
        joint_position=np.zeros(3), lower=np.full(3, -1.), upper=np.ones(3), goal=np.array([0., 0., 2., 0., 0., 0.]),
        previous=np.zeros((1, 3)), dt=.001, limit=.75, rate=10., friction=.8, normal_cap=15.)
    args.update(updates)
    return force_update(**args)


class ForceUpdateTests(unittest.TestCase):
    def test_delta_rate_and_static_cone(self):
        value, audit = problem()
        self.assertGreater(value[0, 2], .009)
        self.assertLessEqual(abs(value).max(), .010001)
        self.assertFalse(audit['rate_conflict'])
        self.assertLessEqual(audit['static_violation'], 1e-6)

    def test_separated_channel_has_zero_wrench_authority_and_unloads(self):
        value, audit = problem(touching=np.array([False]), previous=np.array([[0., 0., .1]]))
        np.testing.assert_allclose(audit['delta_wrench'], 0., atol=1e-12)
        self.assertAlmostEqual(value[0, 2], .09, places=5)
        np.testing.assert_allclose(audit['unmet_wrench'], [0., 0., 2., 0., 0., 0.])

    def test_full_nominal_torque_may_not_exceed_capacity_or_worsen_violation(self):
        for torque in (9.999, 12.):
            with self.subTest(torque=torque):
                value, audit = problem(nominal_torque=np.array([0., 0., torque]))
                self.assertLessEqual(torque+value[0, 2], max(10., torque)+1e-6)

    def test_joint_stop_blocks_outward_residual(self):
        value, audit = problem(joint_position=np.array([0., 0., .995]))
        self.assertLessEqual(value[0, 2], 1e-6)

    def test_constraint_change_rate_conflict_is_exposed(self):
        value, audit = problem(joint_position=np.array([0., 0., 1.]), previous=np.array([[0., 0., .1]]))
        self.assertTrue(audit['rate_conflict'])
        self.assertLessEqual(value[0, 2], 1e-6)

    def test_moment_uses_actual_contact_lever(self):
        value, audit = problem(points=np.array([[.2, 0., 0.]]))
        np.testing.assert_allclose(audit['delta_wrench'][3:], np.cross([.2, 0., 0.], value[0]), atol=1e-12)

    def test_preserves_nominal_preload_when_wrench_wants_to_unload(self):
        value, audit = problem(goal=np.array([0., 0., -10., 0., 0., 0.]), dt=1.)
        self.assertGreaterEqual(2.+value[0, 2], 1.5-1e-6)


class ObservedControllerTests(unittest.TestCase):
    def controller(self, gain):
        scene, old, guard = fixture(inertial=True)
        args = dict(scene=scene, points=old.points, normals_local=old.normals, active=old.active,
                    times=old.times, config=old.config, kp=old.kp)
        base = G1ContactController(**args)
        ctl = StagedObservationController(**args, observed_config=StagedObservationConfig(gain=gain),
            kd=np.array([2.]), planning_poses=dict(object_positions=guard.plan['desired_p'],
                                                  object_wxyz=guard.plan['desired_r']))
        return scene, base, ctl

    def test_disabled_path_exact_and_observation_read_only(self):
        scene, base, ctl = self.controller(0.)
        initial = scene.data.qpos.copy()
        args = (0., np.zeros(1), np.zeros(1), np.array([[-2., 0., 0.]]), np.ones(1))
        a, b = base.step(*args), ctl.step(*args)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(scene.data.qpos, initial)
        np.testing.assert_array_equal(scene.data.qfrc_applied, 0.)
        self.assertEqual(ctl.solves, 0)

    def test_actual_state_changes_goal_without_observed_acceleration(self):
        scene, base, ctl = self.controller(1.)
        first = ctl.desired_delta_wrench(.05)
        scene.data.qpos[0] += .02
        # Deliberately leave cached Cartesian geometry stale, as after mj_step.
        second = ctl.desired_delta_wrench(.05)
        self.assertLess(second[0], first[0])
        before = scene.data.qpos.copy()
        ctl.step(.05, np.zeros(1), np.zeros(1), np.array([[-2., 0., 0.]]), np.zeros(1))
        np.testing.assert_array_equal(scene.data.qpos, before)
        np.testing.assert_array_equal(scene.data.qfrc_applied, 0.)

    def test_current_kinematics_without_mutating_physics_or_contact_cache(self):
        scene, base, ctl = self.controller(1.)
        scene.data.qpos[7] += .2
        cached = scene.data.xpos.copy()
        ctl.step(.05, np.zeros(1), np.zeros(1), np.array([[-2., 0., 0.]]), np.zeros(1))
        np.testing.assert_array_equal(scene.data.xpos, cached)
        fresh = mujoco.MjData(scene.model); fresh.qpos[:] = scene.data.qpos
        fresh.mocap_pos[:] = scene.data.mocap_pos; fresh.mocap_quat[:] = scene.data.mocap_quat
        mujoco.mj_forward(scene.model, fresh)
        np.testing.assert_allclose(ctl.kinematic_data.xpos, fresh.xpos, atol=1e-12)
        np.testing.assert_allclose(ctl.kinematic_data.xipos, fresh.xipos, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
