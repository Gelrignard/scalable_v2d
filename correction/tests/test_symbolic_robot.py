"""Independent G1 model checks against MuJoCo, including a moving root."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]/'outputs/tier1_book/retargeted/contact_aware.npz'
AVAILABLE = ROOT.exists() and all(importlib.util.find_spec(k) for k in ('casadi', 'mujoco', 'pinocchio'))


@unittest.skipUnless(AVAILABLE, 'optional G1 runtime and local book reference required')
class SymbolicRobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from correction.src.preprocessing import retarget_bridge
        from v2d_retarget.core.robot import G1Kinematics
        from correction.src.dynamics.supported_robot import SupportedRobotScene
        from correction.src.dynamics.symbolic_mechanics import BodyPoint, SymbolicRobot
        import trimesh
        with np.load(ROOT) as data:
            cls.original = data['q'][100].copy()
            mesh = trimesh.load(str(data['object_mesh']), force='mesh')
            support = json.loads(str(data['metadata_json']))['source']['support_surface']
        cls.scene = SupportedRobotScene(G1Kinematics(), mesh.bounds, support, cls.original,
                                       move_waist=True, moving_fixture=True)
        points = [BodyPoint(int(cls.scene.model.site_bodyid[i]), cls.scene.model.site_pos[i].copy())
                  for i in cls.scene.tip_ids]
        cls.robot = SymbolicRobot(cls.scene, points)

    def test_forward_kinematics_and_contact_jacobian(self):
        s, r = self.scene, self.robot
        s.set_state(self.original, [10, 10, 2], [1, 0, 0, 0])
        base = np.zeros(25); base[:3] = self.original[:3]; base[3:7] = self.original[[6, 3, 4, 5]]
        q = self.original[7+s.movable]
        np.testing.assert_allclose(np.asarray(r.kinematics(q, base)).T, s.tips_and_jacobian()[0], atol=1e-8)
        np.testing.assert_allclose(np.asarray(r.jacobian(q, base)), s.tips_and_jacobian()[1].reshape(18, -1), atol=1e-8)

    def test_newton_euler_matches_independent_mass_bias_and_passive_terms(self):
        import mujoco
        s, r = self.scene, self.robot
        rng = np.random.default_rng(17)
        for moving in (False, True):
            s.set_state(self.original, [10, 10, 2], [1, 0, 0, 0])
            base = np.zeros(25); base[:3] = self.original[:3]; base[3:7] = self.original[[6, 3, 4, 5]]
            if moving:
                base[7:19] = rng.normal(0, .3, 12)
            v, a = rng.normal(0, .3, (2, r.n))
            rotation = Rotation.from_quat(base[3:7], scalar_first=True).as_matrix()
            s.data.qvel[s.active_v] = v
            s.data.qvel[s.root_v:s.root_v+6] = np.r_[base[7:10], rotation.T@base[10:13]]
            mujoco.mj_forward(s.model, s.data)
            acc = np.zeros(s.model.nv)
            acc[s.active_v] = a
            acc[s.root_v:s.root_v+6] = np.r_[base[13:16], rotation.T@base[16:19]]
            inertial = np.zeros(s.model.nv)
            mujoco.mj_mulM(s.model, s.data, inertial, acc)
            expected = (inertial+s.data.qfrc_bias-s.data.qfrc_passive)[s.active_v]
            actual = np.asarray(r.inverse_dynamics(self.original[7+s.movable], v, a, base)).ravel()
            np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=1e-8)

    def test_horizon_controller_does_not_apply_forces_across_a_missing_contact(self):
        s = self.scene
        times = np.linspace(0, .1, 11)
        commands = np.tile(self.original, (len(times), 1))
        initial = dict(position=np.array([10., 10., 2.]), wxyz=np.array([1., 0., 0., 0.]),
                       linear_velocity=np.zeros(3), angular_velocity_world=np.zeros(3))
        common = dict(initial_object_state=initial, initial_velocity=np.zeros(self.robot.n),
                      command_velocity=np.zeros((len(times), self.robot.n)), inertia_ff=True,
                      force_points=self.robot.contacts, smooth_fixture=True)
        # Deliberately inconsistent observed positions must not create initial
        # object velocity; the known physical initial state wins.
        observed = np.tile([10., 10., 2.1], (len(times), 1))
        observed[1, 2] += .02
        rotation = np.tile([1., 0., 0., 0.], (len(times), 1))
        before = s.rollout(commands, observed, rotation, times, **common)
        after = s.rollout(commands, observed, rotation, times,
                          force_ff=np.full((len(times), 6, 3), 5.), **common)
        np.testing.assert_allclose(before['q'], after['q'], atol=1e-12)
        np.testing.assert_allclose(before['object_positions'], after['object_positions'], atol=1e-12)
        self.assertAlmostEqual(after['object_positions'][-1, 2], 2-.5*9.81*.1**2, delta=.001)
        self.assertEqual(after['max_object_applied_force'], 0.)

    def test_new_controller_paths_preserve_g1_free_space_motion_and_object_fall(self):
        from correction.src.dynamics.contact_controller import ContactControllerConfig
        s = self.scene
        times = np.linspace(0, .1, 11)
        q = np.tile(self.original, (len(times), 1))
        p = np.tile([10., 10., 2.], (len(times), 1))
        r = np.tile([1., 0., 0., 0.], (len(times), 1))
        common = dict(force_points=self.robot.contacts, command_velocity=np.zeros((len(times), self.robot.n)),
                      initial_velocity=np.zeros(self.robot.n), inertia_ff=True, kp_arm=300., kp_hand=20.)
        baseline = s.rollout(q, p, r, times, **common)
        for interface in ('torque', 'position'):
            result = s.rollout(q, p, r, times, **common, force_ff=np.full((len(times), 6, 3), 5.),
                contact_controller=ContactControllerConfig(interface=interface, feedback_gain=.25),
                contact_normals_local=np.tile([1., 0., 0.], (len(times), 6, 1)),
                contact_active=np.ones((len(times), 6), bool))
            np.testing.assert_allclose(result['q'], baseline['q'], atol=1e-10)
            np.testing.assert_allclose(result['object_positions'], baseline['object_positions'], atol=1e-12)
            np.testing.assert_array_equal(result['controller_contact_torque'], 0.)
            np.testing.assert_array_equal(result['controller_feedback_correction_n'], 0.)
            self.assertEqual(len(result['controller_servo_q']), len(times))
            self.assertEqual(result['max_object_applied_force'], 0.)

    def test_contact_surface_faces_object_even_when_nearest_side_is_wrong(self):
        from correction.src.preprocessing.g1_horizon import surface_point
        s = self.scene
        s.set_state(self.original, [10, 10, 2], [1, 0, 0, 0])
        body = self.robot.contacts[0].body
        target = s.data.xpos[body]+[0, 0, .2]
        point, info = surface_point(s, body, target, np.array([0., 0., -1.]))
        self.assertEqual(point.body, body)
        self.assertGreater(info['inward_alignment'], .65)
        self.assertTrue(np.isfinite(point.local).all())


if __name__ == '__main__':
    unittest.main()
