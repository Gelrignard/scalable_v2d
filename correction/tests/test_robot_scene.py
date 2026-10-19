"""Optional real-asset tests for geometry transforms, seams and free dynamics."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'correction/outputs/reorganization_check_20261017'
AVAILABLE = all(importlib.util.find_spec(m) for m in ('mujoco', 'pinocchio')) and (DATA/'human_motion.npz').exists()


@unittest.skipUnless(AVAILABLE, 'requires CPU robot runtime and generated book input')
class RobotSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from correction.src.preprocessing.book_reference import load_book_reference
        from correction.src.dynamics.robot_scene import RobotScene
        from v2d_retarget.core.robot import G1Kinematics
        cls.motion, cls.before, meta, _, cls.surface, _ = load_book_reference(DATA, True)
        cls.kin = G1Kinematics()
        cls.scene = RobotScene(cls.kin, cls.surface.mesh.bounds, meta['source']['support_surface'])

    def test_mujoco_and_pinocchio_body_frames_agree(self):
        names = ['pelvis', 'left_wrist_yaw_link', 'right_hand_middle_1_link']
        for frame in (400, 500, 600):
            q = self.before['q'][frame]
            self.scene.set_state(q, self.motion.object_positions[frame], self.motion.object_wxyz[frame])
            poses = self.kin.poses(q, names)
            for name, pose in zip(names, poses):
                body = self.scene.model.body(name).id
                # MuJoCo's URDF->MJCF serialization rounds fixed transforms.
                np.testing.assert_allclose(self.scene.data.xpos[body], pose.translation, atol=2e-6)
                np.testing.assert_allclose(self.scene.data.xmat[body].reshape(3, 3), pose.rotation, atol=5e-6)

    def test_repeated_rollouts_reset_time_and_object_falls_without_assistance(self):
        q = np.repeat(self.before['q'][400:401], 7, axis=0)
        p = np.tile([10., 10., 2.], (7, 1))
        r = np.tile([1., 0., 0., 0.], (7, 1))
        times = np.arange(7)/60
        first = self.scene.rollout(q, p, r, times)
        second = self.scene.rollout(q, p, r, times)
        self.assertTrue(first['completed'] and second['completed'])
        self.assertEqual(first['steps'], second['steps'])
        self.assertGreater(second['steps'], 0)
        np.testing.assert_allclose(first['object_positions'], second['object_positions'], atol=1e-10)
        self.assertAlmostEqual(first['object_positions'][-1, 2], 2-.5*9.81*.1**2, delta=.003)

    def test_temporal_projection_obeys_acceleration_and_endpoints(self):
        from correction.src.optimization.window_robot import project_temporal, temporal_operators
        ids = np.array([self.kin.joint_names.index('left_wrist_pitch_joint')])
        q = self.before['q'][358:643].copy()
        times = self.motion.timestamps[358:643]
        output, status = project_temporal(self.kin, q, q, times, ids, time_limit=2.)
        self.assertIsNotNone(output, status)
        v, a = temporal_operators(times, 1)
        angle = output[:, 7+ids].ravel()
        self.assertLessEqual(np.max(np.abs(a@angle)), 80.001)
        self.assertLessEqual(np.max(np.abs(v@angle)), .9*self.kin.robot.model.velocityLimit[6+ids[0]]+.001)
        np.testing.assert_array_equal(output[:2], q[:2])
        np.testing.assert_array_equal(output[-2:], q[-2:])

    def test_table_collision_is_detected_in_original_reference(self):
        self.scene.set_state(self.before['q'][450], self.motion.object_positions[450], self.motion.object_wxyz[450])
        contacts = self.scene.contacts()
        self.assertTrue(any('table' in c['pair'] and 'palm' in c['pair'] and c['distance'] < -.01 for c in contacts))

    def test_contact_separation_jacobian_has_correct_sign(self):
        q = self.before['q'][450].copy()
        p, r = self.motion.object_positions[450], self.motion.object_wxyz[450]
        ids = np.arange(len(self.kin.joint_names))
        self.scene.set_state(q, p, r)
        pair = 'table / left_hand_index_1_link_collision_0'
        contact = next(c for c in self.scene.contacts(ids) if c['pair'] == pair)
        jac = contact['jacobian']
        direction = jac/np.linalg.norm(jac)
        distances = []
        eps = 1e-6
        for sign in (1, -1):
            trial = q.copy()
            trial[7:] += sign*eps*direction
            self.scene.set_state(trial, p, r)
            distances.append(min(c['distance'] for c in self.scene.contacts() if c['pair'] == pair))
        self.assertAlmostEqual((distances[0]-distances[1])/(2*eps), float(jac@direction), delta=2e-4)


if __name__ == '__main__':
    unittest.main()
