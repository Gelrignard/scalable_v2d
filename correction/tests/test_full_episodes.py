"""Check supported moving-base invariants and full-episode artifact honesty."""
import importlib.util
import json
from pathlib import Path
import random
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]/'outputs/random_three_full_20261019'
AVAILABLE = all(importlib.util.find_spec(x) for x in ('mujoco','pinocchio')) and (ROOT/'episode_000018/reference.npz').exists()


@unittest.skipUnless(AVAILABLE,'optional G1 runtime and generated full episodes required')
class FullEpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from correction.src.preprocessing import retarget_bridge
        from v2d_retarget.core.robot import G1Kinematics
        from correction.src.dynamics.supported_robot import SupportedRobotScene
        from correction.src.preprocessing.convex_object import decompose
        path = ROOT/'episode_000018'
        with np.load(path/'retargeted/contact_aware.npz') as data:
            cls.raw = {k:data[k].copy() for k in data.files}
        import trimesh
        cls.kin = G1Kinematics()
        meta = json.loads(str(cls.raw['metadata_json']))
        report = json.loads((path/'report.json').read_text())
        mesh = trimesh.load(str(cls.raw['object_mesh']),force='mesh')
        cls.scene = SupportedRobotScene(cls.kin,mesh.bounds,meta['source']['support_surface'],cls.raw['q'][0],
            move_waist=True,moving_fixture=True,object_parts=decompose(str(cls.raw['object_mesh']),path/'collision'),mass=2.,
            floor_z=report['floor_z_m'])

    def test_fixture_cannot_hold_object_and_legs_cannot_collide(self):
        s = self.scene
        self.assertEqual(s.model.neq,1)
        self.assertNotIn(s.book_body,s.model.eq_obj1id)
        self.assertNotIn(s.book_body,s.model.eq_obj2id)
        for i in range(s.model.ngeom):
            name = s.model.geom(i).name
            if any(x in name for x in ('hip','knee','ankle')):
                self.assertEqual(s.model.geom_contype[i],0)
                self.assertEqual(s.model.geom_conaffinity[i],0)
        q = self.raw['q'][0].copy(); q[:3] += [.3,.2,.1]
        s.set_state(q,[10.,10.,2.],[1.,0.,0.,0.])
        expected = np.array([p.translation for p in self.kin.poses(q,s.tip_names)])
        np.testing.assert_allclose(s.tips_and_jacobian()[0],expected,atol=2e-6)

    def test_free_fall_while_pelvis_fixture_translates(self):
        times = np.linspace(0,.2,21)
        q = np.tile(self.raw['q'][0],(len(times),1)); q[:,0] += .1*times
        result = self.scene.rollout(q,np.tile([10.,10.,2.],(len(times),1)),
            np.tile([1.,0.,0.,0.],(len(times),1)),times)
        self.assertAlmostEqual(result['object_positions'][-1,2],2-.5*9.81*.2**2,delta=.0011)
        self.assertLess(np.max(np.linalg.norm(result['q'][:,:3]-q[:,:3],axis=1)),.005)
        self.assertEqual(result['max_object_applied_force'],0.)

    def test_entire_random_sample_has_common_initial_states(self):
        from correction.src.optimization.robot_manipulation import tracking_accepted
        selection = json.loads((ROOT/'selection.json').read_text())
        ids = random.Random(selection['seed']).sample(selection['eligible_episode_ids'],3)
        self.assertEqual(ids,[e['episode_index'] for e in selection['selected']])
        for episode in selection['selected']:
            path = ROOT/f'episode_{episode["episode_index"]:06d}'
            if not (path/'report.json').exists(): self.fail(f'Missing episode {episode["episode_index"]}')
            with np.load(path/'reference.npz') as ref, np.load(path/'baseline.npz') as before, np.load(path/'corrected.npz') as after:
                n = episode['frames']
                np.testing.assert_array_equal(ref['frame_ids'],np.arange(n))
                self.assertEqual(len(before['q']),n); self.assertEqual(len(after['q']),n)
                for key in ('q','object_positions','object_wxyz','joint_velocities'):
                    np.testing.assert_array_equal(before[key][0],after[key][0])
                    self.assertTrue(np.isfinite(after[key]).all())
                np.testing.assert_array_equal(before['timestamps'],after['timestamps'])
                self.assertEqual(float(after['max_object_applied_force']),0.)
                self.assertLessEqual(np.max(abs(after['torques'])/after['torque_caps']),1.+1e-12)
                report = json.loads((path/'report.json').read_text())
                self.assertEqual(report['accepted_tracking'],tracking_accepted(after))
                self.assertFalse(report['ready_for_training'])

if __name__ == '__main__':unittest.main()
