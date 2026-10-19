"""Physical invariants and honest artifact checks for the supported G1 bench."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

OUTPUT=Path(__file__).resolve().parents[1]/'outputs/supported_g1_20261019'
AVAILABLE=all(importlib.util.find_spec(m) for m in ('mujoco','pinocchio')) and (OUTPUT/'reference.npz').exists()


@unittest.skipUnless(AVAILABLE,'requires optional G1 runtime and generated supported robot experiment')
class SupportedRobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from correction.src.preprocessing import retarget_bridge
        from v2d_retarget.core.robot import G1Kinematics
        from correction.src.dynamics.supported_robot import SupportedRobotScene
        with np.load(OUTPUT/'reference.npz',allow_pickle=False) as data:
            cls.ref={k:data[k].copy() for k in data.files}
        cls.kin=G1Kinematics()
        cls.scene=SupportedRobotScene(cls.kin,cls.ref['bounds'],str(cls.ref['support_path']),cls.ref['fixed_q'],move_waist=True)

    def test_supported_robot_has_free_book_and_correct_kinematics(self):
        s,r=self.scene,self.ref
        self.assertEqual(s.model.neq,0,'No weld or equality constraint may hold the book')
        self.assertEqual(len(s.active_v),31)
        self.assertTrue(all(not any(x in self.kin.joint_names[i] for x in ('hip','knee','ankle')) for i in s.movable))
        s.set_state(r['initial_q'],r['object_positions'][0],r['object_wxyz'][0])
        poses=self.kin.poses(r['initial_q'],s.tip_names)
        tips,jac=s.tips_and_jacobian()
        np.testing.assert_allclose(tips,np.array([p.translation for p in poses]),atol=2e-6)
        column=5;epsilon=1e-6
        q=r['initial_q'].copy();q[7+s.movable[column]]+=epsilon
        s.set_state(q,r['object_positions'][0],r['object_wxyz'][0])
        np.testing.assert_allclose((s.tips_and_jacobian()[0]-tips)/epsilon,jac[:,:,column],atol=1e-5)

    def test_book_falls_without_contacts_while_robot_base_stays_fixed(self):
        s,r=self.scene,self.ref
        times=np.linspace(0,.1,11)
        result=s.rollout(np.tile(r['initial_q'],(11,1)),np.tile([10.,10.,2.],(11,1)),
                         np.tile([1.,0.,0.,0.],(11,1)),times)
        self.assertAlmostEqual(result['object_positions'][-1,2],2-.5*9.81*.1**2,delta=.0006)
        self.assertEqual(result['max_object_applied_force'],0.)
        np.testing.assert_array_equal(result['q'][:,:19],np.tile(r['fixed_q'][:19],(11,1)))

    def test_saved_commands_reproduce_physics_without_hiding_tracking_failure(self):
        from correction.src.optimization.robot_manipulation import tracking_accepted
        s,r=self.scene,self.ref
        result=s.rollout(r['q'],r['object_positions'],r['object_wxyz'],r['timestamps'],
                         force_ff=r['force_ff'],kp_arm=300.,kp_hand=20.,inertia_ff=True,
                         initial_q=r['initial_q'],initial_velocity=r['initial_velocity'])
        with np.load(OUTPUT/'corrected.npz',allow_pickle=False) as saved:
            np.testing.assert_allclose(result['object_positions'],saved['object_positions'],atol=1e-12)
            np.testing.assert_allclose(result['q'],saved['q'],atol=1e-12)
        self.assertLessEqual(np.max(np.abs(result['torques'])/result['torque_caps']),1.+1e-12)
        self.assertEqual(result['max_object_applied_force'],0.)
        self.assertLess(result['max_penetration_m'],.001)
        self.assertFalse(tracking_accepted(result),'Current robot trial must not be advertised as passing tracking')
        report=json.loads((OUTPUT/'report.json').read_text())
        self.assertFalse(report['accepted_tracking'])
        self.assertFalse(report['ready_for_training'])
        self.assertLess(report['corrected']['rms_rotation_error_deg'],report['wrench_seed']['rms_rotation_error_deg'])
        self.assertGreater(report['corrected']['max_requested_torque_fraction'],1.,'Report saturation honestly')
        for label in ('baseline','geometry_only','wrench_seed'):
            with np.load(OUTPUT/f'{label}.npz',allow_pickle=False) as data:
                np.testing.assert_array_equal(data['q'][0],result['q'][0])
                np.testing.assert_array_equal(data['joint_velocities'][0],result['joint_velocities'][0])
                np.testing.assert_array_equal(data['object_positions'][0],result['object_positions'][0])

if __name__=='__main__':unittest.main()
