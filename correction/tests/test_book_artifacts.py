"""Regression checks on the real experiment when its local artifacts exist."""
import json
from pathlib import Path
import unittest
import numpy as np


ROOT=Path(__file__).resolve().parents[1]/'outputs/tier1_book'


@unittest.skipUnless((ROOT/'fk_force_audit.json').exists(),'real Tier 1 artifacts not generated')
class BookArtifactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before=np.load(ROOT/'retargeted/contact_aware.npz')
        cls.after=np.load(ROOT/'dynamics_guided.npz')
        cls.metrics=json.loads((ROOT/'robot_repair_metrics.json').read_text())

    def test_object_time_and_base_preserved_exactly(self):
        for key in ('object_positions','object_wxyz','timestamps'):
            np.testing.assert_array_equal(self.before[key],self.after[key])
        np.testing.assert_array_equal(self.before['q'][:,:7],self.after['q'][:,:7])

    def test_only_local_arm_hand_trajectory_changes(self):
        lo,hi=self.metrics['modified_frame_range']
        np.testing.assert_array_equal(self.before['q'][:lo],self.after['q'][:lo])
        np.testing.assert_array_equal(self.before['q'][hi+1:],self.after['q'][hi+1:])
        names=json.loads(str(self.before['metadata_json']))['joint_names']
        fixed=[j+7 for j,n in enumerate(names) if not any(s in n for s in ('shoulder','elbow','wrist','hand'))]
        np.testing.assert_array_equal(self.before['q'][:,fixed],self.after['q'][:,fixed])

    def test_saved_motion_respects_joint_limits_and_velocity(self):
        self.assertEqual(self.metrics['speed_violating_transitions'],0)
        self.assertLessEqual(self.metrics['joint_limit_violation_rad'],1e-6)
        self.assertTrue(np.isfinite(self.after['q']).all())
        self.assertFalse(self.metrics['dynamics_tested'])

    def test_audit_reports_failures_without_hiding_them(self):
        arrays=np.load(ROOT/'fk_force_audit.npz')
        report=json.loads((ROOT/'fk_force_audit.json').read_text())
        for label in ('before','after'):
            residual=arrays[label+'_residual']
            feasible=(np.linalg.norm(residual[:,:3],axis=1)<=.02)&(np.linalg.norm(residual[:,3:],axis=1)<=.002)
            collision_ok=arrays[label+'_probe_max_m']<=.005
            self.assertEqual(report[label]['wrench_feasible_frames'],int(feasible.sum()))
            self.assertEqual(report[label]['frames_passing_wrench_and_probe_checks'],int((feasible&collision_ok).sum()))

    def test_patch_tracking_jacobian_matches_finite_difference(self):
        import pinocchio as pin
        from correction.robot_repair import PatchTrackingTask
        from v2d_retarget.core.robot import G1Kinematics
        kin=G1Kinematics()
        q=self.before['q'][155].copy()
        task=PatchTrackingTask(kin,'right_index_tip')
        normal=np.array([1.,2.,3.]);normal/=np.linalg.norm(normal)
        task.set_target(np.zeros(3),normal,1.3)
        kin.configuration.update(q)
        direction=np.random.default_rng(7).normal(size=kin.robot.model.nv)
        expected=task.compute_jacobian(kin.configuration)@direction
        eps=1e-7
        kin.configuration.update(pin.integrate(kin.robot.model,q,eps*direction))
        positive=task.compute_error(kin.configuration)
        kin.configuration.update(pin.integrate(kin.robot.model,q,-eps*direction))
        negative=task.compute_error(kin.configuration)
        np.testing.assert_allclose((positive-negative)/(2*eps),expected,atol=3e-6,rtol=1e-6)


if __name__=='__main__': unittest.main()
