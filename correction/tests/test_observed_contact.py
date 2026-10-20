from types import SimpleNamespace
import unittest
import mujoco
import numpy as np
from correction.src.dynamics.observed_contact import ContactCharts, observe_patches, allocate_observed


class ObservedContactTests(unittest.TestCase):
    def scene(self, floor=False):
        xml='''<mujoco><option timestep=".001"/>
        <worldbody><body name="object"><freejoint/><geom name="object_geom" type="box" size=".05 .05 .05" mass="1"/></body>
        <body name="finger" pos=".08 0 0"><joint type="slide" axis="1 0 0"/>
        <geom name="finger_geom" type="sphere" size=".01" mass=".1"/></body></worldbody></mujoco>'''
        if floor:xml=xml.replace('</worldbody>','<geom name="floor" type="plane" size="2 2 .1" pos="0 0 -.05"/></worldbody>')
        m=mujoco.MjModel.from_xml_string(xml)
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        s=SimpleNamespace(model=m,data=d,book_q=0,book_v=0,book_body=m.body('object').id,
            active_q=np.array([7]),active_v=np.array([6]),object_geoms={m.geom('object_geom').id},
            environment={m.geom('floor').id} if floor else set(),caps=np.array([10.]))
        points=[SimpleNamespace(body=m.body('finger').id)]
        chart=dict(robot_local=np.array([[-.01,0,0]]),robot_normal=np.array([[-1.,0,0]]),
            object_local=np.array([[.05,0,0]]),inward=np.array([[-1.,0,0]]),
            robot_geom=np.array([m.geom('finger_geom').id]),object_geom=np.array([m.geom('object_geom').id]))
        return s,points,chart

    def test_ground_truth_changes_target_without_moving_robot_or_object(self):
        s,points,chart=self.scene();q=s.data.qpos.copy()
        first=observe_patches(s,points,chart)
        self.assertAlmostEqual(first['gap'][0],.02)
        self.assertFalse(first['touching'][0])
        np.testing.assert_array_equal(s.data.qpos,q)
        s.data.qpos[0]=.01;mujoco.mj_forward(s.model,s.data)
        second=observe_patches(s,points,chart)
        self.assertAlmostEqual(second['gap'][0],.01)
        np.testing.assert_array_equal(first['positions'],second['positions'])
        s.data.qpos[0]=.0201;mujoco.mj_forward(s.model,s.data)
        third=observe_patches(s,points,chart)
        self.assertTrue(third['touching'][0])
        self.assertLess(third['signed_distance'][0],0)
        self.assertLess(third['measured'][0,0],0)

    def test_no_contact_means_no_allocated_load_not_zero_gravity_demand(self):
        s,points,chart=self.scene();obs=observe_patches(s,points,chart)
        f,audit=allocate_observed(s,obs,np.ones(1,bool),s.data.qpos[:3],s.data.qpos[3:7],
              np.zeros(1),np.zeros(1),np.array([20.]),np.array([1.]),np.zeros((1,3)),.1)
        np.testing.assert_allclose(f,0,atol=1e-6)
        self.assertFalse(audit['eligible'][0])
        self.assertGreater(abs(audit['wrench_residual'][2]),9.)
        self.assertFalse(audit['physical_success'])

    def test_wrong_face_is_not_counted_as_selected_contact(self):
        s,points,chart=self.scene();s.data.qpos[0]=.0201;mujoco.mj_forward(s.model,s.data)
        chart['inward']*=-1
        obs=observe_patches(s,points,chart)
        self.assertFalse(obs['touching'][0])
        np.testing.assert_array_equal(obs['measured'],0)

    def test_same_body_and_normal_at_remote_object_patch_is_not_selected_contact(self):
        s,points,chart=self.scene();s.data.qpos[0]=.0201;mujoco.mj_forward(s.model,s.data)
        self.assertTrue(observe_patches(s,points,chart)['touching'][0])
        chart['object_local'][0,1]=.04
        obs=observe_patches(s,points,chart)
        self.assertFalse(obs['touching'][0])
        self.assertLess(obs['signed_distance'][0],0)  # Bodies touch; selected patches do not.

    def test_sliding_support_can_carry_load_and_opposes_measured_slip(self):
        s,points,chart=self.scene(floor=True)
        s.data.qpos[2]=-.0001;s.data.qvel[0]=.02;mujoco.mj_forward(s.model,s.data)
        obs=observe_patches(s,points,chart)
        _,audit=allocate_observed(s,obs,np.zeros(1,bool),s.data.qpos[:3],s.data.qpos[3:7],
              np.zeros(1),np.zeros(1),np.array([20.]),np.array([1.]),np.zeros((1,3)),.1)
        self.assertTrue(audit['solved'])
        support=np.array(audit['environment_forces']).sum(axis=0)
        self.assertGreater(support[2],1.)
        self.assertLess(support[0],0.)
        self.assertLess(support@s.data.qvel[:3],0.)

    def mesh_scene(self):
        import itertools
        vertices=' '.join(str(v) for xyz in itertools.product((-.05,.05),repeat=3) for v in xyz)
        xml=f'''<mujoco><asset><mesh name="cube" vertex="{vertices}"/>
        <mesh name="finger_cube" vertex="{vertices}" scale=".2 .2 .2"/></asset>
        <worldbody><body name="object"><freejoint/><geom name="object_geom" type="mesh" mesh="cube" mass="1"/></body>
        <body name="finger" pos=".08 0 0"><joint type="slide" axis="1 0 0"/>
        <geom name="finger_geom" type="mesh" mesh="finger_cube" mass=".1"/></body></worldbody></mujoco>'''
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
        d.qpos[0]=.0201;mujoco.mj_forward(m,d)
        s=SimpleNamespace(model=m,data=d,book_q=0,book_v=0,book_body=m.body('object').id,
            active_q=np.array([7]),active_v=np.array([6]),object_geoms={m.geom('object_geom').id},
            environment=set(),caps=np.array([10.]))
        points=[SimpleNamespace(body=m.body('finger').id)]
        charts=ContactCharts(s,points)
        prior=charts.get(np.array([[-.01,0.,0.]]),np.array([[.05,.045,0.]]),np.array([[-1.,0.,0.]]))
        return s,points,charts,prior

    def test_explicit_patch_reselection_uses_real_surfaces_with_fixed_prior(self):
        s,points,charts,prior=self.mesh_scene();q=s.data.qpos.copy();original=prior['object_local'].copy()
        self.assertFalse(observe_patches(s,points,prior)['touching'][0])
        candidate,changes=charts.propose_nearby(prior,prior,np.ones(1,bool))
        self.assertEqual(len(changes),1)
        self.assertTrue(observe_patches(s,points,candidate)['touching'][0])
        self.assertLess(changes[0]['object_shift_m'],.08)
        np.testing.assert_array_equal(prior['object_local'],original)
        np.testing.assert_array_equal(s.data.qpos,q)
        stable,next_changes=charts.propose_nearby(candidate,prior,np.ones(1,bool))
        self.assertFalse(next_changes)  # Preserve an already matching patch.
        np.testing.assert_array_equal(stable['object_local'],candidate['object_local'])

    def test_contact_outside_fixed_prior_neighborhood_is_not_adopted(self):
        s,points,charts,prior=self.mesh_scene()
        candidate,changes=charts.propose_nearby(prior,prior,np.ones(1,bool),object_radius=.005)
        self.assertFalse(changes)
        self.assertFalse(observe_patches(s,points,candidate)['touching'][0])
        np.testing.assert_array_equal(candidate['object_local'],prior['object_local'])
