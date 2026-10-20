"""Physical invariants for full-episode independent contact replays."""
import unittest
import numpy as np
from correction.src.dynamics.independent_contacts import IndependentContacts
from correction.src.dynamics.contact_controller import ContactControllerConfig


class IndependentContactsTests(unittest.TestCase):
    def test_patch_preload_moves_only_pad_before_contact(self):
        scene = IndependentContacts(np.array([[-.05]*3, [.05]*3]), floor_z=-10., pad_count=1)
        scene.model.opt.gravity[:] = 0.
        t = np.linspace(0, .005, 6)
        p = np.tile([0.,0.,1.], (len(t),1))
        r = np.tile([1.,0.,0.,0.], (len(t),1))
        targets = np.tile([[[-.07,0.,1.]]], (len(t),1,1))
        normal = np.tile([[[1.,0.,0.]]], (len(t),1,1))
        options = dict(planned_forces=3*normal, inward_normals_local=normal,
                       active=np.ones((len(t),1),bool),
                       config=ContactControllerConfig(patch_distance_gate=True, near_surface_m=.025))
        legacy = scene.rollout(t,targets,p,r,initial_pads=targets[0],**options)
        preload = scene.rollout(t,targets,p,r,initial_pads=targets[0],
            object_local_points=np.tile([[[-.05,0.,0.]]],(len(t),1,1)),**options)
        self.assertFalse(legacy['force_enabled'].any())
        self.assertTrue(preload['force_enabled'].all())
        self.assertFalse(preload['contact_counts'].any())
        self.assertGreater(preload['pad_positions'][-1,0,0], legacy['pad_positions'][-1,0,0])
        np.testing.assert_array_equal(preload['object_positions'], p)
        self.assertEqual(preload['max_object_applied_force'], 0.)
        np.testing.assert_allclose(preload['contact_gate_gap_m'][0], [.014])

    def test_explicit_inertia_and_arbitrary_surface_channel_count(self):
        scene=IndependentContacts(np.array([[-.05]*3,[.05]*3]),pad_count=4,mass=2.,
            object_inertial=dict(com_local_m=[.01,0.,0.],diagonal_kg_m2=[.02,.03,.04]))
        self.assertEqual(len(scene.pad_index),4)
        self.assertEqual(scene.model.body_mass[scene.book_body],2.)
        np.testing.assert_allclose(scene.model.body_ipos[scene.book_body],[.01,0.,0.])
        np.testing.assert_allclose(scene.model.body_inertia[scene.book_body],[.02,.03,.04])
        t=np.linspace(0.,.01,11)
        targets=np.tile([[1.+i,0.,1.] for i in range(4)],(len(t),1,1))
        result=scene.rollout(t,targets,np.tile([0.,0.,1.],(len(t),1)),
            np.tile([1.,0.,0.,0.],(len(t),1)),initial_pads=targets[0])
        self.assertEqual(result['pad_positions'].shape,(len(t),4,3))
        self.assertEqual(result['max_object_applied_force'],0.)

    def test_force_plan_cannot_move_a_distant_object(self):
        scene = IndependentContacts(np.array([[-.05]*3, [.05]*3]), floor_z=-10.)
        t = np.linspace(0, .1, 101)
        p = np.tile([0, 0, 2.], (len(t), 1))
        r = np.tile([1., 0, 0, 0], (len(t), 1))
        targets = np.tile([[[10.+i, 0, 2.] for i in range(6)]], (len(t), 1, 1))
        normal = np.tile([[[2., 0, 0]]*6], (len(t), 1, 1))
        original_normal=normal.copy()
        result = scene.rollout(t, targets, p, r, initial_pads=targets[0],
            planned_forces=100*normal, inward_normals_local=normal, active=np.ones((len(t), 6), bool))
        self.assertEqual(scene.model.neq, 0)
        self.assertEqual(result['max_object_applied_force'], 0.)
        self.assertFalse(np.any(result['force_enabled']))
        np.testing.assert_array_equal(result['commanded_forces'], 0.)
        self.assertAlmostEqual(result['object_positions'][-1, 2], 2-.5*9.81*.1**2, delta=.0006)
        self.assertLessEqual(result['max_motor_force_n'], scene.force_limit+1e-9)
        np.testing.assert_array_equal(normal,original_normal)

    def test_contact_force_direction_and_object_momentum(self):
        scene = IndependentContacts(np.array([[-.05]*3, [.05]*3]), floor_z=-10.)
        scene.model.opt.gravity[:] = 0.
        t = np.linspace(0, .1, 101)
        p = np.tile([0, 0, 1.], (len(t), 1))
        r = np.tile([1., 0, 0, 0], (len(t), 1))
        targets = np.tile([[[10.+i, 0, 1.] for i in range(6)]], (len(t), 1, 1))
        targets[:, 0] = [-.045, 0, 1.]
        initial = targets[0].copy(); initial[0] = [-.056, 0, 1.]
        normal = np.tile([[[1., 0, 0]]*6], (len(t), 1, 1))
        active = np.zeros((len(t), 6), bool); active[:, 0] = True
        result = scene.rollout(t, targets, p, r, initial_pads=initial,
            planned_forces=3*normal, inward_normals_local=normal, active=active)
        self.assertGreater(result['object_positions'][-1, 0], .002)
        self.assertGreater(np.max(result['contact_forces'][:, 0, 0]), 1.)
        self.assertTrue(np.any(result['force_enabled'][:, 0]))
        impulse = np.trapezoid(result['contact_forces'][:, :, 0].sum(axis=1), result['simulation_timestamps'])
        self.assertAlmostEqual(impulse, scene.model.body_mass[scene.book_body]*result['object_linear_velocity'][-1, 0], delta=.015)
        self.assertEqual(result['max_object_applied_force'], 0.)


if __name__ == '__main__':
    unittest.main()
