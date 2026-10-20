"""A solved allocation must not disguise insufficient actuator capacity."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from correction.src.optimization.actuator_force_profile import allocate_profile


class ActuatorForceProfileTests(unittest.TestCase):
    def allocate(self,cap):
        scene=SimpleNamespace(model=SimpleNamespace(body_mass=np.array([1.]),opt=SimpleNamespace(gravity=np.array([0.,0.,-9.81]))),
            data=SimpleNamespace(xipos=np.zeros((1,3)),qfrc_bias=np.zeros(2)),book_body=0,
            active_v=np.array([0,1]),caps=np.array([cap,cap]),set_state=lambda *args:None)
        points=[SimpleNamespace(object_outward=np.array([-1.,0,0])),SimpleNamespace(object_outward=np.array([1.,0,0]))]
        jac=np.zeros((2,3,2));jac[0,0,0]=1.;jac[1,0,1]=1.
        kinematics=(np.array([[-.1,0,0],[.1,0,0]]),np.zeros((2,3)),jac,np.zeros_like(jac))
        with patch('correction.src.optimization.actuator_force_profile.contact_kinematics',return_value=kinematics):
            return allocate_profile(scene,points,np.zeros((3,9)),np.zeros((3,3)),np.tile([1.,0,0,0],(3,1)),
                                    np.arange(3)/30.,np.ones(3),friction=.8)

    def test_feasible_hold_respects_preload_and_joint_caps(self):
        forces,audit=self.allocate(20.)
        self.assertTrue(audit['solved']);self.assertTrue(audit['task_wrench_satisfied'])
        normal=np.c_[forces[:,0,0],-forces[:,1,0]]
        self.assertTrue((normal>=2.-1e-4).all());self.assertTrue((normal<=15.+1e-4).all())
        self.assertLess(audit['constraint_violation'],1e-4)

    def test_small_actuator_caps_report_missing_wrench_despite_solver_success(self):
        forces,audit=self.allocate(6.)
        self.assertTrue(audit['solved']);self.assertFalse(audit['task_wrench_satisfied'])
        self.assertGreater(audit['max_force_residual_n'],2.)
        self.assertLessEqual(audit['max_static_torque_fraction'],.9001)
