"""A one-hand hypothesis must not introduce a target for the other palm."""
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np
from correction.src.optimization.rigid_grasp import RigidGrasp


class SingleHandGraspTests(unittest.TestCase):
    def test_only_participating_palm_has_a_finite_target(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="left_hand_palm_link" pos="0 0 1"><geom size=".02"/>
            <body name="left_hand_index_1_link"><geom size=".01"/></body></body>
          <body name="right_hand_palm_link" pos="1 0 1"><geom size=".02"/></body>
        </worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        scene=SimpleNamespace(model=model,data=data,set_state=lambda *args:None)
        point=SimpleNamespace(body=model.body('left_hand_index_1_link').id,object_outward=np.array([0.,1.,0.]))
        grasp=RigidGrasp(scene,[point],np.zeros(8),np.zeros(3),np.array([1.,0.,0.,0.]))
        self.assertEqual(grasp.sides,['left'])
        self.assertEqual(grasp.local_p.shape,(1,3))
        self.assertTrue(np.isfinite(grasp.outward).all())
        self.assertEqual(len(grasp.local_r),1)
