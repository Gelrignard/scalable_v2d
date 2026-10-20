"""Actual off-center contact moment, independently checked against generalized force."""
import unittest
import mujoco
import numpy as np
from correction.src.dynamics.contact_measurement import object_contact_wrench, object_wrench_partition


class ContactMeasurementTests(unittest.TestCase):
    def test_off_center_force_and_com_moment(self):
        for reverse in (False, True):
            obj = '''<body name="object"><freejoint/>
                <inertial pos="0 -.02 0" mass="1" diaginertia=".01 .01 .01"/>
                <geom name="box" type="box" size=".1 .1 .1" condim="1"/></body>'''
            finger = '''<body name="finger" pos=".1099 .05 0">
                <joint type="slide" axis="1 0 0"/>
                <geom name="pad" type="sphere" size=".01" mass=".1" condim="1"/></body>'''
            bodies = finger+obj if reverse else obj+finger
            model = mujoco.MjModel.from_xml_string(
                '<mujoco><option gravity="0 0 0"/><worldbody>'+bodies+'</worldbody></mujoco>')
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            self.assertEqual(data.ncon, 1)
            body = model.body('object').id
            wrench = object_contact_wrench(model, data, 0, {model.geom('box').id}, body)
            self.assertLess(wrench[0], 0.)
            self.assertGreater(wrench[5], 0.)
            np.testing.assert_allclose(wrench[5], -.07*wrench[0], atol=1e-10)
            # The free joint's constraint torque is about the body origin.
            # Transform our COM moment back to that origin before comparing.
            origin_moment = wrench[3:]+np.cross(data.xipos[body]-data.xpos[body], wrench[:3])
            address = model.body_dofadr[body]
            np.testing.assert_allclose(data.qfrc_constraint[address:address+6],
                                       np.r_[wrench[:3], origin_moment], atol=1e-10)
            robot, support, body_forces = object_wrench_partition(
                model, data, {model.geom('box').id}, body, set())
            np.testing.assert_allclose(robot, wrench)
            np.testing.assert_array_equal(support, 0.)
            self.assertAlmostEqual(body_forces[model.body('finger').id], np.linalg.norm(wrench[:3]))
            finger_id = model.body('finger').id
            _, _, _, detail = object_wrench_partition(model,data,{model.geom('box').id},body,set(),
                detail=True, patches={finger_id:(data.contact[0].pos.copy(), np.array([-1.,0.,0.]), True)})
            np.testing.assert_allclose(detail['body_wrench'].sum(axis=0), wrench)
            np.testing.assert_allclose(detail['selected_patch_wrench'], wrench)
            np.testing.assert_allclose(detail['active_selected_patch_wrench'], wrench)
            _, _, _, wrong = object_wrench_partition(model,data,{model.geom('box').id},body,set(),
                detail=True, patches={finger_id:(data.contact[0].pos.copy(), np.array([1.,0.,0.]), True)})
            np.testing.assert_array_equal(wrong['selected_patch_wrench'], 0.)
            robot, support, _ = object_wrench_partition(
                model, data, {model.geom('box').id}, body, {model.geom('pad').id})
            np.testing.assert_allclose(support, wrench)
            np.testing.assert_array_equal(robot, 0.)


if __name__ == '__main__':
    unittest.main()
