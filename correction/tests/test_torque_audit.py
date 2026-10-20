"""Sub-frame peaks must retain the terms and contact state at that exact step."""
import unittest
import numpy as np
from correction.src.dynamics.torque_audit import TorquePeakAudit


class TorqueAuditTests(unittest.TestCase):
    def test_each_joint_keeps_its_own_peak_time_and_signed_terms(self):
        audit = TorquePeakAudit(3, [2., 4.], contacts=1)
        audit.update(0, .001, np.array([3., -1.]), [[1., 1., 0., 1.], [0., -2., 1., 0.]],
                     enabled=[True], touching=[False], gap=[.01], contact_step=[1., 0.])
        audit.update(0, .002, np.array([-2., -8.]), [[-1., -1., 0., 0.], [1., -10., 1., 0.]],
                     enabled=[False], touching=[True], gap=[0.], contact_step=[-1., .5])
        out = audit.export()
        np.testing.assert_array_equal(out['torque_peak_time_seconds'][0], [.001, .002])
        np.testing.assert_array_equal(out['torque_peak_terms_nm'][0].sum(axis=1), [3., -8.])
        np.testing.assert_array_equal(out['torque_peak_force_enabled'][0, :, 0], [True, False])
        np.testing.assert_array_equal(out['torque_peak_touching'][0, :, 0], [False, True])
        np.testing.assert_array_equal(out['torque_peak_contact_torque_step_nm'][0], [1., .5])
        self.assertTrue(np.isnan(out['torque_peak_time_seconds'][-1]).all(),
                        'The terminal sample is not an additional integration interval')


if __name__ == '__main__':
    unittest.main()
