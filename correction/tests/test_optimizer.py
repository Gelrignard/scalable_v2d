import unittest

import numpy as np

from correction.scripts.demo import pinch_case
from correction.src.optimization.contact_optimizer import CorrectionSettings, PhaseCorrector


class CorrectionTests(unittest.TestCase):
    def make_corrector(self, cap=20, **settings):
        times, patches, reference, required = pinch_case(cap, samples=7)
        return PhaseCorrector(times, patches, reference, required, np.zeros(3), CorrectionSettings(**settings))

    def test_correction_removes_unbalanced_moment(self):
        result = self.make_corrector().solve()
        self.assertFalse(result["before"]["metrics"]["object_wrench_within_tolerance"])
        self.assertTrue(result["after"]["metrics"]["object_wrench_within_tolerance"])
        self.assertLess(result["after"]["metrics"]["max_torque_residual_nm"], .002)
        self.assertLess(np.abs(result["after"]["contact_points_object_m"][:, 1]).max(), .001)
        self.assertGreater(result["after"]["metrics"]["minimum_friction_diamond_slack_n"], .1)
        self.assertFalse(result["report"]["robot_feasibility_checked"])

    def test_insufficient_force_cannot_be_hidden(self):
        # Even before the reserve margin, 2 * mu * cap = 2 N < the 1 kg load.
        result = self.make_corrector(cap=2).solve()
        self.assertFalse(result["after"]["metrics"]["object_wrench_within_tolerance"])
        self.assertGreater(result["after"]["metrics"]["max_force_residual_n"], 7)
        self.assertLessEqual(result["after"]["metrics"]["max_normal_force_n"], 2 + 1e-5)

    def test_contact_neighborhood_limits_correction(self):
        result = self.make_corrector(max_tangent_shift_m=.005).solve()
        self.assertFalse(result["after"]["metrics"]["object_wrench_within_tolerance"])
        np.testing.assert_array_less(.034999, result["after"]["contact_points_object_m"][:, 1])

    def test_analytic_gradient_matches_finite_difference(self):
        corrector = self.make_corrector()
        point = corrector.initial_uv.ravel() / corrector.settings.length_scale_m
        point += np.array([-.05, .02, -.03, -.01])
        _, analytic = corrector.objective(point)
        numeric = np.zeros_like(point)
        for i in range(len(point)):
            delta = np.zeros_like(point)
            delta[i] = 1e-5
            numeric[i] = (corrector.objective(point+delta)[0] - corrector.objective(point-delta)[0]) / (2e-5)
        np.testing.assert_allclose(analytic, numeric, atol=.05, rtol=2e-4)

    def test_known_environment_wrench_is_accounted_for(self):
        times, patches, reference, required = pinch_case(samples=7)
        corrector = PhaseCorrector(times, patches, reference, required, np.zeros(3),
                                    environment_wrench_object=required.copy())
        result = corrector.solve()
        np.testing.assert_allclose(result["after"]["contact_forces_object_n"], 0, atol=1e-8)
        self.assertTrue(result["after"]["metrics"]["object_wrench_within_tolerance"])

    def test_object_motion_demand_is_unchanged(self):
        corrector = self.make_corrector()
        original = corrector.required.copy()
        reference = corrector.reference.copy()
        result = corrector.solve()
        np.testing.assert_array_equal(result["required_wrench_object"], original)
        np.testing.assert_array_equal(result["reference_points_object_m"], reference)

    def test_remote_surface_hypothesis_rejected(self):
        times, patches, reference, required = pinch_case(samples=7)
        reference[:, 0, 0] -= .2
        with self.assertRaisesRegex(ValueError, "too far"):
            PhaseCorrector(times, patches, reference, required, np.zeros(3))

    def test_zero_friction_cannot_lift(self):
        times, patches, reference, required = pinch_case(samples=7)
        from dataclasses import replace
        patches = [replace(patch, friction=0.) for patch in patches]
        result = PhaseCorrector(times, patches, reference, required, np.zeros(3)).solve()
        self.assertFalse(result["after"]["metrics"]["object_wrench_within_tolerance"])
        self.assertGreater(result["after"]["metrics"]["max_force_residual_n"], 9)


if __name__ == "__main__":
    unittest.main()
