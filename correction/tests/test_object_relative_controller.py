"""Safety-relevant bounds of the experimental sensor-feedback adapter."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.dynamics.object_relative_controller import norm_clip,joint_offset


class ObjectRelativeControllerTests(unittest.TestCase):
    def test_translation_cap_is_norm_bounded_and_rotation_equivariant(self):
        r=Rotation.from_rotvec([.4,-.7,.2]);value=np.array([.2,-.3,.5])
        limited=norm_clip(value,.08)
        self.assertAlmostEqual(np.linalg.norm(limited),.08)
        np.testing.assert_allclose(norm_clip(r.apply(value),.08),r.apply(limited),atol=1e-14)
        np.testing.assert_array_equal(norm_clip(np.zeros(3),.08),np.zeros(3))

    def test_rank_deficient_jacobian_yields_finite_bounded_offset(self):
        jac=np.zeros((2,3,4));jac[:,0,0]=1;delta=np.array([[100.,1,0],[100.,1,0]])
        shift=joint_offset(jac,delta,.015,.3)
        self.assertTrue(np.isfinite(shift).all())
        self.assertLessEqual(np.max(abs(shift)),.3)
        np.testing.assert_allclose(shift,[.3,0,0,0])

    def test_feedback_map_is_invariant_to_common_world_rotation(self):
        random=np.random.default_rng(5);jac=random.normal(size=(4,3,7));delta=random.normal(size=(4,3))*.01
        r=Rotation.from_rotvec([.3,.2,-.4]).as_matrix()
        expected=joint_offset(jac,delta,.02,.2)
        actual=joint_offset(np.einsum('ab,kbc->kac',r,jac),delta@r.T,.02,.2)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
