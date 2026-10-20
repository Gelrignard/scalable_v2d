import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.preprocessing.supported_fixture import regularize_fixture


class SupportedFixtureTests(unittest.TestCase):
    def input(self):
        rng=np.random.default_rng(3);q=rng.normal(size=(90,50))*.01
        q[:,:3]+=np.linspace(0,1,90)[:,None]*[.5,.1,0.]
        q[:,3:7]=Rotation.from_rotvec(np.c_[np.zeros((90,2)),np.linspace(0,.7,90)]).as_quat()
        return q

    def test_initial_state_and_non_root_joints_preserved(self):
        q=self.input();result,report=regularize_fixture(q)
        np.testing.assert_array_equal(result[0],q[0]);np.testing.assert_array_equal(result[:,7:],q[:,7:])
        np.testing.assert_allclose(np.linalg.norm(result[:,3:7],axis=1),1.,atol=1e-14)

    def test_quaternion_sign_choices_do_not_change_physical_path(self):
        q=self.input();flipped=q.copy();flipped[::3,3:7]*=-1
        a,_=regularize_fixture(q);b,_=regularize_fixture(flipped)
        error=(Rotation.from_quat(a[:,3:7]).inv()*Rotation.from_quat(b[:,3:7])).magnitude()
        np.testing.assert_allclose(error,0.,atol=1e-14)

    def test_world_frame_is_not_recentered(self):
        q=self.input();r=Rotation.from_rotvec([.2,.4,-.3]);translation=[2.,-4.,1.]
        transformed=q.copy();transformed[:,:3]=r.apply(q[:,:3])+translation
        transformed[:,3:7]=(r*Rotation.from_quat(q[:,3:7])).as_quat()
        a,_=regularize_fixture(q);b,_=regularize_fixture(transformed)
        np.testing.assert_allclose(b[:,:3],r.apply(a[:,:3])+translation,atol=1e-12)
        expected=r*Rotation.from_quat(a[:,3:7]);actual=Rotation.from_quat(b[:,3:7])
        np.testing.assert_allclose((expected.inv()*actual).magnitude(),0.,atol=1e-12)
