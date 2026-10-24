import json
from types import SimpleNamespace
import unittest
import numpy as np
from correction.src.pass_traj.rl_pilot import reference_arrays


class ReferenceContractTests(unittest.TestCase):
    def test_root_quaternion_joint_names_and_missing_frame_are_preserved(self):
        q=np.zeros((3,8));q[:,6]=1.;q[:,7]=[0.,.1,.2]
        kin=SimpleNamespace(joint_names=['example_joint'],poses=lambda row,names:[
            SimpleNamespace(translation=row[:3],rotation=np.eye(3)) for _ in names])
        raw=dict(object_positions=np.ones((3,3)),object_wxyz=np.tile([1.,0.,0.,0.],(3,1)),
                 metadata_json=json.dumps(dict(source=dict(object_observed_mask=[False,True,True],world_from_source_cv=np.eye(4).tolist()))))
        result=reference_arrays(q,np.arange(3,dtype=float),raw,kin,[0,1,2])
        np.testing.assert_array_equal(result['root_wxyz'],np.tile([1.,0.,0.,0.],(3,1)))
        np.testing.assert_array_equal(result['joint_names'],['example_joint'])
        np.testing.assert_array_equal(result['source_frame_indices'],[0,1,2])
        np.testing.assert_array_equal(result['object_observed_mask'],[False,True,True])
        self.assertEqual(result['ee_pose_w'].shape,(3,2,7))
