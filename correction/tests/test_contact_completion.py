"""Candidate extraction uses physical surfaces and rejects unsupported regions."""
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np
from correction.src.optimization.surface_hypotheses import SurfaceLibrary
from correction.src.optimization.contact_completion import extract_contact_patch


class ContactCompletionTests(unittest.TestCase):
    def test_translated_collision_candidate_stays_on_mesh_and_in_human_region(self):
        vertices=' '.join(str(v) for p in [(x,y,z) for x in (-1,1) for y in (-1,1)
                                          for z in (-1,1)] for v in p)
        model=mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0"/>
          <asset><mesh name="finger" vertex="{vertices}" scale=".01 .01 .01"/>
                 <mesh name="box" vertex="{vertices}" scale=".1 .1 .1"/></asset>
          <worldbody><body name="right_hand_index_1_link" pos=".5108 .2 .3">
            <joint type="slide" axis="1 0 0"/>
            <geom name="finger_geom" type="mesh" mesh="finger" margin=".001" mass=".1"/>
          </body><body name="object" pos=".4 .2 .3"><freejoint/>
            <geom name="object_geom" type="mesh" mesh="box" margin=".001" mass="1"/>
          </body></worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        scene=SimpleNamespace(model=model,data=data,book_body=model.body('object').id,
                              object_geoms={model.geom('object_geom').id})
        library=SurfaceLibrary(scene)
        point,audit=extract_contact_patch(scene,library,'right_hand_index_1_link',
            np.array([[.1,0,0],[.1,.01,0],[.1,-.01,0]]),source_channels=[3,4,5])
        self.assertAlmostEqual(point.local[0],-.01,places=6)
        self.assertAlmostEqual(point.object_local[0],.1,places=6)
        self.assertGreater(point.object_outward[0],.99)
        self.assertLess(audit['snapped_surface_gap_m'],.001)
        self.assertLess(audit['evidence_distance_m'],.03)
        with self.assertRaisesRegex(ValueError,'outside human hand region'):
            extract_contact_patch(scene,library,'right_hand_index_1_link',
                np.array([[-.1,0,0]]),source_channels=[3])
        data.qpos[0]=1.;mujoco.mj_forward(model,data)
        with self.assertRaisesRegex(ValueError,'No collision-surface proximity'):
            extract_contact_patch(scene,library,'right_hand_index_1_link',
                np.array([[.1,0,0]]),source_channels=[3])


if __name__=='__main__':
    unittest.main()
