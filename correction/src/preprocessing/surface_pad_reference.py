"""Map selected robot collision patches to independent spherical-pad commands."""
import numpy as np
from scipy.spatial.transform import Rotation
from correction.src.optimization.surface_reference import smoothstep


def derive_pad_targets(scene, reference, report, radius):
    """Keep channel identity and account for pad radius, without moving objects.

    The sphere center lies radius behind the selected robot surface normal.
    This preserves that surface as the intended contact point; placing the
    sphere center directly on it would insert half a sphere into the object.
    The independent pad remains a relaxation of the robot contact geometry.
    """
    points=report['contacts']
    count=len(reference['q'])
    bodies=np.array([p['body'] for p in points])
    channels={body:i for i,body in enumerate(bodies)}
    local_normal=np.tile([p['normal_local'] for p in points],(count,1,1)).astype(float)
    phases=report.get('phase_contacts') or []
    for phase_id,phase in enumerate(phases[1:],start=1):
        begin=phases[phase_id-1]['timing']['force_end']+1
        end=phase['timing']['contact_start']-1
        alpha=smoothstep((np.arange(count)-begin)/(end-begin))[:,None]
        for p in phase['contacts']:
            channel=channels[p['body']]
            local_normal[:,channel]=(1-alpha)*local_normal[:,channel]+alpha*np.asarray(p['normal_local'])
    norms=np.linalg.norm(local_normal,axis=-1)
    if np.any(norms<1e-5):
        raise ValueError('Antipodal surface normal transition needs a separate strategy')
    local_normal/=norms[:,:,None]
    targets, surfaces={},{}
    for name,q in [('original',reference['baseline_command_q']),('corrected',reference['q'])]:
        centers=[]; patch=[]
        for frame,state in enumerate(q):
            scene.set_state(state,reference['object_positions'][frame],reference['object_wxyz'][frame])
            rotation=scene.data.xmat[bodies].reshape(-1,3,3)
            position=scene.data.xpos[bodies]+np.einsum('kij,kj->ki',rotation,reference['robot_local_points'][frame])
            outward=np.einsum('kij,kj->ki',rotation,local_normal[frame])
            centers.append(position-radius*outward)
            patch.append(position)
        targets[name]=np.asarray(centers)
        surfaces[name]=np.asarray(patch)
    np.testing.assert_allclose(targets['original'][0],targets['corrected'][0],atol=1e-10)
    object_rotation=Rotation.from_quat(reference['object_wxyz'],scalar_first=True).as_matrix()
    intended_anchors=reference['object_positions'][:,None,:]+np.einsum(
        'tij,tkj->tki',object_rotation,reference['object_local_points'])
    return dict(original_tip_targets=targets['original'],corrected_pad_targets=targets['corrected'],
        original_surface_targets=surfaces['original'],corrected_surface_targets=surfaces['corrected'],
        intended_object_contact_points=intended_anchors,initial_pads=targets['original'][0],
        channel_names=np.array([scene.model.body(b).name for b in bodies]))
