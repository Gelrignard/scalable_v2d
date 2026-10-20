"""Full MuJoCo geometry audit and contact-witness cuts for a G1 horizon."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from correction.src.optimization.coupled_horizon import sample_solution


def audit_robot_geometry(ref, robot, scene, result, *, samples_per_interval=3, max_cuts=30):
    times = np.unique(np.concatenate([np.linspace(a, b, samples_per_interval+1)
                                      for a, b in zip(ref.times[:-1], ref.times[1:])]))
    sample = sample_solution(ref, result, times)
    root_position = CubicSpline(ref.times, ref.base[:, :3])
    root_rotation = RotationSpline(ref.times, Rotation.from_quat(ref.base[:, 3:7], scalar_first=True))
    com = scene.model.body_ipos[scene.book_body]
    violations = []
    max_pen = 0.
    pair = ''
    for j, time in enumerate(times):
        full_q = scene.fixed_q.copy()
        full_q[:3], full_q[3:7] = root_position(time), root_rotation(time).as_quat()
        full_q[7+scene.movable] = sample['q'][j]
        rotation = Rotation.from_quat(sample['wxyz'][j], scalar_first=True)
        scene.set_state(full_q, sample['position'][j]-rotation.apply(com), sample['wxyz'][j])
        for contact in scene.data.contact:
            if -contact.dist > max_pen:
                max_pen = float(-contact.dist)
                pair = scene.model.geom(contact.geom1).name+' / '+scene.model.geom(contact.geom2).name
            if contact.dist >= -.00005: continue
            normal = np.asarray(contact.frame).reshape(3, 3)[0].copy()
            pa = contact.pos-.5*contact.dist*normal
            pb = contact.pos+.5*contact.dist*normal
            def descriptor(geom, world):
                body = int(scene.model.geom_bodyid[geom])
                if body == scene.book_body:
                    return dict(kind='object', local=rotation.inv().apply(world-sample['position'][j]))
                if body in robot.body_column:
                    local = scene.data.xmat[body].reshape(3, 3).T@(world-scene.data.xpos[body])
                    return dict(kind='robot', body=body, local=local)
                return dict(kind='world', local=world)
            violations.append(dict(time=float(time), penetration=float(-contact.dist),
                pair=(int(contact.geom1), int(contact.geom2)), normal=normal,
                a=descriptor(contact.geom1, pa), b=descriptor(contact.geom2, pb)))
    cuts = []
    seen = set()
    for violation in sorted(violations, key=lambda item: -item['penetration']):
        key = (round(violation['time'], 6), violation['pair'])
        if key in seen: continue
        seen.add(key); cuts.append(violation)
        if len(cuts) >= max_cuts: break
    return dict(full_robot_collision_checked=True, samples=len(times), max_penetration_m=max_pen,
                worst_pair=pair, accepted_geometry=max_pen < .0001, collision_cut_count=len(cuts)), cuts
