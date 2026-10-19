"""Bounded scalar search over finger preload, evaluated through real dynamics.

This deliberately small control refinement is diagnostic. It is neither an
RL policy nor an assertion that the local scene matches the challenge.
"""
from time import perf_counter
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation


def preload_direction(kin, q, quaternions, inward, active):
    from v2d_retarget.core.contacts import TIP_NAMES
    finger = np.array([i for i, n in enumerate(kin.joint_names) if 'hand' in n])
    frame_ids = [kin.robot.model.getFrameId(n) for n in TIP_NAMES]
    rotations = Rotation.from_quat(quaternions, scalar_first=True).as_matrix()
    direction = np.zeros_like(q)
    for t, qt in enumerate(q):
        kin.configuration.update(qt)
        jacobians, targets = [], []
        for j, frame in enumerate(frame_ids):
            if active[t, j]:
                jacobians.append(pin.getFrameJacobian(kin.robot.model, kin.configuration.data, frame,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, finger+6])
                targets.extend(rotations[t]@inward[t, j])
        if jacobians:
            jac = np.vstack(jacobians)
            direction[t, finger+7] = np.linalg.solve(jac.T@jac+1e-4*np.eye(len(finger)), jac.T@targets)
    return direction


def compare_rollouts(scene, kin, before, corrected, positions, quaternions, times,
                     inward, active, *, seconds=20.):
    """At most four rollouts: baseline, corrected, 1 mm and 3 mm preload.

    Position commands obey joint limits. The diagnostic score is object
    tracking; all results include actual robot motion so unsafe/failing
    candidates cannot be passed off as feasible. No pose assistance is used.
    """
    started = perf_counter()
    direction = preload_direction(kin, corrected, quaternions, inward, active)
    proposals = [('before', before), ('after', corrected)]
    for preload in (.001, .003):
        command = corrected+np.clip(preload*direction, -.08, .08)
        command[:, 7:] = np.clip(command[:, 7:], kin.robot.model.lowerPositionLimit[7:], kin.robot.model.upperPositionLimit[7:])
        proposals.append((f'preload_{int(preload*1000)}mm', command))
    records, metrics = {}, {}
    for name, command in proposals:
        if perf_counter()-started >= seconds:
            break
        tick = perf_counter()
        result = scene.rollout(command, positions, quaternions, times, max_steps=int(np.ceil((times[-1]-times[0])/.002))+2)
        count = len(result['q'])
        error = np.linalg.norm(result['object_positions']-positions[:count], axis=1)
        angle = (Rotation.from_quat(result['object_wxyz'], scalar_first=True).inv()
                 *Rotation.from_quat(quaternions[:count], scalar_first=True)).magnitude()
        root_error = np.linalg.norm(result['q'][:, :3]-command[:count, :3], axis=1)
        result['command_q'] = command[:count]
        result['timestamps'] = times[:count]
        records[name] = result
        metrics[name] = dict(seconds=perf_counter()-tick, completed=bool(result['completed']),
            steps=result['steps'], saturation_steps=result['saturation_steps'],
            max_object_position_error_m=float(error.max()), max_object_rotation_error_deg=float(np.rad2deg(angle).max()),
            mean_object_position_error_m=float(error.mean()), max_root_position_error_m=float(root_error.max()),
            diagnostic_score=float(np.mean(error)+.1*np.mean(angle)+.25*np.mean(root_error)),
            initial_max_penetration_m=result['initial_max_penetration_m'],
            initial_state_valid=result['initial_max_penetration_m'] <= .002,
            tracking_pass=bool(result['initial_max_penetration_m'] <= .002 and result['completed'] and error.max() < .05 and angle.max() < np.deg2rad(15) and root_error.max() < .1),
            no_external_object_or_root_assistance=True,
            controller='torque-limited PD: hand kp=2 kd=.2, other kp=40 kd=3; diagnostic assumptions',
            official_challenge_evaluation=False)
    candidates = [n for n in records if n != 'before']
    best = min(candidates, key=lambda n: metrics[n]['diagnostic_score']) if candidates else None
    return records, metrics, dict(seconds=perf_counter()-started, rollout_cap=4,
        selected_diagnostic_candidate=best, accepted_for_training=False,
        budget_exhausted=len(records) < len(proposals),
        scope='scalar command preload search; candidate ranking does not certify feasibility')
