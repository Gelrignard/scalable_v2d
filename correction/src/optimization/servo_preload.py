"""Small executable-command search space and a local load-error proposal.

The proposal uses Kp B delta ~= J.T (desired_force - measured_force).
It is a frozen equilibrium approximation, not an inverse contact simulator.
Both the mechanics proposal and the comparison search use the same four
command directions, controller, force reference and complete episode.
"""
import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline


def phase_envelope(times, boundaries):
    """C2 zero/one/zero envelope, with analytic first and second derivatives."""
    t = np.asarray(times, float)
    a, b, c, d = np.asarray(boundaries, float)
    if not a < b <= c < d:
        raise ValueError('Envelope needs ordered acquisition/hold/release times')
    value = np.zeros_like(t); velocity = value.copy(); acceleration = value.copy()
    value[(t >= b) & (t <= c)] = 1.
    for lo, hi, sign in [(a, b, 1.), (c, d, -1.)]:
        mask = (t > lo) & (t < hi)
        x = (t[mask]-lo)/(hi-lo)
        f = x**3*(10-15*x+6*x*x)
        value[mask] = f if sign > 0 else 1-f
        velocity[mask] = sign*30*x*x*(1-x)**2/(hi-lo)
        acceleration[mask] = sign*60*x*(1-x)*(1-2*x)/(hi-lo)**2
    return value, velocity, acceleration


def command_basis(scene, reference, *, first=300, end=440):
    """Two hand-shape directions and two wrist-pitch directions.

    Finger direction is the seed grasp relative to zero/open joints. Bounds
    are 0.08rad at the largest finger coordinate and 0.02rad wrist pitch.
    This construction uses no executed force or optimizer outcome.
    """
    names = [scene.kin.joint_names[j] for j in scene.movable]
    posture = np.median(reference['q'][first:end+1, 7+scene.movable], axis=0)
    basis = np.zeros((len(names), 4))
    for side_id, side in enumerate(('left_', 'right_')):
        hand = np.array([name.startswith(side) for name in names]) & scene.hand
        scale = np.max(abs(posture[hand]), initial=0.)
        if scale < 1e-6:
            raise ValueError('Seed contains no nonzero hand-shape direction')
        basis[hand, side_id] = .08*posture[hand]/scale
        wrist = [j for j, name in enumerate(names) if name == side+'wrist_pitch_joint']
        if len(wrist) != 1:
            raise ValueError('Expected one wrist pitch joint per hand')
        basis[wrist[0], 2+side_id] = .02
    return basis


def fit_load_proposal(actuator_effect, torque_error, caps):
    """Least-squares projection in normalized actuator-effort units."""
    matrix = np.asarray(actuator_effect)/np.asarray(caps)[:, None]
    target = np.asarray(torque_error)/np.asarray(caps)
    raw = np.linalg.lstsq(matrix, target, rcond=None)[0]
    bounded = np.clip(raw, -1., 1.)
    return bounded, dict(unbounded_parameters=raw.tolist(),
        normalized_load_error=float(np.linalg.norm(target)),
        predicted_remaining_error=float(np.linalg.norm(matrix@bounded-target)),
        approximation='Fixed measured state/Jacobian and fixed other controller terms')


def load_error_proposal(scene, reference, execution, points, basis, *, first=300, end=440):
    """Use every-step body impulse averages, without noisy pose acceleration."""
    torque = np.zeros(len(scene.movable))
    duration = float(execution['interval_timestamps'][end]-execution['interval_timestamps'][first])
    dt = np.diff(execution['interval_timestamps'])
    for frame in range(first, end):
        scene.set_state(execution['q'][frame], execution['object_positions'][frame],
                        execution['object_wxyz'][frame])
        for channel, point in enumerate(points):
            if not reference['contact_active'][frame, channel]:
                continue
            measured = execution['interval_robot_body_wrench_impulse_world'][frame, point.body, :3]/dt[frame]
            desired = .5*(reference['force_ff'][frame, channel]+reference['force_ff'][frame+1, channel])
            local = reference['robot_local_points'][frame, channel]
            world = scene.data.xpos[point.body]+scene.data.xmat[point.body].reshape(3, 3)@local
            jac = np.zeros((3, scene.model.nv))
            mujoco.mj_jac(scene.model, scene.data, jac, None, world, point.body)
            torque += dt[frame]*(jac[:, scene.active_v].T@(desired-measured))/duration
    kp = np.where(scene.hand, 2., 300.)
    parameters, audit = fit_load_proposal(kp[:, None]*basis, torque, scene.caps)
    audit.update(first_frame=first, end_frame=end, average_torque_error_nm=torque.tolist(),
        force_measurement='Dense per-body impulses divided by each video interval; Jacobian at interval start',
        limitation='Unselected patches on each body share this measurement; frozen compliance and friction can be wrong')
    return parameters, audit


def apply_offset(scene, reference, basis, parameters, *, frame_bounds=(192, 270, 480, 525)):
    parameters = np.asarray(parameters, float)
    if parameters.shape != (basis.shape[1],) or not np.isfinite(parameters).all() or np.max(abs(parameters)) > 1.+1e-12:
        raise ValueError('Parameters must be finite and inside the declared unit box')
    t = reference['timestamps']
    envelope, derivative, second = phase_envelope(t, t[list(frame_bounds)])
    offset = basis@parameters
    result = dict(reference)
    result['q'] = reference['q'].copy()
    result['q'][:, 7+scene.movable] += envelope[:, None]*offset
    result['command_velocity'] = reference['command_velocity']+derivative[:, None]*offset
    if 'command_acceleration' in reference:
        result['command_acceleration'] = reference['command_acceleration']+second[:, None]*offset
    # Old global spline coefficients no longer describe this composite command.
    result.pop('spline_knots', None); result.pop('spline_coefficients', None)
    spline = CubicHermiteSpline(t, result['q'][:, 7+scene.movable], result['command_velocity'])
    dense = np.linspace(t[0], t[-1], int(np.ceil((t[-1]-t[0])*150))+1)
    angles = spline(dense); speeds = spline(dense, 1)
    lower = scene.kin.robot.model.lowerPositionLimit[7+scene.movable]
    upper = scene.kin.robot.model.upperPositionLimit[7+scene.movable]
    limit = scene.kin.robot.model.velocityLimit[6+scene.movable]
    position_violation = float(max(0., np.max(lower-angles), np.max(angles-upper)))
    speed_violation = float(max(0., np.max(abs(speeds)-limit)))
    index = np.unravel_index(np.argmax(np.maximum(lower-angles, angles-upper)), angles.shape)
    return result, dict(
        parameters=parameters.tolist(), frame_bounds=list(frame_bounds),
        max_joint_offset_rad=float(np.max(abs(offset))),
        dense_frequency_hz=150, command_position_violation_rad=position_violation,
        worst_position_time=float(dense[index[0]]), worst_position_joint_column=int(index[1]),
        command_speed_violation_rad_s=speed_violation,
        accepted=position_violation <= 1e-4 and speed_violation <= 1e-4,
        scope='Command joint/speed and boundary checks only; actual collision/contact/torque require replay')


def objective(record):
    """Predeclared full-episode ranking objective; physical gates stay separate."""
    task = float(np.mean((record['position_error']/.05)**2+
                         (record['rotation_error']/np.deg2rad(10.))**2))
    effort = float(.1*np.mean(np.maximum(record['frame_peak_torque_fraction']-1., 0.)**2))
    joint = float(100*(max(0., record['max_joint_limit_violation_rad']-.01)/.01)**2)
    return dict(total=task+effort+joint, task=task, saturation=effort, joint_limit=joint,
                ranking_only=True)
