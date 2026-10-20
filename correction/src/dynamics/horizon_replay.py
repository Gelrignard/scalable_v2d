"""Independent free-object MuJoCo replay for the controlled contact benchmark."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .robot_scene import numbers


def replay_pads(ref, robot, command, *, force_guidance=True, dt=.001, mass_scale=1., friction=.6,
                normal_preload_n=0., normal_feedback_gain=0., contact_interface='torque', max_position_offset=.01):
    """Replay torque-limited pads, optionally comparing contact realization.

    ``normal_preload_n`` creates squeeze through a position-setpoint bias only;
    the nominal velocity/acceleration references and baseline compensation stay
    unchanged. It must be compared with force_guidance=False. Normal feedback
    corrects measured force error on top of the existing force feedforward.
    Neither option applies an external force to the object.
    """
    if min(normal_preload_n, normal_feedback_gain) < 0 or not np.isfinite([normal_preload_n, normal_feedback_gain]).all():
        raise ValueError('Controller gains and preload must be finite and nonnegative')
    if normal_preload_n and force_guidance:
        raise ValueError('Use position preload without explicit contact-force guidance')
    if normal_feedback_gain and not force_guidance:
        raise ValueError('Normal feedback extends the force-guidance controller')
    if contact_interface not in ('torque', 'position') or not np.isfinite(max_position_offset) or max_position_offset <= 0:
        raise ValueError('Invalid contact command interface/offset')
    root = ET.Element('mujoco', model='coupled_contact_benchmark')
    ET.SubElement(root, 'option', timestep=str(dt), integrator='implicitfast', gravity='0 0 -9.81',
                  iterations='80', cone='elliptic', impratio='20', tolerance='1e-10', noslip_iterations='10')
    world = ET.SubElement(root, 'worldbody')
    obj = ET.SubElement(world, 'body', name='object')
    ET.SubElement(obj, 'freejoint', name='object_root')
    ET.SubElement(obj, 'geom', name='object_geom', type='box', size=numbers(ref.half_size), mass=str(ref.mass*mass_scale),
                  friction=f'{friction} .00001 .00001', condim='3', solref='.003 1', solimp='.95 .99 .001')
    for i in range(robot.count):
        body = ET.SubElement(world, 'body', name=f'pad{i}')
        for j in range(3):
            ET.SubElement(body, 'joint', name=f'pad{i}_{j}', type='slide', axis=numbers(np.eye(3)[j]))
        ET.SubElement(body, 'geom', name=f'pad{i}_geom', type='sphere', size=str(robot.radius), mass=str(robot.mass),
                      friction=f'{friction} .00001 .00001', condim='3', solref='.003 1', solimp='.95 .99 .001')
    for height in sorted(set(p.support_z for p in ref.contacts if p.support_z is not None)):
        ET.SubElement(world, 'geom', name='support', type='plane', pos=numbers([0, 0, height]), size='2 2 .1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    qadr = np.array([[int(model.joint(f'pad{i}_{j}').qposadr[0]) for j in range(3)] for i in range(robot.count)])
    vadr = np.array([[int(model.joint(f'pad{i}_{j}').dofadr[0]) for j in range(3)] for i in range(robot.count)])
    data.qpos[:7] = np.r_[ref.initial_position, ref.initial_wxyz]
    data.qvel[:3] = ref.initial_velocity
    data.qvel[3:6] = np.linalg.solve(ref.inertia, Rotation.from_quat(ref.initial_wxyz, scalar_first=True).inv().apply(ref.initial_angular_momentum))
    data.qpos[qadr] = ref.initial_q.reshape(robot.count, 3)
    data.qvel[vadr] = ref.initial_joint_velocity.reshape(robot.count, 3)
    mujoco.mj_forward(model, data)
    initial_penetration = max([max(0., -c.dist) for c in data.contact], default=0.)
    spline = CubicHermiteSpline(ref.times, command['q'], command['v']) if 'v' in command else CubicSpline(ref.times, command['q'])
    t = np.linspace(ref.times[0], ref.times[-1], round((ref.times[-1]-ref.times[0])/dt)+1)
    reference_rotation = Slerp(ref.times, Rotation.from_quat(command.get('wxyz', np.tile(ref.initial_wxyz, (len(ref.times), 1))), scalar_first=True))
    filtered_force = np.zeros((robot.count, 3))
    out = dict(times=t, object_positions=[], object_wxyz=[], q=[], v=[], torques=[], contacts=[], contact_forces=[])
    max_penetration = max_external = max_demand = max_conversion_residual = 0.
    for step, time in enumerate(t):
        actual_contacts = np.zeros(robot.count, bool)
        contact_forces = np.zeros((robot.count, 3))
        for ci, contact in enumerate(data.contact):
            max_penetration = max(max_penetration, -float(contact.dist))
            names = (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
            if 'object_geom' in names:
                for name in names:
                    if name.startswith('pad'):
                        pad = int(name[3:name.index('_')])
                        actual_contacts[pad] = True
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(model, data, ci, wrench)
                        sign = -1 if names[0] == 'object_geom' else 1
                        contact_forces[pad] += sign*np.asarray(contact.frame).reshape(3, 3).T@wrench[:3]
        out['object_positions'].append(data.qpos[:3].copy())
        out['object_wxyz'].append(data.qpos[3:7].copy())
        out['q'].append(data.qpos[qadr].ravel().copy())
        out['v'].append(data.qvel[vadr].ravel().copy())
        out['torques'].append(data.qfrc_applied[vadr].ravel().copy())
        out['contacts'].append(actual_contacts)
        out['contact_forces'].append(contact_forces)
        max_external = max(max_external, float(np.max(abs(data.qfrc_applied[:6]))), float(np.max(abs(data.xfrc_applied))))
        if step == len(t)-1: break
        target, velocity, acc = [spline(time, d).reshape(robot.count, 3) for d in range(3)]
        if normal_preload_n:
            # A virtual equilibrium inside the object creates a compliant
            # squeeze; the actual pad is still stopped by physical contact.
            k = min(np.searchsorted(ref.times, time, side='right')-1, len(ref.times)-2)
            for patch in ref.contacts:
                if patch.robot_index >= 0 and patch.active[k] and patch.active[k+1]:
                    # This benchmark starts in contact. Apply the preload from
                    # t=0, just as the force-feedforward arm applies its force
                    # from t=0; a preload-only ramp would make the comparison
                    # unfair by withholding grip while the free object falls.
                    target[patch.robot_index] += normal_preload_n/1200*reference_rotation(time).apply(patch.inward)
        raw = 1200*(target-data.qpos[qadr])+2*np.sqrt(1200*robot.mass)*(velocity-data.qvel[vadr])
        raw += robot.mass*(acc-[0, 0, -9.81])
        if force_guidance:
            f = np.array([np.interp(time, ref.times, command['forces'][:, j]) for j in range(command['forces'].shape[1])]).reshape(-1, 3)
            object_rotation = Rotation.from_quat(data.qpos[3:7], scalar_first=True)
            filtered_force += dt/(.02+dt)*(contact_forces-filtered_force)
            for i, patch in enumerate(ref.contacts):
                if patch.robot_index < 0: continue
                pad = patch.robot_index
                local = object_rotation.inv().apply(data.qpos[qadr][pad]-data.qpos[:3])
                gap = np.linalg.norm(np.maximum(abs(local)-ref.half_size, 0))-robot.radius
                # Enable force guidance near an actual surface, never transmit
                # planned forces across a large gap or directly to the object.
                if actual_contacts[pad] or gap < .002:
                    applied = f[i].copy()
                    if normal_feedback_gain and actual_contacts[pad]:
                        normal = reference_rotation(time).apply(patch.inward)
                        desired = max(0., float(normal@f[i]))
                        measured = max(0., float(normal@filtered_force[pad]))
                        correction = np.clip(normal_feedback_gain*(desired-measured), -2., 2.)
                        normal_command = np.clip(desired+correction, 0., patch.force_limit)
                        applied += (normal_command-desired)*normal
                    if contact_interface == 'position':
                        from .contact_controller import position_target
                        servo, _, residual = position_target(target[pad], applied, 1200.,
                            robot.lower.reshape(robot.count, 3)[pad], robot.upper.reshape(robot.count, 3)[pad], max_position_offset)
                        # Change the servo equilibrium, without adding the
                        # contact force again as feedforward. Physical velocity
                        # and existing gravity/inertia compensation are retained.
                        raw[pad] += 1200*(servo-target[pad])
                        max_conversion_residual = max(max_conversion_residual, float(np.max(abs(residual))))
                    else:
                        raw[pad] += applied
        max_demand = max(max_demand, float(np.max(abs(raw).ravel()/robot.caps)))
        data.qfrc_applied[:] = 0
        data.qfrc_applied[vadr] = np.clip(raw, -robot.caps.reshape(robot.count, 3), robot.caps.reshape(robot.count, 3))
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or any(data.warning[i].number for i in
                (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)):
            raise FloatingPointError('Unstable physical replay; no reset or stitched output allowed')
    out = {key: np.asarray(value) for key, value in out.items()}
    out.update(max_penetration_m=max_penetration, initial_penetration_m=initial_penetration,
               max_object_applied_force=max_external, max_requested_actuator_fraction=max_demand,
               normal_preload_n=normal_preload_n, normal_feedback_gain=normal_feedback_gain,
               contact_interface=contact_interface, max_conversion_residual=max_conversion_residual)
    return out
