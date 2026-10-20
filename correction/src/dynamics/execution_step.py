"""Shared requested-torque computation for execution and shooting prediction."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def requested_torque(scene, controller, time, command, velocity, acceleration,
                     force, kp, kd, *, inertia_ff, inertia_scale,
                     fixture_a=None, fixture_alpha=None, surface_follow=False):
    s, d = scene, scene.data
    raw = kp*(command-d.qpos[s.active_q])+kd*(velocity-d.qvel[s.active_v])
    raw += d.qfrc_bias[s.active_v]
    raw -= d.qfrc_passive[s.active_v]
    if inertia_ff:
        generalized_acceleration = np.zeros(s.model.nv)
        generalized_acceleration[s.active_v] = acceleration
        if fixture_a is not None:
            generalized_acceleration[s.root_v:s.root_v+3] = fixture_a
            rotation = Rotation.from_quat(d.qpos[s.root_q+3:s.root_q+7], scalar_first=True)
            generalized_acceleration[s.root_v+3:s.root_v+6] = rotation.inv().apply(fixture_alpha)
        inertial = np.zeros(s.model.nv)
        mujoco.mj_mulM(s.model, d, inertial, generalized_acceleration)
        raw += inertia_scale*inertial[s.active_v]
    position = kp*(command-d.qpos[s.active_q])
    speed = kd*(velocity-d.qvel[s.active_v])
    compensation = raw-(position+speed)
    tracking = kp*(command-d.qpos[s.active_q])+kd*(velocity-d.qvel[s.active_v])
    servo_q, feedforward = controller.step(time, command, velocity, force, raw-tracking)
    servo_velocity = controller.last['servo_velocity'] if surface_follow else velocity
    raw = kp*(servo_q-d.qpos[s.active_q])+kd*(servo_velocity-d.qvel[s.active_v])+feedforward
    position = kp*(servo_q-d.qpos[s.active_q])
    speed = kd*(servo_velocity-d.qvel[s.active_v])
    return raw, position, speed, compensation, feedforward-compensation
