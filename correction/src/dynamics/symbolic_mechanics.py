"""Small differentiable mechanics models for the coupled horizon optimizer.

CasADi is optional outside this module. Forces always mean robot-on-object;
the robot inverse dynamics therefore adds J.T @ force to actuator demand.
No acceleration is obtained by differentiating observed object poses here.
"""
from dataclasses import dataclass

import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    return ca.vertcat(ca.horzcat(0, -v[2], v[1]),
                      ca.horzcat(v[2], 0, -v[0]),
                      ca.horzcat(-v[1], v[0], 0))


def quat_mul(a, b):
    """Hamilton product, scalar first; rotations map local to world."""
    return ca.vertcat(a[0]*b[0]-ca.dot(a[1:], b[1:]),
                      a[0]*b[1:]+b[0]*a[1:]+ca.cross(a[1:], b[1:]))


def quat_rotation(q):
    # Optimization iterates need not yet satisfy the unit-quaternion dynamics.
    # Keep the physical rotation valid during line search as well as at a solve.
    q = q/ca.sqrt(ca.dot(q, q)+1e-24)
    return ca.MX.eye(3)+2*q[0]*skew(q[1:])+2*ca.mtimes(skew(q[1:]), skew(q[1:]))


def rotation_quat(v):
    angle = ca.sqrt(ca.dot(v, v)+1e-24)
    return ca.vertcat(ca.cos(angle/2), ca.sin(angle/2)/angle*v)


def orientation_error(reference, q):
    delta = quat_mul(ca.vertcat(reference[0], -reference[1:]), q)
    delta = ca.if_else(delta[0] < 0, -delta, delta)
    sine = ca.sqrt(ca.dot(delta[1:], delta[1:])+1e-24)
    return 2*ca.atan2(sine, delta[0])/sine*delta[1:]


@dataclass(frozen=True)
class BodyPoint:
    body: int
    local: np.ndarray


class CartesianPads:
    """Finite-mass, force-limited Cartesian contact pads, for controlled tests.

    The center is actuated, while contact is on the spherical surface. This is
    deliberately identified as a reduced robot model, never as G1 validation.
    """
    def __init__(self, count=2, radius=.006, mass=.04, force_limit=25.):
        self.n = 3*count
        self.count, self.radius, self.mass = count, radius, mass
        self.lower, self.upper = np.full(self.n, -3.), np.full(self.n, 3.)
        self.speed = np.full(self.n, 1.)
        self.acceleration = np.full(self.n, 8.)
        self.caps = np.full(self.n, force_limit)
        q, v, a = [ca.MX.sym(name, self.n) for name in ('q', 'v', 'a')]
        base = ca.MX.sym('base', 25)
        gravity = np.tile([0., 0., -9.81], count)
        self.kinematics = ca.Function('pad_fk', [q, base], [ca.reshape(q, 3, count)])
        self.inverse_dynamics = ca.Function('pad_id', [q, v, a, base], [mass*(a-gravity)])

    def contact_positions(self, q, base, rotation, inward):
        return self.kinematics(q, base)+self.radius*ca.mtimes(rotation, inward)

    def force_torque(self, q, base, world_forces):
        return ca.vec(world_forces)


class SymbolicRobot:
    """World-frame Newton--Euler recursion imported from a supported MJ model.

    All articulated robot joints must be scalar hinges. Frozen legs remain in
    the tree. The externally prescribed root's velocity and acceleration are
    explicit inputs. The object and environment are excluded from this tree.
    Contacts are actual points on the robot collision surface, not tip-site
    offsets. Unit tests compare this recursion against MuJoCo inverse dynamics.
    """
    def __init__(self, scene, contacts):
        model = scene.model
        self.scene, self.contacts = scene, tuple(contacts)
        self.n, self.count = len(scene.movable), len(contacts)
        self.radius = 0.
        self.lower = scene.kin.robot.model.lowerPositionLimit[7+scene.movable].copy()
        self.upper = scene.kin.robot.model.upperPositionLimit[7+scene.movable].copy()
        self.speed = np.minimum(scene.kin.robot.model.velocityLimit[6+scene.movable], 3.)
        self.acceleration = np.full(self.n, 12.)
        self.caps = scene.caps.copy()
        root = int(model.jnt_bodyid[model.joint('robot_root').id]) if scene.moving_fixture else 1
        bodies = [root]
        for body in range(root+1, model.nbody):
            if int(model.body_parentid[body]) in bodies:
                bodies.append(body)
        self.bodies = bodies
        joint_to_col = {int(model.joint(scene.kin.joint_names[i]).id): j
                        for j, i in enumerate(scene.movable)}
        q, v, acc = [ca.MX.sym(name, self.n) for name in ('q', 'v', 'a')]
        # root translation, quaternion wxyz, linear/angular velocity, then
        # linear/angular acceleration (world axes); 6 reserved zeros.
        base = ca.MX.sym('base', 25)
        position = {root: base[:3]}
        rotation = {root: quat_rotation(base[3:7])}
        velocity = {root: base[7:10]}
        omega = {root: base[10:13]}
        acceleration = {root: base[13:16]}
        alpha = {root: base[16:19]}
        axes, pivots = {}, {}
        for body in bodies[1:]:
            parent = int(model.body_parentid[body])
            rp, pp = rotation[parent], position[parent]
            r0 = ca.DM(Rotation.from_quat(model.body_quat[body], scalar_first=True).as_matrix())
            p0 = ca.DM(model.body_pos[body])
            ids = range(int(model.body_jntadr[body]), int(model.body_jntadr[body]+model.body_jntnum[body]))
            joints = [i for i in ids if i in joint_to_col]
            if len(joints) > 1:
                raise ValueError('SymbolicRobot requires at most one hinge per body')
            if joints:
                jid, col = joints[0], joint_to_col[joints[0]]
                if int(model.jnt_type[jid]) != 3:
                    raise ValueError('Only supported scalar hinge joints are implemented')
                axis = ca.DM(model.jnt_axis[jid]); pivot = ca.DM(model.jnt_pos[jid])
                world_axis = rp@r0@axis
                rpj = rp@(p0+r0@pivot)
                jp = pp+rpj
                rotation[body] = rp@r0@(ca.MX.eye(3)+ca.sin(q[col])*skew(axis)
                                         +(1-ca.cos(q[col]))*(skew(axis)@skew(axis)))
                rjo = -rotation[body]@pivot
                position[body] = jp+rjo
                omega[body] = omega[parent]+world_axis*v[col]
                alpha[body] = alpha[parent]+world_axis*acc[col]+ca.cross(omega[parent], world_axis*v[col])
                vj = velocity[parent]+ca.cross(omega[parent], rpj)
                aj = acceleration[parent]+ca.cross(alpha[parent], rpj)+ca.cross(omega[parent], ca.cross(omega[parent], rpj))
                velocity[body] = vj+ca.cross(omega[body], rjo)
                acceleration[body] = aj+ca.cross(alpha[body], rjo)+ca.cross(omega[body], ca.cross(omega[body], rjo))
                axes[col], pivots[col] = world_axis, jp
            else:
                offset = rp@p0
                rotation[body], position[body] = rp@r0, pp+offset
                omega[body], alpha[body] = omega[parent], alpha[parent]
                velocity[body] = velocity[parent]+ca.cross(omega[parent], offset)
                acceleration[body] = acceleration[parent]+ca.cross(alpha[parent], offset)+ca.cross(omega[parent], ca.cross(omega[parent], offset))
        forces, moments = {}, {}
        for body in bodies:
            offset = rotation[body]@ca.DM(model.body_ipos[body])
            com_acc = acceleration[body]+ca.cross(alpha[body], offset)+ca.cross(omega[body], ca.cross(omega[body], offset))
            principal = Rotation.from_quat(model.body_iquat[body], scalar_first=True).as_matrix()
            inertia = principal@np.diag(model.body_inertia[body])@principal.T
            iw = rotation[body]@inertia@rotation[body].T
            forces[body] = float(model.body_mass[body])*(com_acc-ca.DM(model.opt.gravity))
            moments[body] = iw@alpha[body]+ca.cross(omega[body], iw@omega[body])+ca.cross(offset, forces[body])
        torque = [None]*self.n
        for body in reversed(bodies):
            ids = range(int(model.body_jntadr[body]), int(model.body_jntadr[body]+model.body_jntnum[body]))
            for jid in ids:
                if jid in joint_to_col:
                    col = joint_to_col[jid]
                    dof = int(model.jnt_dofadr[jid])
                    torque[col] = ca.dot(axes[col], moments[body]-ca.cross(pivots[col]-position[body], forces[body]))
                    torque[col] += float(model.dof_armature[dof])*acc[col]+float(model.dof_damping[dof])*v[col]
            if body != root:
                parent = int(model.body_parentid[body])
                moments[parent] += moments[body]+ca.cross(position[body]-position[parent], forces[body])
                forces[parent] += forces[body]
        points = ca.horzcat(*[position[c.body]+rotation[c.body]@c.local for c in contacts])
        self.kinematics = ca.Function('robot_surface_fk', [q, base], [points])
        self.jacobian = ca.Function('robot_contact_jac', [q, base], [ca.jacobian(ca.vec(points), q)])
        self.inverse_dynamics = ca.Function('robot_id', [q, v, acc, base], [ca.vertcat(*torque)])
        # Collision witnesses can be transformed without rebuilding mechanics.
        self.body_transforms = ca.Function('robot_bodies', [q, base],
            [ca.horzcat(*[ca.vertcat(position[b], ca.vec(rotation[b])) for b in bodies])])
        self.body_column = {b: i for i, b in enumerate(bodies)}
        # Scalar expression graphs give substantially smaller second-derivative
        # graphs than differentiating nested MX matrix products at every knot.
        self.kinematics = self.kinematics.expand()
        self.jacobian = self.jacobian.expand()
        self.inverse_dynamics = self.inverse_dynamics.expand()
        self.body_transforms = self.body_transforms.expand()

    def contact_positions(self, q, base, rotation, inward):
        return self.kinematics(q, base)

    def force_torque(self, q, base, world_forces):
        return self.jacobian(q, base).T@ca.vec(world_forces)

    def point(self, q, base, body, local):
        transform = self.body_transforms(q, base)[:, self.body_column[body]]
        return transform[:3]+ca.reshape(transform[3:], 3, 3)@local
