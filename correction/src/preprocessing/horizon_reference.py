"""Explicit observation, contact-hypothesis, and initial-state contracts.

Uncertainty scales are declared assumptions, not calibrated probabilities.
Object pose observations are never differentiated to obtain wrench targets.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContactPatch:
    name: str
    center: np.ndarray
    inward: np.ndarray
    tangents: np.ndarray
    active: np.ndarray
    robot_index: int = -1
    radius: float = .015
    sigma: float = .02
    force_limit: float = 20.
    friction: float = .5
    support_z: float | None = None
    observed_center: np.ndarray | None = None

    def validate(self, n):
        self.center = np.asarray(self.center, float)
        self.inward = np.asarray(self.inward, float)
        self.tangents = np.asarray(self.tangents, float)
        self.active = np.asarray(self.active, bool)
        if self.center.shape != (3,) or self.inward.shape != (3,) or self.tangents.shape != (3, 2):
            raise ValueError('Contact center/normal/basis dimensions are invalid')
        if self.active.shape != (n,) or not np.isfinite(np.r_[self.center, self.inward, self.tangents.ravel()]).all():
            raise ValueError('Invalid contact schedule or geometry')
        if not np.allclose(self.tangents.T@self.tangents, np.eye(2), atol=1e-6) or not np.allclose(self.tangents.T@self.inward, 0, atol=1e-6):
            raise ValueError('Contact tangents must be orthonormal and perpendicular to normal')
        if abs(np.linalg.norm(self.inward)-1) > 1e-6 or min(self.sigma, self.force_limit, self.friction) <= 0 or self.radius < 0:
            raise ValueError('Invalid contact normal, uncertainty, or force limits')
        if self.robot_index < 0 and self.support_z is None:
            raise ValueError('Environment contacts require a support plane')
        if self.observed_center is not None:
            self.observed_center = np.asarray(self.observed_center, float)
            if self.observed_center.shape != (3,) or not np.isfinite(self.observed_center).all():
                raise ValueError('Contact observation must be a finite point in object coordinates')
        # One sticking patch per continuous phase; a regrasp needs another patch.
        if np.sum(np.diff(np.r_[False, self.active, False].astype(int)) == 1) > 1:
            raise ValueError('A patch cannot silently bridge separate sticking phases')


@dataclass
class HorizonReference:
    times: np.ndarray
    observation_times: np.ndarray
    observed_positions: np.ndarray
    observed_wxyz: np.ndarray
    joint_reference: np.ndarray
    base: np.ndarray
    initial_q: np.ndarray
    initial_joint_velocity: np.ndarray
    initial_position: np.ndarray
    initial_wxyz: np.ndarray
    initial_velocity: np.ndarray
    initial_angular_momentum: np.ndarray
    goal_position: np.ndarray
    goal_wxyz: np.ndarray
    contacts: list[ContactPatch]
    mass: float
    inertia: np.ndarray
    half_size: np.ndarray
    position_sigma: float | np.ndarray = .012
    rotation_sigma: float | np.ndarray = np.deg2rad(4.)
    goal_position_tolerance: float = .015
    goal_rotation_tolerance: float = np.deg2rad(6.)
    goal_rest: bool = False
    metadata: dict = field(default_factory=dict)

    def validate(self, robot):
        self.times = np.asarray(self.times, float)
        self.observation_times = np.asarray(self.observation_times, float)
        n = len(self.times)
        if n < 3 or len(self.observation_times) < 2 or not np.isfinite(self.times).all() or not np.isfinite(self.observation_times).all() or np.any(np.diff(self.times) <= 0) or np.any(np.diff(self.observation_times) <= 0):
            raise ValueError('At least three increasing trajectory knots are required')
        shapes = dict(joint_reference=(n, robot.n), base=(n, 25), initial_q=(robot.n,),
                      initial_joint_velocity=(robot.n,), initial_position=(3,), initial_wxyz=(4,),
                      initial_velocity=(3,), initial_angular_momentum=(3,), goal_position=(3,),
                      goal_wxyz=(4,), inertia=(3, 3), half_size=(3,),
                      observed_positions=(len(self.observation_times), 3), observed_wxyz=(len(self.observation_times), 4))
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'{name}: expected finite {shape}, received {value.shape}')
            setattr(self, name, value)
        if self.observation_times[0] < self.times[0]-1e-9 or self.observation_times[-1] > self.times[-1]+1e-9:
            raise ValueError('Observations must lie within the horizon')
        for q in (self.observed_wxyz, self.initial_wxyz[None], self.goal_wxyz[None], self.base[:, 3:7]):
            if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-6):
                raise ValueError('Quaternions must be normalized')
        if min(self.mass, self.goal_position_tolerance, self.goal_rotation_tolerance) <= 0:
            raise ValueError('Mass and uncertainty/tolerance scales must be positive')
        for name in ('position_sigma', 'rotation_sigma'):
            scale = np.broadcast_to(np.asarray(getattr(self, name), float), (len(self.observation_times), 3))
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError(f'{name} must contain positive finite observation scales')
        if not np.allclose(self.inertia, self.inertia.T) or np.linalg.eigvalsh(self.inertia).min() <= 0:
            raise ValueError('Object inertia must be symmetric positive definite')
        if np.any(self.half_size <= 0):
            raise ValueError('Box half dimensions must be positive')
        if np.any(self.initial_q < robot.lower-1e-8) or np.any(self.initial_q > robot.upper+1e-8):
            raise ValueError('Known initial robot configuration violates joint limits')
        for contact in self.contacts:
            contact.validate(n)
            if contact.robot_index >= robot.count:
                raise ValueError('Contact references a nonexistent robot surface point')
            axis = int(np.argmax(abs(contact.inward)))
            face_normal = np.zeros(3); face_normal[axis] = np.sign(contact.inward[axis])
            if (not np.allclose(contact.inward, face_normal, atol=1e-7)
                    or abs(contact.center[axis]+contact.inward[axis]*self.half_size[axis]) > 1e-7
                    or np.any(abs(contact.center)+contact.radius*np.sum(abs(contact.tangents), axis=1) > self.half_size+1e-7)):
                raise ValueError('A box contact patch must lie entirely on a physical face')
        for k in range(n):
            ids = [p.robot_index for p in self.contacts if p.active[k] and p.robot_index >= 0]
            if len(ids) != len(set(ids)):
                raise ValueError('One robot point cannot make two simultaneous contacts')
        return self


def stationary_base(count, position=(0., 0., 0.), wxyz=(1., 0., 0., 0.)):
    base = np.zeros((count, 25))
    base[:, :3], base[:, 3:7] = position, wxyz
    return base
