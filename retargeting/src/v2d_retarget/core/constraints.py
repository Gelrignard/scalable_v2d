import numpy as np
import pinocchio as pin
from pink.limits import Limit
from pink.tasks import Task


class OutputVelocityLimit(Limit):
    """Bounds ALL internal iterates against the previous OUTPUT frame.

    Body/finger joints here are scalar revolute coordinates. The floating base
    is deliberately excluded: this is a kinematic reference, not base actuation.
    """

    def __init__(self, model, previous, frame_dt, fraction):
        self.previous = previous.copy()
        self.radius = np.asarray(model.velocityLimit[6:]) * frame_dt * fraction
        self.projection = np.eye(model.nv)[6:]

    def compute_qp_inequalities(self, configuration, dt):
        delta = configuration.q[7:] - self.previous[7:]
        return np.vstack([self.projection, -self.projection]), np.concatenate(
            [self.radius - delta, self.radius + delta]
        )


class HullAvoidanceTask(Task):
    """Soft sphere-probe nonpenetration against the object's convex envelope.

    Conservative for concave objects; this does not cover whole-robot geometry
    or establish collision freedom. Evaluation uses the identical named probes.
    """

    def __init__(self, kin, surface, cost):
        super().__init__(cost=cost, gain=0.5, lm_damping=0.01)
        self.names = kin.probe_names
        self.ids = [kin.robot.model.getFrameId(n) for n in self.names]
        self.radii = np.asarray(kin.probe_radii)
        self.surface = surface
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)
        self._last_q = None

    def set_object(self, rotation, translation):
        self.rotation = rotation
        self.translation = translation
        self._last_q = None

    def _update(self, configuration):
        if self._last_q is not None and np.array_equal(self._last_q, configuration.q):
            return
        points = np.array([configuration.data.oMf[i].translation for i in self.ids])
        local = (points - self.translation) @ self.rotation
        d, n = self.surface.hull_distance_normal(local)
        self._error = np.minimum(d - self.radii, 0.0)
        normals = n @ self.rotation.T
        self._jacobian = np.zeros((len(self.ids), configuration.model.nv))
        for row, frame in enumerate(self.ids):
            if self._error[row] < 0:
                jac = pin.getFrameJacobian(
                    configuration.model,
                    configuration.data,
                    frame,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                self._jacobian[row] = normals[row] @ jac[:3]
        self._last_q = configuration.q.copy()

    def compute_error(self, configuration):
        self._update(configuration)
        return self._error

    def compute_jacobian(self, configuration):
        self._update(configuration)
        return self._jacobian

    def __repr__(self):
        return "HullAvoidanceTask(hand sphere probes)"
