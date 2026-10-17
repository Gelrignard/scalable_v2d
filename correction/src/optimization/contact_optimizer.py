"""Bounded contact-target correction with an inner convex force-trajectory QP.

Scope: one sticking-contact phase, rigid object, planar contact patches, fixed
object motion. Robot kinematics, support reactions, sliding and impacts are not
inferred by this module. Supplied environmental wrenches are subtracted explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import minimize

from correction.src.dynamics.models import ContactPatch, finite_array


def skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


@dataclass(frozen=True)
class CorrectionSettings:
    length_scale_m: float = 0.1
    force_scale_n: float = 10.0
    target_sigma_m: float = 0.02
    max_tangent_shift_m: float = 0.08
    max_normal_projection_m: float = 0.03
    wrench_weight: float = 10000.0
    effort_weight: float = 0.001
    force_rate_weight: float = 0.0001
    friction_utilization_limit: float = 0.85
    max_iterations: int = 60
    force_tolerance_n: float = 0.02
    torque_tolerance_nm: float = 0.002

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.friction_utilization_limit > 1:
            raise ValueError("friction_utilization_limit must be <= 1")
        if not isinstance(self.max_iterations, int):
            raise ValueError("max_iterations must be an integer")


class PhaseCorrector:
    """Optimize 2*K patch coordinates, eliminating forces through a small QP.

    Contact positions are constant in object coordinates throughout the phase.
    Forces are variables at every supplied time sample. The force QP uses a
    conservative diamond friction cone: |ft1| + |ft2| <= mu * fn.
    Wrench balance is a penalized residual, not an exact constraint: acceptance
    always checks residuals in newtons and newton-metres afterwards.
    """

    def __init__(self, times_s, patches: list[ContactPatch], reference_points_object_m,
                 required_wrench_object, com_object_m, settings=None,
                 environment_wrench_object=None):
        self.settings = settings or CorrectionSettings()
        self.patches = list(patches)
        self.times = np.asarray(times_s, dtype=float)
        self.t, self.k = len(self.times), len(self.patches)
        if self.t < 2 or not self.k or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("need >=2 increasing timestamps and >=1 contact patch")
        self.reference = finite_array(reference_points_object_m, (self.t, self.k, 3), "reference points")
        self.required = finite_array(required_wrench_object, (self.t, 6), "required wrench")
        self.com = finite_array(com_object_m, (3,), "COM")
        self.environment = np.zeros_like(self.required) if environment_wrench_object is None else finite_array(
            environment_wrench_object, (self.t, 6), "environment wrench")
        self.scale = np.array([self.settings.force_scale_n] * 3 + [
            self.settings.force_scale_n * self.settings.length_scale_m] * 3)
        self.target = ((self.required - self.environment) / self.scale).ravel()
        self.anchor = np.median(self.reference, axis=0)
        self.uv_reference = np.array([patch.tangents.T @ (point - patch.center_object_m)
                                     for patch, point in zip(self.patches, self.anchor)])
        normal_distance = np.array([abs(np.dot(point - patch.center_object_m, patch.inward_normal_object))
                                    for patch, point in zip(self.patches, self.anchor)])
        if np.any(normal_distance > self.settings.max_normal_projection_m):
            raise ValueError("a candidate is too far from its surface; supply a better patch hypothesis")
        extents = np.array([patch.half_extent_uv_m for patch in self.patches])
        shift = self.settings.max_tangent_shift_m
        self.lower_uv = np.maximum(-extents, self.uv_reference - shift)
        self.upper_uv = np.minimum(extents, self.uv_reference + shift)
        if np.any(self.lower_uv >= self.upper_uv):
            raise ValueError("candidate patch does not overlap the allowed correction neighborhood")
        self.initial_uv = np.clip(self.uv_reference, self.lower_uv, self.upper_uv)
        self.dimension = self.t * self.k * 3
        self.effort_coefficient = self.settings.effort_weight / (self.t * self.k)
        self.wrench_coefficient = self.settings.wrench_weight / self.t
        self.rate_coefficient = self.settings.force_rate_weight / ((self.t - 1) * self.k)
        difference = np.zeros((self.t - 1, self.t))
        rows = np.arange(self.t - 1)
        difference[rows, rows] = -1 / np.diff(self.times)
        difference[rows, rows + 1] = 1 / np.diff(self.times)
        self.rate_matrix = np.kron(difference, np.eye(self.k * 3))
        self.regularizer = self.effort_coefficient * np.eye(self.dimension) + self.rate_coefficient * (self.rate_matrix.T @ self.rate_matrix)
        single_cone = np.zeros((self.k * 4, self.k * 3))
        lower = np.empty(self.k * 3)
        upper = np.empty_like(lower)
        for i, patch in enumerate(self.patches):
            usable_mu = patch.friction * self.settings.friction_utilization_limit
            cone = np.array([[-usable_mu, a, b] for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))])
            single_cone[4*i:4*i+4, 3*i:3*i+3] = cone
            cap = patch.max_normal_force_n / self.settings.force_scale_n
            lower[3*i:3*i+3] = [0, -patch.friction * cap, -patch.friction * cap]
            upper[3*i:3*i+3] = [cap, patch.friction * cap, patch.friction * cap]
        self.inequalities = np.kron(np.eye(self.t), single_cone)
        self.lower_force, self.upper_force = np.tile(lower, self.t), np.tile(upper, self.t)
        self.evaluations = 0

    def _force_solution(self, uv):
        points = np.array([patch.point(coord) for patch, coord in zip(self.patches, uv)])
        grasp = np.column_stack([np.vstack((patch.basis, skew((point - self.com) / self.settings.length_scale_m) @ patch.basis))
                                 for patch, point in zip(self.patches, points)])
        operator = np.kron(np.eye(self.t), grasp)
        hessian = 2 * (self.wrench_coefficient * operator.T @ operator + self.regularizer)
        gradient = -2 * self.wrench_coefficient * operator.T @ self.target
        forces = solve_qp(hessian, gradient, G=self.inequalities,
                          h=np.zeros(len(self.inequalities)), lb=self.lower_force,
                          ub=self.upper_force, solver="daqp", primal_tol=1e-8,
                          dual_tol=1e-8)
        self.evaluations += 1
        if forces is None or not np.isfinite(forces).all():
            raise RuntimeError("force QP failed; no corrected result may be accepted")
        residual = operator @ forces - self.target
        cost = self.wrench_coefficient * (residual @ residual) + forces @ self.regularizer @ forces
        return points, forces, residual, float(cost)

    def objective(self, flat_scaled_uv):
        uv = np.asarray(flat_scaled_uv).reshape(self.k, 2) * self.settings.length_scale_m
        _, forces, residual, force_cost = self._force_solution(uv)
        difference = (uv - self.uv_reference) / self.settings.target_sigma_m
        data_cost = np.sum(difference**2) / self.k
        gradient_uv = 2 * (uv - self.uv_reference) * self.settings.length_scale_m / (self.k * self.settings.target_sigma_m**2)
        local_forces = forces.reshape(self.t, self.k, 3)
        residual = residual.reshape(self.t, 6)
        # Envelope derivative of the strongly convex inner optimum. Its cone
        # constraints and force bounds do not depend on planar patch coordinates.
        for k, patch in enumerate(self.patches):
            object_force = local_forces[:, k] @ patch.basis.T
            for axis in range(2):
                dtorque = np.cross(patch.tangents[:, axis], object_force)
                gradient_uv[k, axis] += 2 * self.wrench_coefficient * np.sum(residual[:, 3:] * dtorque)
        return float(data_cost + force_cost), gradient_uv.ravel()

    def describe(self, uv):
        points, force_vector, normalized_residual, _ = self._force_solution(uv)
        local = force_vector.reshape(self.t, self.k, 3) * self.settings.force_scale_n
        object_forces = np.stack([local[:, i] @ patch.basis.T for i, patch in enumerate(self.patches)], axis=1)
        residual = normalized_residual.reshape(self.t, 6) * self.scale
        force_error = np.linalg.norm(residual[:, :3], axis=1)
        torque_error = np.linalg.norm(residual[:, 3:], axis=1)
        cone_slack = np.stack([patch.friction * local[:, i, 0] - np.abs(local[:, i, 1:]).sum(axis=1)
                               for i, patch in enumerate(self.patches)], axis=1)
        planning_slack = np.stack([patch.friction * self.settings.friction_utilization_limit * local[:, i, 0]
                                   - np.abs(local[:, i, 1:]).sum(axis=1)
                                   for i, patch in enumerate(self.patches)], axis=1)
        normal = local[:, :, 0]
        caps = np.array([patch.max_normal_force_n for patch in self.patches])
        force_constraints_ok = bool(np.min(planning_slack) >= -1e-5 and np.min(normal) >= -1e-5 and np.max(normal - caps) <= 1e-5)
        geometry_ok = bool(np.all(uv >= self.lower_uv - 1e-8) and np.all(uv <= self.upper_uv + 1e-8))
        metrics = {
            "max_force_residual_n": float(force_error.max()),
            "rms_force_residual_n": float(np.sqrt(np.mean(force_error**2))),
            "max_torque_residual_nm": float(torque_error.max()),
            "rms_torque_residual_nm": float(np.sqrt(np.mean(torque_error**2))),
            "minimum_friction_diamond_slack_n": float(cone_slack.min()),
            "minimum_reserved_friction_slack_n": float(planning_slack.min()),
            "max_normal_force_n": float(normal.max()),
            "reference_point_rmse_m": float(np.sqrt(np.mean(np.sum((points[None] - self.reference)**2, axis=2)))),
            "max_anchor_correction_m": float(np.linalg.norm(points - self.anchor, axis=1).max()),
            "force_constraints_satisfied": force_constraints_ok,
            "patch_and_correction_bounds_satisfied": geometry_ok,
            "object_wrench_within_tolerance": bool(force_constraints_ok and geometry_ok and force_error.max() <= self.settings.force_tolerance_n
                                                  and torque_error.max() <= self.settings.torque_tolerance_nm),
        }
        return dict(contact_points_object_m=points, contact_force_components_n=local,
                    contact_forces_object_n=object_forces, wrench_residual_object=residual,
                    metrics=metrics)

    def solve(self):
        started = perf_counter()
        before = self.describe(self.initial_uv)
        scale = self.settings.length_scale_m
        initial = self.initial_uv.ravel() / scale
        initial_cost, _ = self.objective(initial)
        result = minimize(self.objective, initial, jac=True, method="L-BFGS-B",
                          bounds=list(zip(self.lower_uv.ravel()/scale, self.upper_uv.ravel()/scale)),
                          options={"maxiter": self.settings.max_iterations, "ftol": 1e-12, "gtol": 1e-7, "maxls": 30})
        chosen = result.x if np.isfinite(result.fun) and result.fun <= initial_cost else initial
        uv = chosen.reshape(self.k, 2) * scale
        after = self.describe(uv)
        return dict(before=before, after=after, timestamp_s=self.times.copy(),
                    required_wrench_object=self.required.copy(), environment_wrench_object=self.environment.copy(),
                    reference_points_object_m=self.reference.copy(),
                    report={"runtime_seconds": perf_counter()-started, "outer_iterations": int(result.nit),
                            "force_qp_solves": self.evaluations, "optimizer_converged": bool(result.success),
                            "optimizer_message": str(result.message), "initial_objective": initial_cost,
                            "final_objective": float(min(initial_cost, result.fun)),
                            "before": before["metrics"], "after": after["metrics"],
                            "robot_feasibility_checked": False, "physical_rollout_performed": False,
                            "scope": "one sticking phase; planar point-contact patches; fixed object trajectory; object-level model only"})


def validate_dense(result, dense_times, dense_required, com, settings, dense_environment=None):
    """Check interpolated force profiles at EVERY supplied reference frame.

    Interpolation preserves the convex friction/cap constraints for these fixed
    object-frame patches. It does not automatically preserve wrench tracking.
    A successful knot solve must not hide a failed dense check.
    """
    dense_times = np.asarray(dense_times, dtype=float)
    times = result["timestamp_s"]
    if dense_times.ndim != 1 or not len(dense_times) or not np.isfinite(dense_times).all() or np.any(np.diff(dense_times) <= 0):
        raise ValueError("dense times must be finite and increasing")
    if dense_times[0] < times[0] - 1e-10 or dense_times[-1] > times[-1] + 1e-10:
        raise ValueError("dense validation must not extrapolate beyond solved knots")
    required = finite_array(dense_required, (len(dense_times), 6), "dense required wrench")
    environment = np.zeros_like(required) if dense_environment is None else finite_array(dense_environment, required.shape, "dense environment")
    forces = result["after"]["contact_forces_object_n"]
    interpolated = np.stack([np.interp(dense_times, times, column) for column in forces.reshape(len(times), -1).T], axis=1).reshape(len(dense_times), -1, 3)
    points = result["after"]["contact_points_object_m"]
    torque = np.cross(points[None] - np.asarray(com)[None, None], interpolated).sum(axis=1)
    actual = np.column_stack((interpolated.sum(axis=1), torque)) + environment
    residual = actual - required
    max_force = float(np.linalg.norm(residual[:, :3], axis=1).max())
    max_torque = float(np.linalg.norm(residual[:, 3:], axis=1).max())
    return {"dense_contact_forces_object_n": interpolated, "dense_wrench_residual_object": residual,
            "report": {"frames_checked": len(dense_times), "max_force_residual_n": max_force,
                       "max_torque_residual_nm": max_torque,
                       "object_wrench_within_tolerance": bool(result["after"]["metrics"]["force_constraints_satisfied"]
                           and result["after"]["metrics"]["patch_and_correction_bounds_satisfied"]
                           and max_force <= settings.force_tolerance_n and max_torque <= settings.torque_tolerance_nm)}}
