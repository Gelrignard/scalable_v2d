"""Bounded evaluation of reviewed contact explanations with common data/state."""
from dataclasses import replace
from time import perf_counter

import numpy as np

from .coupled_horizon import HorizonSettings, solve_horizon
from correction.src.dynamics.horizon_audit import audit_horizon


def solve_candidates(candidates, settings=None, *, max_candidates=3, seconds=120.):
    """Select only nominally feasible hypotheses, retaining every rejected result.

    Candidate tuples are (name, reference, robot). Geometry and physical replay
    are later gates; this selection is explicitly not execution acceptance.
    The budget stops new solves, but is not a hard limit on solver construction.
    """
    if max_candidates < 1 or seconds <= 0:
        raise ValueError('Candidate count and budget must be positive')
    started = perf_counter()
    results = []
    candidates = list(candidates)[:max_candidates]
    # Validate comparison contracts before paying for any nonlinear solve.
    if candidates:
        initial = candidates[0][1]
        for _, ref, _ in candidates[1:]:
            for field in ('times', 'observation_times', 'observed_positions', 'observed_wxyz', 'base', 'joint_reference',
                          'initial_q', 'initial_joint_velocity', 'initial_position', 'initial_wxyz',
                          'initial_velocity', 'initial_angular_momentum', 'goal_position', 'goal_wxyz',
                          'position_sigma', 'rotation_sigma', 'mass', 'inertia', 'half_size',
                          'goal_position_tolerance', 'goal_rotation_tolerance', 'goal_rest'):
                if not np.array_equal(getattr(initial, field), getattr(ref, field)):
                    raise ValueError(f'Hypothesis comparison must share {field}')
    for name, ref, robot in candidates:
        if perf_counter()-started >= seconds: break
        cfg = settings or HorizonSettings()
        remaining = seconds-(perf_counter()-started)
        cfg = replace(cfg, max_seconds=min(cfg.max_seconds, max(1., remaining/(2 if cfg.lexicographic else 1))))
        result = solve_horizon(ref, robot, cfg)
        audit = audit_horizon(ref, robot, result) if result['success'] else {'accepted_nominal': False}
        results.append(dict(name=name, result=result, audit=audit))
    eligible = [r for r in results if r['audit']['accepted_nominal']]
    selected = min(eligible, key=lambda r: r['result']['data_cost'])['name'] if eligible else None
    return dict(selected=selected, candidates=results, seconds=perf_counter()-started,
                physical_execution_validated=False)
