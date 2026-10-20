"""Deterministic single-contact-interval timing from retargeting evidence."""
import numpy as np
from ..optimization.surface_reference import PhaseTiming


def single_phase(active, times):
    active, times = np.asarray(active, bool), np.asarray(times)
    if active.ndim != 2 or len(active) != len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('Matching contact channels and increasing timestamps required')
    changes = np.diff(np.r_[False, active.any(axis=1), False].astype(int))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)-1
    if len(starts) != 1:
        raise ValueError('This transfer runner requires exactly one observed contact interval')
    first, last = int(starts[0]), int(ends[0])
    dt = float(np.median(np.diff(times)))
    close = max(2, round(.4/dt))
    approach = max(close+1, round(1.2/dt))
    ramp = max(2, round(.8/dt))
    remaining = len(times)-1-last
    if first <= approach or last-first < ramp+2 or remaining < 6:
        raise ValueError('Not enough recorded time for an unassisted approach and release')
    force_end = last+min(close, remaining//3)
    release_end = last+min(3*close, 2*remaining//3)
    hold_end = min(len(times)-1, release_end+2*close)
    phase = PhaseTiming(first-approach, first-close, first, first+ramp,
                        last, force_end, release_end, hold_end)
    phase.validate(len(times))
    frames = sorted(set(int(first+fraction*(last-first)) for fraction in (.25, .55, .8)))
    return phase, frames
