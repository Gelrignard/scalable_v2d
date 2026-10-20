"""Read-only records at each joint's actual peak request, before clipping."""
import numpy as np


class TorquePeakAudit:
    """Different joints can peak at different physics steps in one interval."""
    term_names = ('position_pd', 'velocity_pd', 'compensation', 'contact_feedforward')

    def __init__(self, frames, caps, contacts=0):
        self.caps = np.asarray(caps, float)
        shape = (frames, len(self.caps))
        self.fraction = np.full(shape, -1.)
        self.times = np.full(shape, np.nan)
        self.terms = np.zeros((*shape, len(self.term_names)))
        self.enabled = np.zeros((*shape, contacts), bool)
        self.touching = np.zeros_like(self.enabled)
        self.gap = np.full((*shape, contacts), np.nan)
        self.contact_step = np.zeros(shape)

    def update(self, frame, time, requested, terms, *, enabled=None, touching=None,
               gap=None, contact_step=None):
        fraction = abs(requested)/self.caps
        changed = fraction > self.fraction[frame]
        self.fraction[frame, changed] = fraction[changed]
        self.times[frame, changed] = time
        self.terms[frame, changed] = np.asarray(terms)[changed]
        for destination, value in ((self.enabled, enabled), (self.touching, touching),
                                   (self.gap, gap)):
            if value is not None:
                destination[frame, changed] = value
        if contact_step is not None:
            self.contact_step[frame, changed] = np.asarray(contact_step)[changed]

    def export(self):
        return dict(
            torque_peak_time_seconds=self.times, torque_peak_terms_nm=self.terms,
            torque_peak_term_names=np.asarray(self.term_names),
            torque_peak_force_enabled=self.enabled, torque_peak_touching=self.touching,
            torque_peak_contact_gap_m=self.gap,
            torque_peak_contact_torque_step_nm=self.contact_step)
