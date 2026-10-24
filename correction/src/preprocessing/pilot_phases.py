"""Fixed input-only phase windows for the paired pilot; not contact truth."""
from dataclasses import asdict
import numpy as np
from scipy.ndimage import median_filter
from ..optimization.surface_reference import PhaseTiming


def infer_phases(active, times):
    active, times = np.asarray(active, bool), np.asarray(times)
    if active.shape != (len(times), 6) or len(times)<8 or np.any(np.diff(times)<=0):
        raise ValueError('Six inferred fingertip channels and increasing full timestamps required')
    dt = float(np.median(np.diff(times)))
    # Remove brief on/off spikes without consulting outcomes or clean-tier data.
    width = max(1, round(.2/dt)//2*2+1)
    hands = np.column_stack((active[:,:3].any(axis=1), active[:,3:].any(axis=1)))
    stable = median_filter(hands.astype(int), size=(width,1), mode='nearest').astype(bool)
    mode = stable[:,0].astype(int)+2*stable[:,1].astype(int)
    edges = np.r_[0, np.flatnonzero(np.diff(mode)!=0)+1, len(mode)]
    result = []; rejected = []
    for a,b in zip(edges[:-1],edges[1:]):
        if mode[a] == 0:continue
        first,last=int(a),int(b-1)
        if last-first+1 < max(10, round(.4/dt)):
            rejected.append(dict(first=first,last=last,reason='Short contact-mode segment; original motion retained'))
            continue
        # Neighbouring different-hand modes may touch without a free interval.
        # In that case reserve transition time inside this segment, report it,
        # and do not claim that the observed handover has been solved.
        previous = result[-1]['timing']['hold_end'] if result else 0
        approach = max(previous+1, first-round(1.2/dt), 1)
        contact = max(first, approach+3)
        next_edge = int(b)
        available_after = (next_edge == len(mode) or mode[next_edge] != 0)
        end = min(last, len(mode)-5) if not available_after else min(last-4,len(mode)-5)
        if end-contact < 3:
            rejected.append(dict(first=first,last=last,reason='Insufficient acquisition/release time; original motion retained'))
            continue
        close = max(approach+1,contact-max(1,round(.4/dt)))
        close = min(close,contact-1)
        force_full = min(end,contact+max(2,round(.8/dt)))
        stop = min(len(mode)-1, last if available_after else last+round(.8/dt))
        if not available_after and next_edge < len(mode):
            next_nonzero = np.flatnonzero(mode[next_edge:] != 0)
            if len(next_nonzero): stop=min(stop,next_edge+int(next_nonzero[0])-1)
        stop=max(end+3,stop)
        if stop >= len(mode):stop=len(mode)-1
        timing=PhaseTiming(approach,close,contact,force_full,end,end+1,min(end+3,stop),stop)
        timing.validate(len(mode))
        sides=[s for bit,s in ((1,'left'),(2,'right')) if mode[a]&bit]
        result.append(dict(timing=asdict(timing),sides=sides,source_interval=[first,last],
            frames=sorted(set(int(contact+f*(end-contact)) for f in (.25,.65))),
            compressed_transition=bool(contact!=first or end<last),
            note='Input-only experimental mode segmentation; handover transitions require physical validation'))
    return result,dict(filter_seconds=.2,minimum_mode_seconds=.4,rejected_intervals=rejected,
                       original_contact_frames=int(active.any(axis=1).sum()))
