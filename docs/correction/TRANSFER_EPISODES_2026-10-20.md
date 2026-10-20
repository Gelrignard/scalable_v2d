# Two additional complete episodes — 20 October 2026

The preserved bin result is unchanged. Two different downloaded Tier 1 tasks
were processed to expose transfer limitations: episode 10 (desk) and episode
23 (stroller). These were selected before evaluating outcomes, not chosen for
success. This is exploratory work requested by the user, not the full M8
benchmark or a policy-training experiment.

## Viewer

Use the single [viewer on port 8087](http://127.0.0.1:8087/). Select **Episode**
to switch between the preserved bin, desk and stroller. Each has the same two
channels: robot simulation and independent support-point simulation.

Orange always shows the intended object pose/path. Solid robot/object motion
is the actual simulated result. In the pad channel, cyan shows intended pad
paths, purple planned force and green measured force. All episodes run from
their original pre-approach state without phase resets or direct object help.

```text
python -m correction.scripts.view_episode_catalog --catalog experiments/correction_transfer_20261020.json --port 8087
```

The viewer replaces the old server on the same port. It pauses without a
connected viewer and does not run physical simulation in the background.
Switching episodes resets playback, not a physical experiment: every record
was already simulated continuously from frame 0.

## Results

| Episode | Frames / duration | Original robot position RMS | Diagnostic candidate RMS | Outcome |
|---|---:|---:|---:|---|
| Bin 18, preserved | 858 / 28.57 s | 444.99 mm | 48.48 mm | Partial improvement; physical gate fails |
| Desk 10 | 794 / 26.43 s | 704.28 mm | 738.35 mm | Planning rejected; diagnostic replay worsens |
| Stroller 23 | 946 / 31.50 s | 841.86 mm | 982.19 mm | Planning rejected; diagnostic replay worsens |

The new right-hand views are **rejected diagnostic candidates**, not successfully
optimized trajectories. Neither episode passes the contact-selection gates;
the continuous spline problem is primal infeasible in both. The preceding
finite geometric command was replayed so the failure can be inspected. This
fallback is prominent in the viewer and saved report; it is never exported as
training-ready.

Desk rotation RMS worsens 69.00° to 131.85°; stroller rotation RMS changes
102.79° to 93.47° while its position tracking worsens. Peak requested torque
is 57.71 times the cap for the desk candidate and 10.40 for the stroller.
Actual applied torques remain clipped at the same real caps in both branches.
These are substantial failures, not small residual errors to hide in an average.

Independent pads also fail: corrected desk RMS 747.30 mm and stroller
1050.43 mm. These are separate Cartesian-actuator diagnostics, not substitutes
for the articulated robot experiment.

## What was kept fixed and what was generalized

Both original and candidate use the same supported-pelvis path, initial
robot/object state, gravity/inertia compensation, friction, timestep and motor
gains/caps as the bin benchmark. Model defaults are 1 kg, COM at object origin,
inertia diag(0.01, 0.01, 0.01), friction 1, planning friction 0.8, arm/hand
position gains 300/2. These source-derived defaults are not measured physical
properties of either object and are not official-runtime parity.

The new runner infers one contiguous contact interval from retargeting labels.
It tests three surface families at three predetermined fractions of that
interval, with a 75-second screening budget and 30-second spline-solver budget.
Patch choice uses geometry and static capacity, never physical outcome.
Approach/release timings are deterministic functions of the recorded interval
and sample period. No new episode received the bin's manually selected
two-grasp schedule or its later hand-tuned transition repairs.

The same actual-surface, grasp transport, force allocation and spline modules
are used. Both selected candidates have small positional patch gaps
(0.456 mm desk, 0.904 mm stroller), but normal-cone residuals exceed the
1 mm local gate and their unsupported static support screens leave 9.81 N
unbalanced. Numerical force-profile QPs solve within their constraints, yet
task-wrench residuals reach 18.62 N and 16.97 N. A solved QP is therefore not
a solved task.

The force profile still primarily asks the hands to support gravity. It does
not jointly optimize environment-supported sliding or rolling, nor fit a
latent object trajectory robust to support-plane uncertainty. The stroller
is a single rigid mesh with **unarticulated wheels**. Its replay cannot
establish performance on the actual rolling task.

These results show that the current staged template does not yet transfer
unchanged to these interactions. They do not establish that the original
long-horizon mechanics idea fails. Contact strategy, environment support and
model/controller compatibility remain the missing pieces.

## Cost, reproduction and records

Desk construction/screening/optimization plus two full robot replays:
113.47 s wall / 112.23 CPU-seconds, 1.26 GB peak resident memory. Stroller:
151.73 s / 151.59 CPU-seconds, 1.32 GB peak. The independent-pad pairs add
14.47 s and 18.17 s. Combined recorded process wall time is 297.84 s and CPU
time 297.33 s. The two independent episode processes overlapped, so their sum
is not elapsed user waiting time. Tests, viewer and development are excluded.
No GPU, paid compute or training was used.

```text
python -m correction.scripts.transfer_episode --source correction/outputs/random_three_full_20261019/episode_000010 --output correction/outputs/my_desk --replay-rejected
python -m correction.scripts.compare_support_points --directory correction/outputs/my_desk --output correction/outputs/my_desk_pads
```

Use episode 23 analogously. Always use fresh output directories. The runner
now stops at failed planning gates by default; `--replay-rejected` explicitly
requests the diagnostic behavior used here. This guard was added after the
recordings. Their saved implementation snapshots retain the exact executed
version and all failed gates.

- [Portable result summary and hashes](../../experiments/correction_transfer_20261020_results.json)
- [Viewer catalog](../../experiments/correction_transfer_20261020.json)
- [Desk full report](../../correction/outputs/transfer_20261020/episode_000010_v1/report.json)
- [Stroller full report](../../correction/outputs/transfer_20261020/episode_000023_v1/report.json)
- [Whole-body control boundary, M7.5](WHOLE_BODY_HANDOFF.md)

Three timing tests pass: full contact-interval preservation, rejection of
multiple intervals rather than silently dropping one, and rejection of an
artificial pre-approach when the recording starts in contact. Both actual
physical pairs verify identical initial states, full frame counts and zero
object assistance. Viewer preflight verifies robot/pad reference hashes and
frame counts. Browser checks exercise the episode selector, mid-episode
scrubbing and pad channel. These checks validate the records/UI, not task success.

The preserved code/result checkpoint is commit `bfd818b`. M8–M11 remain assigned
to the user's other agent; these exploratory failures and the M7.5 handoff
should accompany any later learning-cost interpretation.
