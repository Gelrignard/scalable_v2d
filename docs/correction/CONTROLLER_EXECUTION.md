# Controller-aware execution and fallback

Implementation record, 19 October 2026. This extends the
[coupled trajectory formulation](COUPLED_HORIZON.md). It does not replace its
contact feasibility, collision, continuity, or independent execution checks.

## What is preserved

The research hypothesis remains that mechanical trajectory correction can
produce better training references at modest cost. We have not established an
RL speedup. Human motion still constrains plausible contacts; uncertain object
poses are fitted over a horizon. No new force targets are obtained by taking
framewise second differences of those observations.

The physical robot trajectory, servo equilibrium targets, and actual simulated
motion are separate arrays. Forces are always robot-on-object. The object is a
free body throughout execution. Walking remains excluded: the pelvis follows
the recorded externally supported path and the legs cannot introduce balance
contacts. Display coordinates preserve the original world frame.

## Command conversion

For physical reference `q*`, physical velocity `v*`, and planned contact torque
`tau_contact = sum(J_i.T @ f_i)`, two nominally equivalent controllers are:

```text
Torque interface:
  q_cmd = q*
  v_cmd = v*
  tau_ff = compensation + tau_contact

Position interface:
  q_cmd = q* + tau_contact / Kp
  v_cmd = v*
  tau_ff = compensation

Both execute:
  tau = Kp * (q_cmd - q_actual) + Kd * (v_cmd - v_actual) + tau_ff
```

`compensation` contains the existing robot gravity, inertial and applicable
passive-force compensation. This position interface converts the **contact
torque**, not every feedforward term. It is not yet the official challenge
policy adapter. The velocity target remains the physical velocity, not the
derivative of the shifted equilibrium. Adding contact torque again after the
conversion would double-count it.

The conversion is exact before limits for known gains. The runtime limits
offsets to 0.08 rad for arms/waist and 0.10 rad for fingers, and requests an
offset rate below 0.5 rad/s. Joint-target limits have priority. If an invalid
physical reference makes the joint bounds and the rate bound incompatible,
the conflict is recorded and the reference is rejected. It is not silently
called a rate-limited, feasible command. The torque lost through clipping or
rate limiting is saved as `conversion_residual`.

The coupled optimizer has an opt-in servo contract (`servo_kp`,
`servo_max_offset`, `servo_offset_rate`). It constrains the nominal preload,
command joint limits and preload rate at knots; the dense audit checks between
knots. This is not a continuous-time guarantee. Default settings retain the
previous optimization problem.

## Normal-force feedback

The feedback path uses simulated contact forces at the corresponding distal
robot body, within 30 mm of the selected point and facing the intended inward
normal. A different face of the finger does not count as the intended contact.
Multiple channels on one body are rejected because this prototype cannot
separate their measurements.

The measured normal force is filtered with a 20 ms time constant. A proportional
gain of 0.25 produces a correction capped at 2 N, with a 100 N/s rate limit.
There is no integral state. Correction is zero when contact is absent, its
schedule is inactive, or desired compression is zero. The estimate resets on
contact loss. Planned feedforward is gated to actual contact or a point within
2 mm of the object box; it cannot act on the object across an arbitrary gap.

All new controller variants use the same 15 N normal cap and friction limit
of 0.45. Tangential demand is reduced if it exceeds the friction allowance for
the commanded normal force. The feedback/no-feedback comparison consequently
uses the same limiter. This is a bounded prototype; no claim of stability for
arbitrary contact transitions is made. Force measurements are available in the
simulator; hardware sensing/estimation has not been implemented.

## Full-episode test and acceptance

The complete controller comparison uses episode 18, the first episode in the
previously recorded random selection: **858 frames, about 28.57 seconds**. It
contains approach, interaction, and departure at original timing. The same
initial state is used for all variants, with no resets inside an episode.

The input is the **saved historical failed trajectory and force plan**. It is
held fixed to isolate the controller changes. It is not an accepted output of
the new coupled optimizer, and no acceleration-derived force plan is recomputed
for this experiment. A failed controller test on this input cannot establish
that force feedback is generally ineffective.

The variants are original retargeting, historical feedforward, gated
feedforward, position conversion, torque plus force feedback, and position
conversion plus force feedback. New variants use common sensing, force limits,
and contact gating. The old branch remains exactly reproducible.

Promotion requires physical acceptance (object tracking, penetration, joint and
effort limits, no direct object assistance), no worse object RMS error, no more
than a 1% regression in joint/EEF acceleration or EEF step, and a valid command
conversion with at most 0.01 Nm residual. These thresholds are explicit local
diagnostic gates, not the challenge leaderboard metric. Every result remains
`ready_for_training=false`.

The first comparison revealed that normal/friction limiting differed between
the feedback and no-feedback branches. It is retained in
`correction/outputs/g1_controller_full_20261019/` as an exploratory result.
The matched comparison is in
`correction/outputs/g1_controller_full_20261019_v2/`.

### Measured outcome

All six full-episode records fail physical acceptance. Lower position error
alone does not establish successful manipulation.

| Controller | Object RMS error (mm) | Joint acceleration RMS (rad/s²) | Maximum penetration (mm) |
|---|---:|---:|---:|
| Original retargeting | 449.0 | 25.94 | 2.36 |
| Historical corrected reference + feedforward | 509.4 | 98.23 | 1.24 |
| Gated feedforward, common force limits | 742.4 | 99.58 | 2.03 |
| Position conversion, common force limits | 332.2 | 99.77 | 2.05 |
| Gated feedforward + force feedback | 684.3 | 99.82 | 1.05 |
| Position conversion + force feedback | 444.1 | 100.19 | 0.071 |

Feedback modestly reduces position error relative to the matched gated-torque
baseline, but does not make the manipulation pass. It worsens position error
relative to the matched position-conversion baseline. The historical corrected
reference also remains much shakier than original retargeting. No new variant
is promoted, and the retained fallback itself is explicitly marked infeasible.

The saved robot cubic spline already exceeds joint limits by **0.0521 rad**, has
peak speed **1.33 times the declared limit**, and peak commanded joint
acceleration **3306 rad/s²**. These are robot command derivatives, not perceived
object acceleration. Thus the controller is being asked to execute an invalid
reference before considering force regulation. New physical replays request
up to roughly 74–78 times the actuator effort caps; actual applied torque is
clipped and the violation is reported.

Position conversion loses up to **7.67 Nm** to command constraints and records
1443 steps where joint and offset-rate limits conflict. Matching contacts exist
at only 23–30 of the 858 sampled frames in the new variants. Force feedback
cannot supply a missing grasp. These measurements identify the next work as
feasible contact geometry and continuous, actuator-compatible trajectory
planning; they do not support merely increasing feedback gains.

The four new complete replays plus loading the two unchanged baseline records
took **235 s wall time on CPU**. No GPU or policy training was used. The old
controller reproduced its previously saved object trajectory with zero
numerical difference. The regression and focused follow-up runs passed **23
distinct tests**; these include controlled MuJoCo execution, command/torque
equivalence, force direction and gating, clipping, dense servo-contract
rejection, G1 free-space invariants and prior saved-rollout reproduction. Test
success does not override the failed full-episode results.

The single comparison viewer shows **original retargeting on the left** and
**our current candidate on the right**, with all 858 frames at original timing.
Controller ablations remain in saved reports and are not separate UI choices.
Solid objects show simulated motion. An orange ghost, outline and path always
show the intended source object motion in both views; a red segment shows the
position discrepancy. There is no accepted full-episode latent object estimate
from the coupled solver to substitute for that source target.

Keep only one preview server running. The active comparison uses port **8087**;
the obsolete preview servers on **8081–8086** were stopped. Playback starts
paused, does not update frames while paused, and stops when all clients leave.
The two channels load four relevant replay files; they share one server and
one playback clock. On narrow screens, clicking
the **Connected** bar toggles the control panel so it does not cover a robot.

### Independent support-point channel

The **Channel** selector now includes **Independent support-point simulation**.
This is a separate physical comparison for the **same complete episode 18**,
858 frames / 28.57 seconds, not an abstraction of the G1 recorded forces and not
the earlier 1.5-second book segment. Both channels retain the orange intended
object motion. Changing channels pauses playback and preserves the frame.

The independent experiment removes the robot and uses six spherical pads with
three actuated translation degrees of freedom each. Each pad has mass 0.04 kg,
radius 6 mm, stiffness 1200 N/m, critical damping 13.86 N s/m and a 25 N total
actuator force limit. The object retains the same 24 convex collision pieces,
2 kg mass, 0.6 friction, floor height and 1 ms integration step as the G1 test.
Both cases start at the same original fingertip locations, with the same object
pose and zero velocities, **before approach**. There is no reset to an
established grasp, attachment constraint, or force applied directly to the object.

The original uses FK positions from the original retargeted commands. The
corrected case uses the saved pre-IK `tip_targets` and world-frame `force_ff`.
For pad position `x`, desired position `x*`, and desired velocity `v*`, the
actuator command is:

```text
u = norm_limit_25N(Kp (x* - x) + Kd (v* - v) - m_pad g + f_guidance)
```

The original sets `f_guidance = 0`. The corrected trial uses the same bounded
normal-force feedback philosophy as the robot controller (gain 0.25, 2 N
correction limit, 15 N normal cap, friction bound 0.45). Here guidance is
enabled only when a physical pad-object contact faces the intended inward
normal and its schedule is active. PD must establish contact first. The desired
force and actual contact reaction are separate signals; actuator force also
accelerates the pad. All plan data are held fixed. No new optimizer or
observation-acceleration force estimation runs in this experiment.

The physical pads are blue spheres; the simulated object is translucent blue
to reveal contacts on its far side. Cyan points and traces show intended
pad commands during contact phases. Purple arrows show the saved planned
pad-on-object forces; green arrows show measured per-pad resultant forces at
force-weighted contact centroids. Both start at **10 mm per newton**, adjustable
together with the force-scale control, without normalization or clipping.
No measured marker is shown without a force-bearing
contact. Planned forces are not drawn on the original side, which has no force
plan. The table also shows the actual gated force-guidance commands. Aggregated
arrows do not represent every contact location or the complete moment; contact
moments are saved separately in the NPZ records.

| Independent pads | Original | Corrected + force guidance |
|---|---:|---:|
| Object position RMS | 559.72 mm | 566.64 mm |
| Worst object position error, all steps | 911.04 mm | 920.50 mm |
| Frames with physical pad-object contact | 65 | 63 |
| Frames with force guidance enabled | 0 | 62 |
| Worst penetration, all steps | 0.118 mm | 0.250 mm |
| Tracking gate | failed | failed |

Generation took **45.4 s CPU wall time**, with no GPU or RL. This shows that the
current fixed candidate also fails in the reduced contact model; it does not
establish that an improved optimization cannot work. Nor does success in such
a reduced model, if achieved later, establish G1 reachability or joint feasibility.
The viewer itself only loads these recordings and remains idle while paused.
The old short successful book experiment is not used in this comparison.

Reproduce into a new directory:

```text
python -m correction.scripts.compare_support_points --output correction/outputs/my_support_points
python -m correction.scripts.view_controller_comparison --support-directory correction/outputs/my_support_points --port 8087
```

This runner currently rejects episodes containing additional support geometry,
rather than silently omit it. Recorded physics timestamps are also saved; their
rounding error relative to source frames is at most half the 1 ms physics step.
Checks cover absence of remote object actuation and agreement between measured
contact impulse and object momentum in a controlled push. Those checks verify
the simulator interface, not the quality of this failed manipulation.

## Fallback and reproduction

Nothing automatically changes the default controller. Omit `contact_controller`
when calling `SupportedRobotScene.rollout` to execute the historical code path.
Set `servo_kp=None` to use the original coupled optimization problem. Optional
controller ablations in the coupled runner require `--compare-controllers`.
Original retargeting files and old result directories are never overwritten.

Keeping the historical controller is a **behavioral rollback**, not a claim
that its manipulation is feasible. If no candidate meets acceptance, the
decision records `promoted=false` and also records whether the fallback itself
is feasible. The working copies of the three original replay files are backed
up locally in `correction/outputs/controller_checkpoint_20261019/`.

```sh
python -m unittest correction.tests.test_contact_controller correction.tests.test_horizon_replay correction.tests.test_symbolic_robot correction.tests.test_supported_robot correction.tests.test_full_episodes -v

# Complete physical episodes; choose an unused output directory.
python -m correction.scripts.compare_g1_controllers --output correction/outputs/my_controller_comparison
python -m correction.scripts.view_controller_comparison --directory correction/outputs/my_controller_comparison --port 8087

# Coupled G1 integration experiment (first pickup only, not a full demonstration).
python -m correction.scripts.coupled_book --controller-aware-position --compare-controllers --initialize-contacts --output correction/outputs/my_controller_aware_pickup
```

`--reuse-fixed-baselines PATH` can reuse the original and historical-controller
rollouts only when the input file hashes match. New controller variants are
always simulated. `--assess-existing` recomputes acceptance from unchanged saved
physical trajectories and retains the earlier report.

## Code and output contract

| File | Purpose |
|---|---|
| `correction/src/dynamics/contact_controller.py` | Torque/position conversion, contact sensing, bounded feedback, execution diagnostics |
| `correction/src/dynamics/supported_robot.py` | Optional controller hook; unchanged default execution |
| `correction/src/optimization/coupled_horizon.py` | Optional servo constraints inside trajectory optimization |
| `correction/src/dynamics/horizon_audit.py` | Dense servo and mechanics checks |
| `correction/src/pass_traj/controller_reference.py` | Physical references and nominal actuator targets exported separately |
| `correction/scripts/compare_g1_controllers.py` | Complete controller ablations and explicit fallback decision |

Each physical record contains actual robot/object poses, applied torques,
contact forces, and timing. New records additionally contain requested/applied
servo targets, physical velocity targets, compensation, contact torque,
conversion residuals, matched contacts and force-feedback corrections. The
last frame holds the preceding command; it does not add an extra physics step.
The export is diagnostic and does not bypass the trajectory acceptance gates.
