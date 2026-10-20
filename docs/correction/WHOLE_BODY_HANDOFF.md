# Correction to whole-body control: milestone 7.5

This handoff follows the user's 20 October decision: this chat develops and
checks correction through milestone 7; policy/batch/Tier 3/official evaluation
work (M8–M11) can be handled by the other agent. The small additional episode
comparisons requested here are exploratory transfer checks, not the complete
M8 benchmark.

## What changed overnight

The preserved implementation is commit `bfd818b` on `main`. The
[saved bin comparison](../../correction/examples/bin_checkpoint_20261020/README.md)
contains its derived commands and physical recordings. The
[morning report](OVERNIGHT_REPORT_2026-10-20.md) records every remaining failure.

The core idea is unchanged: human motion constrains plausible contact regions;
object motion specifies task progress; mechanics constrains forces and robot
motion before learning. However, the displayed result is a **staged local
approximation**, not the complete coupled horizon optimizer originally proposed.

| Stage | Actual implementation | Remaining limitation |
|---|---|---|
| Contact evidence | Human-derived regions, directions and intervals | Not a validated perception-uncertainty model |
| Robot contact hypotheses | Actual G1/Dex3 and object collision surfaces; bounded patch optimization | Finite local candidates and task-specific choices in the bin prototype |
| Force allocation | Friction/normal/actuator constraints; force effort and rate regularization | Mostly quasistatic; no observed per-frame acceleration demand, but no complete dynamic task model either |
| Motion construction | Object-relative grasp transport, arm/waist IK, approach/release paths | Fixed grasp templates can miss sliding, rolling or changing contacts |
| Trajectory optimization | Sparse C2 spline QP with local contact Jacobians, nominal robot dynamics, joint/rate bounds; later collision constraints in the same variables | Frozen local linearization omits actual servo tracking error and contact-mode changes |
| Validation | Full physical episode with finite torque caps and contact solver | Supported pelvis; no walking/balance validation |

The later phase-prediction and bounded surface-feedback experiments are separate
ablations. They did **not** replace the retained 48.48 mm bin result. In
particular, the 57.72 mm feedback trial uses a perfect simulated object-pose
sensor and is not the displayed best reference.

## The robot already has dynamics

In the robot channel, arm, wrist, finger and waist joints have inertia,
gravity, damping, collisions and finite torque limits. PD tracking and contact
guidance produce motor torques; the simulator determines whether contact occurs
and which force it delivers. Requested force is not directly applied to the
object and is not guaranteed to be realized.

The root follows an external mocap weld, and the leg joints are frozen with
their contacts disabled. That fixture can supply support and reaction loads
that a freely balancing robot would have to produce through its feet. Thus the
remaining question is not whether to enable arm dynamics; it is whether the
same manipulation remains achievable with a finite, balancing whole-body
controller and its actual command interface.

The independent-pad channel is a different experiment. It has dynamic
finite-mass pads and a free object, but no articulated arm or balance problem.
Do not use its results as whole-body validation.

## What SONIC provides

SONIC is a learned general motion-tracking controller for humanoid whole-body
movement, with motion representations that can also support manipulation
interfaces. It is not simply a walking IK solver.
[Official SONIC project](https://nvlabs.github.io/GEAR-SONIC/).

The V2D whole-body implementation describes reference encoding through SONIC,
decoded joint actions with RL residuals, and separate finger residual support.
Its contact/wrench rewards are objectives for learning; they are not promises
that a desired hand wrench is exactly delivered.
[Official V2D whole-body interface](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/README.md).

Our inference is that successful reference tracking alone cannot establish
force compatibility: balancing may change the pelvis, wrist tracking and
available joint torque. The local torque-guidance adapter must not be silently
discarded when exporting motion to SONIC, or assumed to be equivalent to its
joint-action interface. Nor should an upper-body torque controller and SONIC
independently command the same joints.

## M7.5: controller compatibility before the RL handoff

This is a joint boundary milestone that either agent can implement. Its result
must be available to both sides.

1. **Pin the executable interface.** Record SONIC/runtime/checkpoint versions,
   joint order, frames, update rates, gains, actuator limits, action units,
   finger-control ownership and reference horizon. Decide explicitly which
   controller owns each joint. Keep intended object motion separate from all
   optimized/reference/predicted/executed arrays.
2. **Zero-residual replay first.** Run original and corrected references through
   the same pretrained whole-body controller from the original episode start.
   Enable foot contacts and free base; remove the external fixture. Use one
   uninterrupted episode with a free physical object. No retraining is required
   for this compatibility measurement.
3. **Measure what transfers.** Record base tracking/balance, hand/contact gaps,
   actual object motion, actual contact impulses, joint saturation, falls,
   joint continuity, collisions and command latency. Compare against the
   supported benchmark without presenting its fixture as free support.
4. **Resolve force guidance.** If the downstream interface accepts only joint
   commands, test a bounded compatible impedance/residual mapping with the
   same gains and rates. Export force references as guidance/targets with
   measured realization, not as guaranteed controls. Test original and
   corrected references with the same adapter to isolate correction benefit.
5. **Return failures to correction.** Persistent balance or wrist/force errors
   become constraints/margins for the next correction pass. Retiming must
   preserve task/contact meaning and be declared. Do not assume RL will cheaply
   repair an incompatible interface.

The supported replay should export a versioned bundle containing timestamps,
original and corrected q/dq, root and EEF references, selected robot/object
patch IDs, contact schedule, desired force frame/sign, model assumptions,
controller parameters, initial state, acceptance failures and provenance.
The receiver must retain the **actual simulated** trajectory as a separate
validation record. A successful M7.5 replay does not itself prove lower RL cost.

Acceptance requires the explicit downstream action contract to execute without
hidden assistance, collapse or interface errors; tracking and torque criteria
must be declared in advance. If it fails, keep the reference as a diagnostic
and report the discrepancy before policy learning. The existing 10 mm / 5°
local gate is a research gate, not a claim about official leaderboard scoring.

No whole-body runtime, weights or GPU stack is installed by this handoff.
No training job is launched here. The preservation commit and additional
supported-episode diagnostics remain usable while the other agent prepares
the downstream environment.
