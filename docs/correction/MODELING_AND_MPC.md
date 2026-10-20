# Modeling checkpoint and the first coupled MPC experiment

The retained bin reference is produced by a **staged** pipeline: select robot
surface patches, construct an object-relative hand path with IK, allocate
quasistatic forces, then refine a smooth robot command. It is not the earlier
joint contact-motion nonlinear program. Physical replay is a separate acceptance
test. The bin checkpoint must remain available while testing alternatives.

## Modeling corrections before MPC

- Environment support candidates come from the actual MuJoCo collision
  manifold, shared by the geometry screen and force allocator. A small positive
  geometric gap does not imply a supporting force.
- The optional environment-aware allocator includes gravity even while the
  hands are inactive. Environment forces enter the object wrench balance, but
  do not enter the robot actuator torque through a hand Jacobian.
- Sliding environment friction opposes the nominal contact slip. The allocator
  cannot select arbitrary tangential support to improve its residual.
- Sequential arm/waist IK can constrain each step relative to its predecessor,
  within physical position and velocity limits. The observed posture remains
  a soft preference. This does not certify finger transitions, acceleration,
  or the interpolated trajectory; those need the subsequent spline gate.
- A fitted trajectory is not modified at its first sample after optimization.
  Initial-state constraints must be satisfied by the fitted curve itself.

These fixes do not solve uncertain scene registration, unknown object inertial
parameters, missing wheel articulation, grasp selection, or infeasible task
geometry. A quasistatic allocator cannot represent all dynamic task motions.
Failed candidates are recorded and cannot be exported as training-ready.

Local raw audits and experiment scripts are deliberately ignored by Git:
`correction/outputs/model_audit_20261020_v1/`,
`correction/outputs/staged_model_20261020/`, and
`correction/local_experiments/`. Reusable implementation stays versioned.

## One-optimization MPC: proposed test contract

Let the state $s_k$ contain robot position and velocity, object pose and
velocity, and the state of the execution controller. The desired object path
$T_k^\star$ and the nominal robot commands come from the demonstration and
the saved correction. At each planning time, only the initial state is replaced
by the **current** simulated observation. Future simulated outcomes are not
read from a saved rollout.

The optimization is a single shooting problem:

$$
\min_\theta\ \sum_{j=1}^{H}
 \left[
   \|p(s_j)-p_j^\star\|_{Q_p}^{2}
   +\|\operatorname{Log}((R_j^\star)^T R(s_j))\|_{Q_R}^{2}
   +\ell_{\mathrm{limits}}(s_j,u_j)
 \right]
 +\ell_{\mathrm{deviation}}(\theta)
 +\ell_{\mathrm{continuity}}(\theta),
$$

$$
s_0=s_{\mathrm{observed}},\qquad
u_j=u_{\mathrm{reference},j}+\delta u_j(\theta),\qquad
s_{j+1}=F_{\mathrm{contact}}(s_j,u_j).
$$

Here $F_{\mathrm{contact}}$ includes the supported robot, free object, actual
collision surfaces, friction, torque clipping and controller state. Contact
forces and future states are eliminated by physical integration, rather than
chosen in a separate force allocation problem. The optimization cannot apply
forces directly to the object.

The first test restricts $\theta$ to a small set of smooth palm-motion and
force-guidance adjustments around the existing trajectory. This is a restricted
control space, not arbitrary grasp discovery. It still optimizes one common
object-task objective through coupled physical dynamics. The retained grasp
schedule is an initialization/structural restriction, not a scalable solution
to contact-mode selection.

Only a short command prefix is executed; then the horizon is solved again from
the new observed state. Predicted branches may be reset to the same observed
state for optimization, but the **executed episode must never be reset**.
The predictor must copy controller memory as well as mechanical state.

Before claiming benefit:

1. Validate that a copied prediction reproduces execution and leaves the live
   simulation untouched.
2. Check that zero changes reproduce the saved full-episode controller result.
3. Compare complete physical episodes with the same initial state, intended
   object motion, limits and environment.
4. Record solve evaluations, wall time, failures, tracking, contacts, limits and
   continuity. A reduced predicted objective alone is not physical success.

If this bounded coupled test fails, preserve the staged checkpoint and report
the failure before deciding whether to add measured-state feedback to the
staged trajectory optimization. Do not describe unsuccessful trials as a
working replacement.

## Implemented first test

`correction/src/optimization/contact_mpc.py` uses eight parameters per solve:
world translation offsets for each palm (six), and multiplicative changes to
each hand's nominal force guidance (two). A damped palm Jacobian maps translation
offsets into an arm joint-offset basis at the current state. It does not solve
another IK optimization inside the MPC.

The limits in this initial experiment are 20 mm per translation coordinate,
0.15 rad joint offset, and force scales between 0.5 and 1.5. A quintic command
offset matches position, velocity and acceleration from the preceding command
segment; delta actions alone would not provide that continuity.

The default horizon is 0.75 seconds with a 0.5-second executed prefix. The
bounded search evaluates a previous-target candidate, a return-to-reference
candidate, and six seeded perturbations when the reference anticipates hand
forces. This is a small shooting search, **not a converged nonlinear-program
solution**. Its inability to improve an episode would not prove that MPC in
general cannot work.

The previous endpoint is projected into the new local palm basis; this does not
guarantee a feasible shifted solution. Candidates violating the joint-offset
bound are rejected. If all candidates fail, the experiment stops and reports a
failure rather than silently resetting execution or switching controllers.

Prediction copies MuJoCo integration state and the contact controller's filter,
correction, engagement and command-offset memory. The shared
`correction/src/dynamics/execution_step.py` computes requested torque in both
prediction and execution. Actual applied torque is clipped identically.
The live simulator is never reset by the optimizer; predicted branches are
discarded. Each executed prefix is checked against the selected prediction.

The objective normalizes object translation error by 30 mm and rotation error
by 0.15 rad. It adds penalties for physical joint-limit violation, requested
torque beyond saturation, penetration beyond 2 mm, command deviation and
command-velocity changes. Final horizon error has weight three. These scales
are declared initial choices, not learned or validated universal weights.
Contact forces are simulator outcomes, and force guidance is a robot actuator
command, not a direct force applied to the object. Torque clipping and simulator
contact laws are enforced; the other penalty terms are not hard feasibility
certificates.

The fixed grasp schedule and demonstration object target remain restrictions.
This experiment does not infer new contact phases, estimate object inertia,
fit uncertain Tier 2/3 observations, run a whole-body walking controller, or
measure reinforcement-learning savings. Full-episode outputs and provenance are
stored locally under `correction/outputs/mpc_20261020/`.

## Reading the bin example: what changes, and what does not

Use the second bin manipulation as the example. The demonstration supplies a
desired object pose sequence $T_k^\star=(R_k^\star,p_k^\star)$ and a retargeted
robot posture sequence $q_k^{\mathrm{human}}$. Neither is a measured future
outcome of our simulator.

The retained staged method first selects plausible object/robot surface
patches near the demonstrated grasp. If $c_i$ is a selected point expressed in
the object frame, its nominal world target is

$$
x_{i,k}^{\mathrm{target}}=p_k^\star+R_k^\star c_i.
$$

IK constructs a robot hand path near those targets while preferring the
retargeted posture. A force allocation stage searches for friction-constrained
loads that approximately balance the object wrench and respect its actuator
model. Command refinement then constructs a smooth robot command. These are
stages with different decision variables and rejection gates, not one joint
optimization. The bin also has a saved two-phase contact sequence; it is not an
automatic solution to arbitrary contact-mode discovery.

The resulting plan consists of robot commands $\bar q_k$ (and their
derivatives), force guides $\bar f_{i,k}$, and contact activity/patch references.
At execution, joint PD and contact-force feedback generate bounded robot
torques. The physical object moves only through actual contacts. Thus even
the staged controller has local feedback, but its task-level plan is fixed:
it does not re-optimize the next grasp when the object deviates.

For example, at frame 645 of the experimental MPC rollout the bin is about
90 mm and 5.9 degrees from its intended pose. A contact target computed from
$T_k^\star$ can now differ substantially from the actual surface. This is a
recorded deviation, not a proposed 90-mm palm adjustment: the tested MPC bounds
each palm translation coordinate to 20 mm.

At that instant the MPC copies the actual robot/object pose, velocities and
controller memory. It tests small smooth modifications to $\bar q$ and
$\bar f$ for the next 0.75 seconds. A hypothetical candidate could shift a
palm by 5 mm and scale that hand's force guide by 1.1. The simulator must
predict whether the actual fingers touch, what force results after torque
clipping, and whether the resulting bin motion approaches $T^\star$.
Increasing a force guide does not ensure any object force is delivered.

All candidate scores use the same desired future object path. Only the first
0.5 seconds of the selected command is executed; then the prediction starts
from the new actual state. Both robot dynamics and object dynamics participate
in each prediction. The pelvis still follows an externally supported recorded
path; this is not walking control.

| Quantity | Staged execution | Experimental MPC |
| --- | --- | --- |
| Desired object motion | Demonstration reference | Same reference; not replaced by current motion |
| Future robot/force-guide commands | Saved offline schedule | Smooth bounded changes selected from current state |
| Initial state of each future prediction | No online task prediction | Current robot/object state and controller memory |
| Future object motion/contact loads | Outcomes of the executed physics | Candidate physics predictions, then actual execution |
| Contact phases, finger shapes and selected patches | Saved staged plan | Still restricted by that plan |
| Low-level joint/contact controller | Joint PD, compensation, force feedback and torque caps | Same controller inside prediction and execution |

Consequently this implementation replaces **open-loop selection of the next
task-command segment**, not the whole offline correction pipeline. Within each
MPC solve, future contact forces emerge from coupled simulation instead of a
separate future wrench QP. It does not optimize new grasp phases or independent
finger motions.

## Current result and viewer checkpoint

The first full-episode restricted MPC test is rejected: bin position RMS rises
from 48.48 mm for the retained staged method to 190.32 mm, and orientation RMS
from 5.91 to 37.31 degrees. The first grasp improves, but the second loses
contact and the object tips. Prediction and execution match exactly in this
deterministic simulator; that check does not make the selected motion good.
The active run takes 670.84 seconds of local CPU wall time, with 247 candidate
evaluations. There is no demonstrated learning-cost reduction.

See the local [MPC experiment report](../../correction/outputs/mpc_20261020/REPORT.md)
for failures and torque decomposition. Excessive PD torque demand, short
lookahead across release/reacquisition, and the restricted candidate search
remain concerns. No single causal explanation has been isolated.

The viewer on port 8087 shows the staged method only. Its bin entry retains
the original saved reference, whose complete physical replay under current
execution code is bitwise identical. Its desk/stroller entries now expose the
new modeling-fix trials as **rejected pre-spline diagnostics**, together with
matching independent-pad runs. These failed trajectories are visible for
inspection, not promoted into training export. Raw reports are preserved.

The precise status is: **identified modeling inconsistencies were corrected;
general geometric/contact feasibility is still unresolved; MPC is experimental
and its first tested variant failed.** The new planning fixes did not regenerate
or improve the saved bin trajectory. Their effect was tested on the additional
episodes, where contact and smoothness checks still fail.

## Decision after the contact-feasibility discussion

The next development priority is **contact acquisition and persistence using
current simulated state**, with accumulated, bounded command increments.
Retain the staged solver as the starting point and make its local geometry,
force and command updates share the same observed state and physical
validation. Do not continue broad eight-parameter shooting sweeps as the
default next experiment. This is a design decision, not an implemented or
validated controller change.

### What the implementation already observes

- The ordinary controller uses actual robot FK and object pose to rotate
  normals and measure selected-patch gaps. It measures actual contact forces,
  but still follows the offline motion/contact schedule.
- The shooting MPC initializes each prediction from actual robot/object state
  and controller memory. Its fixed finger shapes, fixed contact schedule and
  small palm-offset search can still fail to acquire a displaced object.
- The orange viewer ghost uses the entire object's convex hull, whereas the
  physical object uses convex decomposition and the displayed solid uses the
  source mesh. Consequently a hollow/complex object can appear box-like in the
  intended overlay. That proxy is inappropriate for judging contact surfaces.
- Current surface hypotheses are taken from individual simulator collision
  pieces, not directly from the orange global hull. Their exposure checks and
  collision approximation still require validation. Perfect pose feedback
  does not correct a wrong mesh, missing articulation or inertial model.

Zero measured force is a symptom with several possible explanations: separation,
unloaded touching, a different contact patch, or sensor matching failure. The
controller needs signed gap, closest points, opposing normals, relative surface
velocity and actual contact identity as well as force. MuJoCo also distinguishes
detected contacts from force-producing constraints; see its
[contact model documentation](https://mujoco.readthedocs.io/en/latest/computation/index.html#contact).

### State, contact geometry and task intent

At each update read only the current simulated state: robot q/dq, object SE3
and twist, environment/body poses, actual contact records and controller memory.
For articulated objects, each relevant link needs its own pose/joint state;
a single rigid object pose cannot stand in for that structure. Do not use a
previous recorded rollout's future states as observations.

For an object-local candidate point c and outward unit normal n, construct an
approach target from the current object pose:

$$
x^{\mathrm{approach}}=p^{\mathrm{obs}}+R^{\mathrm{obs}}(c+d n),\qquad d>0.
$$

This specifies an object-side approach location, not a complete robot grasp.
The actual robot patch, its normal, finger configuration and swept collisions
must also be checked. Candidate identity should remain stable across updates;
do not jump to whichever triangle is nearest at each frame. Alternative nearby
patches are selected using the demonstrated interaction neighborhood,
reachability, approach clearance and mechanical capability.

The desired object motion T-star remains the demonstration task objective.
Within a prediction, contact geometry follows the **predicted** object pose
initialized from the observed pose, rather than the desired pose or a frozen
observed pose throughout the horizon. This separates where the object is from
where it should go. Contact recovery must not reset the episode, overwrite the
object, or silently shift the reference clock.

Before contact, optimize approach distance, orientation and approach speed,
including finger motion. A task-only object-motion objective may provide no
useful distinction between several candidates that all miss the object.
Treat prospective wrench capacity as conditional on making the contact, not
as an available applied load. After contact, use the observed geometry and
measured loading to update the force/command plan toward the original object
goal. Losing contact returns to geometric recovery; a larger force target
alone is not a recovery motion. Use persistence/hysteresis around transitions
to prevent chatter. These transitions should follow measured geometry and
load, with the demonstration as a prior, not hand-labeled episode frames.

### Exact meaning of delta actions

Use accumulated upper-body command increments (including necessary finger
adjustments), with an EEF/contact objective:

$$
q^{\mathrm{cmd}}_{k+1}=q^{\mathrm{cmd}}_k+\Delta q_k.
$$

The observation initializes physical prediction; it does not overwrite the
previous command at each update. Likewise retain the previous command velocity
and acceleration when joining the next interpolated segment. This differs
from defining every target as observed q plus a fresh delta. Both conventions
exist; for example [robosuite's documented default](https://robosuite.ai/docs/modules/controllers.html)
uses deltas from the current state. Our chosen convention must be explicit.

For update interval h, constrain increments and their differences, componentwise:

$$
|\Delta q_k|\le v_{\max}h,\qquad
|\Delta q_k-\Delta q_{k-1}|\le a_{\max}h^2.
$$

Also constrain joint limits, interpolation derivatives, force-guide rates,
tracking error and predicted actuator demand with reserve. Command accumulation
must not wind up against an unreachable contact or a saturated joint. Reduce
or brake command progression smoothly when the physical robot cannot follow;
do not hard-reset the target to measured q or clip it after fitting and hide
the discontinuity. If no bounded continuation is feasible, report failure.

Ground-truth feedback itself does not force command jumps. The problem is how
new observations are converted into commands. The existing MPC already uses
continuous quintic residuals; changing its action name alone cannot repair
contact loss. The substantial changes are recoverable contact geometry,
finger/approach freedom and actuator-aware command evolution.

### Next validation, before another performance claim

First resolve the source-mesh/collision-surface/ghost discrepancy for the same
desk and stroller assets; do not tune gains to compensate for an invalid model.
Then run one continuous full bin episode using current-state contact recovery
and bounded delta commands, preserving the old controller result. Include one
declared small physical disturbance or initial-placement offset as a separate
test, with matched initial conditions for its baseline and no execution reset.
Measure reacquisition, actual load, task progress, saturation and continuity,
along with complete computation cost. Disturbed and nominal episodes must be
reported separately. No additional viewer channel is needed.

Cache collision surfaces and local patch data; update their transforms and
local optimization from observation. Use a limited number of physical
predictions to validate proposed prefixes. Runtime remains a measured result,
not a claim that all such feedback solves will be cheap. These tests assess
perfect-state simulation feedback first, not robustness to perception errors
or a demonstrated reduction in RL training cost.

## Implemented observed-state recovery prototype

The design now has an opt-in implementation:

See [the implemented equations, example and reproduction command](OBSERVED_CONTACT_RECOVERY.md).

- `correction/src/dynamics/observed_contact.py`: cached real collision patches,
  actual object/robot geometry, selected-patch contact sensing, and conditional
  hand/environment wrench allocation. Distance and force are separate signals.
- `correction/src/optimization/delta_command.py`: accumulated command endpoints
  with whole-curve position/velocity/acceleration constraints on quintic
  Bernstein coefficients. New curves start from previous commanded q/dq/ddq.
- `correction/src/optimization/contact_recovery.py`: observed-state geometry
  proposals including fingers, force allocation, and physical comparison with
  the shifted previous command. No random shooting search is used here.
- `correction/scripts/replay_contact_recovery.py`: full-episode replay with
  source snapshots, timing, actual physics metrics and explicit initial offsets.
- `correction/scripts/audit_recovery_geometry.py`: selected contact patches
  compared with source mesh and the simulator's collision geometry.

The predictor and executor use the same `requested_torque` function. Predictions
copy robot/object state and controller memory. Every selected 0.15-second
prefix is compared with actual execution; each prediction extends 0.45 seconds.
Desired object motion and episode time remain unchanged. Force guidance acts
only through robot torques and actual contacts. A candidate is not physically
accepted merely because it improves the prediction cost or its QP solves.

The force allocator includes a bounded object pose-error/velocity-feedback
wrench, not differentiated observed acceleration. Its actuator constraint is
an instantaneous approximation including joint PD and bias/passive forces;
the final physical prediction also includes robot inertia and the moving
fixture. Command constraints cover entire polynomials conservatively. The
geometry and force QPs remain separate stages, followed by coupled physical
validation. This is not a single joint nonlinear program.

Patch sensing now checks proximity to both the selected robot surface and the
selected object patch, as well as normal direction and active physical contact.
A remote contact on the same robot link is not sufficient. The default saved
controller is unchanged; the stricter behavior is enabled by the recovery
reference override. Source-derived contact neighborhoods and phase priors
remain restrictions. Stroller wheels remain unarticulated.

The first completed recovery trial (`nominal_v4`) is **rejected**: 303.19 mm /
69.75 degrees object RMS, compared with 211.72 mm / 38.95 degrees for the
matched contact-gating baseline and 48.48 mm / 5.91 degrees for the retained
method. It has all 858 frames and zero prediction/execution discrepancy, but
loses useful contact and task progress. Its recorded wall time is 307.52 s
(local CPU), not real-time execution. It also exposed remote-patch attribution
in the initial observer; the stricter fix was validated separately below.

The strict-patch replay (`nominal_v5`) also completed all 858 frames and is
**rejected**: 467.73 mm / 86.54 degrees RMS, 570.66 s wall / 501.67 CPU-seconds,
about 1.13 GB peak process memory. It found no matching selected contact and
applied no force guidance. The predictor/executor discrepancy remained zero.
This means the shared dynamics implementation is consistent; it does not mean
the contact strategy works. Merely tracking fixed saved patches at the observed
object pose is insufficient in this trial.

The bounded extension explicitly proposes nearby actual collision patches
using physical contacts or closest-geometry points. The center of each search
remains the original human-derived prior (80 mm object / 40 mm robot radii),
and a currently matched patch is retained. The candidate geometry is adopted
only with its command after physical comparison against the unchanged
continuation. This is a limited patch-selection stage, not free grasp search.

Its full replay (`nominal_v6`) is also **rejected**: 391.97 mm / 69.59 degrees
RMS, 675.60 s wall / 577.06 CPU-seconds, 1.132 GB peak process memory. It restores
90 sampled frames of selected-patch contact and accepts eight patch changes,
but does not produce a reliable grasp or task. All 858 frames are simulated,
with no object assistance or resets. All predicted prefixes match execution
exactly. Peak requested torque is 2.198 times capacity, joint-limit excursion
is 10.34 mrad, and penetration is 1.113 mm. The controller selects 110/191
proposals; 49 command QPs fail, and eight attempted force allocations fail.
Solved force QPs can still have large residuals because balance is soft.

There are 19 passing focused tests, including real-mesh patch reselection,
fixed-neighborhood rejection and command continuity/bounds. These tests support
implementation semantics, not manipulation success. The initial-placement
disturbance test is deferred after nominal rejection; no robustness claim is
made. Next work must preserve the approach/controller baseline while coupling
opposing contact acquisition and achievable support to task progression.
Current endpoint commands and a fixed demonstration clock do not implement
that coupling. No broad gain search or additional episode batch was started.
The contact-gating baseline matches near-surface/engagement settings and keeps
the original reference/sensor path; it does not isolate all other recovery
changes individually. The observed failures should not be attributed to one
parameter or to the feedback concept as a whole.

The complete default-controller regression reproduces the retained robot q,
joint/generalized velocities, torque and object SE3 arrays bit for bit. Its
48.48 mm / 5.91 degree result remains preserved, and still fails the strict
physical acceptance criteria. The viewer now draws intended objects using the
source mesh rather than a global hull; its existing data and two channels are
unchanged. Source-mesh distances at selected collision patches are up to
3.03 mm for desk, 1.70 mm for stroller and 9.74 mm for bin. This is a local audit,
not certification of the complete physical model.

The local [experiment report](../../correction/outputs/contact_recovery_20261020/REPORT.md)
records diagnostic starts, source versions, geometry discrepancies, complete
physical results and next decisions. The saved bin reference is not replaced,
and no training export or learning-efficiency claim is made.
