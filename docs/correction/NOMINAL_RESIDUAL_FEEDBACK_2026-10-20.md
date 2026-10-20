# Preserve the nominal correction and add bounded observed-state feedback

## Decision

Following the user's review, preserve the best staged trajectory and its
executor. Test feedback as small changes around this reference before any
joint contact/MPC redesign. Keep the current contact schedule, robot surface
families, finger shapes, force guidance, controller gains and original clock.
The retained result remains physically rejected despite useful object tracking.

The analogy to a KL-constrained policy update is a trust region: preserve useful
behavior while allowing local improvement. This controller is deterministic;
we are not computing a KL divergence between policies. In command space the
corresponding design is

$$
q^c(t)=q_0(t)+r(t),\qquad
\|r\|_\infty\leq r_{max},\quad
\|\dot r\|_\infty\leq v_{max},\quad
\|\ddot r\|_\infty\leq a_{max}.
$$

Small command changes do not guarantee small contact-force or task changes.
Contact transitions can amplify them. Full physical episodes remain necessary.

## What was already observed

The retained G1 contact controller already reads actual joint positions and
velocities for PD feedback, actual object rotation for contact normals, actual
robot Jacobians, geometric gaps and measured contact forces. It was never
entirely open loop. Its saved command path, force profile, contact identity and
phase clock remain nominal. Replacing all of these at once caused a confounded
comparison in the rejected v6 trial.

## First isolated change: observed surface displacement

This experiment generalizes the existing surface-feedback adapter to the full
inherited contact schedule. It adds no new manually selected frame interval.
The inherited staged schedule and grasp candidates still contain prototype
choices; this is not a claim of automatic annotation-free planning.

Let a_i be the saved object-local patch, b_i the corresponding robot-local
surface point, and let the planning object pose be (p_0,R_0). Read the actual
pose (p,R) from simulation. Form

$$
c_i^{obs}=p+Ra_i,\qquad c_i^{plan}=p_0+R_0a_i,
\qquad d_i=c_i^{obs}-c_i^{plan}.
$$

At the actual robot state, compute J_i at b_i. For inherited active patches
within 30 mm of the actual object anchor, solve damped least squares

$$
r^*=\arg\min_r\sum_{i\in E}
\|J_i r-\gamma\operatorname{clipnorm}(d_i,10\text{ mm})\|^2
+0.025^2\|r\|^2.
$$

The implementation solves unconstrained damped least squares then clips the
target components to 0.04 rad; it is not an exact box-constrained QP. A damped
second-order filter integrates this target at 1 ms with residual speed bound
0.2 rad/s and acceleration bound 1 rad/s2. Targets update at 50 Hz. The
declared nonzero trial uses gamma=0.25, so each Cartesian target displacement
is at most 2.5 mm before the Jacobian mapping. Residual joint bounds do not
constitute a global EEF displacement bound.

Only arm/waist joints with sufficient continuous nominal joint-limit margin
receive corrections. The margin is checked with cubic-Hermite Bezier control
hulls over the entire episode, rather than only sampled joint positions.
Finger joints retain their planned shape. Residuals decay through the same
filter when no patch is eligible. Any rate/limit conflicts are recorded and
reject the residual contract; clipping is not hidden as successful continuity.

The actuator receives q_0+r and matching velocity dq_0+dr. Existing nominal
inertial compensation and force guidance remain unchanged in this ablation;
residual acceleration is not added to inertial feedforward. Bounds refer to
the discrete residual sequence, not C2 continuity of the total robot motion.
No observed q is copied directly into the command and no object state is
overwritten. All object forces arise from simulated contact.

This displacement feedback preserves the nominal local hand/object relation.
It does **not** guarantee a mechanically useful grasp, create missing remote
contacts, allocate a new task wrench, or drive a deviated object back to the
task. In particular, following the object can worsen task error. This narrow
test measures whether observed geometry can be added without losing the saved
nominal behavior.

## Preserve the actual planning reference

The saved best bin sequence already contains an offline correction of the
second planning path: (-77.00, -40.43, -5.25) mm, decaying to zero during
transport. The original object target was retained for scoring. Subtracting
that scoring target in this new feedback loop would conflate this saved
correction with execution error.

The runner reconstructs the planning pose from the recorded ancestor
`proximal_sequence_v1`, using its saved shift and phase schedule. It verifies
source ancestry and identical timestamps, intended poses, object-local patches
and contact schedule. Switching the planning chart must occur during an
inactive interval. Both the reconstructed planning poses and unchanged scoring
reference are saved separately. The previous simulated outcome is used only
for comparison, never as the controller's object reference.

## Experiments and acceptance

Predeclared cases on the preserved `rest_matched_v1` reference:

1. Gain zero, full episode: require exact equality of physical robot positions,
   velocities, object poses, applied torques and requested torque peaks.
2. Gain 0.25, full episode: require identical physical prefix before scheduled contact activation,
   identical initial state/reference/controller/model, no object assistance,
   no within-episode reset, and report all physical acceptance failures.

The focused suite passes 12 tests, including actual MuJoCo geometry/sign,
schedule-only activation, zero-gain equivalence, offset conflicts, continuous
nominal margin, planning/scoring pose separation, force-controller tests and
torque audit. These tests do not establish task improvement.

Both complete 858-frame / 28.57-second episodes finished. The zero-gain case
matches the retained physical arrays exactly across the entire episode. The
nonzero case preserves all 236 frames before scheduled activation exactly but
is rejected. Actual incidental robot contact can occur before that schedule.

| Measure | Zero residual / preserved nominal | Gain 0.25 surface residual |
|---|---:|---:|
| Object position RMS | 48.484 mm | 214.420 mm |
| Object rotation RMS | 5.912 deg | 39.437 deg |
| Peak requested torque / capacity | 2.307 | 1.777 |
| Saturated 1 ms steps | 2498 | 1037 |
| Maximum penetration | 0.000011 mm | 2.036 mm |
| Joint-limit excursion | 9.049 mrad | 9.207 mrad |
| Physical acceptance | Fail | Fail |

The residual reaches 0.01898 rad, 0.09218 rad/s and 1 rad/s2, with zero
residual-bound conflicts. It updates 513 times. Staying inside these bounds
does not preserve task performance. Neither case is promoted as physically
feasible; the saved nominal remains the default and both viewers are unchanged.

### Where the failure occurs

Read-only analysis uses contact intervals obtained from the saved schedule,
not newly annotated phase windows. In the first active interval (236–500),
position RMS changes only from 25.45 to 26.64 mm. Object paths first differ by
over 10 mm at frame 501, at the first release. Before the second grasp (frame
645), their positions differ by 5.75 mm, but object orientation errors are
1.03 versus 3.92 degrees. Thus the second interaction starts from different
physical states even though its nominal command is unchanged.

During the second active interval (646–745), position RMS increases from
61.43 to 296.37 mm, and sampled bilateral-contact frames decrease from 46 to
24. At frame 680 the retained run has contacts on both hands and supports the
object without environmental support. The residual run has only left-hand
contact and substantial environmental support. By frame 725 its orientation
error is 90.44 degrees and it has lost robot contact. It ends approximately
512 mm from the intended location. These samples locate the failure but do
not isolate whether changed first-release state or the second residual is
the dominant cause; no phase-isolation experiment was performed.

Less torque demand is not necessarily improvement: in this trial it coincides
with worse load support and task completion. The evidence rejects unconditional
geometry following as a sufficient correction method. It does not reject
observed-state feedback or establish that all small residuals fail.

### Records and compute

- [Zero-residual complete replay and exact comparison](../../correction/outputs/nominal_residual_20261020/zero_v2/report.json)
- [Nonzero residual complete replay](../../correction/outputs/nominal_residual_20261020/surface_v1/report.json)
- [Read-only interval and sample diagnosis](../../correction/outputs/nominal_residual_20261020/comparison.json)
- [Experiment log](../../correction/outputs/nominal_residual_20261020/REPORT.md)

The zero case took 161.57 s wall / 141.33 CPU-seconds. The nonzero case took
81.09 s wall / 80.06 CPU-seconds. Both peaked at approximately 1.14 GB RAM and used no GPU.
The differing runtimes are not a controlled timing benchmark. The first zero
attempt failed during setup because an optional old report field was absent;
it never entered simulation. That setup failure and its fix are retained in
the log. Large arrays and source snapshots remain ignored.

## Next decision

Do not enable this geometry-only residual by default or tune a collection of
gains until one episode happens to pass. The trust region needs a task/contact
check as well as a command bound. The next isolated step is to evaluate the
unchanged nominal continuation and a bounded residual from the **same current
physical state**, using the preserved controller and a shared local prediction.
Adopt a residual only when it improves declared task/support criteria without
worsening torque/collision limits. Returning toward nominal must itself be
rate-limited; disabling feedback must not jump an accumulated residual to zero.

This retains the staged method and does not require a general all-in-one
contact solver. Physical prediction must cover the relevant support transfer;
an instantaneous lower position error is insufficient (frame 680 already
illustrates that). It also cannot guarantee recovery by selecting between two
bad candidates. If the predicted contact set cannot support the required
wrench, report that limitation and generate a mechanically different bounded
candidate rather than declaring continuation safe.

A separate subsequent intervention is a bounded force residual driven by actual
object pose/twist and achievable task wrench, retaining nominal guidance and
verifying joint torque feasibility. Change it independently to keep attribution
clear. A force plan must not credit separated contacts with object support.
Pre-contact torque on the robot is still a legitimate means of approaching or
preloading a surface.

If small staged updates help, the staged method with observation feedback is a
valid development direction in its own right. A joint finite-horizon optimizer
is an option when coupling is necessary, not a prerequisite imposed regardless
of evidence. The current v6 implementation was itself staged proposal plus
prediction; it was never a successfully solved all-in-one contact optimization.

## Reproduction

```powershell
python -m correction.scripts.replay_nominal_residual --source correction/outputs/overnight_20261020/rest_matched_v1 --planning-sequence correction/outputs/overnight_20261020/proximal_sequence_v1 --output correction/outputs/new_nominal_residual --gain 0.25
```

Use the existing optional robot environment and a fresh output directory.
Set gain to zero for the complete-episode regression. No GPU or RL is used.
