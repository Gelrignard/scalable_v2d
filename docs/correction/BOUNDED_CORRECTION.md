# Bounded correction experiment

The implementation now processes the book interval at frames **400–600**,
using a bounded CPU correction loop and separately timed physical diagnostics.
It is a research experiment, **not a successful manipulation or a training-ready
reference**. The measured run below shows affordable computation but failed
physical acceptance. This document supplements METHOD.md; it does not claim
the older derivation has acquired missing robot dynamics constraints.

## Reproduce

Use the existing CPU retargeting environment and optional NVIDIA G1 assets.
Install these additional dependencies into that environment:

```text
python -m pip install -r correction/requirements.txt
python -m pip install -r setup/correction-simulation-requirements.txt
python -m correction.scripts.bounded_book --input correction/outputs/tier1_book --output correction/outputs/bounded_book --start 400 --stop 601 --budget 120 --rollout
python -m correction.scripts.audit_bounded --directory correction/outputs/bounded_book --rollout
python -m correction.scripts.view_bounded --directory correction/outputs/bounded_book --port 8083
python -m unittest discover -s correction/tests -v
```

`--stop` is exclusive. The recorded development run instead used
`correction/outputs/reorganization_check_20261017` as input and
`correction/outputs/bounded_book_20261019_final` as output. Those local datasets
and generated outputs are ignored by Git. No GPU, paid job, RL training, or
external service is used.

## What is optimized

The input is the saved robot reference, source skeleton fingertips, object
SE(3) poses, mesh, timestamps, and table description. The original retargeting
output remains unchanged. The correction currently changes arm and hand
joints; root, legs, waist, object poses and timing remain prescribed references.
This restriction is important: a bad foot placement cannot be repaired by
moving only the hands.

### 1. Contact hypotheses

For each short window, the rigid book is approximated by its bounding box.
Source fingertips and current robot fingertips suggest nearby surface points.
Candidates include individually nearest faces, alternative faces for a whole
hand, and opposing thumb/finger pinch configurations. A previous contact
assignment can continue into the next window. Candidate points must remain
within the configured demonstration neighborhood (8 cm here).

This is a small beam, **not exhaustive contact-mode search**. Five candidates
are screened by default and two are sent to robot refinement. Surface anchors
are fixed within each candidate window; they are not continuously optimized by
this new solver. Source contact labels are geometric hypotheses, not measured
forces. The current implementation does not infer reliable palm normals from
the complete skeleton or fit probabilistic observation uncertainty.

### 2. Sparse force allocation

For fixed surface points and normals, solve for object-frame forces $f_t$:

\[
\min_{f_{0:T}} \sum_t\|S^{-1}(G_t f_t-w_t)\|^2
+\epsilon\sum_t\|f_t\|^2
+\gamma\sum_t\|(f_{t+1}-f_t)/\Delta t\|^2.
\]

The wrench map stacks force and moment about the object COM. Required wrench
comes from rigid-body inverse dynamics. Constraints enforce nonnegative normal
force, a normal-force cap, and the conservative friction diamond

\[
|f_{u}|+|f_{v}|\leq0.85\mu f_n.
\]

An inactive contact has exactly zero force. Possible table/floor support is
restricted to box corners within 3 mm of an actual support surface. No arbitrary
environment wrench is available. Cylinder side/edge support is not modeled.

OSQP receives sparse block matrices. For a fixed number of contacts, matrix
storage grows linearly with the number of samples. This does **not** imply
linear worst-case solve time. Independent physical-unit checks require force
residual at most 0.02 N and moment residual at most 0.002 Nm. A small optimization
objective by itself is not acceptance.

Forces and replay use the **same raw exported object poses**. This version
uses spline derivatives without smoothing, so noisy accelerations remain a
limitation. The previous implementation's smoothed-force/raw-playback mismatch
is not carried into this experiment.

### 3. Temporal robot refinement

The default window has 31 saved frames with overlap. Robot forward kinematics,
contact target errors, and collision distances are locally linearized. Each
SQP step solves a sparse QP near the saved reference. Contact-force magnitudes
weight contact tracking. Joint limits, velocity bounds, acceleration bounds,
and fixed boundary samples constrain the temporal motion.

The default velocity bound is 90% of the URDF limit. Acceleration limits are
engineering choices: 80 rad/s² for arms/wrists and 200 rad/s² for fingers; they
are **not verified actuator limits**. A global sparse temporal projection
initializes the segment and repairs export seams. If fixed boundary motion is
inconsistent with these constraints, padding is expanded by up to 60 frames,
with every attempt recorded. This happened in the measured book run.

Collision slack allows optimization to start from a penetrating reference.
Slack is penalized heavily but **never interpreted as collision freedom**.
The independent audit checks actual resulting configurations. A candidate is
ranked using manipulation penetration, fingertip gaps and attained wrench
checks. This heuristic still trades contact quality against penetration and
does not enforce sticking as a hard constraint. The first experiment shows
that this tradeoff is inadequate: it can reduce penetration by losing contact.

Two bounded rounds repeat robot refinement and attained-force checks. The
second round updates load weights using attained contacts. It is not full
coupled robot-object inverse dynamics. The baseline window remains an explicit
candidate and unresolved windows/failures are recorded.

The correction budget is cooperative wall time, with solver time limits and
a five-second reserve for final temporal projection. Mandatory final audit and
physical rollouts are timed separately. A budget-limited result is marked
incomplete; it is not promoted to training data.

### 4. Independent geometry and physical diagnostics

The scene imports G1 URDF collision geometry and inertia into MuJoCo, with a
floating robot base, free rigid book box, cylindrical table and floor. Mesh
contacts use convex collision representations. MuJoCo's same-body/parent
collision exclusions apply. Geometry checks include self, robot-object and
environment contacts, at saved frames and their midpoints. These finite checks
are **not a continuous swept-volume guarantee**. The 2 mm reporting tolerance
does not mean an exact nonpenetration constraint has been satisfied.

The force audit uses eligible fingertip contacts and support corners. It does
not credit every possible palm or proximal-finger contact. Object-relative
tip speed is reported only where consecutive eligible contacts have compatible
normals. That is a diagnostic, not a complete sticking/sliding/rolling model.

Optional physical replay resets once, initializes velocities from the reference,
then applies torque-limited joint PD. Root and object poses are never assigned
after initialization. Book mass defaults to a provisional 1 kg, friction to
0.6, and uniform-box inertia. PD gains are diagnostic assumptions. The scene
is **not the official Isaac challenge environment**, and no balancing controller
or robot inverse-dynamics optimization has been added.

A bounded command search tests corrected commands with 0, 1 and 3 mm nominal
fingertip preload, converted through damped finger Jacobians. Together with the
baseline this is at most four rollouts. Joint command offsets are capped and
clipped to joint limits. These extra commands are experimental controls, not
the exported corrected reference. All rollouts and initial-penetration flags
are saved. The diagnostic ranking never certifies training readiness.

## Measured book trial, 2026-10-19

The final correction run took **97.8 seconds** locally. It covered the requested
201 frames plus transition context. The audited context was frames 358–642
(285 frames, 569 frame/midpoint collision samples). No correction budget
exhaustion occurred. Original and corrected references used identical object
motion and timestamps.

| Measurement over audited context | Before | After |
|---|---:|---:|
| Maximum joint speed / URDF limit | 0.90 | 0.90 |
| Maximum joint acceleration, rad/s² | 648.0 | 200.0 |
| Sum of squared worst hand/arm penetration per sample, m² | 0.8033 | 0.1757 |
| Maximum hand/arm penetration, mm | 67.8 | 55.9 |
| Mean absolute fingertip surface gap, mm | 11.0 | 27.2 |
| Frames passing fingertip wrench checks | 0 / 285 | 0 / 285 |

The aggregate penetration reduction is about 78%; this is **not a success
rate**, and maximum penetration remains unacceptable. Contact-relative tip
motion also worsened. The inherited feet penetrate the floor by up to about
64 mm in this diagnostic geometry, independently of arm/hand correction.
The Pinocchio and imported MuJoCo body frames agree to micrometre-scale
serialization tolerance, so this is not explained by a gross simulator frame
mapping error.

All four diagnostic rollouts failed tracking. Both baseline and corrected
robots lose whole-body support, and initial penetrations confound any attempt
to isolate grasp-controller quality from these rollouts. They are useful
failure diagnostics, not evidence of RL improvement or a fair policy benchmark.

The supported conclusion is: **the bounded computation fits this CPU budget;
the current local contact/kinematic formulation is insufficient for this book
interval.** The implementation makes these failures observable rather than
showing prescribed object playback as successful dynamics.

## Files and outputs

| Responsibility | File |
|---|---|
| CLI orchestration | `correction/scripts/bounded_book.py` |
| Bounded experiment loop | `correction/src/optimization/bounded_book.py` |
| Contact alternatives | `correction/src/optimization/contact_hypotheses.py` |
| Sparse mechanical check | `correction/src/dynamics/sparse_forces.py` |
| Temporal SQP and seam projection | `correction/src/optimization/window_robot.py` |
| Collision model / physical replay | `correction/src/dynamics/robot_scene.py` |
| Independent audit | `correction/src/dynamics/window_audit.py` |
| Bounded command search | `correction/src/optimization/rollout_refinement.py` |
| Reference export | `correction/src/pass_traj/bounded_reference.py` |
| Revalidation / viewer | `correction/scripts/audit_bounded.py`, `view_bounded.py` |

Outputs include `bounded_reference.npz`, `bounded_contacts.npz`,
`bounded_report.json`, and `rollout_*.npz`. The report records candidates,
solver failures, selected modes, actual changed frames, budget coverage,
physical assumptions and failed audits. Reference metadata explicitly sets
`ready_for_training=false` and `accepted_as_feasible_manipulation=false`.
Stale body/probe diagnostics are removed rather than reused for modified joints.

## Next method decision supported by this experiment

More optimization iterations alone are not an established remedy. The next
change should address the observed failure: maintaining useful realized
contacts while clearing full collision geometry, and obtaining valid support
states for whole-body evaluation. Until then, comparisons of RL convergence
would be premature. A geometry-only ablation remains necessary before claiming
the force calculation causes a learning-efficiency benefit.
