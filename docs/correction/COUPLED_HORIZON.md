# Coupled correction with uncertain object observations

Implementation and validation note, 19 October 2026.

This is the new experimental correction path. The previous `full_episode.py`
and book experiments remain historical baselines. They solve geometry and
forces separately and do **not** implement the formulation described here.

The new solver jointly estimates object motion, robot motion, contact locations,
and contact forces over a phase. There is no target wrench calculated from the
second derivative of observed object poses. It runs on CPU using CasADi/IPOPT.

## Evidence and current limits

- A manufactured, two-pad carry passes an independent MuJoCo replay and tests
  with a smaller timestep, changed mass, and changed friction. This is a
  controlled mechanics experiment with an intentionally deficient baseline,
  **not a challenge episode or successful G1 manipulation**.
- The real G1 forward kinematics, contact Jacobians, and inverse dynamics agree
  with MuJoCo in independent tests, including a moving prescribed pelvis.
- The recorded first book pickup is an integration test. The tested coupled
  solves exhausted their time budgets. The latest facing-surface initialization
  run still has a **56.8 mm contact gap and 44.4 mm mesh penetration**. It is
  rejected. This implementation has **not yet produced a valid G1 correction**.
- No policy training, leaderboard validation, or reduction in RL compute has
  been demonstrated. `ready_for_training` remains false.

Generated reports are authoritative for numerical results. They distinguish
solver convergence, dense nominal checks, mesh collision checks, and actual
physical replay. None of these flags is inferred from another.

### Measured experiments

The controlled case is a 2 s, 0.5 kg box carry, initially held by two finite-mass
Cartesian pads. It uses 21 optimization knots and 61 noisy pose observations.
Position errors below are Euclidean errors against the known clean trajectory,
not against observations that the optimizer is fitting.

| Quantity | Measured result |
|---|---:|
| Noisy observation RMS position error | 6.84 mm |
| Estimated trajectory RMS position error | 3.10 mm |
| Actual guided replay RMS position error | 3.13 mm |
| Actual guided replay maximum position error | 5.26 mm |
| Actual guided replay maximum orientation error | 0.066 degrees |
| Maximum penetration | 0.069 mm |
| Peak requested actuator force / limit | 0.214 |
| Object assistance during replay | Zero |
| Half-timestep replay maximum position error | 5.26 mm |
| 10% heavier object maximum position error | 5.43 mm |
| Lower-friction replay maximum position error | 6.10 mm |

These values are from `correction/outputs/coupled_controlled/report.json`.
The final isolated local run took **11.82 s** to construct and solve the two
optimization stages, and **17.44 s** for the experiment including all physical
variants. It used CPU only. These are single-run measurements of a small model,
not a dataset throughput estimate.
The intentionally poor baseline opens its grasp, and there is no catching
floor in this experiment. Its large falling error is **not** evidence of an
improvement on challenge data. The comparison without force guidance also
loses the grasp. This first comparison lacked a position-setpoint preload;
the additional controller comparison below shows why it does not establish
that explicit force commands are necessary.

The recorded test covers **4 s of the first pickup**, not a complete episode.
The latest run (`coupled_book_facing_surfaces`) took 137 s end to end, including
127 s building/solving the optimization, and failed its 120 CPU-second solver
budget. The remaining violations rule out accepting or replaying that iterate
as a corrected demonstration. More optimization time has not been shown to
solve this problem; neither has GPU acceleration.

A separate geometry-only diagnostic increased the local initialization budget
to 300 function evaluations per knot and compared the three declared modes:

| Contact hypothesis | Worst contact error | Worst mesh penetration |
|---|---:|---:|
| Thumb/finger pinch across book thickness | 76.8 mm | 4.29 mm |
| Two hands on opposite book edges | 32.2 mm | 0.84 mm |
| Two hands on opposite broad faces | 67.3 mm | 6.42 mm |

None passes the initialization tolerances. This is a diagnosis of these local
initializations, **not a proof that a feasible grasp does not exist**. The next
G1 work must improve contact correspondence and initialization, then repeat
the coupled solve and independent replay. It should not tune loss weights to
conceal penetration or treat solver termination as physical success. Detailed
per-knot results are in `correction/outputs/contact_initialization_audit/report.json`.

The final regression run passed **24 tests in 76.18 s**. It includes corrupted
translation and rotation observations, missing observations, objective-weight
variation, mechanically impossible contact modes, dense collision rejection,
independent MuJoCo execution, G1 model checks, and the existing supported-robot
and full-episode regression tests. The log is
`correction/outputs/coupled_final_tests.log`. Passing these tests does not
override the recorded pickup's failed acceptance checks.

## Inputs and frames

`HorizonReference` contains all source observations, a coarser knot sequence,
retargeted joint poses, an externally supported pelvis path, explicit initial
states, a goal region, a fixed contact hypothesis, and assumed object mechanics.
Object positions are **centers of mass in the original world frame**. Object
quaternions are scalar-first, mapping object coordinates to world coordinates.
The mesh-origin/COM conversion is recorded; it is not a moving display frame.

Object pose uncertainty may be a scalar, three axis scales, or an observation
count by three array. These are declared tolerances, not calibrated posterior
covariances. Missing observations are omitted, while physical state integration
continues across the gap. Goal and initial conditions have separate contracts.
The solver cannot invent initial momentum to make an episode succeed.

Contact hypotheses specify a surface patch, its inward normal, tangent basis,
active interval, friction coefficient, force cap, and associated robot contact
point. For G1, the robot point is on a distal collision mesh's convex hull,
matching the simulator's collision shape. Its surface normal must face into
the object at the selection pose. Selecting the nearest point from a penetrated
retargeted pose can otherwise choose the finger's back side. This selection is
only an initialization: the full mesh must still pass collision checks after
the robot moves. The human contact observation may
lie off the surface; its discrepancy is a cost, not a fictitious contact.

The first dataset adapter explicitly covers frames **0–120** of episode 11,
with original timing, a box approximation of the book, and a moving supported
pelvis. It declares contact onset and support release hypotheses rather than
claiming to infer them automatically. A shared initial resting pose is estimated
from the table and the first observation, with limits of 15 mm / 3 degrees.
The source observations are retained unchanged. This registration is recorded
and applied to both physical comparisons only at initialization.

## Optimization

The unknowns are robot spline controls, object position $x_k$, orientation
$R_k$, linear velocity $v_k$, angular momentum $L_k$, contact forces $f_{ik}$,
and two-dimensional coordinates for movable contact anchors $c_i$.

Robot configuration is a cubic B-spline $q(t)=\sum_j B_j(t)C_j$. Velocity and
acceleration are its derivatives, not independent per-frame guesses. The curve
is $C^2$; jerk is piecewise constant. Bounds are imposed at knots and audited
between knots. This is not a continuous-time collision certificate.

The same world force appears in both sets of mechanics:

$$
m\dot v=\sum_i f_i+mg
$$

$$
\dot L=\sum_i (R c_i)\times f_i,
\qquad \omega=R I_B^{-1}R^T L
$$

$$
\tau=M_a(q;B)\ddot q+h_a(q,\dot q;B,\dot B,\ddot B)
       +\sum_{i\in\mathrm{hand}}J_i(q)^T f_i
$$

The last sign follows the convention that $f_i$ is the robot's force **on the
object**. The robot experiences its opposite reaction. Robot mechanics include
URDF inertias, joint armature and damping, and prescribed root inertial terms.

Object translation uses an integrated cubic/trapezoidal transcription. Angular
momentum uses trapezoidal integration and orientation uses a Lie-group update.
Observed acceleration does not appear in any of these equations. Quaternion
normalization inside rotation evaluation protects intermediate solver iterates;
the orientation integration itself preserves unit norm at a feasible solution.

For an active sticking contact:

$$
b_i(q_k;B_k)=x_k+R_k c_i
$$

Inactive contacts carry zero force. Contact forces are compressive and lie in a
conservative four-sided friction pyramid inside the Coulomb cone. The force is
zero at scheduled acquisition/release boundary knots, preventing interpolation
from applying force before the active interval. A contacting robot point must
reach the selected surface; the inverse-dynamics solver cannot supply a force
across a gap.

During approach, nonpenetration is enforced against the box rather than the
infinite half-space of the eventual contact face. This allows a finger to move
around an edge. Full robot collision geometry is handled by an independent
MuJoCo audit and additional local separating-plane constraints. The candidate
is rejected when the audit still fails.

The initial support mode is a stationary sticking face on a horizontal plane.
Net object wrench is zero during that mode, and integration preserves rest.
This avoids both redundant corner constraints and hidden between-knot motion.
Sliding, impacts, later touchdown, multiple finite supports, arbitrary meshes,
and articulated objects require additional mode models. They are not silently
approximated as sticking contacts.

### Objective and target provenance

$$
J_D=J_{\mathrm{object}}+J_{\mathrm{human\ contact}}+J_{\mathrm{posture}}
$$

Object observations contribute pseudo-Huber losses on translation error and
rotation-log error, normalized by their uncertainty scales. All observations
are evaluated, including those between optimization knots. The human contact
term penalizes anchor deviation from the uncertain human contact observation;
this includes the distance to alternative object faces. The posture term
penalizes changes from the original retargeting. Initial states and reliable
terminal goal regions are constraints, not disposable soft preferences.

$$
\begin{aligned}
J_R={}&\lambda_a\int\|\ddot q\|^2\,dt
      +\lambda_j\int\|\dddot q\|^2\,dt\\
     &+\lambda_f\int\|f\|^2\,dt
      +\lambda_{\dot f}\int\|\dot f\|^2\,dt\\
     &+\lambda_\tau\int\|\tau/\tau_{\max}\|^2\,dt
\end{aligned}
$$

The code normalizes these terms by physical scales and duration. Its first
stage minimizes $J_D+\epsilon J_R$ with a small, explicit regularizer to remove
flat force directions. Its second stage minimizes $J_R$ subject to
$J_D\le J_D^{(1)}+\epsilon_D$. The first-stage value is a local solver result,
not a global optimum or a recovered human force profile. A second-stage failure
retains the first-stage solution only if it genuinely passed the numerical
constraints; failure is reported.

Defaults and every run's settings are serialized. The data allowance controls
how much demonstration fit may be traded for regularity. Tests vary objective
weights and corrupt observations; they do not establish universal weights for
Tier 2 or Tier 3. `solve_candidates` compares only hypotheses with common
observations, task goals, and initial states, and retains rejected outcomes.

## Execution and acceptance

The physical comparisons are:

1. Original commands with the baseline controller.
2. Corrected trajectory with that controller, without contact-force guidance.
3. Corrected trajectory with contact-force guidance.

In the controlled pinch, tracking the nominal touching trajectory without
grip preload loses the grasp. A position-setpoint offset also realizes squeeze
and passes the controlled tracking gate. It would be incorrect to infer from
the original ablation that a position-command interface cannot manipulate.

The object remains a free body. It is initialized once, receives gravity and
contact forces, and is neither pose-assigned nor welded during replay. Force
guidance acts through robot actuators and is disabled across a large physical
contact gap. G1's legs remain frozen and the pelvis is externally supported.

The planner uses ideal sticking contact. MuJoCo's regularized contact model can
drift even inside a friction cone. An earlier pyramidal-contact replay drifted
24 mm and failed the controlled tracking gate. The current controlled replay
uses elliptic cones, `impratio=20`, a tight solver tolerance and ten no-slip
iterations, consistently across all comparison arms. These are documented
[MuJoCo contact-model controls](https://mujoco.readthedocs.io/en/stable/modeling.html).
The earlier failure remains in the report. Simulator sensitivity is evidence
of a modeling limitation, not proof of real-world robustness.

Dense checks measure contact gaps, inactive forces, friction violations,
actuation limits, joint bounds, continuity, and object force/moment residuals.
They use box clearance during inactive approach, not distance to an eventual
contact face's infinite half-space. Active signed normal gap is reported
separately. All box contact patches are validated to lie on physical faces.
G1 additionally requires a full collision-mesh audit. Physical replay then
checks tracking, collisions, actuator saturation, joint continuity, and the
absence of object assistance. Failed plans remain diagnostic artifacts.

### Contact motion versus actuator commands

The subsequent [controller implementation, full-episode comparison and fallback
record](CONTROLLER_EXECUTION.md) now contains the optional G1 execution paths,
nominal command export, and servo constraints. Its full-episode input remains a
historical failed reference; no accepted full-episode coupled G1 result exists.

The optimizer's physical trajectory and a servo's equilibrium targets are
different quantities. For a PD-controlled joint, the declared model is

$$
\tau=K_p(q_{\mathrm{cmd}}-q)+K_d(\dot q_{\mathrm{cmd}}-\dot q)+\tau_{\mathrm{ff}}.
$$

If the motion and contact optimization require torque

$$
\tau^*=M(q^*)\ddot q^*+h(q^*,\dot q^*)+J(q^*)^T f^*,
$$

then a nominal position command compatible with that same PD law is

$$
q_{\mathrm{cmd}}=q^*+K_p^{-1}
\left[\tau^*-\tau_{\mathrm{ff}}-K_d(\dot q_{\mathrm{cmd}}-\dot q^*)\right].
$$

This algebra assumes known nonsingular gains and no command/effort saturation;
it is a nominal controller conversion, not a robustness guarantee. The physical
trajectory remains subject to collision constraints. A servo equilibrium can
lie beyond a contact surface without the physical finger passing through it.
For G1, actuator delay, target scaling/clipping and joint limits must also be
represented and tested. Merely exporting `q*` as `q_cmd` discards the preload.

The additional comparison reuses exactly the same corrected two-pad motion,
initial state, controller gains, simulator and limits. It does not retrain or
re-optimize the trajectory. The position-preload arm adds a 5 mm inward position
offset to each pad (1200 N/m stiffness, nominal 6 N squeeze), with no explicit
contact-force feedforward. All variants retain the same existing inertial and
gravity compensation. Since the experiment starts grasped, preload and force
feedforward are both active from the initial control step.

| Controller | Maximum object error | Controlled gate |
|---|---:|---|
| Nominal position targets without preload | Object falls | Fail |
| Position targets with grip preload | 7.04 mm | Pass |
| Planned contact-force feedforward | 5.26 mm | Pass |
| Feedforward plus bounded normal-force feedback | 5.26 mm | Pass |

`correction/outputs/contact_controller_comparison/report.json` contains the
measurements. The feedback prototype uses a 20 ms low-pass estimate of measured
pad force, proportional gain 0.25, and a correction capped at 2 N and by the
contact-force cap. It acts only while actual contact exists. This nominal test
does **not** demonstrate a benefit from force feedback over feedforward; model
errors and disturbances need separate evaluation. This was initially implemented
for the reduced pad replay. Optional G1 feedback and position-conversion paths
are now implemented; see the execution record above for full-episode failures
and the retained fallback. The coupled G1 optimization candidate still fails
geometry and dynamics acceptance before replay.

Unitree's [official G1 low-level example](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py)
exposes joint targets, gains and feedforward torque. These are not measurements
or direct commands of fingertip contact force. In [CHORD Appendix E.2](https://arxiv.org/html/2607.00033v2#A5.SS2),
the whole-body policy outputs residual joint-position targets; its wrench reward
does not imply a force-valued policy action. A practical next interface is to
export both physical motion and controller-compatible targets, retaining contact
forces as mechanics guidance or privileged supervision. RL must learn actions
that generate useful contacts, but need not reproduce one exact force allocation.

**Full-episode delivery requirement:** future robot demonstrations must cover
the complete source episode at its original timing, including acquisition,
transitions and release, with one continuous physical rollout and no hidden
resets. Chunked optimization may be used internally, but its boundaries must
remain continuous. The pelvis stays externally supported along its recorded
path. Short controlled tests remain diagnostics and are not substitutes for
that deliverable. Full-episode coupled G1 success is still unachieved.

## Reproduction

Install the optional CPU optimizer into the existing retargeting environment:

```sh
python -m pip install -r setup/coupled-correction-requirements.txt
python -m correction.scripts.coupled_controlled --verify
python -m correction.scripts.compare_contact_controllers
python -m correction.scripts.view_coupled --port 8085
python -m correction.scripts.diagnose_horizon_contacts
python -m correction.scripts.coupled_book --stride 12 --seconds 120 --hypothesis two_hand_edges --initialize-contacts --output correction/outputs/coupled_book_facing_surfaces
python -m unittest correction.tests.test_coupled_horizon correction.tests.test_horizon_replay correction.tests.test_symbolic_robot -v
```

The G1 adapter also requires the existing optional NVIDIA/robot asset setup and
the saved book retargeting. No upstream code, model weights, or dataset files
are added to Git. Exact G1 second derivatives are an opt-in experiment; they
used excessive memory locally. The G1 default uses quasi-Newton updates.
The optional geometry-only initialization supplies an optimizer starting guess;
it never bypasses dynamics, collision, or replay gates. It preserves the known
physical initial state. A bad initial guess is reported, not called a correction.

Each output directory contains `problem.json` (inputs, assumptions, contact
hypotheses and goals), `observations.npz`, `plan.npz`, physical rollout files
when available, and `report.json`. Failed optimizer status is retained. A warm
start is recorded and cannot change the physical initial state.

## Files

| Location | Responsibility |
|---|---|
| `src/preprocessing/horizon_reference.py` | Validated observation and phase contract |
| `src/preprocessing/g1_horizon.py` | Recorded pickup adapter and actual robot surface points |
| `src/preprocessing/controlled_horizon.py` | Known-ground-truth diagnostic inputs |
| `src/dynamics/symbolic_mechanics.py` | Quaternion operations, pad mechanics, G1 Newton–Euler model |
| `src/optimization/coupled_horizon.py` | Joint nonlinear horizon optimization |
| `src/optimization/horizon_candidates.py` | Bounded hypothesis comparison |
| `src/optimization/horizon_initialization.py` | Geometry-only initial guesses and their residuals |
| `src/dynamics/horizon_audit.py` | Dense mechanics and trajectory audit |
| `src/dynamics/horizon_collision.py` | Full robot mesh audit and collision witnesses |
| `src/dynamics/horizon_replay.py` | Independent reduced-model physical replay |
| `src/dynamics/g1_horizon_replay.py` | G1 controller ablations and execution gates |
| `src/pass_traj/horizon_result.py` | Reproducible diagnostic export |

The next research gate is a consistently accepted recorded G1 manipulation.
Reduced-model success and correct rigid-body algebra do not establish that gate,
and neither establishes faster RL convergence or industrial-scale throughput.
The existing comparison website has not been switched to this experimental
method, and no failed G1 iterate has been published there as a successful result.

The separate `view_coupled` viewer reads the saved controlled MuJoCo replay and
the rejected G1 optimizer poses. Its result selector explicitly distinguishes
**physical replay** from **kinematic diagnostics**. The “Inspect worst G1
collision” button pauses at the largest sampled penetration. The camera focuses
on manipulation without shifting world coordinates. Solid blue is the simulated
controlled object; its orange wireframe is the clean reference. The G1 orange
object is an optimizer estimate, not a physical rollout.
