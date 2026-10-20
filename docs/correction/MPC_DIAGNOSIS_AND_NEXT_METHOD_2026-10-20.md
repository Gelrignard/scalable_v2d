# Current feedback/MPC: diagnosis and proposed next method

## Viewer and scope

[Current feedback/MPC replay](http://127.0.0.1:8088/) shows the original
retargeting on the left and `nominal_v6` on the right. Both solid objects are
physically simulated; orange meshes and paths are the intended motion. All
858 frames / 28.57 seconds are available. The controller diagnosis panel
reports the last optimization decision and its proposed wrench residual.
The viewer replays saved closed-loop simulation; it does not run optimization.

[Port 8087](http://127.0.0.1:8087/) retains the previous comparison and its
independent-pad channel. No corresponding pad replay exists for v6, so the
new port has only the robot comparison. The source mesh is rendered, while
physics uses convex collision pieces. Walking remains externally supported.

The current controller is a staged receding-horizon prototype: geometry and
force QPs propose commands, then full physical prediction decides whether to
execute them. It is **not one jointly solved contact/robot/object trajectory
optimization**. The implementation is documented in
[OBSERVED_CONTACT_RECOVERY.md](OBSERVED_CONTACT_RECOVERY.md).

## What the completed experiment establishes

| Full bin episode | Object RMS position | Object RMS rotation |
|---|---:|---:|
| Original retargeting | 444.99 mm | 64.60 deg |
| Preserved staged correction | 48.48 mm | 5.91 deg |
| Staged commands with revised near-contact/engagement settings | 211.72 mm | 38.95 deg |
| Current feedback/MPC v6 | 391.97 mm | 69.59 deg |

Every row still fails strict physical acceptance. The retained result is a
useful reference, not a fully feasible robot trajectory. Its peak requested
torque is 2.307 times capacity; v6 reaches 2.198 times capacity. Actual torques
are clipped at the actuator limits in both runs.

V6 makes 191 replans, accepts 110 command proposals and eight patch changes,
and has 90 video-rate frames with a matching active selected patch. There are
49 failed command QPs and eight failed force allocations. Joint acceleration
RMS decreases from 3.99 to 2.10 rad/s2 relative to the retained run, but EEF
acceleration RMS increases from 0.81 to 0.93 m/s2. Smooth commanded joints do
not establish useful object manipulation.

The predicted selected prefixes reproduce actual execution exactly. This
checks shared simulator/controller implementation. It does not establish
physical-model accuracy, a good objective, or a useful contact strategy.

The new full replay took 675.60 seconds wall / 577.06 CPU-seconds and about
1.13 GB peak process memory. Planning alone took 505.18 seconds, approximately
2.64 seconds per replan for a requested 0.15-second execution interval. This
implementation is an offline experiment, not a real-time controller. No GPU
or paid compute was used; no RL-cost reduction has been measured.

## Failure analysis: evidence versus inference

### 1. The experiment changed a successful approach/controller combination

Disabling the original near-contact torque guidance and changing engagement
already raises position RMS from 48.48 to 211.72 mm without online command
correction. That baseline matches those two settings only; it retains the
original sensor/reference path. It is not a full factorial ablation of all
recovery changes.

V6 also reconstructs the entire command as repeated quintics ending at zero
velocity and acceleration. At the first intended contact, frame 236, its
object error is 25.96 mm versus 11.28 mm for the retained/controller-ablation
runs. It has 36 earlier robot-contact frames versus 25. Therefore the approach
was already changed before the intended grasp, independently of later patch
selection. The evidence does not isolate whether timing, collision sequence,
or tracking change is the dominant cause.

Robot torque before contact is physically legitimate. It can close the hand
or establish preload. Its existence must be distinguished from claiming that
the object receives force while separated. I imposed contact-only guidance
too early as a controller change, which made this comparison harder to
interpret. Preserve the original executor first when testing the feedback
idea; separately assess any proposed replacement of its torque guidance.

### 2. A nearby contact is not necessarily a useful collective grasp

The new patch search minimizes local gap and distance from the human prior
for each selected body. It does not jointly choose the set of contacts for
the wrench needed by the task. At the first pickup it establishes left-hand
contacts while the selected opposing-hand channel remains unavailable.

At the recorded 9.75-second replan, the soft force QP reports about 10.14 N of
unmet vertical wrench for the 1 kg object. That returned force profile is
insufficient. A large residual from a **soft** objective alone does not prove
that no feasible force vector exists, because effort/regularization can trade
away balance. A separate diagnostic was therefore performed.

### 3. Additional frozen-state force diagnostics

The saved samples at frames 266, 293 and 320 were reconstructed without
integrating physics or modifying any trajectory. Their selected touching
flags matched the recorded flags exactly. For each sample, three force QPs
were compared: the implemented soft objective, minimum normalized wrench
residual with identical constraints, and that minimum-residual problem with
the approximate actuator rows removed.

The target wrench, contact eligibility, geometry, friction cones and force
caps stay the same within each comparison. Scale is 10 N for force and 1 Nm
for moment. The minimum-residual objective retains only a numerical 1e-10
force regularizer. Solver constraint violations are at most 1.32e-6.

| Saved time | Eligible selected patches | Soft vertical deficit | Vertical deficit at minimum full-wrench error | Normalized full-wrench error, with / without actuator rows |
|---|---|---:|---:|---:|
| 8.867 s | Two left-hand patches | 3.754 N | 1.312 N | 0.349416 / 0.349416 |
| 9.767 s | Three left-hand patches | 9.424 N | 9.399 N | 1.005697 / 1.005697 |
| 10.667 s | One left-hand patch | 3.194 N | 0.262 N | 0.148788 / 0.148788 |

These are minima of the scaled **six-dimensional wrench error**, not minima
of vertical error alone. None achieves exact wrench balance. At 9.767 s the
large deficit persists after effort penalties and approximate torque rows are
removed. Within this frozen selected-contact model, those rows are not the
binding cause of the failure. The available contacts, normals, friction/force
caps and requested wrench are incompatible with exact balance. This supports
improving the collective contact set. It does not prove that the object or G1
cannot be manipulated using other patches, bodies, schedules or trajectories.

At 8.867 and 10.667 s, regularization materially worsens the vertical component,
so the current objective also needs a separate feasibility priority. At
10.667 s the better local balance comes largely from environment support;
it is not evidence of a successful airborne grasp.

Limitations: saved samples are not the exact 9.75-second internal replan
state. Contacts are recomputed, the actuator estimate is instantaneous and
omits robot inertia/controller history, and environmental complementarity is
not enforced. These QPs characterize the current local model, not full robot
feasibility. The additional diagnostic cost was 20.52 s wall / 17.36 CPU-seconds,
about 1.12 GB peak process memory including the imported robot runtime.

### 4. Force feasibility does not modify the geometry decision

The command QP is solved first. The force QP then uses the current observed
contacts and current command to estimate a load. A poor wrench residual does
not cause a new joint/contact geometry solve within that update. A future
contact that the command might create is also absent from this current-state
force allocation. This is a structural separation, even though a physical
rollout subsequently evaluates the combined proposal.

The rollout compares only one proposal with continuation, over 0.45 seconds.
A correct prediction of two inadequate options cannot produce the missing
grasp. Longer prediction alone is not guaranteed to help if candidate
generation continues to exclude the needed contact coordination.

### 5. Timing and fallback can preserve the wrong behavior

Desired motion and most contact intent still advance with demonstration time.
The controller has a support-based release delay, but no general task-progress
variable to coordinate acquisition, transport and release. While a hand is
still acquiring the object, the nominal pose and desired object can move on.

The command endpoint is constrained to stop at the end of each horizon.
Changing the remaining horizon and target need not preserve recursive
feasibility of this restricted curve family. There are 49 command-QP failures;
their individual causes have not yet been classified. On failure, the previous
segment continues and eventually holds its endpoint. That preserves command
continuity but can stall manipulation. Relative prediction guards also permit
requests above actuator capacity; their acceptance is not a feasibility
certificate.

### 6. Geometry still has modeling uncertainty

Selected collision points agree with the simulator, but their distances from
the source mesh reach 9.74 mm for this bin, 3.03 mm for the desk and 1.70 mm
for the stroller. The intended ghost previously exaggerated shape by drawing
a global convex hull; it now uses the source mesh. That rendering repair
does not validate all collision surfaces, inertia, friction or articulation.

## Proposed method: preserve the nominal motion, optimize coordinated recovery

I recommend a warm-started residual trajectory optimization with actual-state
feedback, joint selection of useful contact sets, and explicitly controlled
task progress. Retain the existing mechanics pipeline to generate the nominal
trajectory and plausible contact regions. Do not rebuild the complete nominal
motion from stop-at-end command segments.

### State and commands

Let sigma be progress along the demonstrated path, with sigma in [0,1]. Let
x include actual robot q/dq, object SE3/twist and controller memory. At each
update initialize x_0 from simulation observation. Let q_0(sigma) be the saved
nominal command and r a bounded correction:

$$
q^c(t)=q_0(\sigma(t))+r(t),\qquad
\dot q^c=q_0'(\sigma)\dot\sigma+\dot r,
$$

$$
\ddot q^c=q_0''(\sigma)\dot\sigma^2+q_0'(\sigma)\ddot\sigma+\ddot r.
$$

Optimize a few trajectory knots for r and, where needed, a bounded robot
torque-guidance correction. New commands match the previous commanded q/dq/ddq
at the update boundary. Measured q initializes predicted physics, not the
command. Delta actions parameterize these continuous corrections.

Zero residual at nominal progress must reproduce the original executor when
feedback/new constraints are disabled. This is an implementation regression,
not a claim that the original trajectory meets new strict torque constraints.
Enforcing physical limits can require changing an already saturated nominal.

### Select contacts by task wrench capability

Use the human-derived object regions, approach directions and contact order
as priors. Generate a small number of **collective** surface-pair candidates
from actual collision geometry, including reachable alternative robot hand
links where appropriate. Do not greedily accept each nearest contact in isolation.

For a predicted configuration and candidate set C, define its local achievable
wrench set W(x,C) using unilateral forces, friction, contact geometry and
actuator limits. Measure the task-directed deficiency

$$
e_w(x,C)=\min_{w\in\mathcal W(x,C)}
\|S^{-1}(w-w_{task})\|.
$$

First minimize this deficiency; then optimize effort and force variation
within a declared residual tolerance. This prevents cheap forces from being
mistaken for sufficient forces. Evaluate the required wrench for the intended
short future motion, including gravity/support transfer, using the predicted
state and smooth goal trajectory. Do not differentiate noisy framewise
observations to prescribe exact acceleration.

This is a task-wrench test, not a universal requirement that both hands touch
or that every grasp provide arbitrary six-axis force closure. Pushing may be
valid with one contact. In this bin example, the demonstrated opposing contact
is a plausible way to remedy the deficient selected set. The criterion is
mechanical capability, not a manual label such as “bin requires two hands.”

A currently separated hand cannot supply object force. A future contact can
enter W only when the proposed approach reaches the actual surface with
compatible normals and feasible robot motion. Environmental support must
disappear consistently when the object lifts; it cannot remain a fictitious
support decision through separation.

### Couple future execution and progress

For each of a few contact candidates, optimize residual command knots and
bounded progress rate through one shared finite-horizon evaluation:

$$
x_{j+1}=\Phi(x_j,u_j),\qquad x_0=x_{observed},
\qquad \sigma_{j+1}=\sigma_j+h\nu_j.
$$

Here Phi is the robot/controller/contact/object execution model. Applied object
forces are consequences of contact, not freely imposed controls. A torque
guide acts through robot joints and may establish preload before contact;
it does not count as a realized object wrench before touching.

Use a task objective of the form

$$
J=\sum_j\left[
\|\operatorname{Log}(T_d(\sigma_j)^{-1}T_j)\|_Q^2
+\lambda_w e_w(x_j,C_j)^2
+\|r_j\|_R^2+\|\Delta u_j\|_D^2
+\lambda_s(\sigma_j-\sigma_{nom}(t_j))^2
\right]+V_{terminal}.
$$

Use separate, declared bounds on force and moment residual for task segments
requiring load support. Include joint/actuator/collision limits and valid
contact transitions; soft penalties remain diagnostic when these cannot be
met. Continuity covers the whole interpolated command, not only its knots.
Do not perform an independent final IK that can invalidate the optimized
contact/collision trajectory.

The terminal condition should value a reachable useful contact configuration
or continued supported motion, rather than always stopping the command. A
candidate that creates the necessary grasp can then justify temporary tracking
error. Use the staged sequence as a warm start and comparison candidate.

Contact families can remain an outer bounded discrete choice. Within a family,
geometry/robot motion, controller execution and object task progress must be
evaluated together. This is a targeted coupled continuous problem, not a claim
that we already possess a scalable unrestricted contact-implicit solver.

Start with the original clock and a residual-only ablation. Introduce variable
progress only after that comparison is interpretable. If progress is retimed,
the externally supported pelvis path and upper-body/object references need a
consistent phase; otherwise the pelvis may move away while the hand waits.
This is an explicit experimental change. Keep the original intended trajectory
visible, report both original-time and phase-aligned error, report delay and
completion time, and retain a completion deadline. Pausing forever is failure,
not improved tracking. No observed object state is reset or overwritten.

### Computation and staged validation

1. Preserve executor behavior with zero correction and classify the first
   divergence. Test contact gating, command reconstruction and patch selection
   as separate ablations before combining them.
2. Use cheap frozen minimum-residual QPs to rank a bounded number of contact
   sets and distinguish geometry, force capacity, torque and objective effects.
3. Optimize a small number of residual knots with warm starts and sparse/local
   Jacobians. A horizon covering acquisition/support transfer is the target;
   1–2 seconds is an initial experimental range, not a validated setting.
   Broad random search and a larger horizon alone are not the proposed fix.
4. Validate candidate prefixes with the same physical controller, then replan
   from the next actual observation. Report infeasibility explicitly and use
   a continuous braking/continuation policy whose physical failure is still
   counted. A previous command is not automatically safe when contacts change.
5. Evaluate the complete bin episode with identical initial conditions and
   model parameters. Assess acquisition, wrench residual, task progress,
   saturation, collision, continuity and CPU cost. Then use a declared small
   disturbance and more episodes. Current v6 is not ready for those claims.

Increasing the horizon and coordinating contacts will cost more than the
current endpoint QPs. CPU feasibility and lower cost than RL must be measured.
The useful components already implemented are actual-state observation,
surface caching, continuous command joins, shared physical prediction and
auditable complete-episode records. The missing piece is a reliable joint
contact-acquisition/task-progress decision.

## Artifacts and reproduction

- [Current implementation and equations](OBSERVED_CONTACT_RECOVERY.md)
- [Full experiment report](../../correction/outputs/contact_recovery_20261020/REPORT.md)
- [Frozen-state diagnostic output](../../correction/outputs/contact_recovery_20261020/wrench_diagnosis_v1/report.json)
- Viewer configuration: `experiments/current_contact_mpc_viewer.json`.
- Viewer command: `python -m correction.scripts.view_episode_catalog --catalog experiments/current_contact_mpc_viewer.json --port 8088`.
- Diagnostic command: `python -m correction.scripts.diagnose_recovery_wrench --source correction/outputs/contact_recovery_20261020/nominal_v6 --output correction/outputs/new_wrench_diagnosis --frames 266 293 320`.

Commands require the existing optional robot environment and locally generated
input files. The diagnostic source and summary are versioned; large arrays and
local reports remain ignored. No new physical rollout or optimizer redesign
was implemented for this diagnosis/viewer request.
