# Observed-state contact recovery: implemented prototype

20 October 2026. This is an opt-in research controller, not an accepted
manipulation result. It preserves the retained staged method and its data.
The current outcome and limitations are recorded in [MODELING_AND_MPC.md](MODELING_AND_MPC.md)
and the local [experiment report](../../correction/outputs/contact_recovery_20261020/REPORT.md).

## What changes when the bin is somewhere unexpected?

Suppose the intended bin is 5 cm ahead of the actual bin. The original object
trajectory remains the task target. The controller transforms the candidate
contact locations using the **actual** bin pose. Its approach objective therefore
aims at the actual bin surface. A load is allocated only to a matching physical
contact; a zero load and a positive gap are distinct observations.

If the hand touches a nearby part of the bin, that contact is not counted as
force at the old intended location. It can propose a new pair of actual robot
and object surface points, within a fixed neighborhood of the original prior.
The pair changes the force lever arm, surface normal and robot Jacobian.
The new pair and its command are adopted only if the physical prediction
improves the objective and satisfies the relative guards described below.

The controller predicts from actual robot positions and velocities, actual
object pose and twist, and current controller memory. Its commanded position,
velocity and acceleration start from the previous command. These are two
different states: measured state initializes physics, while command state
preserves continuity.

## Inputs, observations and priors

- The staged reference supplies nominal upper-body joints, intended object
  SE3, force guidance, robot/object local contact points, inward normals and
  contact phases. These arrays remain immutable during the replay.
- MuJoCo supplies actual robot q/dq, object SE3/twist, contacts, environmental
  support and the moving external pelvis fixture. The robot's upper body and
  the object remain dynamic. The lower body is externally supported.
- `SurfaceLibrary` caches the actual convex collision pieces. Contact points
  must project onto exposed collision surfaces within 15 mm. This is an
  agreement check against the simulator, not a guarantee of source-mesh fidelity.

For object pose (p,R), object-local anchor a_i, robot surface point b_i and
robot configuration q, define

$$
c_i^o=p+Ra_i,\qquad c_i^r=\operatorname{FK}_i(q,b_i),
\qquad g_i=\|c_i^r-c_i^o\|.
$$

The observer reports g_i, signed geometry distance, normal alignment, whether a
matching contact constraint exists, and measured force separately. Matching
requires the correct robot body, a physical constraint, normal dot product at
least 0.5, and contact-position distances of at most 25 mm from both selected
points. The lower-level controller retains its 30 mm robot-side gate and adds
the same 25 mm object-side gate when recovery is enabled. The stricter observer
is consequently conservative relative to that controller gate.

## Bounded contact alternatives

`ContactCharts.propose_nearby` considers actual contacts on the same selected
robot body and the closest points of the current geometry pair. The latter can
guide approach even when separated; it does not create a contact load.

Each candidate is projected onto exposed object collision geometry and a
facing robot collision triangle. The default maximum displacement from the
fixed original prior is 80 mm on the object and 40 mm on the robot. Normals must
remain compatible with the prior, and the candidate must reduce the current
gap by at least 1 mm. A currently matched patch is retained. The search center
does not drift with successive accepted patches. These are declared prototype
hyperparameters, not tuned or demonstrated universal values.

The selected body family is still fixed by the staged reference. This is not
arbitrary grasp discovery or a complete contact-sequence optimizer. The
demonstration phase schedule supplies the initial contact intent with 0.3 s
lookahead. Release can be delayed until measured upward environmental support
exceeds 70% of weight and object speed stays below 0.05 m/s and 0.2 rad/s for
0.15 s. Progress is still tied to demonstration time.

## Accumulated delta commands

At replan time t_k, retain the last command state

$$
z_k^c=(q_k^c,\dot q_k^c,\ddot q_k^c).
$$

The command decision is an endpoint increment d, not an overwrite by measured q:

$$
q_H^c=q_k^c+d.
$$

A quintic joins z_k^c to (q_H^c,0,0) over H=0.45 s. Thus adjacent segments
are C2 in command space. Zero terminal velocity/acceleration is a prototype
restriction: repeatedly applying it can delay the original approach. Actual
robot motion is not guaranteed C2 at impacts, nor does this guarantee smooth
torque when contact identities change.

The endpoint QP minimizes

$$
\|q_H^c-q_H^{nom}\|^2+
\sum_i\|W_i(J_i(q_H^c-q^{obs})+c_i^r-c_i^{target})\|^2
+L_{normal}+L_{separation}+L_{tracking}.
$$

For a separated intended contact, c_i^target is the actual object anchor plus
1 mm inward. For a matching contact, add a bounded step toward the intended
future object pose (at most 15 mm translation and 0.06 rad rotation). The
position residual scale is 70; normal alignment is 1.5. Currently penetrating
contacts contribute a linearized separation residual with scale 150 and
0.5 mm tolerance. A soft tracking term uses measured q/dq and gain-to-torque-cap
ratios. These terms remain local approximations.

All Bernstein coefficients of the command polynomial and its first two
derivatives must satisfy position, speed and acceleration limits. This is a
conservative sufficient bound over the whole curve. Default arm speed/acceleration
are 1.5 rad/s and 12 rad/s2; hand values are 2.5 rad/s and 20 rad/s2. Speed is
also limited by 80% of model velocity limits. Infeasible QPs are reported;
there is no post-fit clipping or jump back to measured q.

## Conditional wrench allocation

At the same observed state, a separate force QP allocates a soft target wrench.
It does not differentiate the observed object trajectory:

$$
a^*=\operatorname{clip}(12(p^{des}-p)-4v,\,-0.8,\,0.8),
$$

$$
\alpha^*=\operatorname{clip}(8\operatorname{Log}(R^{des}R^T)-3\omega,\,-2,\,2),
$$

$$
w^*=\begin{bmatrix}m(a^*-g)\\I\alpha^*+\omega\times I\omega\end{bmatrix}.
$$

Clipping is componentwise. Hand forces exist only at currently matching
contacts with compatible robot surface normals. Separated channels are
structurally zero. The QP includes measured environment contact locations,
unilateral normal forces and conservative sticking friction constraints;
sliding environment friction opposes the measured slip direction.

With wrench map G, force vector f and diagonal scale S (10 N force, 1 Nm torque),
the objective is

$$
\|S^{-1}(Gf-w^*)\|^2+0.002\|f-f_{preload}\|^2
+0.002\|f-f_{previous}\|^2.
$$

The normal cap is 15 N per hand point. An instantaneous actuator constraint
requires PD + bias - passive + J^T f within 90% of torque caps. This allocation
omits robot inertial torque and contact complementarity; therefore a feasible
QP is not a certificate of physical feasibility. Residual wrench and allocation
failure are saved. The subsequent simulation includes inertial and fixture
effects. Force-guide changes use a quintic blend limited to 60 N/s per world
component; contact gating and patch changes can still alter realized torques.

Disabling pre-contact force guidance is a deliberate conservative controller
choice in these trials. Applying a robot torque before contact is physically
legitimate and is not equivalent to applying a force directly to the object.
The matched baseline separately measures the effect of changing that choice.

## Coupled physical prediction and execution

The geometry and force QPs remain staged. Their combined result is evaluated
by copied full physical simulation, using exactly the same `requested_torque`
function, limits and controller memory as execution. Two predictions are used:
continuation of the previous command with its current patches, and the proposed
command with its candidate patches. No object pose is forced to the reference.

The prediction score uses actual future object translation/rotation error,
contact gap and normal alignment, deviation from nominal commands, saturation,
joint-limit violation and penetration. It samples every 30 ms, weights the
terminal sample by 3, and uses scales of 50 mm / 0.2 rad for object error,
25 mm for contact gap, 0.1 rad for nominal commands, 0.01 rad for joint violation
and 5 mm for penetration. Contact gap is capped at 0.3 m in this prototype.
Nominal-command weight is 1 during free motion and 0.03 during contact intent;
normalized saturation weight is 2.

Adoption requires lower cost, no unstable solver state, peak requested torque
no greater than max(1, previous peak)+0.15 times capacity, and penetration no
greater than max(3 mm, previous peak)+0.5 mm. These are **relative diagnostic
guards**, not strict physical acceptance. Both candidates may be poor. If the
proposal fails, continue the previous bounded segment and report the failure.
When that segment ends it holds position; there is no hidden trajectory reset.

Execute 0.15 s, observe again, and verify the predicted prefix against the
actual q/dq and force-feedback memory. The equality check tests implementation
consistency under deterministic simulation, not real-world prediction quality.
Full-episode physical acceptance is evaluated separately.

## Reproduce and inspect

Use the existing retarget environment; no GPU is required for this prototype:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PYTHONPATH=(Get-Location).Path
& ./.tools/Library/bin/micromamba.exe run -p ./.retarget-env "$PWD/.retarget-env/python.exe" -m correction.scripts.replay_contact_recovery --source correction/outputs/overnight_20261020/rest_matched_v1 --output correction/outputs/contact_recovery_new
```

The source example is a locally generated reference, not a bundled public
dataset. The output must not already exist. `--baseline` uses the same revised
near-surface and engagement settings without recovery. It retains the original
reference/sensor path, so it does not isolate the stricter recovery patch
observer or contact reselection as individual factors. `--object-offset x y z` declares a physical
initial-placement offset without modifying the intended trajectory; it is not
an execution reset.

Each result saves a source snapshot and hashes, complete reference, actual
`mechanics.npz`, QP/prediction logs, configuration, CPU/wall cost and memory.
The nominal `command_q` field remains the staged input; actual online commands
are `recovery_command`, with their derivatives and selected local patches.
`object_positions` in mechanics is simulated motion; in reference it is intended
motion. Tests and solver success never set `ready_for_training` to true.

The next substantive limitation is approach preservation and actuator-feasible
contact acquisition over time. A fixed endpoint with zero terminal derivatives,
a fixed demonstration clock and a short horizon can all prevent recovery.
The first pickup in the bounded-patch trial acquired left-hand surface pairs
without completing the opposing grasp. At 9.75 s, three left-hand channels were
eligible but the force QP still left about 10.14 N of vertical wrench unmet for
the 1 kg object. Numerical solution of a soft force-allocation problem does not
mean its selected contacts can support the task. Coordinated grasp acquisition
and a progress variable conditioned on achievable support remain necessary
research steps; this prototype does not implement them.
Disturbance robustness, multiple-episode improvement and reduced RL cost remain
unverified.
