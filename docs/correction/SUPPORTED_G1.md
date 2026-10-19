# Supported G1/Dex3 manipulation physics

19 October 2026. This experiment combines the object/contact correction with
the G1 robot model, while removing walking and balance from the test.
It demonstrates physical book carrying, **but does not pass the tighter
trajectory-tracking or demonstration-locality gates**. No policy is trained.

## What is simulated

The pelvis world transform and all twelve leg joints are fixed in the model.
They are not stabilized by a hidden controller. The three waist joints, fourteen
arm joints and fourteen Dex3 finger joints remain dynamic: **31 actuated DOFs**.
The book has a free six-DOF joint. Gravity, robot self-contact, hand/object
contact, floor and recorded table contact are enabled.

MuJoCo imports the optional NVIDIA G1 URDF's inertias and collision geometry.
Meshes use MuJoCo's convex collision treatment. The book collision shape is a
rigid box fitted to the dataset mesh. The assumed mass is 1 kg, friction 0.6,
and timestep 1 ms. These are local diagnostic assumptions, not an official
challenge environment or hardware validation.

There is no object weld, equality attachment, applied book wrench, or assignment
of object poses after reset. Only robot joint torques are applied. The viewer
plays saved physics states; the gold wireframe is the desired object pose.
The grey fixture post symbolizes the fixed pelvis; it is decorative.

## Trial and measured results

Episode 11, frames **520–565**. The book's pose path is preserved, but its timing
is slowed by a factor of two: **1.5 seconds becomes 3 seconds**. The path covers
about 52 cm and 100 degrees of rotation. This is a mid-episode carry segment,
starting from a prepared two-hand grasp. Acquisition and release are excluded.

All comparisons share the initial robot/object state, initial velocities,
time scale, motor limits and simulator parameters. The raw retargeted command
comparison blends from the prepared grasp over 0.4 seconds after a 0.1-second
handover delay. It is not a comparison against a trained CHORD policy.

| Commands | Worst book position error | Worst orientation error |
|---|---:|---:|
| Raw retargeted commands | 1056.86 mm; book drops | 170.95 degrees |
| Robot geometry correction, no contact-force feedforward | 1038.11 mm; book drops | 90.67 degrees |
| Robot geometry plus inverse-dynamics force seed | 13.93 mm | 19.68 degrees |
| Force seed plus rollout-based command correction | **10.88 mm** | **10.52 degrees** |

Corrected RMS errors are **5.39 mm / 7.11 degrees**. Maximum geometric overlap
is **0.0124 mm**, a small initial finger/finger overlap. Contacts have a 0.5 mm
margin per geom, so compliant contact can act at a positive geometric gap.
Both hands exert book contact forces at 45 of the 46 saved frames; the remaining
sample has one hand in contact. This count is sampled, not a continuous-contact
certificate. Worst pose error, overlap and joint speed are checked at every
integration step. Maximum measured joint speed is 4.46 rad/s.

The full method run took **31.59 seconds on the local CPU**, including model
construction, contact IK, force planning, correction and comparison rollouts.
This excludes Python/import/input-loading time and the four extra sensitivity
replays. It is one local timing measurement, not a dataset-scaling result.
No GPU, RL or paid compute was used.

### Why the output is still marked unready

- The tracking gate remains **<10 mm and <5 degrees**. This trial fails it.
- Some robot motors saturate at every integration step. Actual torques are
  clipped to their limits, but requests reach **8.89 times** the limit for the
  most constrained joint. This has insufficient actuator headroom.
- The simulator's soft joint limits allow up to 0.00802 rad (0.46 degrees)
  beyond the URDF range. The command angles themselves stay within the range.
- Maximum realized fingertip-site displacement from the source human tips is
  **123.1 mm**. It exceeds the earlier 80 mm locality criterion. The robot uses
  a larger regrasp; this is not yet a faithful local repair of every fingertip.
- The force QP is a soft-residual guidance problem. Only 13/46 frames satisfy
  its separate 0.02 N / 0.002 Nm residual test. The largest residuals are
  0.373 N and 0.0669 Nm. Numerical convergence is not wrench feasibility.

The output sets `accepted_tracking=false` and `ready_for_training=false`.
Passing software tests does not change those flags.

Replaying identical commands without reoptimization gives:

| Variation | Worst position error | Worst orientation error |
|---|---:|---:|
| 0.5 ms timestep | 11.42 mm | 10.30 degrees |
| 0.9 kg book | 9.70 mm | 10.17 degrees |
| 1.1 kg book | 12.63 mm | 11.18 degrees |
| Friction 0.5 | 10.52 mm | 12.95 degrees |

All four still fail the strict tracking gate. They do show that carrying in
this local experiment is not solely an artifact of the 1 ms timestep.

## Implemented method

The retargeting output supplies timestamps, robot configuration, human/robot
fingertip trajectories, object poses, mesh and table metadata. No CV or RL is
run here. The fixed pelvis/leg pose is taken from the first selected frame.

### Realizing a grasp with the actual hand

Independent pad contact locations cannot simply be copied to Dex3. The robot's
finger triangle, link thickness, wrist reach and self-collisions constrain them.
This prototype chooses opposing hand-side contacts on the book's thin faces.
It projects the retargeted fingertip sites toward those faces, with a 14 mm
offset for the fingertip frame's location within the real distal geometry.
A bounded tangential candidate shifts the left and right contact patches by
40 mm in opposite local X/Z directions. This book-specific regrasp aligns their
force-support regions. It is an experimentally selected initialization, not
a general automatic contact-mode planner.

Collision-aware IK minimizes

\[
\|W(p_{\mathrm{tip}}(q)-p^*)\|^2
+w_q^2\|q-q_{\mathrm{prior}}\|^2
+w_c^2\sum_{a,b}[\max(0,-d_{ab}(q))]^2,
\qquad q_{\min}\le q\le q_{\max}.
\]

Here signed distances come from MuJoCo's collision manifold; fingertip and
contact Jacobians are analytic. This is a local least-squares solve, not a
global collision-free IK guarantee. The established grasp is propagated along
the book pose path with warm starts, allowing the waist to turn.

### Contact-force guidance and robot control

Object inverse dynamics is recomputed at the slowed timing. Translational and
angular acceleration terms scale down with the squared time scale; gravity
does not. The existing sparse force QP allocates six contact forces with
unilateral normal forces, conservative friction diamonds and a 20 N normal-force
cap. Net force/moment residuals are penalized and separately reported.

For active robot coordinates, the applied motor command is

\[
\tau=\operatorname{clip}_{[-\tau_{\max},\tau_{\max}]}
\left[
M_{rr}(q)\ddot q^*+h_r(q,\dot q)
+K_p(q^*-q)+K_d(\dot q^*-\dot q)
+\sum_i J_i(q)^T f_i^*
\right].
\]

The feedforward acceleration is the cubic command spline's second derivative.
The inertia and bias terms come from the current MuJoCo state. Fingertip forces
are mapped to robot torques, **never applied directly to the book**. Their
effect on the book is mediated by simulated collisions and friction.

Finger torque caps are 0.76 Nm, following the local challenge actuator config;
other joint caps come from the URDF. Diagnostic gains are 300/8 for arm/waist
position/velocity errors and 20/0.4 for fingers. These gains are explicitly
experimental and are not a hardware-tuned controller.

The online controller reads robot joint state, not object pose error. Object
tracking error is used only by the offline correction between rollouts.

### Offline correction

Each rollout supplies object position and orientation error. Smoothed error
proposes a shared SE(3) bias for the six contact commands, bounded by 30 mm
translation and 0.4 rad rotation. Damped Jacobian updates realize the biased
sites on the robot, clipping command joints to limits. A fresh physical rollout
accepts a proposal only when its tracking objective improves and maximum
overlap stays below 1 mm. Gains 0.6 and 0.25 provide backtracking.

This is approximate shooting/iterative command correction, not a differentiable
simulator gradient or a full constrained optimal-control solution. The inner
Jacobian update is not collision-certified; the outer physical replay checks
the realized result. The next useful improvement is joint contact/torque-aware
grasp refinement: prevent infeasible load allocation before asking the tracking
controller to correct it. Increasing gains alone did not solve that problem.

## SONIC's role

[NVIDIA GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) is a learned
whole-body motion-tracking controller. A motion encoder supplies latent motion
tokens and a state-conditioned decoder produces robot joint actions. Its
separate kinematic locomotion planner provides movement references.
It is broader than a walking IK solver. The local challenge integration sends
non-hand joint commands through SONIC and treats hand commands separately.

For this bench, the fixed pelvis removes balance from the experiment, so SONIC
is unnecessary. Later, releasing the fixture would introduce whole-body tracking
and balance. SONIC can then provide a pretrained execution layer, but it does
not automatically certify book contact forces, grasp stability or torque margins.
Neither the SONIC policy nor a walking policy has been loaded for this demo.

## Reproduce and inspect

Use the optional G1/Pinocchio runtime described in [SETUP.md](../SETUP.md) and
the CPU simulation dependencies. From the repository root:

```text
python -m correction.scripts.supported_robot --verify
python -m correction.scripts.view_supported_robot --port 8084
python -m unittest correction.tests.test_supported_robot -v
```

Default input is the saved retargeted episode under
`correction/outputs/reorganization_check_20261017/retargeted/contact_aware.npz`.
Use `--input`, `--start`, `--stop`, `--time-scale` and `--output` to select a
trial. This remains a book-specific prototype; other objects are not supported
just by changing the filename. `--prepared-reference` is an optional cache of
the initial G1 path; its construction time is then excluded and flagged.

The main output directory is `correction/outputs/supported_g1_20261019/`:

- `reference.npz`: commanded robot path, initial state, desired object path,
  force feedforward and input metadata.
- `baseline.npz`, `geometry_only.npz`, `wrench_seed.npz`, `corrected.npz`:
  actual simulated robot/object states, torques, velocities, contact resultants
  and errors. They are not interchangeable with command trajectories.
- `report.json`: results, rejected tracking status, timing and sensitivity runs.
- `scene.xml`: exact local physics model.

The viewer includes corrected-only, raw/corrected, geometry/corrected and
force-seed/corrected modes, source-frame scrubbing, desired pose outlines and
measured contact-force arrows. Tests verify model/FK agreement, free fall when
the book has no contacts, frozen root/legs, deterministic replay and honest
failure flags. They do not certify manipulation quality or faster RL learning.
