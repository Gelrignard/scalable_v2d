# Staged correction with observation feedback, without MPC

## Decision and scope

At the user's request, stop predictive control experiments and return to the
preserved staged trajectory and executor. The first isolated change updates
the **force allocation stage** using current simulator observations. It does
not replan the hand path, select new grasps, retime the episode, or simulate
candidate futures. The saved original, corrected planning reference, and
actually simulated object motion remain separate.

This is a deliberately bounded implementation of staged feedback, not a claim
that the whole correction problem is now closed loop. The previous unconditional
hand-position following worsened the second grasp. Therefore it stays disabled
while this force-stage substitution is tested independently.

The fixed nominal controller already observes joint state, actual Jacobians,
object rotation and matched contact forces. Its nominal trajectory, force
profile, contact bodies and clock are retained. The adapter adds object pose
and twist feedback through a small instantaneous force QP, using observed
physical contact points and normals. There are **zero predictive rollouts**.

## Example and variables

Suppose the saved motion raises the blue bin. The robot follows the same
corrected arm/finger trajectory and uses the same nominal joint guidance. If
the actual bin lags below its planning reference, the controller requests a
small additional upward object wrench. It allocates this only through patches
currently touching the bin, using their actual lever arms and robot Jacobians.
If only the left patch touches, the right patch has zero object-wrench authority.
If neither touches, allocation cannot lift the bin: the unmet wrench stays in
the log, and nominal approach guidance continues. There is no invented object
force and no claim that force feedback alone can recover a missing grasp.

Let the corrected planning pose be (p_d, R_d), and the actual simulated object
pose and twist be (p, R, v, omega). The controller transforms both poses to
their centers of mass. In the following equations p and v denote those COM
quantities. Reference velocity uses a smoothed position spline and a centered
orientation secant; **observed acceleration is never used**.

$$
e_p=p_d-p,\qquad e_R=\operatorname{Log}(R_dR^T),
$$

$$
\Delta w_d=
\begin{bmatrix}
\operatorname{clipnorm}\{\gamma m(K_p e_p+K_v(v_d-v)),3\ \mathrm{N}\}\\
\operatorname{clipnorm}\{\gamma I_W(K_R e_R+K_\omega(\omega_d-\omega)),0.15\ \mathrm{Nm}\}
\end{bmatrix}.
$$

This is a feedback **increment**, not the whole gravity/inertial wrench. The
saved nominal force guidance still carries its original role. Translational
gains are 8 and 3; angular gains are 12 and 3. The single declared active trial
uses gamma=1. No gains are fitted to the outcome during this experiment.

The planning pose is reconstructed from saved ancestry, including the original
second-grasp planning translation. Original intended motion remains the scoring
target. Subtracting the original target indiscriminately would count that
offline correction as an execution error. This inherited shifted planning
reference is a limitation to revisit, not a new observation-derived target.

## Instantaneous force-stage optimization

For each inherited contact channel i, let c_i and n_i be the current physical
contact point and inward normal. They are force-weighted measurements of
contacts matched to the inherited robot body, local surface neighborhood, and
normal direction. This is the base controller's existing match; it is not a
new grasp classifier. If the channel is active and physically touching, define

$$
G_i=\begin{bmatrix}I\\[c_i-p]_{\times}\end{bmatrix}.
$$

Otherwise G_i=0. J_i is computed at the current robot state and contact point;
for a separated channel its saved robot surface point is used only for smooth
unloading of any residual joint guidance. Let f_i^0 be the base controller's
force command, after its original feedback and gating. At each 1 ms physics
step solve for a small force change delta_f:

$$
\min_{\delta f}\quad
\left\|D^{-1}(G\delta f-\Delta w_d)\right\|^2
+0.1\|\delta f\|^2,
\qquad D=\operatorname{diag}(3,3,3,0.15,0.15,0.15).
$$

Constraints:

- Each world-axis force increment is bounded by 0.75 N and changes by at most
  10 N/s. These are componentwise bounds, not bounds on the vector norm.
- At matched contacts, total nominal-plus-increment force satisfies an
  eight-sided friction approximation, a 15 N normal cap, and a normal floor
  of 75% of nominal compression. If nominal guidance already violates a
  constraint, the increment may not worsen that particular violation.
- Let tau_0 be the **full current** nominal torque request: PD tracking,
  gravity minus passive force, inertial feedforward, and nominal contact
  guidance. Require

$$
|\tau_0+J^T\delta f|\leq
\max(\tau_{cap},|\tau_0|).
$$

- Within 0.01 rad of a joint stop, the added joint torque may not push farther
  toward that stop. This uses actual joint positions; it does not certify
  future joint-limit avoidance.

The no-worse envelope does **not** turn an already excessive nominal request
into a valid actuator command. Simulation still clips actual torque at the
unchanged physical caps, and overall acceptance still rejects excess requests.
Likewise the friction check is an approximate local command check, not proof
that the simulator will realize the requested forces.

Execute tau_0 + J^T delta_f on the robot. Do not change object state or apply
object forces directly. Motion commands and their velocities remain the saved
nominal commands, so observed q is never copied abruptly into the command.
The updated action is a force/torque increment; this does not claim that a
future RL policy must use the same action representation.

On contact loss, existing residual robot guidance unloads through the force
rate constraint, while the separated channel has zero modeled object-wrench
authority. Changing constraints can make the rate box infeasible. Then the
solver prioritizes the static constraints, records a rate conflict and rejects
the continuity contract. It never labels that abrupt fallback as smooth.

## What was reused and what was left out

Reused from the feedback experiments: preserve nominal behavior; distinguish
planning/scoring poses; evaluate geometry at the actual state; include full
actuator demand; use actual contacts for wrench authority; constrain action
increments; expose constraint conflicts and unmet wrench.

Left out: prediction, candidate future selection, phase search, new contact
families, automatic grasp recovery, and geometry-only arm following. The
inherited contact schedule and grasp choices remain prototype-specific. No
scalability or RL speedup claim follows from this experiment alone.

## Predeclared validation

1. Full 858-frame / 28.57-second replay with gain zero: require exact equality
   of robot motion, object poses, torques and requested torque peaks.
2. Same uninterrupted episode with gain one: require identical physical prefix
   before contact activation, record all physical/continuity failures, and
   compare position **and** orientation against the preserved nominal.
3. Promote only after physical acceptance, no bound conflicts and no worse
   position/orientation tracking. The old nominal itself is not physically
   accepted; preserving it does not confer acceptance on the new run.

Both use the saved initial state, dynamic upper body and object, and the
recorded pelvis path as an externally supported fixture. No walking simulation,
resets, artificial grasp or direct object assistance is introduced. Source
snapshots, input hashes, runtime and peak process memory are recorded per run.

Implementation: `correction/src/optimization/observed_force_update.py`,
`correction/src/dynamics/staged_observation.py`, and
`correction/scripts/replay_staged_observation.py`.

```powershell
python -m correction.scripts.replay_staged_observation --source correction/outputs/overnight_20261020/rest_matched_v1 --planning-sequence correction/outputs/overnight_20261020/proximal_sequence_v1 --output correction/outputs/new_staged_observation --gain 1
```

Use the existing optional robot environment and a fresh output directory.
Set gain to zero for regression. Generated outputs stay ignored by Git.

## State-timing correction during review

MuJoCo's `mj_step` advances qpos, but its Cartesian/contact caches describe the
last solved step. The initial adapter read COM from that cache. Version 2
computes object COM directly from the current qpos and refreshes robot
kinematics/Jacobians in a separate `MjData` using `mj_kinematics` and `mj_comPos`.
This is a current-state kinematic evaluation, with no dynamics prediction and
no mutation of the nominal controller's physics/contact data. It does not
change the saved baseline executor.

Contact forces, points and normals remain the latest **solved** contact
measurement, with a one-integration-step sensing delay. Contact existence is
not certified for a future state. The new tests deliberately leave Cartesian
caches stale to check current-pose feedback and verify that kinematic refresh
does not alter the physical data. Version 2 uses the same gains as version 1.

## Initial completed trial

The zero-feedback full episode exactly reproduces all six checked physical
arrays. It takes 139.99 s wall / 126.27 CPU-seconds / 1.138 GB peak RAM.
Before the timing correction, `active_v1` gives 48.735 mm / 5.936 degrees RMS,
versus 48.484 mm / 5.912 degrees nominal. It is not promoted. Requested torque
peaks at 2.349 times capacity, and 24 rate/static-constraint conflicts produce
a maximum residual rate of 123.05 N/s, exceeding the declared 10 N/s bound.
All 12,101 static QPs solve after the exposed fallback; that is not continuity
or physical success. No direct object assistance or reset occurs.

Version 1 takes 196.52 s wall / 186.06 CPU-seconds / 1.139 GB peak RAM;
force allocation accounts for 61.04 s. It uses no GPU. Of the video-rate
scheduled-contact samples, 52 have no matched active patch; every one has
exactly zero allocated additional object wrench. An unmatched selected patch
does not mean the entire robot has no incidental contact: frame 495 has a
right-hand contact elsewhere but no contact matching the selected channels.

The first grasp's position RMS is almost unchanged (25.455 to 25.411 mm).
Second-grasp rotation RMS worsens from 5.979 to 6.191 degrees. These results
show preservation of most nominal behavior, not a demonstrated improvement.
They do not establish disturbance robustness, transfer or learning savings.

The initial saved event logs retain the fallback solve status, not the first
rate-constrained solve status. Thus the recorded `rate_conflicts` count means
that the bounded-rate solve failed validation and required fallback; it does
not prove every event was mathematically infeasible rather than numerically
unconverged. The current code additionally records `rate_status` for future
runs. This is a logging-only change after the active-run source snapshots.
The measured rate excess itself is unambiguous.

## Completed current-state rerun and decision

Version 2 completes the same full episode with the first 236 physical frames
exactly unchanged. No parameters were retuned after version 1. It is **not
promoted**: aggregate tracking, continuity and physical acceptance do not pass.

| Metric | Preserved staged nominal | Observed force stage v2 |
|---|---:|---:|
| Object position RMS | 48.484 mm | 51.533 mm |
| Object orientation RMS | 5.912 deg | 6.022 deg |
| Requested torque / capacity, peak | 2.307 | 2.273 |
| Saturated 1 ms steps | 2498 | 2548 |
| Joint-limit excursion | 9.049 mrad | 9.568 mrad |
| Residual rate fallback events | 0 | 16 |
| Physical acceptance | Fail | Fail |

The 12,104 force updates have no final static-solve failures. Maximum force
increment is 0.1403 N per axis. However, bounded-rate solve fallbacks reach
104.98 N/s against the declared 10 N/s limit. These runs therefore do **not**
demonstrate smooth residual execution through every transition. The saved
maximum static inequality violation is approximately 1.08e-7.

Second-grasp orientation RMS improves from 5.979 to 4.415 degrees, and sampled
bilateral contacts increase from 46 to 66 frames. Yet its position RMS worsens
from 61.431 to 63.725 mm. After the second release, position RMS increases from
42.043 to 60.309 mm. Better contact count or an improved interval is insufficient
to claim a better whole manipulation. Forty scheduled-contact samples have no
matched active patch; additional allocated wrench is exactly zero on all of them.

Runtime: 248.45 s wall / 227.59 CPU-seconds / 1.138 GB peak process RAM, no GPU.
Force QPs account for 77.71 s (approximately 6.42 ms per update on this machine).
The implementation is an offline feedback experiment, not a real-time 1 kHz
controller. There are no prediction rollouts and no RL training. Compared with
the previous 528.81 s guarded experiment this run is cheaper, but it is a
different algorithm and one episode; this is not a general scaling benchmark.

The focused suite passes **18 tests**, covering force/torque/rate constraints,
contact loss, preload, joint-stop direction, COM timing, read-only current
kinematics, disabled equivalence, and the existing controller/torque audit.
After adding the diagnostic-only first-solve status field, all 10 adapter/QP
tests were rerun and passed. Neither suite substitutes for the failed physical
and continuity acceptance above.

- [Disabled full replay](../../correction/outputs/staged_observation_20261020/zero_v1/report.json)
- [Initial active replay](../../correction/outputs/staged_observation_20261020/active_v1/report.json)
- [Current-state active replay](../../correction/outputs/staged_observation_20261020/active_v2/report.json)
- [Interval diagnosis](../../correction/outputs/staged_observation_20261020/comparison_active_v2.json)
- [Experimental viewer](http://127.0.0.1:8088/): original retargeting left,
  current staged feedback right, original intended object ghost/path orange.
- [Preserved comparison and independent support-point channel](http://127.0.0.1:8087/).

No additional viewer server is created. Large arrays and temporary diagnosis
scripts remain ignored. The existing experimental catalog filename retains
`mpc` for compatibility, but its current entry contains no MPC.

## Interpretation and next staged step

This tests force allocation with observation feedback, not every possible
observed-state staged method. Force-only correction cannot recover a missing
selected contact, and an instantaneous no-worse torque check does not prevent
larger torque demands later as the physical state changes. The next useful
stage to investigate is bounded geometric reacquisition from the actual
object pose, followed by force allocation on the acquired patches. It should
retain the saved grasp family and task goal, constrain changes from the last
command, and expose failed acquisition rather than following an assumed object
trajectory. This differs from the previously rejected unconditional hand
following. That next step has not been implemented in this trial.

The force-stage implementation remains opt-in. Neither the preserved nominal
nor a new run with physical/continuity failures is a training-ready certificate.
