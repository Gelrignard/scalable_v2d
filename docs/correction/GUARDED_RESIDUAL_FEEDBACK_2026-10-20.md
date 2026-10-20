# Predictive acceptance around the preserved staged controller

## Implemented four-step contract

1. Keep the saved nominal joint trajectory, velocity, inertial feedforward,
   contact force guidance, finger shapes, contact schedule and source clock.
2. Use current simulated robot/object state to compute the existing bounded
   observed-surface residual. Its definition and failed unguarded trial are in
   [the residual note](NOMINAL_RESIDUAL_FEEDBACK_2026-10-20.md).
3. Predict two closed-loop futures from identical physical state and controller
   memory: smoothly return the residual toward zero, or enable surface feedback.
   Apply the proposal only if declared task, impulse and physical checks pass.
4. Execute 0.3 seconds, observe again and repeat. Every selected predicted
   prefix is compared with executed state and controller memory. Rejecting an
   update never overwrites measured state or abruptly clears residual memory.

The default horizon is 1.2 seconds. There are two deterministic candidates,
not a random search or a general nonlinear optimizer. The planner is an
optional supervisor; old execution defaults remain unchanged.

## State and command continuity

The physical state contains robot joints/velocities, object pose/twist and all
MuJoCo integration state. Controller state contains force-feedback filtering,
engagement gain, previous contact torque, any position offset, and the residual
position/velocity, target and update clock. Prediction clones every mutable
array in this state; it shares only the model and immutable reference data.

Both candidates use the same current residual r and rate dr. The return branch
sets its target to zero and retains the damped, bounded residual integration.
The proposal recomputes targets from predicted observations at 50 Hz, exactly
as in execution. Both pass q0+r and dq0+dr to the unchanged torque controller.
Nominal inertial feedforward remains unchanged in this isolated experiment.

When r=dr=0, the return branch is exact nominal execution. After a correction,
returning toward nominal is a braking/decay trajectory, not an instantaneous
jump back. This distinguishes a continuous fallback from resetting commands.

## Prediction and acceptance

Each branch uses the actual supported-G1 physics and the same requested-torque
function, clipping and 1 ms integration as execution. Walking remains externally
supported along the recorded pelvis path. The object is free and receives no
direct actuation. All object wrenches arise from physical contacts.

The task objective averages

$$
e(t)=\frac{\|p(t)-p_d(t)\|^2}{(0.05\text{ m})^2}
+\frac{\|\operatorname{Log}(R_d(t)^TR(t))\|^2}{(0.2\text{ rad})^2}.
$$

Evaluate every 20 ms and at the final predicted state; give the final sample
weight 3. Nominal/proposal forecasts use the unchanged desired object motion,
not the observed motion as their task goal.

### Windowed contact impulse

The check uses all physically realized robot and environment contact forces,
including their moment about the object COM. For each 150 ms window [a,b], form

$$
J_{contact}=\int_a^b\begin{bmatrix}f_{contact}\\
\tau_{contact}^{COM}\end{bmatrix}dt,
\qquad
J_{target}=\begin{bmatrix}
m(v_d(b)-v(a))-mg(b-a)\\
L_d(b)-L(a)
\end{bmatrix}.
$$

Here v is COM velocity and L is world angular momentum about COM. Desired
velocities use a 0.2-second centered secant of the intended SE3 trajectory
(one-sided near endpoints), not differentiated observed acceleration. Desired
angular momentum uses the model inertia at the desired orientation. Residuals
are scaled by weight times window duration for linear impulse, and that scale
times 0.1 m for angular impulse; the force scale has a 1 N numerical minimum.

This is a task-momentum check on simulated contact impulses, not a certificate
of force closure or robustness. It can include legitimate environmental
support. A hand that is separated contributes zero force; a floor reaction
cannot substitute for upward task motion without leaving a momentum/tracking
error. No particular hand count is required. The model's inertia, friction,
collision geometry and privileged state remain assumptions.

### Adoption rules

Require all of the following:

- Stable prediction and at least 1% improvement in average task cost.
- Terminal task cost and mean normalized contact-impulse error no greater than
  102% of the return branch plus 0.01 numerical allowance.
- Peak requested torque fraction no greater than max(1, return-branch peak)
  plus 1e-6. No relaxed actuator clipping.
- Penetration and joint-limit excursion no greater than return-branch peaks
  (with 10 micrometers/10 microradians numerical floors and 1e-6 allowances).
- No new residual speed/acceleration/position-limit conflict.

These are **relative guards**. The retained nominal has physical violations;
selecting a better candidate does not make either trajectory feasible. If both
choices are poor, the supervisor can only retain the less disruptive branch.
An unstable nominal forecast aborts with an explicit failure, never a reset.
Full-episode physical acceptance remains separate and is not relaxed.

## Verification and provenance

Seventeen focused tests pass, including actual MuJoCo branch isolation, exact
prefix reproduction with nonzero residual/controller memory, continuous
rejection, bad-support/torque/collision rejection and the previous controller
tests. An independent check against MuJoCo object velocity covers a rotated,
anisotropic inertia and offset COM. Full-episode validation is additionally required.

Predeclared full-episode trials use the saved bin reference and identical rest
initialization: supervisor wired but disabled, followed by the active supervisor.
Results, exact source snapshots, progress decisions and costs are saved under
`correction/outputs/guarded_residual_20261020/`. The inactive trial must reproduce
the preserved physical arrays exactly. The active trial must retain the same
initial conditions, full 858 frames and zero direct object assistance.

## Completed full-episode result

The disabled-supervisor replay matches every retained physical sample exactly.
The active run completes all 858 frames / 28.57 seconds with zero direct object
assistance, no within-episode reset and the original target trajectory.

| Measure | Preserved nominal | Unguarded residual (previous trial) | Guarded residual |
|---|---:|---:|---:|
| Object position RMS | 48.484 mm | 214.420 mm | 45.845 mm |
| Object rotation RMS | 5.912 deg | 39.437 deg | 6.603 deg |
| Peak requested torque / capacity | 2.307 | 1.777 | 2.307 |
| Saturated 1 ms steps | 2498 | 1037 | 2484 |
| Joint-limit excursion | 9.049 mrad | 9.207 mrad | 9.049 mrad |
| Physical acceptance | Fail | Fail | Fail |

The supervisor made 56 decisions / 112 predictions, accepted one at 16.5 seconds
near the first release, and then smoothly decayed the residual. Nine internal
50 Hz target updates occurred while enabled. Maximum residual position is
0.003076 rad (about 0.176 degrees), rate 0.03554 rad/s and acceleration 1 rad/s2;
there are zero residual-bound conflicts. All selected predicted physical and
controller-memory prefixes match execution exactly (maximum difference 0).

This avoided the previous unguarded collapse and modestly improved position
tracking, but orientation tracking worsened. The preserved reference is not
replaced. Full-episode joint acceleration RMS changes from 3.988 to 4.012 rad/s2;
EEF acceleration RMS changes from 0.811 to 0.799 m/s2. Peak penetration remains
the inherited startup numerical residual, approximately 0.000011 mm.

The first active interaction remains nearly identical (25.45->25.49mm RMS).
In the second, position RMS changes from 61.43 to 59.32 mm and peak requested
torque changes from 1.900 to 1.088 times capacity. The final inactive interval has lower
position error (42.04->16.99mm RMS) but higher orientation error
(3.72->8.77degrees RMS). These changes result from the earlier accepted update;
the controller did not accept a second-grasp correction. An improvement within
one horizon does not guarantee improvement in every later metric.

### What the rejection record tells us

Rejection reasons can overlap: 50 insufficient task improvements, 31 terminal
regressions, 25 joint-excursion regressions, 6 torque regressions and 1 impulse-error
regression. At 23.1 seconds, one proposal predicts a large task-cost improvement
(0.964 to 0.192) and reduced peak torque (1.088 to 0.919 times capacity), but predicts
joint excursion increasing from 0.383 to 1.985 mrad. The declared guard rejects it. We did not relax
the guard after seeing this result.

The next candidate-generation improvement should account for predicted joint
limits while seeking a useful contact correction. This record does not justify
silently allowing greater violation. Nor does it establish whether the present
relative tolerance is the best final criterion for a simulator with soft joint
limits; that is a separate modeling/acceptance decision. We have not yet tested
disturbance robustness, different episodes, force-residual proposals or RL cost.

### Runtime and artifacts

- Active: 528.81 s wall / 496.03 CPU-seconds / 1.139 GB peak RAM, no GPU.
- Prediction: 412.92 s total, 7.37 s per decision on average for a 0.3 s execution
  interval. This implementation is an offline experiment, not real-time MPC.
- Disabled regression: 150.45 s wall / 136.34 CPU-seconds / 1.138 GB peak RAM.
- [Active replay report](../../correction/outputs/guarded_residual_20261020/active_v1/report.json)
- [Complete decision log](../../correction/outputs/guarded_residual_20261020/active_v1/progress.json)
- [Interval comparison](../../correction/outputs/guarded_residual_20261020/comparison.json)
- [Experiment log](../../correction/outputs/guarded_residual_20261020/REPORT.md)

[Port 8088](http://127.0.0.1:8088/) now opens this experimental full-episode
comparison: original retargeting on the left, guarded method on the right,
original intended object ghost/path in orange. The previous v6 replay is a
dropdown option. [Port 8087](http://127.0.0.1:8087/) retains the preserved result
and independent support-point channel. Neither viewer runs optimization.

## Reproduce the guarded experiment

```powershell
python -m correction.scripts.replay_nominal_residual --source correction/outputs/overnight_20261020/rest_matched_v1 --planning-sequence correction/outputs/overnight_20261020/proximal_sequence_v1 --output correction/outputs/new_guard_trial --guard
```

Use the existing optional robot environment and a fresh output directory.
`--guard-disabled` checks nominal regression; `--horizon` and `--period` declare
changed prediction settings. No GPU, training, new contact family, force
reallocation, object reset or trajectory retiming is part of this trial.
