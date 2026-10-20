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
