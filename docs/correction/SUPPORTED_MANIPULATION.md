# Supported EEF manipulation: what the current demo establishes

19 October 2026. This experiment answers a narrower, testable question:
**can demonstration-guided contact commands produce the desired object motion
through actual contact dynamics, without walking or balance getting in the way?**

The answer is yes for the short book segment described below. This is not a
full G1 hand or arm simulation. The old falling-robot replay remains a failed
experiment; passing software tests did not establish manipulation feasibility.

## Evidence and acceptance

The main demo uses Tier 1 book episode 11, **frames 520–565**, lasting **1.5 s**.
The recorded book moves about **52 cm** and rotates about **100 degrees** between
the endpoints. Its desired object poses and timing are unchanged. The crop was
chosen to examine a relatively stable contact pattern during substantial
motion; it is not a complete pick-and-place episode.

| Commands, same supported contact system | Worst position error | Worst orientation error |
|---|---:|---:|
| Raw retargeted contact paths | 78.87 mm | 121.22 degrees |
| Surface projection / constant anchors only | 953.94 mm | 87.64 degrees |
| Inverse-dynamics force seed with preload search | 9.37 mm | 3.45 degrees |
| Force seed plus rollout-based trajectory correction | **1.65 mm** | **2.07 degrees** |

These are simulated object errors, not hand IK errors. Worst errors and
penetration are checked at every integration step, not just displayed frames.
The corrected result has **0.324 mm** maximum contact penetration and **16.29 N**
peak pad actuator force, below the **25 N** limit. No control step saturates.
Maximum commanded pad speed is 0.538 m/s and acceleration 5.67 m/s².

The correction stage, including four contact-mode force solves, preload
selection, and all **15 comparison/optimization rollouts**, took **5.87 s on
the local CPU**. This excludes interpreter/import/input-loading time and the
four additional verification runs. No GPU, RL, or paid compute was used.

Acceptance requires position error <10 mm, orientation error <5 degrees,
penetration <1 mm, initially nonpenetrating geometry, zero externally applied
object wrench, command speed <1 m/s, and command acceleration <10 m/s². A
separate demonstration-locality gate requires every source-frame contact command
to stay within 80 mm of its corresponding human fingertip. The main demo
passes this gate: maximum change is **61.2 mm** from the human fingertips and
**84.1 mm** from the retargeted robot tips. These are meaningful corrections,
not submillimeter preservation of the noisy hand trajectory.

Replaying the **same final commands without reoptimization** also passes with
0.5 ms rather than 1 ms integration, mass 0.9 or 1.1 kg rather than 1 kg, and
friction 0.5 rather than 0.6. Across these four individual changes the maximum
position error is 1.76 mm and orientation error 2.23 degrees. This is a small
sensitivity check, not a robustness or generalization study.

## What is physically simulated

The book is a six-DOF free rigid body with the mesh's bounding-box dimensions,
uniform-box inertia, assumed mass 1 kg, and friction coefficient 0.6. The
recorded table and floor remain collision geometry. These parameters are
explicit local assumptions, not an identified official challenge environment.

There are six 6 mm-radius spherical pads. Each has 40 g mass and three
actuated translational DOFs. Pad–book, pad–pad, and environment collisions are
enabled. A supported translation stage stands in for each fingertip; the
supports' reachable workspace is idealized. There are no legs or arm joints.

At reset, both comparisons receive exactly the same book and pad positions,
orientations, and velocities. Pads start on the book surface without overlap.
Their initial velocities are consistent with the initial rigid-body object
motion. This represents an already established grasp midway through an
episode, not picking up the book. A shared 250 ms command handover avoids a
discontinuous change when entering the cropped reference.

After reset, **the book's position, orientation, velocity, and applied force
are never commanded**. MuJoCo integrates gravity and contact forces. There is
no object weld, pose servo, or attachment. Removing the contacts makes the
book fall under gravity. Controller forces are applied only to the 18 pad
translation DOFs. Actual poses, commands, and measured contact forces are
saved separately; the viewer plays those measured poses.

## Inputs and compact contact description

The input is the saved retargeting NPZ, with timestamps, object SE(3), human
fingertip locations, and retargeted robot fingertip locations, plus the book
mesh and recorded table description. Loading the experiment does not require
the G1/NVIDIA runtime.

Let the object pose be $(p_t,R_t)$ and a human fingertip be $h_{i,t}$. Convert
the human evidence to object coordinates:

$$a^{\rm human}_{i,t}=R_t^\top(h_{i,t}-p_t).$$

Take a temporal median, clip tangential coordinates to the book face, and
enumerate four opposing-face assignments: each hand's thumb may be on either
thin face, with its index and middle fingers on the opposite face. Human
distance and required-wrench residual rank these assignments. A candidate
farther than 80 mm from the median human evidence is rejected.

This produces **six fixed object-relative contact anchors** $a_i$ and inward
normals $n_i$. Constant anchors encode one sticking-grasp hypothesis. This
representation cannot faithfully explain arbitrary regrasping or sliding;
the per-frame locality audit is necessary even when the median fit is good.

## Dynamics guidance and compliant EEF commands

Using a locally filtered derivative estimate of the demonstrated object poses,
inverse dynamics gives the required non-gravitational wrench about its COM:

$$w_t=\begin{bmatrix}mR_t^\top(\ddot p_{C,t}-g)\\
I\dot\omega_t+\omega_t\times I\omega_t\end{bmatrix}.$$

The derivative filter spans 11 source frames. It only estimates the guidance
wrench; the physical rollout is compared against the original, unfiltered
object poses. A sparse, time-coupled QP solves for contact forces:

$$\min_f\sum_t\|W(G(a)f_t-w_t)\|^2+
\lambda_f\|f_t\|^2+\lambda_{\Delta f}\|D_tf\|^2,$$

subject to nonnegative normal force, a conservative friction pyramid, and
20 N maximum planned normal force per pad. The grasp matrix is

$$G_i(a_i)=\begin{bmatrix}I_3\\[a_i-c]_\times\end{bmatrix}.$$

For pad radius $r$, contact stiffness $k_p$, and a small preload $s$, initialize
the world command as

$$u_{i,t}=p_t+R_t(a_i-rn_i+f_{i,t}/k_p+sn_i).$$

The pad center lies outside the surface by $r$; the force/preload terms move
the *controller target* inward. They do not place the physical pad at that
target. Finite pad mass, actuator force, and collision dynamics determine its
actual position. A four-value preload search tries 0, 2, 4, and 6 mm.

The pad controller is

$$F_i=\operatorname{clip}_{\|F\|\le25\,\mathrm N}
\{k_p(u_i-x_i)+k_d(\dot u_i-\dot x_i)-m_{\rm pad}g\},$$

with $k_p=1200$ N/m and $k_d=2\sqrt{k_pm_{\rm pad}}$.
All targets use a cubic spline, so the commanded path is continuous through
velocity and acceleration. No object-state feedback is used during final
playback; the correction is computed offline.

## Rollout-based trajectory correction

The search changes a smooth six-component translation/rotation bias shared
by the six contact commands, while preserving the chosen grasp arrangement.
Translation bias is bounded to 30 mm and rotation bias to 0.3 rad. The physical
rollout objective is

$$J=\frac1T\sum_t\left[
\frac{\|p_t^{\rm sim}-p_t^{\rm ref}\|^2}{(0.01\,\mathrm m)^2}+
\frac{\|\log((R_t^{\rm sim})^\top R_t^{\rm ref})\|^2}{(5\pi/180)^2}
\right].$$

After each rollout, position and rotation tracking errors provide an
**approximate inverse-response search direction**. Smooth it in time and try
step sizes 0.7, 0.35, and 0.15. Accept a candidate only if a fresh physics rollout
reduces $J$ and either meets the penetration limit or reduces the existing
penetration violation. The final result must independently pass the strict
acceptance checks. Stop after eight rounds, no accepted improvement, or the
configured time budget. The time budget is checked between shooting rollouts;
setup and the initial comparisons run first, and a running rollout is not
interrupted. The number of candidate modes and rollouts is also bounded.

This is a bounded shooting/iterative-learning trajectory search. It is **not**
an exact differentiable optimal-control solver, and there is no claim of a
global optimum or general convergence. Its useful evidence is the measured
physical replay, its baseline/geometry ablations, and its CPU cost.

## Why the full 400–600 interval is not the main success claim

The full 6.67-second interval can be carried in this reduced model using a
fixed replacement grasp. However, that solution moves some contact commands
far outside the 80 mm neighborhood of the recorded human fingers. It therefore
fails the demonstration-locality gate even if object tracking succeeds.

The interval also begins with the book/finger geometry intersecting the
recorded table. The full-interval diagnostic explicitly raises the entire
reference by about 9 mm, using the same registration for all comparison
conditions. This preserves object velocities and rotations and is performed
before simulation. It does not silently remove the table or impose the object
pose during a rollout. **The main 520–565 demo needs no such registration.**

We do not claim that the full original contact sequence has been repaired.
Contact-mode transitions, releases, and regrasping remain the next limitation
to address. Later G1 IK must additionally preserve contact, satisfy arm/hand
joint limits and continuity, and avoid actual link collisions. The spherical
pad experiment cannot certify any of those properties.

## Files and reproduction

From the repository root, with the CPU environment active:

```text
python -m pip install -r setup/correction-simulation-requirements.txt
python -m correction.scripts.supported_book --verify
python -m correction.scripts.view_supported --port 8083
python -m unittest correction.tests.test_supported_contacts -v
```

The optional full-interval diagnostic is:

```text
python -m correction.scripts.supported_book --start 400 --stop 601 --output correction/outputs/supported_book_20261019 --verify
```

The implementations follow the existing module boundaries:

- `src/preprocessing/supported_reference.py`: saved retargeting inputs.
- `src/dynamics/supported_contacts.py`: physical plant and force-limited control.
- `src/optimization/supported_manipulation.py`: contact selection, wrench QP,
  preload search, rollout correction, and acceptance/locality checks.
- `src/pass_traj/supported_reference.py`: separate reference, command, measured
  rollout, scene, and JSON report outputs.
- `scripts/supported_book.py`, `scripts/view_supported.py`: run and inspect.

Generated artifacts live in `correction/outputs/supported_book_local_20261019/`.
The report deliberately keeps `ready_for_training=false` and
`robot_hand_or_arm_feasibility_validated=false`. This is evidence that a small
CPU correction can produce a physically useful **contact reference**. The
effect on RL convergence, full robot feasibility, and dataset-wide cost remains
to be measured.
