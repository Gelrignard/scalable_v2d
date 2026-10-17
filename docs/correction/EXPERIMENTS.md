# Correction experiments and measured limitations

See [METHOD.md](METHOD.md) for the mathematical derivation, required inputs,
measured limitations, and proposed policy-learning comparison.

The correction stages use the existing retargeting output and CPU robot runtime.
The PowerShell launcher activates that runtime; it does not install packages.
See [the module README](../../correction/README.md) for the current layout.

## Implemented scope

1. Load a real Tier 1 object trajectory, respecting w-first quaternions, visibility,
   object-origin/COM differences, timestamps, and coordinate frames.
2. Estimate the required object wrench using Newton–Euler dynamics. Gravity is
   included; the resulting wrench is the demand on **all other contacts**, not
   necessarily on the hands alone.
3. For an explicitly selected sticking phase, correct contact targets inside
   bounded planar surface patches while holding the smoothed object trajectory
   fixed. Inner QPs allocate contact forces; a small outer optimization moves
   contact locations. Human/retargeted point proximity remains in the objective.
4. Export corrected contact targets, force profiles, numerical residuals, and
   explicit failure reports. If a dense reference is supplied, validate every
   reference frame after interpolating the optimized force samples.

The generic phase optimizer solves for contact targets. The real-book integration
below also corrects G1 arm/hand joint trajectories through the existing robot
model. It does not impose full self-collision, actuator Jacobians, whole-body
balance, or controller dynamics. No physical robot rollout, policy training, or proof of
reduced training cost has been performed. A low object-wrench residual is only a
necessary model-level check; it is not a robot-execution certificate.

## Run from the workspace root

```powershell
.\correction\scripts\run.ps1 -Mode test
.\correction\scripts\run.ps1 -Mode demo
.\correction\scripts\run.ps1 -Mode fetch -Episode 11
.\correction\scripts\run.ps1 -Mode diagnostic -Episode 11 -MassKg 1
```

The sample is the real Tier 1 `light_blue_book` episode 11, 901 frames. The fetcher
downloads only its poses, mesh and metadata, approximately 12 MB, from immutable
dataset revision `5f68335f3acc802033d1e80728c1633197521de8`. A SHA-256 manifest
records the downloaded bytes and is checked before the diagnostic. The diagnostic
does not require body-model assets. The real-robot experiment below downloads
pinned SOMA assets into this directory and uses a separate CPU Torch environment.
No cloud compute or GPU training is used.

## Real Tier 1 robot comparison

The integration uses the actual 900 visible frames of episode 11. The final
invisible frame is excluded. `soma_tools/decode_tier1.py` applies the official SOMA model,
including time-varying flexible bone parameters, then the documented constant
CV/SOMA/G1 coordinate transforms and supplied ground-plane leveling. Object poses
remain prescribed. `.soma-env` is separate from the working Pinocchio environment.

The first correction trial uses frames 135–175 (4.50–5.83 s), an airborne carry
segment with six unchanged inferred contact identities. Contact optimization has
12 object-local patch coordinates and a force QP over every selected frame.
Tangential placement balances human anchors with the retargeted robot's FK, within
a 3 cm coordinate bound. This handles hand morphology without choosing a new grasp.
The resulting targets are projected onto the actual mesh. A load-weighted local
joint projection prioritizes normal contact closure; base, legs, object and timing
remain unchanged. Fifteen transition frames at each end connect to the original
trajectory under saved-frame joint velocity limits.

The robot projection penalizes changes in the joint correction between frames
and uses a nonlinear line search. Surface-normal closure receives more weight
than exact tangential placement. This improves actual surface contact, but does
not enforce sticking: the current result still has object-relative fingertip
motion. The force audit tests instantaneous wrench existence, not sliding-friction
direction, closed-loop stability, or full-robot dynamic feasibility.

```powershell
# Existing CPU SOMA environment: torch 2.8.0+cpu, py-soma-x 0.2.1, pyarrow, scipy.
.\.soma-env\Scripts\python.exe -m soma_tools.decode_tier1
.\retarget.ps1 run --input correction/outputs/tier1_book/human_motion.npz --method contact_aware --output correction/outputs/tier1_book/retargeted
.\.tools\Library\bin\micromamba.exe run -p ./.retarget-env python -m correction.scripts.book_trial
.\.tools\Library\bin\micromamba.exe run -p ./.retarget-env python -m correction.scripts.repair_robot
.\.tools\Library\bin\micromamba.exe run -p ./.retarget-env python -m correction.scripts.audit_book
.\.tools\Library\bin\micromamba.exe run -p ./.retarget-env python -m correction.scripts.view_book
```

Outputs are in `correction/outputs/tier1_book/`: the original and corrected robot
NPZ/JSON trajectories, `wrench_plan.json`, `robot_repair_metrics.json`, and
`fk_force_audit.json`. The viewer is at `http://127.0.0.1:8082` with synchronized
before/after replay. `correction/src/dynamics/contact_audit.py` removes force authority from fingertips more
than 3 mm away from the 2 mm-radius tip contact configuration, then solves force
balance again at the attained FK geometry. It separately reports probe penetration.
A successful target-level wrench solve is **not** accepted as successful robot
contact execution. Remaining gaps, penetrations and wrench failures are reported.

Measured final trial (41 selected frames): mean absolute surface gap 10.76 →
0.40 mm; maximum hand-probe penetration 26.84 → 0.99 mm; instantaneous wrench
and 5 mm probe checks 0/41 → 41/41. No joint-position or saved-frame velocity
violations were detected. Object-relative tip speed is still 45 mm/s on average;
the output is explicitly **not accepted as a feasible manipulation**. Probe
coverage is incomplete. The 3 mm contact allowance and assumed mass/friction
must be considered when interpreting the wrench result.

CPU timings: full 900-frame retargeting approximately 117 s; phase contact
optimization approximately 6.3 s; final local robot projection approximately
12.2 s including preparation and projection, before final geometry evaluation
and the separate force audit. These are measurements
on this sample, not evidence of reduced RL cost or large-dataset scalability.

Convenience commands: `.\correction\scripts\run.ps1 -Mode book` reruns correction and
audits from the saved reference; `-Mode view-book` starts the comparison viewer.

The **1 kg mass is an explicit experimental assumption**, not a measurement.
The diagnostic assumes a uniform solid bounding box for COM/inertia and gravity
normal to the supplied ground plane. These parameters must be replaced with the
evaluation simulator's parameters before making simulator-specific claims. The
G1 support stage uses a different world frame and is not silently combined with
source-world object poses.

## Results and what they mean

Initial CPU run on this machine, 2026-10-12:

| Check | Result |
|---|---|
| Controlled 16-sample pinch correction | 4 outer iterations, 9 force QPs, about 0.07 s inside the solver |
| Maximum force mismatch in that controlled example | 1.56 N before, 0.000015 N after |
| Maximum torque mismatch in that controlled example | 0.390 N m before, 0.00116 N m after |
| Actual Tier 1 episode 11 dynamics diagnostic | 901 frames, 900 visible, 890 usable interior frames; about 0.30 s excluding interpreter/import startup |
| Change from smoothing that reference | 0.70 mm RMS translation; this is a displacement measurement, not an accuracy improvement |
| Analytic mechanics, constraints, and integration tests | 21 passed |

The controlled reference was deliberately displaced 4 cm sideways and 1 cm away
from each surface; the largest resulting target correction was 4.11 cm. The
surface/force model is synthetic. These numbers do not measure challenge success,
policy training speed, or full-episode robot feasibility. Timings vary and exclude
process startup; the JSON files record the latest individual runs.

- `outputs/controlled_pinch/report.json`: controlled synthetic box-lift case with
  a deliberately offset grasp. This checks that mechanics can drive a contact
  correction. Its contacts are **not** inferred from the challenge video.
- `outputs/controlled_pinch/correction.npz`: original targets, geometric
  projection baseline, corrected targets, and allocated contact forces.
- `outputs/tier1_episode_000011/report.json`: actual challenge-episode dynamics
  diagnostic and all physical assumptions. This is **not** a corrected robot demo.
- `outputs/tier1_episode_000011/object_dynamics.npz`: raw and smoothed poses,
  validity masks, and object-frame force/torque demands. No contacts are invented.

In the controlled example, geometry-only projection reaches the box surface but
leaves the grasp displaced sideways. The resulting vertical force creates an
unwanted moment. The optimizer moves the two points within their allowed patches
to remove that moment. It preserves the prescribed object wrench and respects
compressive force limits and a conservative friction cone with a reserve.

The real-episode diagnostic compares derivative sensitivity before/after an
11-frame local smoother. A smaller acceleration or torque spike is not evidence
that the true motion was recovered. Raw poses are retained, displacement is
reported, and derivatives near missing observations and segment boundaries are
excluded. No scoring trajectory is exported or altered.

## Optimization definition

Each contact has one 2-D coordinate on its object-local planar patch throughout
the phase. The force trajectory has three components per contact per force knot.

The objective penalizes:

- distance of contact locations from the median demonstrated object-relative
  targets, scaled by `target_sigma_m`;
- normalized object force/torque imbalance at all optimization knots;
- force magnitude and the rate of change of force components in object axes.

Hard constraints bound patch coordinates, correction size, normal force, and
friction. The inner force QP uses
`abs(ft1) + abs(ft2) <= 0.85 * mu * fn` by default, a conservative subset of a
Coulomb cone. Net wrench balance uses a soft residual: feasibility is checked
after optimization, in N and N m, rather than inferred from optimizer success.

The outer L-BFGS-B search optimizes only `2 * contact_count` variables, using an
analytic envelope derivative of the inner QP. This is a local, nonconvex solve.
The implementation does not build a full wrench polytope or search for a grasp
from scratch. Contact hypotheses, physical parameters and phase selection remain
explicit inputs.

Force caps are inputs; they are not automatically certified robot force limits.
Environmental wrench defaults to zero, which is appropriate only when deliberately
assuming the selected robot contacts provide all support. Supported phases need
an explicit environmental model/wrench. Sliding, rolling, impacts, contact-mode
inference, and articulated objects are outside this first version.

See [INTEGRATION.md](INTEGRATION.md) for the interface to the retargeting work.
