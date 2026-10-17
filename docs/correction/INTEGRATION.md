# Interface to the retargeting pipeline

No dependency on the retargeter's code, viewer, Python object serialization, or
joint ordering is required. Export geometric arrays with explicit frame IDs.
Use NPZ with `allow_pickle=False`; strings must be NumPy Unicode arrays.

## Reference bundle

| NPZ key | Shape | Meaning |
|---|---|---|
| `timestamp_s` | `[T]` | Original increasing, approximately uniform timestamps |
| `frame_index` | `[T]` | Increasing original frame IDs |
| `object_pose_xyz_wxyz` | `[T,7]` | `world_T_object`, metres, quaternion **w first** |
| `object_visible` | `[T]` bool | False for unusable object poses |
| `reference_points_world_m` | `[T,K,3]` | Candidate robot surface/fingertip points in the same world frame |
| `point_names` | `[K]` string | Corresponding contact/robot-link identifiers |
| `world_frame_id` | scalar string | Explicit world-frame name |
| `object_frame_id` | scalar string | Exact object mesh/body frame name |
| `environment_wrench_object` | optional `[T,6]` | Known environmental force and COM torque in object axes, excluding gravity |

Do not use wrist origins as fingertip contacts. Use actual candidate contact
locations on robot links, with their relationship to the kinematic model retained
by the caller. The corrector returns point targets; the retargeter must check
whether G1/Dex3 can reach them with acceptable motion, collision and actuation.

## Config JSON

`contact_mode` must be `sticking`. Provide `world_frame_id`, `object_frame_id`,
`provenance`, and a `body_model` with:

```json
{
  "mass_kg": 1.0,
  "com_object_m": [0, 0, 0],
  "inertia_com_object_kg_m2": [[0.02, 0, 0], [0, 0.03, 0], [0, 0, 0.04]],
  "gravity_world_m_s2": [0, 0, -9.81],
  "provenance": "Replace example parameters with verified simulation parameters."
}
```

Gravity must use the declared world frame; the example above is Z-up only.
Inertia must be about the COM and expressed in object axes, not about the mesh
origin or in principal axes unless those frames coincide.

Provide `patches`, one per point, with these fields:

- `name`: identical to the corresponding `point_names` entry.
- `center_object_m`: a point **on the surface**, in object coordinates.
- `inward_normal_object`: compressive-force direction into the object.
- `tangent_u_object`, `tangent_v_object`: unit orthogonal surface directions.
- `half_extent_uv_m`: two positive bounds on this local planar approximation.
- `friction`: assumed Coulomb coefficient.
- `max_normal_force_n`: explicit cap, to be checked against robot actuation later.

Optional `settings` keys are documented by `CorrectionSettings` in
[contact_optimizer.py](../../correction/src/optimization/contact_optimizer.py).
`correction/outputs/controlled_pinch/input_config.json` is a working patch/settings example;
add the body/world fields above for the geometric reference adapter. Its physics
and patches are synthetic, not appropriate parameters for the challenge book.

## Prepare and solve a phase

Run these modules with the CPU environment used by `correction/scripts/run.ps1`:

```text
python -m correction.scripts.prepare_phase --reference reference.npz --config config.json --output phase.npz --start-frame 100 --stop-frame 160 --stride 3
python -m correction.scripts.run_phase --phase phase.npz --config config.json --output correction/outputs/my_phase
```

The selected frame interval is start-inclusive and stop-exclusive. Select one
plausibly sticking interaction, excluding grasp establishment/release and missing
poses. Phase detection is not implemented. The adapter rejects invalid boundary
derivatives, mismatched frame IDs, point-name order, and more than 64 force knots.

It first smooths object poses over the complete reference, records the difference
from the original input, and calculates the object's required wrench. The
contact optimizer holds that smoothed reference fixed. It does not retime or
freely modify the object path to make a grasp easier.

## Returned data and interpretation

`correction.npz` includes:

- `corrected_contact_targets_world_m [N,K,3]` for every selected dense frame;
- `after_contact_points_object_m [K,3]`, constant over the sticking phase;
- force components/forces at knots, interpolated forces at dense frames;
- `dense_object_pose_xyz_wxyz`, unchanged by the contact optimizer;
- `dense_original_object_pose_xyz_wxyz` and `dense_frame_index` for audit;
- force/torque residuals and frame IDs.

`report.json` separates numerical optimizer convergence, force/correction
constraints, wrench tolerances, and dense validation. Do not accept a phase solely
because `optimizer_converged` is true. Do not accept an apparently successful knot
solve if `dense_validation.object_wrench_within_tolerance` is false. Dense checks
cover reference frames, not continuous time or faster simulator substeps.

The caller still needs to solve robot kinematics/dynamics for the corrected
targets and test controller rollouts. If the robot cannot realize a candidate,
its reachability/actuation restrictions must return to the optimization; blindly
applying point targets would not complete the method.

Keep Tier 1 and Tier 2/3 reference bundles separate. Clean Tier 1 trajectories
must never become correction targets for Tier 2/3.
