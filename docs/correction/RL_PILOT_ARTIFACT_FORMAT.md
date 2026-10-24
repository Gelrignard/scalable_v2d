# Paired pilot artifact format

Schema: `v2d_rl_pilot_v1`. These are experimental references for comparing
downstream policy learning, not certified demonstrations or challenge submissions.
Read the root `manifest.json` first. Its twelve entries identify tier, episode,
status, reference availability and the hash of each episode manifest.
`RUN_REPORT.md` and `batch_summary.json` describe this batch's actual results;
the method handoff alone does not certify these automatically generated cases.

The same six episode IDs occur in both tiers. Keep them paired when comparing
noise sensitivity. Do not replace a failed corrected case with a successful
different episode. Use explicit fallback/ablation labels instead.

## Episode directory

| File | Purpose |
|---|---|
| `original_reference.npz` | Contact-aware retargeting baseline; shared regularized pelvis path used in the supported comparison |
| `corrected_reference.npz` | Staged candidate, if available; check `reference_kind` before describing it as spline-optimized |
| `contact_guidance.npz` | Planned force schedule, body-local/object-local patches, named contact bodies and stage flags |
| `native_controller_reference.npz` | Original local controller arrays for reproduction; full `q` has xyzw root quaternion |
| `planning_poses.npz` | Object planning reference; separate from simulated motion |
| `original.npz`, `mechanics.npz` | Actual full physical baseline/candidate outcomes and controller diagnostics |
| `human_motion_source.npz` | Decoded same-tier human/object input with original provenance |
| `correction_report.json`, `preparation_report.json` | Optimization failures, configuration, runtime and source hashes |
| `assets/object.glb`, `assets/support.usda`, `assets/collision/` | Supplied object/support and local convex collision pieces |
| `manifest.json` | File hashes, units, names, source transforms, status and model assumptions |

If correction is unavailable, the manifest says so and there is no falsely
named corrected file. A successful process exit only means execution completed.
`physical_accepted: false` and `ready_for_training: false` remain explicit.
An RL experiment can deliberately use such imperfect references as soft goals;
it must not relabel failed rollouts as expert behavior.

## Normalized reference NPZs

Load with `numpy.load(path, allow_pickle=False)`. Fields:

- `timestamps` and `source_frame_indices`: full original sequence, no silent
  removal of occluded frames; seconds and integer dataset row indices.
- `joint_names`, `joint_positions`, `joint_velocity`: named joint order,
  radians and radians/second. Velocity is a command derivative, not measurement.
- `root_position`, `root_wxyz`: robot root position in metres and quaternion
  in wxyz order. These normalized files differ from native `q` ordering.
- `ee_link_names`, `ee_pose_w`: final-reference palm FK, xyz plus wxyz. A palm
  frame is not the selected finger collision patch.
- `object_position`, `object_wxyz`: intended same-tier object input, not the
  actual simulated result. Tier 2 values are not clean Tier 1 scoring targets.
- `object_observed_mask`: false for interpolated/unobserved frames. The missing
  book endpoint is retained; it must not become a confident observation.
- `world_from_source_cv`: constant coordinate transform saved by decoding.
- `correction_coverage_mask` in the corrected file: samples affected by a
  constructed phase candidate. It does not mean those samples passed physics.

All normalized quaternions are wxyz. Reference arrays contain no machine-specific
file dependencies. Source/native artifacts retain historical absolute paths;
use the episode manifest's `asset_paths` when relocating them. Robot assets and
the official policy runtime remain external dependencies. Check the URDF hash
and joint names before importing; never map 31 supported-model actuator columns
onto a whole-body controller by index alone.

## Force guidance and measured transitions

`force_ff` is the nominal robot-on-object force in world coordinates. During
physical replay, the base controller may gate/limit it and add normal-force
feedback; the observation adapter may add another force residual. Neither is
guaranteed to become a physical contact force. The actual measured contact and
wrench fields are in `mechanics.npz`.

The phase flags `phase_has_accepted_static_seed` and
`phase_has_accepted_nominal_wrench` describe stages of the planner. They are not
probabilities and do not certify contact during execution. The first only
reports the selected static candidate; the grasp may fail later in the phase.
`command_spline_accepted` and `physical_episode_accepted` are separate gates.
The episode manifest's `accepted` additionally requires planning and feedback
checks; `physical_accepted` reports the physical replay gate alone.

Current recordings are approximately 30 Hz plus interval impulse/peak audits;
they are not complete 1 kHz action-transition datasets. A learned world model
needs a logger matched to its control timestep and actual executed actions.

## Suggested first downstream check

1. Read a single episode manifest, verify hashes and full-frame correspondence.
2. Recompute palm FK in the training robot and check mapping/units against the
   supplied `ee_pose_w`; use the supplied object/support assets.
3. Choose an explicit action/controller contract. Position-only targets discard
   planned preload unless the learner/controller separately uses the force goal.
4. Compare original and corrected references under identical policy training.
   Keep object-task reward, observation/action interface and evaluation fixed;
   label any added force reward or teacher controller as a separate ablation.
5. Report official task performance and total compute, including failed cases.

The local replay fixes/supports the lower body and uses prototype object
properties. Its RMS errors are diagnostics, not official evaluation scores.
Whole-body policy execution and simulator parity remain downstream work.
