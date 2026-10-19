# Three complete Tier 1 episodes

19 October 2026. The experiment tests generalization beyond the short prepared
book grasp. It includes approach, contact acquisition, manipulation and release,
at the original timing. It does not simulate walking or balance, train a policy,
or constitute an official challenge evaluation.

## Selection and inputs

The selection was made before evaluating the episodes:

```python
random.Random(20261019).sample(list(range(30)), 3)  # [18, 10, 23]
```

Dataset: `nvidia/video_to_data_challenge`, revision
`5f68335f3acc802033d1e80728c1633197521de8`, Tier 1.

| Episode | Object | Frames (all retained) | Duration |
|---|---|---:|---:|
| 18 | Blue trash can | 858 | 28.5667 s |
| 10 | White desk | 794 | 26.4333 s |
| 23 | Toy stroller | 946 | 31.5000 s |

All selected object poses were visible and finite. SOMA decoding and contact-aware
retargeting used every source frame. No episode was replaced after viewing results.
Retargeting required a numerical fix: project joint-limit residues below 0.0001 rad
onto the bounds after an IK iteration. Larger violations still raise an error.
All three saved retargeting diagnostics report zero speed-violating transitions.

## Physical comparison

Both sides start from the identical first robot pose, object pose and velocities.
The robot is the G1/Dex3 URDF with manufacturer collision geometry and inertias.
The pelvis follows the retargeted recording through an explicitly external mocap
weld. That weld constrains only the pelvis, never the object. Twelve leg joints
are frozen at their initial angles and leg collisions are disabled. Waist, arms
and fingers remain dynamic (31 actuated joints). Frozen legs are displayed as a
fixture, not as a walking motion. The robot may be translated and yaw-aligned for
viewing; this display transform does not modify the saved physical trajectory.

The object is a free rigid body with 24 V-HACD convex collision pieces made from
the supplied mesh. This approximates concavities; it is not exact mesh contact.
Assumed masses are 2 kg (trash can), 10 kg (desk) and 3 kg (stroller), with friction
0.6, a 1 ms timestep, and uniform density across convex pieces. These masses are
not provided by the dataset. Overlapping pieces count twice in the assumed mass
distribution. The stroller's wheels are not articulated, so rolling is not modeled.

Initial floor registration is common to both sides: if the lowest collision
vertex in the first recorded pose is within 30 mm of z=0, place the floor at that
height. Otherwise keep the floor at z=0. This rule uses only the initial pose;
it does not optimize the plane against rollout outcomes. It prevents a roughly
21 mm initial desk/floor intersection from launching the desk. The plane offset
is recorded in `floor_z_m`; reference object poses remain unchanged. Remaining
penetrations and later reference/floor inconsistencies are reported.

Only arm/hand/waist motor torques drive manipulation. Torques include PD tracking,
gravity/bias compensation and inertia feedforward, clipped to URDF limits and a
0.76 Nm finger cap. There is no applied object wrench, no object attachment, no
object pose assignment after initialization, and no reset at phase boundaries.
Invalid MuJoCo states abort the run rather than making a stitched replay.

**Before** uses the contact-aware retargeter's saved commands. **After** adds the
generic contact-repair and force-guidance candidate below. This is not a CHORD
policy comparison and it is not the earlier book-specific regrasp optimizer.

## What correction actually does

1. Use the retargeted contact activity, blend, object-local anchors and normals.
   In free motion preserve the robot's nominal fingertip targets. In contact blend
   toward the human-derived surface anchors, with a 10 mm outward site offset and
   an 80 mm maximum nominal target change. The offset is a heuristic for the
   fingertip frame's location inside the distal link, not an exact pad model.
2. At every frame, perform up to 12 damped Jacobian least-squares updates for
   fingertip error, posture deviation and actual MuJoCo collision penetration.
   Bound commanded joints and inter-frame joint displacement. The first pose is
   untouched. These soft collision terms generate a candidate; they do not certify
   collision-free execution or constrain every continuous-time derivative.
3. Estimate object inertial wrench from filtered pose derivatives and assumed
   inertia. Allocate forces among six hand contacts and four approximate ground
   support points, with unilateral friction cones and force caps. Use 48-frame
   planning windows to bound solver cost. Dynamics remain continuous across all
   windows. Ten endpoint frames use an explicitly labeled quasi-static fallback
   because filtered derivatives are unreliable there. No endpoint is discarded.
4. Convert planned hand forces to joint feedforward with fingertip Jacobians.
   Run a complete physical replay. A solver returning `solved` does not mean that
   wrench residuals satisfy the stricter physical feasibility thresholds.
5. Try one bounded command adjustment based on the replay's object position error,
   only where the object remains within 50 mm of its reference. Keep it only if
   the combined tracking score improves without increasing maximum penetration
   beyond the existing error (or 1 mm). This is local refinement, not recovery
   planning after a failed grasp. Export the rejected candidate as well.

The resulting object is never pulled toward the reference by an external force.
The optional orange outline in the viewer is only the recorded target. The solid
mesh always follows the saved actual physics states.

## Output and viewing

Local generated files are under `correction/outputs/random_three_full_20261019/`:

- `selection.json`: all 30 eligible IDs, seed, pinned revision and selected metadata.
- `episode_XXXXXX/retargeted/contact_aware.npz`: complete retargeting interface.
- `reference.npz`: complete corrected commands, source timestamps, frame IDs,
  object targets, force guidance and validity mask.
- `baseline.npz`, `corrected.npz`, `wrench_seed.npz`, `candidate.npz`: actual physics
  states, torques, contact measurements and errors for the full episode.
- `report.json`, `scene.xml`, `collision/`: parameters, measured results and replay
  model. The model references local optional robot assets and cached convex pieces.

With the existing retargeting environment activated:

```text
python -m correction.scripts.random_full_episodes
python -m correction.scripts.view_full_episodes --port 8084
```

The commands consume the downloaded, decoded and retargeted files listed above;
they do not fetch the dataset automatically. Optional `--episode 18` limits the
batch to one already selected episode. V-HACD requires the optional `vhacdx`
package. Object decomposition is cached by mesh hash and decomposition settings.

The episode selector preserves complete timelines. Before/after, corrected-only
and before-only modes share the same scrubber. The camera target is set after
camera position (Viser's position setter also moves its target). Robot centering
and hand/object/whole-robot focus controls keep orbit and zoom usable near the
manipulation scene. Playback starts paused at frame zero.
On a narrow browser panel, click the **Connected** header to collapse the
controls while orbiting; click it again to select an episode or scrub time.
Use **Corrected only** for a larger close-up, and expand **Camera and overlays**
for hand/object focus. Camera fitting accounts for the canvas aspect ratio.

## Interpretation

The final common-floor-aligned runs give these **whole-episode** measurements:

| Object | RMS position error, before → after | RMS rotation error, before → after | Correction + three replays |
|---|---:|---:|---:|
| Trash can | 449 → 509 mm | 64.8 → 66.2° | 44.5 s |
| Desk | 634 → 706 mm | 59.9 → 58.9° | 44.6 s |
| Stroller | 825 → 698 mm | 106.5 → 121.8° | 72.1 s |

**All three fail the physical tracking gate.** Position tracking worsens on the
trash can and desk. Stroller position improves but its orientation worsens. No
bounded feedback refinement was accepted in these final runs. The external pelvis
fixture stays within 2.0, 3.3 and 0.95 mm of its respective commanded path, so robot
falling is no longer the source of these failures.

Force QPs return `solved` in approximately 3.4, 3.1 and 4.7 seconds, but **zero
frames pass the combined strict force/moment-residual and cone checks**. The
soft objective and coarse contact hypotheses cannot be interpreted as an exact
Newton–Euler certificate. The controller saturates, and actual joint limits
overshoot by as much as 0.091, 0.185 and 0.086 rad respectively. Maximum rollout
penetrations after correction are 1.24, 2.27 and 1.23 mm. These are additional
failure evidence, not quantities hidden by successful software checks.

The generic full-episode candidate has therefore **not validated the method's
effectiveness**. Contact acquisition, force-feasible placement, torque-limited
trajectory continuity and scene-model fidelity still need work before a policy
training comparison would be meaningful.

A visually plausible retargeted hand, a solved force QP and passing software
tests are not manipulation success. The tracking gate still checks actual object
position/orientation, penetration, initial overlap, absence of object assistance
and joint-limit error. The report always states `ready_for_training: false`.
The tests separately check complete frame counts, common initial conditions,
torque clipping, the pelvis-only weld, disabled leg contacts and free fall of an
uncontacted object while the supported robot moves.

This experiment can expose acquisition, contact geometry, torque and scene-model
failures. An improvement in position error alone does not demonstrate a better
manipulation when orientation or collisions worsen. Runtime excludes decoding and
retargeting; cached collision construction may also be excluded. It is not a
measurement of policy-training savings.
