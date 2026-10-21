# Additional complete episodes with staged observation feedback

## Scope and selection

The user requested more episodes after the blue-bin non-MPC trial. Add the
two other locally prepared episodes from the earlier recorded random selection
(seed 20261019): episode 10, desk lowering; episode 23, stroller back-and-forth.
This is not a newly drawn random sample or a selection by new rollout outcome.

The exact same `StagedObservationConfig(gain=1)` is used, with no per-episode
gain tuning. The actual pose/twist, contact and current-Jacobian force update is
unchanged from the final bin implementation. The only reusable implementation
change resolves each episode's actual planning reference without assuming the
bin's special two-grasp planning translation.

The transfer pipeline plans directly against its recorded object poses. The
loader recognizes that saved provenance and checks timestamps and poses against
the source retargeting output before using them. Unknown saved formats fail
explicitly. The bin still requires its verified planning-sequence ancestry.
No previous simulated object outcome is used as a reference.

## Important inherited failures

These are tests of feedback around **rejected staged plans**, not successful
nominal trajectories like the useful bin reference:

- Desk: saved contact/force and spline checks failed. Replay retains the finite
  pre-spline diagnostic, with its original limitations.
- Stroller: force-profile allocation failed and used an explicitly recorded
  zero-guidance fallback; spline checks also failed. Wheels are part of a single
  rigid object, not articulated rolling bodies.
- Both use the inherited prototype mass of 1 kg and diagonal inertia of
  0.01 kg m2, not validated official object parameters. Floor registration and
  convex collision approximation also retain their known limitations.

The original retargeting and saved staged outcomes are reused for comparison.
The new feedback run must reproduce their common pre-contact prefix exactly,
complete the full episode, and apply no object assistance or within-episode
resets. Original, planning/intended and actual simulated motion stay distinct.
The pelvis follows the externally supported recorded path; upper body and
object dynamics are enabled. Walking and balance are excluded.

The saved source planning rejection propagates to the new report: even a
numerically successful feedback solve cannot promote an invalid upstream plan.
No RL/training-readiness or scalability claim follows from these examples.

## Reproduction

```powershell
python -m correction.scripts.replay_staged_observation --source correction/outputs/staged_model_20261020/desk_v1 --output correction/outputs/new_desk_feedback --gain 1
python -m correction.scripts.replay_staged_observation --source correction/outputs/staged_model_20261020/stroller_v1 --output correction/outputs/new_stroller_feedback --gain 1
```

Use the existing robot environment and fresh output directories. These two
single-interval sources need no `--planning-sequence`. Source files, input
hashes, physical metrics, command-bound failures and compute are recorded in
each report. Output arrays and temporary analysis scripts remain ignored.

The focused test run passes 13 tests: three planning-contract tests and ten
existing observed-force/controller tests. Full physical results, not these
test passes, determine the interpretation below.

## Completed results

Both uninterrupted runs completed, with zero direct object assistance and
exact pre-feedback agreement with the saved staged physical motion (137 desk
frames and 97 stroller frames). Neither is accepted or promoted.

| Episode | Frames / duration | Original retargeting RMS | Saved staged RMS | Staged + observations RMS |
|---|---|---|---|---|
| 10, desk lowering | 794 / 26.43 s | 704.28 mm / 69.00 deg | 1162.99 mm / 162.28 deg | 1162.99 mm / 162.28 deg |
| 23, stroller back-and-forth | 946 / 31.50 s | 841.86 mm / 102.79 deg | 811.99 mm / 113.40 deg | 816.27 mm / 112.25 deg |

Desk: none of the 16,900 force updates finds a matched active patch. The
force increment stays exactly zero, and the entire robot/object motion and
torque arrays match the saved staged failure exactly. There are no rate
conflicts because no correction is applied. This is a contact-acquisition
failure; successful zero-force QPs do not make the manipulation successful.
Peak requested torque is 160.78 times capacity, with actual torque clipped.

Stroller: 21,675 of 26,967 force updates have no matched active patch. At video
rate, 161 scheduled-contact frames have a match and 649 do not. Increments
reach 0.404 N per axis. Sixty-three rate-constrained solves require fallback;
the first 20 logged events report primal infeasibility. The maximum measured
increment rate is 185.26 N/s versus the declared 10 N/s bound. Peak requested
torque is 21.96 times capacity. The slight orientation improvement accompanies
worse position error and does not establish a successful manipulation.

On every sampled unmatched active frame in both episodes, allocated additional
object wrench is exactly zero. Contacts on other patches can still exist;
absence of a selected match is not a claim of no robot/object contact anywhere.

The additional runs strengthen the current diagnosis: observed force feedback
cannot repair upstream contact/trajectory failures on its own. A subsequent
bounded geometry/contact reacquisition stage remains necessary to investigate.
These models and rejected starting references do not support a general claim
about benchmark performance or the best achievable method.

## Compute and artifacts

The two independent CPU replays ran concurrently with one BLAS/OpenMP thread
each; this is not a controlled timing benchmark. No new GPU resources or
viewer ports were used.

| Run | Wall time | CPU time | Force-QP time | Peak process RAM |
|---|---:|---:|---:|---:|
| Desk | 307.72 s | 264.80 s | 103.55 s | 1.127 GB |
| Stroller | 470.72 s | 409.98 s | 200.17 s | 0.944 GB |

Desk's zero-authority solves reveal avoidable overhead: future code can skip a
force QP when no patch is matched and the previous force increment is zero.
That optimization was not introduced during this fixed-code comparison.

- [Desk full replay report](../../correction/outputs/staged_observation_20261020/desk_v1/report.json)
- [Stroller full replay report](../../correction/outputs/staged_observation_20261020/stroller_v1/report.json)
- [Detailed cross-episode comparison](../../correction/outputs/staged_observation_20261020/transfer_comparison.json)
- [Existing experimental viewer](http://127.0.0.1:8088/): choose desk, stroller,
  or bin in **Episode**. Original retargeting is left; staged feedback is right;
  the orange ghost/path is intended object motion. All are full episodes.

The preserved viewer on port 8087 remains unchanged. Its independent pad
channel is not relabeled as a pad simulation of these new feedback controllers.
