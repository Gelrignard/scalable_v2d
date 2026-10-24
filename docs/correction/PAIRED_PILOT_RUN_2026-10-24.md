# Paired Tier 1/2 pilot run — 24 October 2026

User authorization: run six matched episodes in each tier for downstream policy
experiments. Selection was fixed before physical rollouts: episodes 3, 5, 11,
16, 18 and 25 (the original five-episode RL plan plus the bin). No replacements
are selected based on outcomes. No cloud resources, GPU training or extra viewer
servers are used.

Working outputs: `correction/outputs/rl_pilot_20261024_v1/` (Git-ignored).
Each tier/episode has decode/retarget logs, source hashes, preparation status,
and a `correction/` directory with source snapshots and stage/physics reports.
The preparation and correction orchestrators each use at most two worker
processes, with one BLAS/OpenMP thread per correction/retarget worker.

## Changes needed to run the requested pilot

- Tier-aware fetching and verified decoding of the actual synthetic-noise
  Parquets. No clean trajectory is loaded into a Tier 2 correction.
- Full-frame retention with visibility masks. Book episode 11 has one missing
  object pose per tier; interpolation does not relabel it as observed. All
  other selected inputs have visible object poses throughout.
- Input-only hand-mode segmentation for repeated pickups/handovers. Short or
  unconstructable phases retain original motion and are reported. Mode
  transitions sometimes need shortened acquisition/release windows; this is an
  experimental automatic extension, not the manually refined bin sequence.
- Single-hand surface hypotheses and palm targets. The old bimanual method
  remains unchanged for candidates involving both hands.
- Reuse of staged geometry, quasistatic force allocation, spline refinement and
  instantaneous observed-force feedback. Failed force solves use explicit zero
  force guidance; failed splines retain a labeled finite pre-spline candidate.
- Portable named-joint reference arrays, local object/support assets, hashes,
  guidance and measured outcome separation for downstream experiments.

The preserved successful-looking bin is not silently substituted into this
automatic batch. Its historical manual refinement and physics limitations
remain documented in the existing checkpoint/handoff.

## Run control and records

Preparation command:

```text
python -m correction.scripts.prepare_rl_pilot --root correction/outputs/rl_pilot_20261024_v1 --jobs 2
```

The first Tier 1 mug correction was launched separately to exercise integration.
Remaining runs use:

```text
python -m correction.scripts.run_rl_pilot_corrections --root correction/outputs/rl_pilot_20261024_v1 --jobs 2 --skip 1:3
```

The orchestrator refuses to restart an existing correction directory. Inspect
`prepare_report.json`, `correction_process.json`, active processes and outputs
before resuming; completed numerical experiments must not be overwritten.

For export after all runs terminate:

```text
python -m correction.scripts.export_rl_pilot --root correction/outputs/rl_pilot_20261024_v1 --output correction/outputs/rl_pilot_handoff_20261024
```

Twenty-one focused tests passed: missing-observation completion, phase segmentation,
single-hand target selection, existing observed-force constraints/controller,
planning provenance, and the named-joint/quaternion reference contract. Full physical replay determines outcome quality.

Initial finding: the first complete Tier 1 mug replay has object position RMS
650.06 mm for original retargeting and 692.37 mm for the diagnostic staged
candidate; planning and physical gates fail. This is not a successful correction.
The fixed roster continues so downstream selection cannot hide this failure.

## Interpretation and limits of this pilot

This automatic runner adds input-only hand-mode segmentation and single-hand
handling to the existing staged modules. It is not an exact replay of the
historically refined bin. It fixes a roster before observing outcomes, keeps
both successful and failed stages, and does not use per-episode manual grasp
labels. The lower body follows an externally supported recorded pelvis path;
this is not a walking or whole-body controller experiment.

Object mass (1 kg), local centre of mass (zero), diagonal inertia (0.01 kg m^2
per axis) and friction (1.0 in physics, 0.8 for planned hand forces) are prototype
assumptions shared by each original/corrected pair, not calibrated dataset
properties. No direct object forces or in-episode state resets are used.

A numerically solved force QP can still have an unacceptable wrench residual.
A finite geometry candidate can fail its static screen. When the final spline
QP fails, `corrected_reference.npz` contains the explicitly labeled pre-spline
candidate; it must not be described as an accepted optimized trajectory.
Observed-state feedback runs during physical replay and is not baked into the
saved joint-position reference. A position-only downstream importer discards
that controller behavior.

Our physical gate requires maximum position error below 10 mm, maximum rotation
error below 5 degrees, maximum penetration below 1 mm, initial penetration below
0.05 mm, joint-limit violation below 0.01 rad, no applied object assistance, and
requested torque within capacity. These are local prototype gates, not official
challenge evaluation. The full-episode RMS diagnostics below compare with the
same-tier input; Tier 2 errors do not measure agreement with clean ground truth.

The useful first downstream experiment is an explicit imperfect-reference
ablation with identical learner/action/task settings, keeping force guidance as
a separate ablation. These files alone do not establish faster learning or a
scalable successful correction method.


## Completed batch results

All 12 preparation/correction processes completed. There are 9,758 original frames (324.87 seconds) across twelve references and 24 complete physical rollouts. Each original/candidate pair starts from the same robot/object state. No case was replaced after failure.

**6/12 candidates improve object-position RMS; 6/12 improve rotation RMS. Zero pass planning or physical acceptance. All twelve spline QPs report `primal infeasible`, so all corrected references are `diagnostic_pre_spline_candidate`.** This is a negative generalization result for the automatic extension, not evidence that RL will need fewer samples.

| Tier | Episode | Frames | Original position RMS (mm) | Candidate position RMS (mm) | Original rotation RMS (deg) | Candidate rotation RMS (deg) | Corrected coverage | Inferred phases |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 781 | 650.1 | 692.4 | 82.9 | 109.8 | 665/781 | 2 |
| 1 | 5 | 1003 | 464.1 | 680.0 | 72.8 | 83.3 | 815/1003 | 8 |
| 1 | 11 | 901 | 1084.1 | 812.5 | 143.3 | 119.4 | 875/901 | 2 |
| 1 | 16 | 477 | 922.4 | 753.8 | 116.3 | 106.1 | 347/477 | 1 |
| 1 | 18 | 858 | 470.9 | 443.9 | 65.0 | 65.1 | 483/858 | 3 |
| 1 | 25 | 859 | 1281.1 | 1377.4 | 115.5 | 124.2 | 731/859 | 15 |
| 2 | 3 | 781 | 838.2 | 699.7 | 89.4 | 68.4 | 662/781 | 2 |
| 2 | 5 | 1003 | 340.0 | 526.6 | 72.0 | 147.3 | 719/1003 | 11 |
| 2 | 11 | 901 | 1132.2 | 798.1 | 137.9 | 114.5 | 864/901 | 7 |
| 2 | 16 | 477 | 921.8 | 756.2 | 115.3 | 99.4 | 312/477 | 4 |
| 2 | 18 | 858 | 310.2 | 348.4 | 76.3 | 76.5 | 242/858 | 2 |
| 2 | 25 | 859 | 631.1 | 673.4 | 119.1 | 118.6 | 728/859 | 14 |

Coverage denotes samples changed by a constructed candidate, not successful physical contact. Force guidance is nonzero in every episode, but no phase passes the nominal task-wrench residual gate. Satisfying the QP inequalities does not mean the desired wrench was produced.

Automatic hand-mode segmentation produces many short phases (15 in the Tier 1 water-bottle case and 14 in Tier 2). The box and water-bottle candidates worsen position RMS in both tiers. Large tracking errors and actuator saturation remain; these are material failures, not merely stringent acceptance thresholds. The run retains the intermediate geometry/force candidates to make those failures inspectable.

The same-tier input audit found exactly one unobserved endpoint in each book input (frame 900). It is held from the last valid observation, with its observation mask false. All 12 timelines retain every source frame. The six noisy-source hashes differ from their clean counterparts. Ninety-six stored source-file hashes match the implementation used for this batch.

### Compute and verification

- Local CPU only, at most two workers, no GPU hours or cloud charges.
- Sum of preparation worker wall times: 37.65 minutes. Sum of correction plus physical replay worker wall times: 58.53 minutes; summed correction process CPU time: 58.55 CPU-minutes. These sums are not elapsed calendar time.
- Span from the first correction start to the last finish: 29.79 minutes, including overlap and orchestration gaps. Per-case correction/replay costs: 1.83 to 9.21 minutes.
- Largest measured correction process peak resident memory: 1.34 GiB; excludes other processes and preparation workers.
- 21 focused unit tests pass, compilation checks pass, all full-episode arrays are finite, and no object assistance or reset warnings were found. Unit tests do not establish manipulation success.
- All batch experiment processes have terminated; no viewer was added.

### Handoff locations

- Working run: `correction/outputs/rl_pilot_20261024_v1/`.
- Portable bundle: `correction/outputs/rl_pilot_handoff_20261024/`; start with `manifest.json`, `README.md`, and `batch_summary.json`.
- Archive: `correction/outputs/rl_pilot_handoff_20261024.zip`, with sibling `.zip.sha256`.
- [Artifact contract](RL_PILOT_ARTIFACT_FORMAT.md) and [method handoff](RL_HANDOFF_2026-10-24.md).

The portable export passed validation for all 12 entries: file hashes, complete
frame order, finite arrays, quaternion conventions, force/patch shapes, paired
object targets, identical physical initial conditions, and zero applied object
assistance. The six source pairs have distinct clean/noisy input hashes. This
integrity result does not change the physical rejection labels.

The decoder logs contain trimesh texture-processing `All-NaN slice` warnings.
The exported reference and physics arrays are finite; these texture warnings
must not be confused with a successful physics check or suppressed from the
record. No physics reset/instability warning or traceback appeared in the
completed correction logs.

The archive SHA-256 is in its `.zip.sha256` sidecar; archive byte count and ZIP
CRC verification are recorded in `rl_pilot_handoff_20261024.packaging.json`.
Generated data and archives remain Git-ignored; source, tests and documentation
are committed locally on main. No push was performed in this task.
