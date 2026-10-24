# Scalable V2D robot grounding

Research prototype for the [NVIDIA Video to Data Challenge, Track 2](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/): use human and object observations to construct a contact-aware robot reference, then refine contact geometry, force guidance and robot motion in staged optimization before policy training.

The research question is whether this preprocessing reduces the GPU time and environment steps needed to learn useful manipulation policies. That benefit has **not yet been measured**.

## Start here

- [Current method and downstream RL handoff: equations, artifacts, results, SuperMimic and batch readiness (24 October)](docs/correction/RL_HANDOFF_2026-10-24.md)
- [Paired six-episode Tier 1/2 pilot: full timelines, automatic stages and execution records](docs/correction/PAIRED_PILOT_RUN_2026-10-24.md)
- [Current non-MPC observed-force adapter](docs/correction/STAGED_OBSERVATION_FEEDBACK_2026-10-20.md) and [three-episode findings](docs/correction/STAGED_FEEDBACK_TRANSFER_2026-10-20.md)
- [Earlier coupled correction experiment: uncertain observations and validation status](docs/correction/COUPLED_HORIZON.md)
- [Earlier method and mathematical derivation](docs/correction/METHOD.md)
- [G1/Dex3 supported physics: robot integration, errors, and SONIC's role](docs/correction/SUPPORTED_G1.md)
- [Three random full episodes: supported pelvis, before/after physics and failed tracking results](docs/correction/FULL_EPISODE_COMPARISON.md)
- [Supported EEF demo: measured contact dynamics and scope](docs/correction/SUPPORTED_MANIPULATION.md)
- [Bounded correction: implementation, runtime and remaining failures](docs/correction/BOUNDED_CORRECTION.md)
- [CPU setup and reproduction](docs/SETUP.md)
- [Compute, challenge requirements, and implementation handoff](docs/COMPUTE_AND_IMPLEMENTATION_HANDOFF.md)
- [Parallel branches and agent ownership](docs/PARALLEL_WORK.md)
- [Tier 2 RL experiment plan](training/README.md) and [five-episode manifest](experiments/tier2_pilot.json)
- [Tier 3 reconstruction plan](reconstruction/README.md)

## Current implementation

| Component | Status |
|---|---|
| [Contact-aware retargeting](retargeting/README.md) | CPU body/hand IK, explicit contact targets, canonical input, diagnostics, viewers, and reference export |
| [Dynamics-guided correction](correction/README.md) | Current method uses staged surface fitting, quasistatic force allocation and robot spline refinement; optional observed-state force QP during replay |
| Current G1 validation | Preserved bin improves position RMS from 444.99 to 48.48 mm but fails physical acceptance. Desk/stroller transfers fail; observation adapter is not promoted. No accepted full G1 episode in this comparison |
| Earlier experiments | Coupled-horizon, MPC and independent-pad results remain separate historical diagnostics |
| Tier 1/2 policy comparison | Paired six-episode data preparation and automatic staged correction implemented; reference quality remains experimental and the trainer remains to be implemented |
| Tier 3 reconstruction | Planned; no integrated video-to-motion reconstruction pipeline yet |

Kinematic playback is not physical simulation. Current G1 comparisons simulate arm/hand/waist and object dynamics with torque limits while an external fixture follows the recorded pelvis path. Walking and balance are excluded. These are experimental learning references, not certified expert trajectories or official whole-body rollouts. No trained policy or leaderboard result is included. Read the [current handoff](docs/correction/RL_HANDOFF_2026-10-24.md) before using historical method documents or exporting to training.

## Optional NVIDIA dependency

[NVIDIA Video to Data](https://github.com/nvidia-isaac/video_to_data) is an
optional external dependency. Its source is not included in this repository or
registered as a Git submodule. A locally downloaded `upstream/` directory is
excluded by `.gitignore`.

- **Core dynamics/contact optimization:** no NVIDIA checkout is needed. Install
  `correction/requirements.txt`, then run `python -m correction.scripts.demo` or
  the standalone mechanics/optimization tests in [SETUP.md](docs/SETUP.md).
- **G1 retargeting and the book robot trial:** the current adapters require
  NVIDIA's robot code and assets. Follow the documented
  [optional dependency and compatibility patch setup](docs/NVIDIA_COMPATIBILITY.md),
  then run `python -m data_tools.fetch_preview_assets`.

The tested external version and compatibility edits are described in that guide.
Cloning this repository does not download NVIDIA's source, datasets, or weights.

## Repository layout

```text
data_tools/        Dataset downloads, SOMA setup/decoding, assets, and diagnostics
retargeting/       Human-motion adapters, robot IK, export, viewer, and tests
correction/        Stage implementations in src/, runnable scripts/, and tests/
training/          Tier 2 policy-training implementation assignment
reconstruction/    Tier 3 reconstruction implementation assignment
experiments/       Experiment definitions and proposed episode roster
setup/             CPU environment recipe, Windows package lock, and pip packages
docs/              Method, interfaces, experiment notes, setup, and third-party notices
```

See [dataset and SOMA tools](data_tools/README.md), [SOMA setup](docs/SOMA_SETUP.md), and
[correction](correction/README.md) for their commands and file responsibilities.

The optional local `upstream/` folder holds NVIDIA's original project. The
[compatibility guide](docs/NVIDIA_COMPATIBILITY.md) explains the edits and links
the patch, which is stored under `docs/`.

Datasets, model weights, Python environments, checkpoints, generated outputs, and the NVIDIA checkout are excluded from Git. Follow [SETUP.md](docs/SETUP.md) to obtain dependencies and generate local artifacts. Public commands use Python directly; machine-specific convenience launchers are not included.

The two development branches are `work/tier2-rl` and `work/tier3-reconstruction`. Use a separate worktree for each agent. Keep the same downstream learner and compute budget across the NVIDIA body baseline, our retargeting, and our retargeting plus correction.

See [third-party notices](docs/THIRD_PARTY_NOTICES.md) before redistributing upstream source or assets.
- [Preserved 20 October result and overnight report](docs/correction/OVERNIGHT_REPORT_2026-10-20.md)
- [Correction to whole-body control: milestone 7.5](docs/correction/WHOLE_BODY_HANDOFF.md)
- [Recover the saved bin comparison](correction/examples/bin_checkpoint_20261020/README.md)
- [Observed-state contact recovery: implementation, equations and limitations](docs/correction/OBSERVED_CONTACT_RECOVERY.md)
- [Current MPC diagnosis, frozen-state checks and proposed next method](docs/correction/MPC_DIAGNOSIS_AND_NEXT_METHOD_2026-10-20.md)
- [Preserve the nominal correction: bounded observed-state feedback](docs/correction/NOMINAL_RESIDUAL_FEEDBACK_2026-10-20.md)
- [Predictive acceptance of staged feedback: implementation and validation](docs/correction/GUARDED_RESIDUAL_FEEDBACK_2026-10-20.md)
- [Staged correction with observation feedback, without MPC](docs/correction/STAGED_OBSERVATION_FEEDBACK_2026-10-20.md)
- [Additional desk and stroller full-episode feedback tests](docs/correction/STAGED_FEEDBACK_TRANSFER_2026-10-20.md)
