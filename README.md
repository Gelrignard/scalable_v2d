# Scalable V2D robot grounding

Research prototype for the [NVIDIA Video to Data Challenge, Track 2](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/): use human and object motion to construct a contact-aware robot reference, then correct it using object inverse dynamics and trajectory optimization before policy training.

The research question is whether this preprocessing reduces the GPU time and environment steps needed to learn useful manipulation policies. That benefit has **not yet been measured**.

## Start here

- [Method and mathematical derivation](docs/correction/METHOD.md)
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
| [Dynamics-guided correction](correction/README.md) | Object inverse dynamics, sparse force allocation, bounded contact hypotheses, temporal robot refinement, collision checks, and diagnostic CPU rollouts |
| Tier 1 demonstration | Supported G1/Dex3 carries a free book with fixed pelvis/legs; correction improves tracking but misses the 10 mm / 5 degree gate. Six-pad experiment passes its separate reduced-model checks |
| Tier 2 policy comparison | Planned; general episode/noisy-input support and the trainer remain to be implemented |
| Tier 3 reconstruction | Planned; no integrated video-to-motion reconstruction pipeline yet |

Kinematic playback is not physical simulation. The [supported G1 experiment](docs/correction/SUPPORTED_G1.md) uses actual arm/hand/waist dynamics and torque-limited control with a fixed pelvis and legs. It demonstrates book carrying, but remaining tracking errors, torque saturation and contact changes prevent certification as a training reference. The [six-pad experiment](docs/correction/SUPPORTED_MANIPULATION.md) passes its separate reduced-model checks. No trained policy, unsupported whole-body execution certificate, or leaderboard result is included.

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
