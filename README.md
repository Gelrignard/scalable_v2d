# Scalable V2D robot grounding

Research prototype for the [NVIDIA Video to Data Challenge, Track 2](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/): use human and object motion to construct a contact-aware robot reference, then correct it using object inverse dynamics and trajectory optimization before policy training.

The research question is whether this preprocessing reduces the GPU time and environment steps needed to learn useful manipulation policies. That benefit has **not yet been measured**.

## Start here

- [Method and mathematical derivation](docs/correction/METHOD.md)
- [CPU setup and reproduction](docs/SETUP.md)
- [Compute, challenge requirements, and implementation handoff](docs/COMPUTE_AND_IMPLEMENTATION_HANDOFF.md)
- [Parallel branches and agent ownership](docs/PARALLEL_WORK.md)
- [Tier 2 RL experiment plan](training/README.md) and [five-episode manifest](experiments/tier2_pilot.json)
- [Tier 3 reconstruction plan](reconstruction/README.md)

## Current implementation

| Component | Status |
|---|---|
| [Contact-aware retargeting](retargeting/README.md) | CPU body/hand IK, explicit contact targets, canonical input, diagnostics, viewers, and reference export |
| [Dynamics-guided correction](correction/README.md) | Object inverse dynamics, contact-force optimization, local robot trajectory repair, and independent checks of the resulting geometry |
| Tier 1 demonstration | One short book phase exercised end to end; its physical audit still reports failures |
| Tier 2 policy comparison | Planned; general episode/noisy-input support and the trainer remain to be implemented |
| Tier 3 reconstruction | Planned; no integrated video-to-motion reconstruction pipeline yet |

Kinematic playback is not physical simulation. The current corrected book reference is **not accepted as a dynamically feasible manipulation**, and no trained policy or leaderboard result is included.

## Repository layout

```text
retargeting/       Human-motion adapters, robot IK, export, viewer, and tests
correction/        Stage implementations in src/, runnable scripts/, and tests/
training/          Tier 2 policy-training implementation assignment
reconstruction/    Tier 3 reconstruction implementation assignment
experiments/       Experiment definitions and proposed episode roster
docs/              Method, interfaces, experiment notes, setup, and third-party notices
scripts/           Dependency bootstrap and selected asset downloads
patches/           Reproducible changes to the pinned NVIDIA CPU dependency
upstream.lock.json Source revisions, compatibility-patch hash, and asset provenance
```

Direct dataset/model utilities live at the repository root: `fetch_sample.py`,
`fetch_soma_assets.py`, `decode_tier1.py`, and `tier1_diagnostic.py`.
The correction layout and updated commands are described in
[correction/README.md](correction/README.md).

Datasets, model weights, Python environments, checkpoints, generated outputs, and the NVIDIA checkout are excluded from Git. Follow [SETUP.md](docs/SETUP.md) to obtain dependencies and generate local artifacts. Existing Windows users can continue using `retarget.ps1`; portable Python commands are documented in setup.

The two development branches are `work/tier2-rl` and `work/tier3-reconstruction`. Use a separate worktree for each agent. Keep the same downstream learner and compute budget across the NVIDIA body baseline, our retargeting, and our retargeting plus correction.

See [third-party notices](docs/THIRD_PARTY_NOTICES.md) before redistributing upstream source or assets.
