# Documentation

| Document | Purpose |
|---|---|
| [Setup](SETUP.md) | Reproduce the CPU environment and example |
| [Optional NVIDIA dependency and patch](NVIDIA_COMPATIBILITY.md) | Tested source version, compatibility edits, and manual installation |
| [SOMA setup](SOMA_SETUP.md) | Body-model assets, dependencies, and decoding commands |
| [Correction architecture](correction/ARCHITECTURE.md) | Stage boundaries, file map, and command migration |
| [Method](correction/METHOD.md) | Mathematical derivation and research claims |
| [Retargeting interface](correction/INTEGRATION.md) | Geometric input bundle, units, frames, and output contract |
| [Experiment notes](correction/EXPERIMENTS.md) | Book trial, synthetic example, measured results, and limitations |
| [Compute and implementation handoff](COMPUTE_AND_IMPLEMENTATION_HANDOFF.md) | Challenge requirements and remaining training/reconstruction work |
| [Parallel development](PARALLEL_WORK.md) | Branches, ownership, and comparison design |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Dependency terms and original upstream license |

Keep explanatory Markdown and text documents here. Component READMEs stay beside
their code. Dependency manifests such as `requirements*.txt` remain with the
environment or package they configure; those are executable setup inputs.
