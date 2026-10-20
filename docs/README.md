# Documentation

| Document | Purpose |
|---|---|
| [Setup](SETUP.md) | Reproduce the CPU environment and example |
| [Optional NVIDIA dependency and patch](NVIDIA_COMPATIBILITY.md) | Tested source version, compatibility edits, and manual installation |
| [SOMA setup](SOMA_SETUP.md) | Body-model assets, dependencies, and decoding commands |
| [Correction architecture](correction/ARCHITECTURE.md) | Stage boundaries, file map, and command migration |
| [Method](correction/METHOD.md) | Mathematical derivation and research claims |
| [Research review, 19 October](correction/RESEARCH_REVIEW_2026-10-19.md) | Current challenge requirements, implementation audit, revised milestones, and delta-action analysis |
| [Overnight implementation report, 20 October](correction/OVERNIGHT_REPORT_2026-10-20.md) | Final morning physical results, costs, failures and remaining work |
| [Whole-body controller handoff, M7.5](correction/WHOLE_BODY_HANDOFF.md) | What changed overnight, existing robot dynamics, SONIC interface and pre-RL compatibility checks |
| [Additional full episodes, 20 October](correction/TRANSFER_EPISODES_2026-10-20.md) | Desk/stroller transfer failures, full comparisons, costs and one-server viewer catalog |
| [Overnight implementation log](correction/OVERNIGHT_LOG_2026-10-20.md) | Detailed experiment history and continuation instructions |
| [Contact sequences and command dynamics](correction/SEQUENCE_CORRECTION.md) | Phase/controller contract, regrasping, continuous transitions and the bounded command QP |
| [Literature review](LITERATURE_REVIEW.md) | Research positioning, novelty risks, proposed directions, and evaluation design |
| [Annotated literature](LITERATURE_REFERENCES.md) | 46 primary-source papers on contact correction and policy learning |
| [Retargeting interface](correction/INTEGRATION.md) | Geometric input bundle, units, frames, and output contract |
| [Experiment notes](correction/EXPERIMENTS.md) | Book trial, synthetic example, measured results, and limitations |
| [Compute and implementation handoff](COMPUTE_AND_IMPLEMENTATION_HANDOFF.md) | Challenge requirements and remaining training/reconstruction work |
| [Parallel development](PARALLEL_WORK.md) | Branches, ownership, and comparison design |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Dependency terms and original upstream license |

Keep explanatory Markdown and text documents here. Component READMEs stay beside
their code. Dependency manifests such as `requirements*.txt` remain with the
environment or package they configure; those are executable setup inputs.
