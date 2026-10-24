# Tier 2 RL experiment workstream

Branch: work/tier2-rl. This directory is a handoff scaffold, not a trainer.

Start with the [current method and RL handoff](../docs/correction/RL_HANDOFF_2026-10-24.md).
It specifies the staged optimization, optional observed-state feedback, exact
artifact meanings, failed physical checks, and proposed controlled RL comparison.
Also read the compute handoff, docs/SETUP.md, docs/PARALLEL_WORK.md, and
experiments/tier2_pilot.json before implementation. Earlier coupled/MPC documents
are historical experiments, not the current training contract.

The [paired six-episode pilot](../docs/correction/PAIRED_PILOT_RUN_2026-10-24.md)
extends the original roster with episode 18 and supplies Tier 1/2 reference
bundles. Its manifest and physical rejection labels must accompany training.

## Deliverables

1. Prepare a separate FlashCHORD runtime compatible with the current submission
   kit. Pin its revision and container; do not replace the CPU upstream checkout.
2. Use the explicit `--tier 2 --all-frames` decoder and paired-pilot manifests;
   preserve observation masks, timestamps and source-to-world transforms.
3. Import our robot references into motion_v1 with correct joint mapping,
   object physics, supports, and SONIC/controller semantics.
4. Implement baseline/our-retargeting/corrected-reference runs under identical
   learner settings. Profile 24 GB memory fit before longer training.
5. Evaluate unassisted learned-policy rollouts and export the exact toolkit
   format. Track environment steps, updates, wall time, VRAM, and success curves.
6. Extend to all six matched pilot episodes and paired seeds after the first works.

## Existing gaps that must not be hidden

- Fetch/decode now support explicit Tier 2 provenance. The full-frame decoder
  interpolates missing object observations with masks; it is not a validated
  general denoiser or a substitute for uncertainty handling in learning.
- Our canonical-input Parquet export needs scene metadata integration; the
  default cached example export does not establish arbitrary-episode support.
- correction/scripts/correct_rl_pilot_episode.py adds automatic hand-mode
  phases, with explicit failed/unchanged segments. This is experimental;
  the preserved bin still has a separately refined two-phase sequence.
- Pilot NPZs have named-joint exports and manifests, but need a learner/action
  adapter and official-runtime validation. Planned, commanded and measured
  forces must remain separate. The
  preserved bin is useful for an imperfect-reference experiment but does not
  pass physical acceptance; desk/stroller transfers fail upstream planning.
- Sticking, grasp/release, environmental support and full physical execution
  remain unresolved. Record unsupported phases/failures instead of claiming
  whole-episode dynamic feasibility.

If correction generalization is blocked, the runtime, raw/retargeted baselines,
exporter, and resource profiler are still useful independent work. Keep the
corrected-reference interface explicit so the core-method agent can supply it.

The only clean Tier 1 access in Tier 2 experiments should be the official local
evaluation/packing path. Do not use clean references or same-episode Tier 1
policies for Tier 2 reconstruction, correction, or initialization.

The offered server is $0.32/h, 24 GB VRAM, 16 CPU cores, 120 GB RAM, and only
30 GB disk allowance. Resolve storage and model/runtime access before paid runs.
No claim that six GPU-hours covers all training has been established.
