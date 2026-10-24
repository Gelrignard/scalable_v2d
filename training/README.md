# Tier 2 RL experiment workstream

Branch: work/tier2-rl. This directory is a handoff scaffold, not a trainer.

Start with the [current method and RL handoff](../docs/correction/RL_HANDOFF_2026-10-24.md).
It specifies the staged optimization, optional observed-state feedback, exact
artifact meanings, failed physical checks, and proposed controlled RL comparison.
Also read the compute handoff, docs/SETUP.md, docs/PARALLEL_WORK.md, and
experiments/tier2_pilot.json before implementation. Earlier coupled/MPC documents
are historical experiments, not the current training contract.

## Deliverables

1. Prepare a separate FlashCHORD runtime compatible with the current submission
   kit. Pin its revision and container; do not replace the CPU upstream checkout.
2. Add an explicit Tier 2 download/decode adapter with correct source provenance,
   masks, timestamps, missing-observation treatment, and saved world transforms.
3. Import our robot references into motion_v1 with correct joint mapping,
   object physics, supports, and SONIC/controller semantics.
4. Implement baseline/our-retargeting/corrected-reference runs under identical
   learner settings. Profile 24 GB memory fit before longer training.
5. Evaluate unassisted learned-policy rollouts and export the exact toolkit
   format. Track environment steps, updates, wall time, VRAM, and success curves.
6. Extend to the five specified episodes and paired seeds after the first works.

## Existing gaps that must not be hidden

- data_tools/fetch_sample.py and data_tools/decode_tier1.py hardcode Tier 1 provenance.
  Passing Tier 2 files without adapting this code is not an acceptable solution.
- The body decoder currently requires one contiguous visible object segment.
  Tier 2 dropouts need explicit handling; do not silently omit scored frames.
- Our canonical-input Parquet export needs scene metadata integration; the
  default cached example export does not establish arbitrary-episode support.
- correction/scripts/transfer_episode.py supports one contact interval; the
  preserved bin has a separately constructed two-phase sequence. Neither is
  an automatic general corrector for all five episodes. The older book script
  is not the current method.
- Current staged NPZs are diagnostic artifacts, not complete portable training
  bundles. Planned, commanded and measured forces must remain separate. The
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
