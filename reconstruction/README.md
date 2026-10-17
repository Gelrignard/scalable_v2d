# Tier 3 reconstruction workstream

Branch: work/tier3-reconstruction. This is an implementation handoff scaffold.

Read docs/correction/METHOD.md, docs/COMPUTE_AND_IMPLEMENTATION_HANDOFF.md,
docs/SETUP.md, and the canonical HumanMotion contract in retargeting/README.md.

## Deliverables

1. Choose and pin one reconstruction implementation. Initially prefer V2D's
   CARI4D MHR integration; do not mix it with the original SMPL-based repository.
2. Resolve model access, the permitted Tier 3 mesh-input condition, and how to
   map reconstruction coordinates into the released evaluation world.
3. Generate human/object masks and reconstructed human/body/hand and object
   motion. Run pretrained inference, not foundation-model training.
4. Preserve episode/frame identity, timestamps, units, object-body frame, model
   versions, confidence/visibility, and all coordinate transforms.
5. Convert to the retargeter's documented canonical NPZ. MHR parameter output
   must be explicitly decoded/mapped; it is not interchangeable by filename
   with the SOMA-X dataset fields.
6. Prepare the custom Tier 3 frame reference expected by the submission packer.
7. Measure stage time, peak memory, host RAM, storage, and reconstruction failures
   on one full clip before expanding to all 30.

Cache reconstruction once per episode and reuse it across downstream experiments.
Keep models, videos, masks, meshes, and generated arrays out of Git.

## Open conditions

- Tier 3 motion is reconstructed from the same videos as the released tiers.
  It must not use clean Tier 1 trajectories as targets.
- CARI4D's current V2D command consumes a mesh. Whether the clean released mesh
  is permitted for Tier 3 initialization needs an explicit rule interpretation.
- The evaluation world transform needs a permitted calibration route; fitting
  reconstructed trajectories to the clean target would hide input errors.
- A full-clip CARI4D memory profile on 4090 is not yet available. Start by
  controlling per-stage batches; windowed refinement can change quality.
- The quoted server has 30 GB disk allowance. Establish persistent storage and
  checkpoint access before installing the complete stack.

Use the accepted canonical interface to keep this work independent of RL.
Do not modify retargeting objectives or the contact-wrench optimizer merely to
make a reconstruction adapter pass.
