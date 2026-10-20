# Preserved bin comparison — 20 October 2026

This archive preserves the derived command and complete physical recordings
shown in the morning comparison. It contains no upstream repository, robot
asset, object mesh, raw video or model weights.

The 858-frame episode is Tier 1 episode 18, blue trash can. The original
retargeting has 444.99 mm object position RMS; the retained mechanics candidate
has 48.48 mm RMS. **Both fail physical acceptance.** This is a supported-pelvis
manipulation benchmark, not official whole-body execution or a trained policy.

`manifest.json` records the archive and individual file SHA256 hashes. The
archive includes `rest_command_v2`, `rest_matched_v1` and `rest_support_v1`,
including the original and corrected physical trajectories. They are complete
episodes with separate intended object motion and no direct object assistance.

To restore on a machine with the optional retargeting runtime and authorized
dataset assets, extract `saved_comparison.zip` into a fresh output directory.
Existing local outputs should not be overwritten. Run:

```text
python -m correction.scripts.view_controller_comparison --directory <extracted>/rest_matched_v1 --support-directory <extracted>/rest_support_v1 --port 8087
```

The recordings preserve their original paths for provenance. On another
machine, `reference.npz`'s `object_mesh` and report input-directory paths must
be resolved to that machine's downloaded assets before viewing or recomputing.
Do this in a working copy; retain the archive unchanged. The robot/object
assets remain optional external dependencies described in the main README.

See the [morning report](../../../docs/correction/OVERNIGHT_REPORT_2026-10-20.md)
for limitations and the [mathematical implementation](../../../docs/correction/SEQUENCE_CORRECTION.md)
for the staged method. The repository commit containing this archive preserves
the code version before the subsequent multi-episode work.
