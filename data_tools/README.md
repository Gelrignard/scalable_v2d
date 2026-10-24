# Dataset and SOMA utilities

| File | Purpose |
|---|---|
| `fetch_sample.py` | Fetch one pinned Tier 1 or Tier 2 episode, object mesh, and scene metadata; verify cached hashes |
| `fetch_preview_assets.py` | Fetch selected NVIDIA robot/example LFS assets and verify their hashes |
| `tier1_diagnostic.py` | Read a raw episode and report object-wrench demand under explicit physical assumptions |
| `fetch_soma_assets.py` | Download and verify the SOMA-X/MHR model assets |
| `decode_tier1.py` | Decode SOMA parameters; explicit `--tier 2` provenance and optional `--all-frames` completion |
| `observation_completion.py` | Fill missing object poses from same-tier observations while retaining validity masks |
| `requirements-soma.txt` | Packages for the separate SOMA environment |

Run from the repository root:

```text
python -m data_tools.fetch_sample --episode 11
python -m data_tools.fetch_sample --tier 2 --episode 11
python -m data_tools.fetch_preview_assets
python -m data_tools.tier1_diagnostic --episode 11 --mass-kg 1
python -m data_tools.fetch_soma_assets
```

The first two commands use Python's standard library. The diagnostic uses the
[CPU correction environment](../docs/SETUP.md). Its mass is an experimental
assumption, not a dataset measurement. Prepare the optional NVIDIA checkout
using [the compatibility guide](../docs/NVIDIA_COMPATIBILITY.md) before fetching
its preview assets.

The existing Tier 1 cache remains at `correction/data/tier1/` so saved references
to its meshes and scene files stay valid. Downloads are excluded from Git.
SOMA utilities live in this same folder. Follow [SOMA setup](../docs/SOMA_SETUP.md)
for its separate environment, then run `python -m data_tools.decode_tier1` there.

For a full Tier 2 timeline, use `--tier 2 --data correction/data/tier2 --all-frames`.
The decoder verifies the download manifest and noised-parquet metadata. Missing
object positions are linearly interpolated and orientations use SLERP, with
nearest-pose holds at missing endpoints. Imputed frames remain explicitly marked
as unobserved; clean Tier 1 trajectories are never read for this operation.
