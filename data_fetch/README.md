# Dataset utilities

| File | Purpose |
|---|---|
| `fetch_sample.py` | Fetch one pinned Tier 1 episode, object mesh, and scene metadata; verify cached hashes |
| `fetch_preview_assets.py` | Fetch selected NVIDIA robot/example LFS assets and verify their hashes |
| `tier1_diagnostic.py` | Read a raw episode and report object-wrench demand under explicit physical assumptions |

Run from the repository root:

```text
python -m data_fetch.fetch_sample --episode 11
python -m data_fetch.fetch_preview_assets
python -m data_fetch.tier1_diagnostic --episode 11 --mass-kg 1
```

The first two commands use Python's standard library. The diagnostic uses the
[CPU correction environment](../docs/SETUP.md). Its mass is an experimental
assumption, not a dataset measurement. Prepare the pinned NVIDIA checkout with
`python scripts/bootstrap_upstream.py` before fetching its preview assets.

The existing Tier 1 cache remains at `correction/data/tier1/` so saved references
to its meshes and scene files stay valid. Downloads are excluded from Git.
SOMA model assets and decoding are handled by [soma_tools/](../soma_tools/README.md).
