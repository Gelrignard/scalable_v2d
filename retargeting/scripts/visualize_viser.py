import _bootstrap  # noqa: F401 -- initialize local source/runtime paths
import argparse
from pathlib import Path
from v2d_retarget.paths import MODULE
from v2d_retarget.viewer import serve

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare human, targets, final robot FK, and inferred contacts"
    )
    p.add_argument("--results", type=Path, default=MODULE / "outputs/snack_box")
    p.add_argument("--port", type=int, default=8081)
    a = p.parse_args()
    paths = [
        a.results / f"{name}.npz"
        for name in ("baseline", "contact_aware")
        if (a.results / f"{name}.npz").exists()
    ]
    serve(paths, a.port)
