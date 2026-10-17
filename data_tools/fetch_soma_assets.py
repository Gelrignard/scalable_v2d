"""Fetch the upstream-pinned SOMA rig/MHR assets into an isolated local cache."""
from pathlib import Path
import json
import urllib.request
from data_tools.fetch_sample import sha256

REVISION = "466879a83d57eabf3d875ded2d869f2075f90348"
FILES = ["SOMA_neutral.npz", "correctives_model.pt", "MHR/mhr_model_lod1.pt",
         "MHR/base_body_lod1.obj", "MHR/SOMA_wrap_lod1.obj"]


def main():
    root = Path("correction/data/soma_assets").resolve()
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/nvidia/SOMA-X/tree/{REVISION}?recursive=true&expand=false"
    with urllib.request.urlopen(url, timeout=30) as response:
        tree = {item["path"]: item for item in json.load(response)}
    manifest = {"repository": "nvidia/SOMA-X", "revision": REVISION, "files": {}}
    for relative in FILES:
        destination = root / relative
        expected = tree[relative].get("lfs", {}).get("oid")
        if not destination.exists() or (expected and sha256(destination) != expected):
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/nvidia/SOMA-X/resolve/{REVISION}/{relative}"
            temporary = destination.with_suffix(destination.suffix + ".part")
            print(f"Downloading {relative}: {tree[relative]['size']/1e6:.1f} MB", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                for block in iter(lambda: response.read(1024*1024), b""):
                    stream.write(block)
            digest = sha256(temporary)
            if expected and digest != expected:
                raise ValueError(f"asset checksum failed: {relative}")
            temporary.replace(destination)
        manifest["files"][relative] = {"sha256": sha256(destination), "bytes": destination.stat().st_size}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Verified assets: {root}", flush=True)


if __name__ == "__main__":
    main()
