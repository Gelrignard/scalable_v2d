"""Download one pinned Track 2 Tier 1/2 episode and its assets; no video."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
REPO = "nvidia/video_to_data_challenge"
PREFIX = "track_2/tier_1_multiview_caption"
PREFIXES = {1: PREFIX, 2: "track_2/tier_2_synthetic_noise"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(output: Path, episode=11, tier=1):
    if tier not in PREFIXES:
        raise ValueError("Only Track 2 tiers 1 and 2 are supported")
    prefix = PREFIXES[tier]
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "download_manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {
        "repository": REPO, "revision": REVISION, "tier": tier, "files": {}}
    if manifest["revision"] != REVISION or manifest["tier"] != tier:
        raise ValueError("cache provenance does not match requested pinned tier")

    def get(relative):
        destination = output / relative
        previous = manifest["files"].get(relative)
        if destination.exists() and previous:
            if sha256(destination) != previous["sha256"]:
                raise ValueError(f"cached file changed: {destination}")
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{prefix}/{relative}"
        request = urllib.request.Request(url, headers={"User-Agent": "v2d-cpu-correction-prototype"})
        temporary = destination.with_suffix(destination.suffix + ".part")
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
        temporary.replace(destination)
        manifest["files"][relative] = {"url": url, "sha256": sha256(destination),
                                        "bytes": destination.stat().st_size}
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        return destination

    for relative in ("README.md", "meta/info.json", "meta/episodes_metadata.jsonl"):
        get(relative)
    records = [json.loads(line) for line in (output / "meta/episodes_metadata.jsonl").read_text().splitlines() if line]
    entry = next(record for record in records if record["episode_index"] == episode)
    for relative in (f"data/chunk-000/episode_{episode:06d}.parquet", entry["mesh"], entry["ground_plane"], entry["support_surface"]):
        get(relative)
    print(json.dumps({"episode": episode, "tier": tier, "revision": REVISION, "directory": str(output.resolve()),
                      "downloaded_bytes": sum(item["bytes"] for item in manifest["files"].values()),
                      "metadata": entry}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tier", type=int, choices=(1, 2), default=1)
    parser.add_argument("--episode", type=int, default=11)
    args = parser.parse_args()
    fetch(args.output or Path(f"correction/data/tier{args.tier}"), args.episode, args.tier)
