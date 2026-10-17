"""Fetch and verify only the G1 and one demo's Git LFS assets."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "upstream"
ASSETS = REPO / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
DEMO = "2026-03-06_10-24-18_snack_box_pick_and_place_01"
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def fetch(path: Path) -> int:
    if path.stat().st_size > 200:
        return 0
    raw = path.read_bytes()
    if not raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return 0
    fields = dict(line.split(" ", 1) for line in raw.decode().splitlines())
    expected = fields["oid"].split(":", 1)[1]
    url = "https://media.githubusercontent.com/media/nvidia-isaac/video_to_data/" + COMMIT + "/" + urllib.parse.quote(path.relative_to(REPO).as_posix(), safe="/")
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read()
    if len(content) != int(fields["size"]) or hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError(f"LFS validation failed: {path}")
    path.write_bytes(content)
    return len(content)


if __name__ == "__main__":
    roots = [ASSETS / "meshes/g1", ASSETS / "human_motion_data/whole_body/soma" / ("sequence_id=" + DEMO)]
    paths = [p for root in roots for p in root.rglob("*") if p.is_file()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        sizes = list(pool.map(fetch, paths))
    print(f"Verified {sum(size > 0 for size in sizes)} assets; downloaded {sum(sizes)/1e6:.1f} MB; commit {COMMIT}")
