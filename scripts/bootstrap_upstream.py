"""Prepare the pinned CPU dependency without replacing an existing checkout.

Uses only Python's standard library and Git. Downloads source, not LFS assets.
Run python -m data_fetch.fetch_preview_assets afterward for the selected robot/example assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def git(directory: Path, *args: str, check: bool = True):
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        env=env, text=True, capture_output=True, check=check,
    )


def prepare(destination: Path, check_only: bool = False):
    lock = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
    patch = ROOT / lock["patch"]
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    if digest != lock["patch_sha256"]:
        raise ValueError("Compatibility patch differs from upstream.lock.json")
    destination = destination.resolve()
    if not (destination / ".git").exists():
        if check_only:
            raise FileNotFoundError(f"No dependency checkout at {destination}")
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Refusing to populate nonempty directory: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        git(destination, "init")
        git(destination, "remote", "add", "origin", lock["repository"])
        git(destination, "config", "core.autocrlf", "false")
        git(destination, "config", "core.longpaths", "true")
        git(destination, "fetch", "--depth=1", "--filter=blob:none",
            "origin", lock["retarget_commit"])
        git(destination, "sparse-checkout", "init", "--cone")
        git(destination, "sparse-checkout", "set", *lock["sparse_paths"])
        git(destination, "checkout", "--detach", lock["retarget_commit"])
    commit = git(destination, "rev-parse", "HEAD").stdout.strip()
    if commit != lock["retarget_commit"]:
        raise ValueError(f"Existing checkout is at {commit}; expected {lock['retarget_commit']}. "
                         "Use a different destination; existing files were not reset.")
    # Existing Windows checkouts may contain CRLF or mixed line endings.
    # Ignore context whitespace when detecting/applying this source patch.
    patch_args = ("-c", "core.autocrlf=false", "apply", "--ignore-space-change")
    already_applied = git(destination, *patch_args, "--reverse", "--check",
                          str(patch), check=False).returncode == 0
    if not already_applied:
        if check_only:
            raise ValueError("Pinned source exists, but the compatibility patch is not applied")
        result = git(destination, *patch_args, "--check", str(patch), check=False)
        if result.returncode:
            raise ValueError("Compatibility patch conflicts with existing edits; "
                             "nothing was overwritten.\n" + result.stderr)
        git(destination, *patch_args, str(patch))
    print(json.dumps({"directory": str(destination), "commit": commit,
                      "patch_sha256": digest, "patch_applied": True,
                      "lfs_assets_downloaded": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "upstream")
    parser.add_argument("--check", action="store_true", help="Verify without changing files")
    args = parser.parse_args()
    try:
        prepare(args.destination, args.check)
    except subprocess.CalledProcessError as exc:
        parser.exit(1, f"Git failed: {exc.stderr or exc.stdout}\n")
