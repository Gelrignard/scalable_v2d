# Reproduce the CPU prototype

Run commands from the repository root. The tested host is Windows; Linux
environment creation must be validated independently. No GPU, Isaac Lab, model
training, or cloud account is needed for the CPU correction tests.

## 1. Source dependency

With Git and Python 3.11+ available:

~~~text
python scripts/bootstrap_upstream.py
python -m data_fetch.fetch_preview_assets
~~~

The bootstrap fetches the revision in upstream.lock.json into ignored upstream/
and applies the tracked compatibility patch. It refuses a conflicting existing
checkout instead of resetting it. The second command fetches only G1 meshes and
the pinned snack-box example, checking the LFS object hashes.

The patch enables CPU NumPy readers, avoids unnecessary Torch imports, and
limits the historical viewer to loopback. It does not change baseline IK costs.
The separate training/evaluation revision in the lock file is intentionally
different; do not replace the CPU dependency with it.

Here, **upstream** means NVIDIA's original source project, which we use as a
dependency. A **patch** is a text file listing edits relative to that source
version. The bootstrap combines the pinned source with our compatibility patch
so another checkout can reproduce the same local dependency.

## 2. CPU retargeting environment

Use Conda or Micromamba. The following uses Micromamba on PATH:

~~~text
micromamba create -y -p ./.retarget-env -f preview-environment.yml
micromamba run -p ./.retarget-env python -m pip install -r requirements-cpu.txt
micromamba run -p ./.retarget-env python -m pip install --no-deps pin-pink==4.2.0
micromamba run -p ./.retarget-env python -m pip install --no-deps -e ./retargeting
micromamba run -p ./.retarget-env python -m unittest discover -s retargeting/tests -v
micromamba run -p ./.retarget-env python -m unittest discover -s correction/tests -v
~~~

The existing Windows launchers expect .tools/Library/bin/micromamba.exe.
The direct commands above also work with a separately installed executable.
preview-conda-win64.lock records the tested Conda package builds for Windows;
it is not a Linux lock file.

Tests requiring generated local audit artifacts explicitly skip on a fresh
clone. No generated artifacts or model weights are committed.

The optional scientific plot command, `python -m correction.scripts.plot_book`, also
requires Matplotlib (`python -m pip install "matplotlib>=3.8,<4"`). It is not
needed by the optimizer, viewer, or tests.

## 3. Pinned example and synthetic correction

~~~text
micromamba run -p ./.retarget-env python retargeting/scripts/run.py --method both
micromamba run -p ./.retarget-env python -m correction.scripts.demo
~~~

The retargeting CLI compares the pinned NVIDIA body baseline with our
contact-aware reference. Both variants are also selected by default.

## 4. Actual Tier 1 book input

The current fetch/decode commands are deliberately Tier 1 specific:

Download utilities are grouped in `data_fetch/`. SOMA model download, decoding,
and dependency requirements are grouped in `soma_tools/`; the latter name avoids
shadowing the installed `soma` library. Existing environment and cache paths are
preserved, including `.soma-env` and `correction/data/soma_assets`.

~~~text
python -m data_fetch.fetch_sample --episode 11
python -m soma_tools.fetch_assets
python -m venv .soma-env
~~~

Use .soma-env/Scripts/python.exe on Windows or .soma-env/bin/python on Linux:

~~~text
<soma-python> -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
<soma-python> -m pip install -r soma_tools/requirements.txt
<soma-python> -m soma_tools.decode_tier1
micromamba run -p ./.retarget-env python retargeting/scripts/run.py --input correction/outputs/tier1_book/human_motion.npz --method contact_aware --output correction/outputs/tier1_book/retargeted
micromamba run -p ./.retarget-env python -m correction.scripts.book_pipeline
~~~

The source dataset and model asset revisions are pinned. Review the upstream
model/dataset terms when obtaining them. Downloads remain local.

The book correction is one short airborne phase. Its output is not accepted as
a dynamically feasible manipulation. It is not a Tier 2 adapter or a trained
policy; those implementation gaps are listed in the workstream plans.

## 5. Submission tooling and excluded outputs

Download the current official kit from
https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip
when implementing evaluation. The 17 October kit SHA-256 was
8d2bce64ecab6da4eee938fe2859bd60e767535af30068b386e2dc20a72cf43e.
The kit itself is not committed.

METHOD.md and the handoff retain descriptions of measured local artifacts.
Their generated-output links become usable after reproducing the experiment;
they do not imply that those artifacts ship with this repository.
