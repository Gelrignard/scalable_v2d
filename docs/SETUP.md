# Reproduce the CPU prototype

Run commands from the repository root. The tested host is Windows; Linux
environment creation must be validated independently. No GPU, Isaac Lab, model
training, or cloud account is needed for the CPU correction tests.

## 1. Standalone correction without NVIDIA source

With a Python 3.11+ environment active:

~~~text
python -m pip install -r correction/requirements.txt
python -m correction.scripts.demo
python -m unittest correction.tests.test_dynamics correction.tests.test_optimizer correction.tests.test_integration -v
~~~

These commands run the generic dynamics/contact optimizer without an `upstream/`
checkout, G1 assets, Pinocchio, or SOMA. The robot retargeting/book integration and
viewers use the optional dependency below.

## 2. Optional NVIDIA source dependency

Skip this section for standalone correction. Follow it for G1 retargeting,
the book robot-trajectory trial, or robot viewers. NVIDIA's source is downloaded
locally on request, excluded from Git, and not registered as a submodule.

Follow [the optional dependency and patch guide](NVIDIA_COMPATIBILITY.md) for
the tested source version, Git commands, and compatibility patch. The patch and
its explanation are kept in `docs/`. The CPU retargeting dependency and the
separate training/evaluation runtime use different source versions.

## 3. CPU retargeting environment

Use Conda or Micromamba. The following uses Micromamba on PATH:

~~~text
micromamba create -y -p ./.retarget-env -f setup/preview-environment.yml
micromamba run -p ./.retarget-env python -m pip install -r requirements-cpu.txt
micromamba run -p ./.retarget-env python -m pip install --no-deps pin-pink==4.2.0
micromamba run -p ./.retarget-env python -m pip install --no-deps -e ./retargeting
micromamba run -p ./.retarget-env python -m unittest discover -s retargeting/tests -v
micromamba run -p ./.retarget-env python -m unittest discover -s correction/tests -v
~~~

Use your own Micromamba installation and shell. Machine-specific shortcuts are
not distributed. `setup/preview-conda-win64.lock` records the tested Conda
package builds for Windows; it is not a Linux lock file.

Tests requiring generated local audit artifacts explicitly skip on a fresh
clone. No generated artifacts or model weights are committed.

The optional scientific plot command, `python -m correction.scripts.plot_book`, also
requires Matplotlib (`python -m pip install "matplotlib>=3.8,<4"`). It is not
needed by the optimizer, viewer, or tests.

## 4. Pinned example and synthetic correction

~~~text
micromamba run -p ./.retarget-env python retargeting/scripts/run.py --method both
micromamba run -p ./.retarget-env python -m correction.scripts.demo
~~~

The retargeting CLI compares the pinned NVIDIA body baseline with our
contact-aware reference. Both variants are also selected by default.

## 5. Actual Tier 1 book input

The current fetch/decode commands are deliberately Tier 1 specific:

Dataset downloads, SOMA model downloads/decoding, and SOMA requirements are
grouped together in `data_tools/`. This package does not shadow the installed
`soma` library. Existing environment and cache paths are preserved, including
`.soma-env` and `correction/data/soma_assets`.

~~~text
python -m data_tools.fetch_sample --episode 11
python -m data_tools.fetch_soma_assets
python -m venv .soma-env
~~~

Use .soma-env/Scripts/python.exe on Windows or .soma-env/bin/python on Linux:

~~~text
<soma-python> -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
<soma-python> -m pip install -r data_tools/requirements-soma.txt
<soma-python> -m data_tools.decode_tier1
micromamba run -p ./.retarget-env python retargeting/scripts/run.py --input correction/outputs/tier1_book/human_motion.npz --method contact_aware --output correction/outputs/tier1_book/retargeted
micromamba run -p ./.retarget-env python -m correction.scripts.book_pipeline
~~~

The source dataset and model asset revisions are pinned. Review the upstream
model/dataset terms when obtaining them. Downloads remain local.

The book correction is one short airborne phase. Its output is not accepted as
a dynamically feasible manipulation. It is not a Tier 2 adapter or a trained
policy; those implementation gaps are listed in the workstream plans.

## 6. Submission tooling and excluded outputs

Download the current official kit from
https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip
when implementing evaluation. The 17 October kit SHA-256 was
8d2bce64ecab6da4eee938fe2859bd60e767535af30068b386e2dc20a72cf43e.
The kit itself is not committed.

METHOD.md and the handoff retain descriptions of measured local artifacts.
Their generated-output links become usable after reproducing the experiment;
they do not imply that those artifacts ship with this repository.

## Environment setup files

| File | Purpose |
|---|---|
| `setup/preview-environment.yml` | Conda/Micromamba environment recipe: Python, Pinocchio, and numerical libraries |
| `setup/preview-conda-win64.lock` | Exact Windows Conda package builds and hashes for reproducing the tested environment |
| `setup/preview-requirements.txt` | Additional pinned pip packages used by retargeting and visualization |
| `requirements-cpu.txt` | Combined pip dependency list for retargeting and correction; includes the other requirements files |

The `preview` prefix dates from the first CPU demonstration. Those environment
files still support the current retargeting setup. Use the public Python commands
above and create any convenience shortcuts in your own local environment.
