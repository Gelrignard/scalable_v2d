# Third-party dependencies

NVIDIA Video to Data:
https://github.com/nvidia-isaac/video_to_data

The CPU prototype imports a separately downloaded pinned checkout. Its source
is not vendored here. The compatibility patch in patches/windows_cpu_preview.patch
contains modifications to NVIDIA source under Apache License 2.0; the upstream
license is retained in licenses/NVIDIA-LICENSE.txt. That original file also
contains the CC-BY-4.0 terms for upstream documentation/skills. Preserve the
original copyright and license notices when redistributing patched source.

Robot assets, pretrained models, and datasets are downloaded separately and
retain their respective upstream terms. They are not relicensed or included
by publishing this project. This notice does not assign a new project-wide
license to the original code in this repository.
