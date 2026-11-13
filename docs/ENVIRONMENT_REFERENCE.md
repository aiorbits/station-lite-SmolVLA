# Environment reference and installation boundaries

The following is an **observed reference**, not a portable lockfile or a fresh-machine installation certificate. See `releases/right7_20261113/environment_snapshot.json` for machine-readable versions and validation status.

## Two separate computers

| Role | Reference |
|---|---|
| Training / policy host | Hans22, Linux x86_64, RTX 4080 Laptop 12GB, `station_act`, Python 3.12.15 |
| Robot | `ido`, Linux aarch64, `lumosteleop`, Python 3.10, manufacturer SDK/runtime and CAN |
| Model | Native right7 10k checkpoint; VLM backbone cache also required |
| Software-only validation | System Python 3.10, NumPy/OpenCV/Pillow/SciPy; 66/66 contract tests passed |

Do not install the workstation's CUDA packages onto the ARM robot or replace the vendor runtime with host dependencies.

## Pinned source and observed packages

LeRobot checkout: commit `3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`, package metadata `0.6.2` (`v0.6.1-44-g3f2c29ef`). The reviewed `src/` and `pyproject.toml` had no tracked diff from that commit.

Observed host versions include PyTorch `2.11.0+cu130`, torchvision `0.26.0`, Transformers `5.22.2`, Accelerate `1.15.0`, datasets `4.8.5`, huggingface-hub `1.33.0`, NumPy `2.2.6`, Pillow `12.3.0`, OpenCV-headless `4.13.0.92`, num2words `0.5.14`, torchcodec `0.11.1` and PyAV `15.1.0`.

The host has a `.deps` overlay (including num2words/NumPy) in addition to its conda environment. Without the overlay, importing the processor can fail. `station_act` currently lacks SciPy, which is needed by one geometric contract test; no packages were installed or upgraded during this documentation work.

**Important:** the pinned source's SmolVLA extra requests Transformers `>=5.4,<5.6`, whereas the observed environment contains `5.22.2`. Therefore installing the extra afresh does not reproduce the observed environment. Keep the working environment unchanged; prepare and validate a separate reproduction environment before claiming a clean install is supported. The snapshot is not an instruction to force conflicting dependencies with `--no-deps`.

For a new workstation, the reproducible source starting point is:

```bash
git clone https://github.com/huggingface/lerobot.git lerobot_source
cd lerobot_source
git checkout 3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e
```

Choose a compatible CUDA PyTorch build using the [official installation instructions](https://pytorch.org/get-started/locally/). The [official SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla) describes the `.[smolvla]` extra. Treat an install using that extra as a **new candidate environment** until CUDA loading, processor round-trip, one training update, checkpoint reload and robot software contracts pass. These fresh-environment checks were NOT RUN here. Do not upgrade the current ACT/SmolVLA environment in place.

## Cache the base model and VLM separately

Recorded revisions:

| Artifact | Recorded revision |
|---|---|
| `lerobot/smolvla_base` | `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |

On the host, from the `SmolVLA` repository directory, prepare caches while connected:

```bash
export HF_HOME="$PWD/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
hf download lerobot/smolvla_base --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 \
  --local-dir "$PWD/base_snapshot"
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467
```

The VLM code references a model ID without an explicit revision. On a fresh cache, make sure its offline `main` resolution points to the approved snapshot, not an unrelated newer snapshot. For example, after confirming the listed snapshot is complete:

```bash
python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['HF_HUB_CACHE'])/'models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct'
revision = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
assert (p/'snapshots'/revision).is_dir(), 'Prepare the complete snapshot first'
(p/'refs').mkdir(exist_ok=True)
(p/'refs/main').write_text(revision)
PY
```

Do this in the dedicated task cache, not a cache shared by unrelated models. The fine-tuned checkpoint is supplied separately; downloading the base is not downloading the fine-tuned policy. Enable offline mode only after all required files are present. Cache download commands are preparation instructions, not operations performed during this review.

## Before any policy computation

Run `tools/verify_baseline.py --role host --checkpoint PATH --gpu` in the actual inference environment. It verifies source/checkpoint hashes and requires `torch.cuda.is_available()`. The documentation session returned CUDA unavailable, so no fresh inference/training was executed; this alone does not diagnose the machine's NVIDIA driver or invalidate earlier live GPU runs.

Use ext4 for training checkpoints. Preserve free disk space and keep model/data storage separate from the source bundle. Never put credentials, raw robot recordings or model caches in the repository.
