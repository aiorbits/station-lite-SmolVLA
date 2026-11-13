# Resource map

The public export contains the active path below. Historical files remain in the full workspace; no recordings or checkpoints were moved or deleted during organization.

## Current runtime

| File | Purpose |
|---|---|
| `station_lite_smolvla_right7_server.py` | CUDA-only HTTP policy service; `/health`, `/predict` |
| `station_lite_smolvla_right7_guarded.py` | Robot observation, phase control, planning guards and execution |
| `build_grasp_replay_cases.py` | Color parsing/detection, calibration/projection, approach/grasp/place conditioning |
| `station_lite_smolvla_single_chunk.py` | Shared IK, arm bounds, image encoding and frame helpers |
| `joint_trace.py` | Timestamped measured-state logging and per-chunk plans |
| `tools/verify_baseline.py` | Source/checkpoint integrity; optional CUDA availability gate; never connects SDK |
| `releases/right7_20261113/` | Frozen hashes, data split, environment observation and results summary |

## Data and training

| Resource | Purpose |
|---|---|
| `woodblock_20261027_v1/manifest.csv` | Approved historical episode selection and split |
| `convert_right_only_dataset.py` | Approved raw recordings → right7 LeRobot view |
| `train_right_only_woodblock.py` | CUDA smoke test and fine-tuning; reference-host executable path |
| `dataset_standard.py` | Legacy manifest/raw schema audit; review its schema before a new dataset |
| `docs/DATA_CLEANING_PROTOCOL.md` | Independent human review; original-frame A/C/M/L annotations |
| `derived_datasets/lumos_woodblock_right7_v1/` | Local-only train/holdout; not exported |
| Approved checkpoint + VLM/base caches | Supplied separately; see environment and local resources |

## Tests and diagnostic tools

Run `python -m unittest test_right7_guarded_contract test_adaptive_phase_timing test_joint_trace`. These are software/mocked-interface checks, not physical motion validation. `diagnose_sdk_ik_isolated.py` is available in the full project for representative SDK planning tests in a network namespace without CAN. Do not substitute the live constructor for isolated planning.

## Historical / experimental, outside the quickstart

- `station_lite_smolvla_server.py` and earlier guarded clients: 14D protocol, incompatible with the native right7 server.
- `train_right_woodblock.py`, `continue_right_woodblock.py`: earlier 14D training.
- `compare_guided_rtc.py`, RTC reports: controlled research experiments, not this live baseline.
- `experiments_v070/`, `docs/LEROBOT_V070_Experiment_PLAN.md`: future/isolated work.
- `docs/new*.md`, `3-waypoints-*.md`, `diagnostics/`: investigation and field evidence; old commands are not current deployment instructions.
- `HANDOFF_20261104.md` and archived README: historical context. `HANDOFF.md` contains newer updates.

## Local-only assets on Hans22

See [LOCAL_RESOURCES.md](LOCAL_RESOURCES.md) for absolute paths. The reference raw data is in `../20261027/raw/`; the approved checkpoint is in `/home/hans/cc/ckpt/smolvla_woodblock_right7_train35_v1_10000/checkpoints/010000/pretrained_model`. New9 raw run archives still require return from the robot. The manufacturer SDK/runtime must be provisioned on each robot separately.

## Publication preparation

Use `prepare_shared_repository.py --profile public` for the concise pipeline or `--profile full` for the historical sharing list. Both export only explicitly listed files and produce checksums. Read [SHARING.md](SHARING.md). Exporting and local synchronization do not commit or push to GitHub.
