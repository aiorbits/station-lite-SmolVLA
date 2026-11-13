# Data → fine-tuning → deployment

This reproduces the **right7** experiment, not the earlier 14D training. Run project commands from the directory containing `SmolVLA/` unless stated otherwise.

## 1. Freeze the task and data contract

Reference task: pick a red wooden block from the white grid and place it on the brown target area. The operator reports that local training demonstrations used red blocks at standard, unrotated orientations; they covered grid positions and small within-cell offsets. This was not a single fixed-position dataset. Per-frame yaw labels were not recorded, so the orientation statement comes from the collection protocol, not a new annotation audit.

| Field | right7 contract |
|---|---|
| `observation.state` | Six measured right joints, radians; measured gripper opening, metres |
| `action` | Absolute EEF `[x,y,z,roll,pitch,yaw,gripper]`; metres/radians/metres |
| Camera 1 | `observation.images.cam_high` → `observation.images.camera1` |
| Camera 2 | `observation.images.cam_right_wrist` → `observation.images.camera2` |
| Images | 640×480 RGB; dataset storage HWC, server converts to CHW |
| Sampling | Nominal 30 Hz; old raw episodes lack real capture timestamps |
| Language | Stable task string supplied with each frame/episode |

Joint state and EEF action have different representations. Never send the seven predicted values as joint commands. Retain base frame, TCP, extrinsic Euler `xyz` convention and calibration versions across recording, conversion and runtime.

## 2. Audit without changing raw data

Keep 45 raw episodes. The approved manifest selects **35 training episodes / 13,231 frames** and **10 diagnostic holdout episodes / 3,705 frames**. Split whole episodes, never individual frames. The holdout has already informed debugging and is not an untouched final benchmark.

Use [the demonstration review protocol](DATA_CLEANING_PROTOCOL.md). Review videos plus commanded/measured gripper motion; record original A/C/M/L frames and concrete issues in a separate audit log. A historical `quality_pass=true` is not proof that an episode has undergone this manual review. Do not overwrite raw files or silently change the fixed split. A newly cleaned subset is a new dataset/model version.

The converter below is specific to the approved 35/10 manifest and Station Lite gripper scaling. It is not a universal converter for arbitrary robots or episode counts.

## 3. Convert the approved right-arm view

With the pinned LeRobot environment and raw data supplied separately:

```bash
python SmolVLA/convert_right_only_dataset.py \
  --manifest SmolVLA/woodblock_20261027_v1/manifest.csv \
  --raw 20261027/raw --split train \
  --out SmolVLA/derived_datasets/lumos_woodblock_right7_v1/train \
  --repo-id lumos/lumos_woodblock_right7_train35_v1
python SmolVLA/convert_right_only_dataset.py \
  --manifest SmolVLA/woodblock_20261027_v1/manifest.csv \
  --raw 20261027/raw --split holdout \
  --out SmolVLA/derived_datasets/lumos_woodblock_right7_v1/holdout \
  --repo-id lumos/lumos_woodblock_right7_holdout10_v1
```

The output must not exist. On Hans22 these datasets already exist: inspect them instead of rerunning conversion. The converter preserves raw data, maps joint/gripper feedback into the right7 state, bounds historical gripper action labels to 0–80mm and records provenance. Its gripper scale is hardware-specific. The generated provenance, manifest hash, feature names and image correspondence are part of the dataset identity.

## 4. Smoke test, then fine-tune on CUDA

Use [the environment reference](ENVIRONMENT_REFERENCE.md), the approved base snapshot and a new output directory on ext4. The wrapper has a Hans22-specific training executable path; adapt it explicitly on another computer or use the direct equivalent printed by `--dry-run`.

```bash
python SmolVLA/train_right_only_woodblock.py --steps 1 \
  --output-dir /tmp/lumos_right7_smoke_NEW --dry-run
python SmolVLA/train_right_only_woodblock.py --steps 1 \
  --output-dir /tmp/lumos_right7_smoke_NEW
python SmolVLA/train_right_only_woodblock.py \
  --steps 10000 --batch-size 1 --save-freq 2000 --seed 42 \
  --output-dir /home/hans/cc/ckpt/lumos_right7_NEW_10000
```

The dry run prints a command; it does not validate the GPU. The executed wrapper requires CUDA and refuses CPU fallback. Do not rerun into an existing trained output directory.

The reference checkpoint's saved configuration records:

| Setting | Value |
|---|---|
| Starting weights | `lerobot/smolvla_base`, separate from this repository |
| Updates / batch / workers / seed | 10,000 / 1 / 0 / 42 |
| Action chunk / action dimension | 50 / 7 |
| Learning rate / warmup / decay schedule | 1e-4 / 1,000 / 30,000 steps |
| Training selection | Vision frozen; expert and state projection enabled |
| Checkpoint interval | 2,000 updates |

This is pretrained-model adaptation using 35 **local** episodes, not training from scratch using only 35 examples. A low loss is not evidence of a successful grasp.

## 5. Preserve the entire checkpoint

Keep `model.safetensors`, `config.json`, `train_config.json`, pre/postprocessor JSON and statistics safetensors, and the tokenizer directory together. The VLM cache is an additional dependency. Do not move weights alone or substitute another dataset's normalization statistics.

`verify_baseline.py --checkpoint` verifies the **approved reference weights**. A newly trained candidate will intentionally have a different hash; version its own manifest after validation rather than presenting it as the unchanged reference checkpoint.

Before deployment, check native output shape `(50,7)`, camera mapping, state scaling, finite values, units and saved processors. Use CUDA for any numerical model comparison. The historical evaluator `evaluate_right7_candidate_gpu.py` also expects a particular archived **14D** live snapshot; it is not a generic evaluator for every new right7 capture.

## 6. Validate in increasing scope

1. Run software contracts and validate checkpoint identity.
2. Check known-pose FK/IK consistency, calibration and representative continuous IK/vendor planning without policy commands.
3. Observe one fresh input and a guarded short approach with an operator present.
4. Validate hover alignment, descent, closure, lift and placement separately.
5. Repeat a single-block baseline; then vary position, color, yaw and distractors one factor at a time.

Follow [the deployment quickstart](QUICKSTART_STATION_LITE.md). Keep runtime failures as evidence. Record raw model predictions, conditioned/sent actions, fresh observations and measured tracking so the model and execution layer can be diagnosed separately.
