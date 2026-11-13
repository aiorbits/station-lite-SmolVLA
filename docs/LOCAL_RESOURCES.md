# Local resources and dependencies

Project entry points: [quickstart](QUICKSTART_STATION_LITE.md), [environment reference](ENVIRONMENT_REFERENCE.md), and [resource map](RESOURCE_MAP.md). The current baseline is the new9 right7 system; historical 14D results are not the deployment recipe.

These paths belong to Hans22. Verify their existence; the assets are not supplied by a repository clone.

| Resource | Reference location |
|---|---|
| Project parent | /media/hans/Bridge/Lumos Station Lite |
| Original 45 demonstrations | Project parent/20261027/raw/ |
| Original 14D train/holdout | Project parent/20261027_training/datasets/block_pick_place_20261027_train35 and block_pick_place_20261027_holdout10 |
| right7 derived datasets | SmolVLA/derived_datasets/lumos_woodblock_right7_v1/train and holdout |
| Base snapshot | SmolVLA/base_snapshot/ |
| Approved right7 checkpoint | /home/hans/cc/ckpt/smolvla_woodblock_right7_train35_v1_10000/checkpoints/010000/pretrained_model |
| Historical 14D checkpoint | /home/hans/cc/ckpt/smolvla_woodblock_train35_resume10k_to20k/checkpoints/020000/pretrained_model |
| Full experimental evidence | SmolVLA/diagnostics/ and SmolVLA/woodblock_20261027_v1/ |
| Host environment | /home/hans/miniconda3/envs/station_act/ |
| Dependency overlay / cache | SmolVLA/.deps/ and .hf_cache/ |
| ACT tools | Project parent/lumos_act_toolkit/; historical vendor code in archive/original/ |

Hans22 uses an RTX 4080 Laptop. Recheck CUDA and package versions each session. The observed environment is not a portable lockfile; do not upgrade station_act in place.

Robot hostname: ido. Historical LAN addresses: robot 192.168.0.101, host 192.168.0.109. Check hostname, IP and MAC before connecting. New9 used /home/lumos/replay_work/smolvla_right7_live_20261105. Deploy the release bundle into a separate /home/lumos/replay_work/SmolVLA_release_20261113 directory.

Vendor SDK: /home/lumos/startouch_sdk/interface_py. Application runtime: /home/lumos/lumosteleop/.app/runtime. Supply credentials separately, never in Git. SDK initialization can enable motors; distinguish static inspection from hardware preflight.

On a new machine, verify the data split, base weights, checkpoint, saved processors and checksums. Never substitute another normalization configuration.
