#!/usr/bin/env python3
"""GPU-only training entry point for the right7 derived dataset."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SMOLVLA = PROJECT / "SmolVLA"
DATASET = SMOLVLA / "derived_datasets/lumos_woodblock_right7_v1/train"
BASE = SMOLVLA / "base_snapshot"
TRAIN_BIN = Path("/home/hans/miniconda3/envs/station_act/bin/lerobot-train")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--save-freq", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/lumos_smolvla_right7_smoke"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if min(args.steps, args.batch_size, args.save_freq) < 1:
        ap.error("steps, batch-size and save-freq must be positive")
    info = json.loads((DATASET / "meta/info.json").read_text())
    if info.get("total_episodes") != 35 or info["features"]["observation.state"]["shape"] != [7] or info["features"]["action"]["shape"] != [7]:
        ap.error("Expected the validated right7 train35 dataset")

    input_features = {
        "observation.state": {"type": "STATE", "shape": [7]},
        "observation.images.camera1": {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.images.camera2": {"type": "VISUAL", "shape": [3, 480, 640]},
    }
    rename_map = {
        "observation.images.cam_high": "observation.images.camera1",
        "observation.images.cam_right_wrist": "observation.images.camera2",
    }
    output = args.output_dir.resolve()
    if args.steps > 1 and not args.dry_run:
        probe = output
        while not probe.exists():
            probe = probe.parent
        fs = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(probe)], text=True).strip()
        if fs.lower() == "exfat":
            ap.error("Checkpoint output must be on ext4, not Bridge exFAT")
    command = [
        str(TRAIN_BIN), "--policy.path=" + str(BASE),
        "--policy.input_features=" + json.dumps(input_features, separators=(",", ":")),
        "--policy.output_features=" + json.dumps({"action": {"type": "ACTION", "shape": [7]}}, separators=(",", ":")),
        "--policy.device=cuda", "--policy.push_to_hub=false",
        "--dataset.repo_id=lumos/lumos_woodblock_right7_train35_v1",
        "--dataset.root=" + str(DATASET),
        "--rename_map=" + json.dumps(rename_map, separators=(",", ":")),
        "--batch_size=" + str(args.batch_size), "--num_workers=0", "--steps=" + str(args.steps),
        "--save_checkpoint=" + ("false" if args.steps == 1 else "true"),
        "--wandb.enable=false", "--seed=" + str(args.seed), "--output_dir=" + str(output),
    ]
    if args.steps > 1:
        command.append("--save_freq=" + str(args.save_freq))
    if args.dry_run:
        print(json.dumps({"command": command, "dataset": str(DATASET), "GPU_REQUIRED": True}, indent=2))
        return 0
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required; training was not started.")
    print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SMOLVLA / ".deps") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["HF_HOME"] = str(SMOLVLA / ".hf_cache")
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
