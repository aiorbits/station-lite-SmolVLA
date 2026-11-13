#!/usr/bin/env python3
"""Create a versioned single-right-arm LeRobot view from approved raw episodes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

TASK_TEXT = "Pick up the red wooden block from the white grid and place it on the brown target area."
CAMERAS = {
    "observation.images.cam_high": "head_camera",
    "observation.images.cam_right_wrist": "right_wrist_camera",
}
STATE_NAMES = [f"right_joint_{i}" for i in range(1, 7)] + ["right_gripper"]
ACTION_NAMES = [f"right_{k}" for k in ("x", "y", "z", "rx", "ry", "rz", "gripper")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("SmolVLA/woodblock_20261027_v1/manifest.csv"))
    ap.add_argument("--raw", type=Path, default=Path("20261027/raw"))
    ap.add_argument("--split", choices=("train", "holdout"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-id", required=True)
    args = ap.parse_args()

    if args.out.exists():
        ap.error(f"Output already exists: {args.out}")
    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = sorted((r for r in csv.DictReader(f) if r["split"] == args.split), key=lambda r: r["episode_id"])
    expected = 35 if args.split == "train" else 10
    if len(rows) != expected:
        ap.error(f"Expected {expected} {args.split} episodes, found {len(rows)}")
    for row in rows:
        if row["active_arm"] != "right" or row["task_outcome"] != "success" or row["quality_pass"].lower() != "true":
            ap.error(f"Unexpected episode metadata: {row['episode_id']}")
        if row["action_schema_version"] != "station_lite_pose14_v1" or float(row["nominal_fps"]) != 30:
            ap.error(f"Unsupported schema/fps: {row['episode_id']}")
        rel = Path(row["relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            ap.error(f"Unsafe relative path: {row['episode_id']}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES, "info": None},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES, "info": None},
    }
    for key in CAMERAS:
        features[key] = {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channel"], "info": None}
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=30, features=features, root=args.out.resolve(),
        robot_type="station_lite_right_arm", use_videos=False, image_writer_threads=8,
    )
    frame_total = 0
    for row in rows:
        episode = args.raw / row["relative_path"]
        payload = json.loads((episode / "data.json").read_text())
        frames = payload["data"]
        if len(frames) != int(row["frame_count"]):
            raise ValueError(f"Manifest/raw frame count mismatch: {row['episode_id']}")
        if payload.get("text", {}).get("goal", "").strip() != row["task_text"]:
            raise ValueError(f"Manifest/raw task mismatch: {row['episode_id']}")
        for fr in frames:
            st = np.asarray([*map(float, fr["states"]["right_joint"]["qpos"]), float(fr["states"]["right_gripper"][0]) / (1.5 * (0.046 / 0.117))], dtype=np.float32)
            pose = [float(x) for x in fr["actions"]["right_arm"]["pose"]]
            ac = np.asarray(pose[:6] + [min(0.08, max(0.0, pose[6]))], dtype=np.float32)
            if st.shape != (7,) or ac.shape != (7,) or not np.isfinite(st).all() or not np.isfinite(ac).all():
                raise ValueError(f"Invalid right-arm vector: {row['episode_id']}")
            sample = {"observation.state": st, "action": ac, "task": TASK_TEXT}
            for key, camera in CAMERAS.items():
                image_path = episode / fr["colors"][camera]
                with Image.open(image_path) as image:
                    if image.size != (640, 480):
                        raise ValueError(f"Unexpected image size {image.size}: {image_path}")
                    sample[key] = np.asarray(image.convert("RGB")).copy()
            ds.add_frame(sample)
        ds.save_episode()
        frame_total += len(frames)
        print(f"{row['episode_id']}: {len(frames)} frames", flush=True)
    ds.finalize()
    provenance = {
        "view_version": "lumos_woodblock_right7_v1",
        "source_split": args.split,
        "repo_id": args.repo_id,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "episode_ids": [r["episode_id"] for r in rows],
        "episodes": len(rows), "frames": frame_total,
        "state_source_indices": [7, 8, 9, 10, 11, 12, 13],
        "action_source_indices": [7, 8, 9, 10, 11, 12, 13],
        "cameras": list(CAMERAS), "task_text": TASK_TEXT,
        "purpose": "Single-right-arm state/action ablation; original 14D data remains unchanged.",
    }
    (args.out / "station_lite_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Converted {len(rows)} {args.split} episodes / {frame_total} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
