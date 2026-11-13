#!/usr/bin/env python3
"""Create and audit the Station Lite demonstration manifest.

The CSV is the stable lab contract. LeRobot's internal parquet layout is not.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "20261027/raw"
LEGACY = ROOT / "20261027_training/student_technical_packet_20261030/01_plan_and_manifest/manifest_20261027.csv"
SPLIT = ROOT / "20261027_training/split_and_gripper_audit.csv"
DATASETS = ROOT / "20261027_training/datasets"
CAMERAS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
FIELDS = (
    "episode_id", "relative_path", "dataset_version", "split", "session_id",
    "collected_at", "active_arm", "object_type", "start_location_id",
    "target_id", "task_text", "task_outcome", "quality_pass", "frame_count",
    "nominal_fps", "timestamps_present", "camera_keys", "camera_config_version",
    "robot_calibration_version", "action_schema_version",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def migrate(args: argparse.Namespace) -> int:
    legacy = read_csv(args.legacy)
    split_rows = read_csv(args.split)
    split = {row["episode_id"]: row for row in split_rows}
    if len(split) != len(split_rows) or {r["episode_id"] for r in legacy} != set(split):
        raise ValueError("Legacy manifest and split audit do not have the same unique episode IDs")
    rows = []
    for item in legacy:
        episode_id = item["episode_id"]
        if item["task_outcome"] != "success" or item["active_arm"] != "right":
            raise ValueError(f"Unexpected historical label: {episode_id}")
        payload = json.loads((args.raw / item["relative_path"] / "data.json").read_text())
        task = payload.get("text", {}).get("goal", "").strip()
        if not task:
            raise ValueError(f"Missing task text: {episode_id}")
        if int(item["frame_count"]) != len(payload["data"]):
            raise ValueError(f"Frame count mismatch: {episode_id}")
        audit = split[episode_id]
        if int(audit["frame_count"]) != len(payload["data"]):
            raise ValueError(f"Split audit frame count mismatch: {episode_id}")
        split_name = {"train35": "train", "holdout10": "holdout"}[audit["split"]]
        target = item["start_location_id"].split("_", 1)[-1]
        rows.append({
            "episode_id": episode_id, "relative_path": item["relative_path"],
            "dataset_version": "woodblock_20261027_v1", "split": split_name,
            "session_id": item["session_id"], "collected_at": item["collected_at"],
            "active_arm": "right", "object_type": "wood_block",
            "start_location_id": item["start_location_id"], "target_id": target,
            "task_text": task, "task_outcome": "success", "quality_pass": "true",
            "frame_count": item["frame_count"], "nominal_fps": item["nominal_fps"],
            "timestamps_present": item["timestamps_present"], "camera_keys": item["camera_keys"],
            "camera_config_version": "unknown_legacy",
            "robot_calibration_version": "unknown_legacy",
            "action_schema_version": "station_lite_pose14_v1",
        })
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force only after review")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} episodes to {args.output}: {dict(Counter(r['split'] for r in rows))}")
    return 0


def check_vector(values: object, n: int, label: str) -> None:
    if not isinstance(values, list) or len(values) < n or not all(math.isfinite(float(x)) for x in values[:n]):
        raise ValueError(f"{label}: expected {n} finite values")


def check_frame(frame: dict, episode: Path, deep: bool) -> None:
    state, action = frame["states"], frame["actions"]
    for side in ("left", "right"):
        check_vector(state[f"{side}_joint"]["qpos"], 6, f"{side} joints")
        check_vector(state[f"{side}_gripper"], 1, f"{side} gripper feedback")
        check_vector(action[f"{side}_arm"]["pose"], 7, f"{side} pose action")
        gripper = float(action[f"{side}_arm"]["pose"][6])
        if not 0 <= gripper <= 0.08 + 1e-5:
            raise ValueError(f"{side} gripper action outside 0–0.08 m: {gripper}")
    from PIL import Image
    for camera in CAMERAS:
        rel = Path(frame["colors"][camera])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe image path: {rel}")
        path = episode / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        if deep:
            with Image.open(path) as image:
                if image.size != (640, 480):
                    raise ValueError(f"Wrong image size: {path}: {image.size}")
                image.verify()


def audit_dataset(path: Path, rows: list[dict[str, str]], split: str, errors: list[str]) -> dict:
    info_path = path / "meta/info.json"
    if not info_path.is_file():
        errors.append(f"{split}: missing {info_path}")
        return {}
    info = json.loads(info_path.read_text())
    expected = sorted((r for r in rows if r["split"] == split), key=lambda r: r["episode_id"])
    counts = (len(expected), sum(int(r["frame_count"]) for r in expected))
    if (info["total_episodes"], info["total_frames"]) != counts:
        errors.append(f"{split}: dataset counts differ from manifest: {counts}")
    if info["fps"] != 30 or info["robot_type"] != "station_lite":
        errors.append(f"{split}: wrong fps or robot_type")
    features = info["features"]
    state_names = [f"left_joint_{i}" for i in range(1, 7)] + ["left_gripper"] + [f"right_joint_{i}" for i in range(1, 7)] + ["right_gripper"]
    action_names = [f"{s}_{v}" for s in ("left", "right") for v in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
    for key, names in (("observation.state", state_names), ("action", action_names)):
        if features.get(key, {}).get("shape") != [14] or features[key].get("names") != names:
            errors.append(f"{split}: wrong {key} feature contract")
    for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        if features.get(f"observation.images.{key}", {}).get("shape") != [480, 640, 3]:
            errors.append(f"{split}: wrong {key} image contract")
    try:
        import pyarrow.parquet as pq
        episode_files = sorted((path / "meta/episodes").glob("chunk-*/file-*.parquet"))
        if not episode_files:
            raise FileNotFoundError("No episode metadata parquet files")
        records = [record for file in episode_files for record in pq.read_table(file, columns=["episode_index", "length", "tasks"]).to_pylist()]
        if len(records) != len(expected):
            errors.append(f"{split}: episode metadata count mismatch")
        for index, (record, manifest_row) in enumerate(zip(records, expected)):
            if record["episode_index"] != index or record["length"] != int(manifest_row["frame_count"]):
                errors.append(f"{split}: episode {index} length/order mismatch")
            if manifest_row["task_text"] not in record["tasks"]:
                errors.append(f"{split}: episode {index} task text mismatch")
    except Exception as exc:
        errors.append(f"{split}: cannot inspect episode metadata: {exc}")
    return {"path": str(path), "episodes": info["total_episodes"], "frames": info["total_frames"], "fps": info["fps"], "codebase_version": info["codebase_version"]}


def audit(args: argparse.Namespace) -> int:
    rows = read_csv(args.manifest)
    errors: list[str] = []
    seen: set[str] = set()
    counts = Counter()
    frame_total = 0
    for row in rows:
        episode_id = row.get("episode_id", "")
        try:
            missing = [key for key in FIELDS if not row.get(key)]
            if missing:
                raise ValueError(f"empty fields: {missing}")
            if episode_id in seen:
                raise ValueError("duplicate episode ID")
            seen.add(episode_id)
            if row["split"] not in ("train", "holdout", "excluded"):
                raise ValueError("split must be train, holdout, or excluded")
            if row["active_arm"] not in ("left", "right"):
                raise ValueError("active_arm must be left or right")
            if row["quality_pass"].lower() not in ("true", "false"):
                raise ValueError("quality_pass must be true or false")
            if row["split"] != "excluded" and (row["task_outcome"] != "success" or row["quality_pass"].lower() != "true"):
                raise ValueError("train/holdout must contain approved successes")
            if row["action_schema_version"] != "station_lite_pose14_v1":
                raise ValueError("unsupported action schema")
            if abs(float(row["nominal_fps"]) - 30) > 1e-6:
                raise ValueError("expected 30 fps")
            if set(row["camera_keys"].split(";")) != set(CAMERAS):
                raise ValueError("camera keys mismatch")
            rel = Path(row["relative_path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("relative_path must stay under raw root")
            episode = args.raw / rel
            payload = json.loads((episode / "data.json").read_text())
            frames = payload["data"]
            if len(frames) != int(row["frame_count"]):
                raise ValueError("frame_count differs from data.json")
            if payload.get("text", {}).get("goal", "").strip() != row["task_text"]:
                raise ValueError("task_text differs from data.json text.goal")
            for frame in (frames if args.deep else (frames[0], frames[-1])):
                check_frame(frame, episode, args.deep)
            counts[row["split"]] += 1
            frame_total += len(frames)
        except Exception as exc:
            errors.append(f"{episode_id or '<missing ID>'}: {type(exc).__name__}: {exc}")
    datasets = {}
    if args.train_dataset:
        datasets["train"] = audit_dataset(args.train_dataset, rows, "train", errors)
    if args.holdout_dataset:
        datasets["holdout"] = audit_dataset(args.holdout_dataset, rows, "holdout", errors)
    report = {"manifest": str(args.manifest), "raw_root": str(args.raw), "episodes": len(rows), "frames": frame_total, "split_counts": dict(counts), "datasets": datasets, "deep_images": args.deep, "errors": errors, "passed": not errors}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("migrate", help="Create a versioned manifest for the 2026-10-27 data")
    m.add_argument("--legacy", type=Path, default=LEGACY)
    m.add_argument("--split", type=Path, default=SPLIT)
    m.add_argument("--raw", type=Path, default=RAW)
    m.add_argument("--output", type=Path, default=Path(__file__).parent / "woodblock_20261027_v1/manifest.csv")
    m.add_argument("--force", action="store_true")
    a = sub.add_parser("audit", help="Check raw episodes and optional LeRobot datasets")
    a.add_argument("--manifest", type=Path, default=Path(__file__).parent / "woodblock_20261027_v1/manifest.csv")
    a.add_argument("--raw", type=Path, default=RAW)
    a.add_argument("--train-dataset", type=Path, default=DATASETS / "block_pick_place_20261027_train35")
    a.add_argument("--holdout-dataset", type=Path, default=DATASETS / "block_pick_place_20261027_holdout10")
    a.add_argument("--raw-only", action="store_true", help="Skip LeRobot dataset checks before initial conversion")
    a.add_argument("--deep", action="store_true", help="Decode every camera image, not just check paths")
    a.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "audit" and args.raw_only:
        args.train_dataset = None
        args.holdout_dataset = None
    return migrate(args) if args.command == "migrate" else audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
