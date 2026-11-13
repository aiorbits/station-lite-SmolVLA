#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One live SmolVLA observation and, optionally, one right-arm-only policy chunk.

The DualTouchRobot constructor and final go_home() still move both arms. This
script never sends a policy trajectory to the left arm.
"""

import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/lumos/startouch_sdk/interface_py")
sys.path.insert(0, "/home/lumos/lumosteleop/.app/runtime")

SCHEMA = "station_lite_smolvla_shadow_v1"
TASK = "Pick up the block and place it in the target zone."
JOINT_LOW = np.array([-2.824, -0.366, -3.193, -1.710, -1.710, -2.856])
JOINT_UP = np.array([2.824, 3.508, 0.000, 1.710, 1.710, 2.856])
GRIPPER_MAX_M = 0.085

EXPECTED_CHECKPOINT = (
    "/home/hans/cc/ckpt/smolvla_woodblock_train35_resume10k_to20k/"
    "checkpoints/020000/pretrained_model"
)
MAX_JOINT_STEP_RAD = 0.25


def enc_jpeg_b64(img):
    if isinstance(img, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(img)).decode("ascii")
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def wait_frame(getter, timeout=6.0):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        frame, _ = getter()
        if frame is not None:
            return frame
        time.sleep(0.05)
    raise RuntimeError("no camera frame (timeout)")


def solve_with_retreat(arm, pos, euler, seed):
    q, ok = arm.solve_ik_euler(list(pos), list(euler), list(seed))
    if ok:
        return np.asarray(q, dtype=float).reshape(-1), True, 1.0
    try:
        cur_pos, cur_euler = arm.get_ee_pose_euler()
        cur_pos = np.asarray(cur_pos, dtype=float).reshape(3)
        cur_euler = np.asarray(cur_euler, dtype=float).reshape(3)
        tpos = np.asarray(pos, dtype=float).reshape(3)
        teul = np.asarray(euler, dtype=float).reshape(3)
        delta = (teul - cur_euler + np.pi) % (2 * np.pi) - np.pi
        for alpha in (0.7, 0.4, 0.15):
            p = cur_pos + alpha * (tpos - cur_pos)
            e = cur_euler + alpha * delta
            q2, ok2 = arm.solve_ik_euler(list(p), list(e), list(seed))
            if ok2:
                return np.asarray(q2, dtype=float).reshape(-1), True, alpha
    except Exception:
        pass
    return np.asarray(seed, dtype=float).reshape(-1), False, 0.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--img-host", default="127.0.0.1")
    parser.add_argument("--expected-checkpoint", default=EXPECTED_CHECKPOINT)
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--exec-steps", type=int, default=15)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument(
        "--max-soft-clip-rad", type=float, default=0.0,
        help="Allow bounded clipping to inner joint limits; default 0 rejects any clip",
    )
    parser.add_argument("--switch-delay", type=float, default=0.05)
    parser.add_argument(
        "--snapshot-dir", default="",
        help="Save the exact observation and prediction for an IK-only diagnosis",
    )
    parser.add_argument(
        "--send-right-chunk", action="store_true",
        help="Actually command the right arm; without this flag only check IK",
    )
    return parser.parse_args()


def check_right_targets(arm, actions, margin, solve_with_retreat, joint_low,
                        joint_up, gripper_max_m, max_soft_clip_rad=0.0,
                        max_initial_retreat_targets=0,
                        max_open_gripper_overshoot_m=0.0,
                        max_close_gripper_undershoot_m=0.0):
    """Check every target before sending any policy command to either arm."""
    lo = np.asarray(joint_low, dtype=float) + margin
    hi = np.asarray(joint_up, dtype=float) - margin
    seed = np.asarray(arm.get_joint_positions(), dtype=float).reshape(-1)
    if seed.shape != (6,) or not np.isfinite(seed).all():
        raise ValueError("Invalid current right-arm joint state")

    waypoints, grippers = [], []
    failures = 0
    clipped_grippers = 0
    soft_clipped_targets = 0
    initial_retreat_targets = 0
    action_width = np.asarray(actions).shape[-1]
    if action_width not in (7, 14):
        raise ValueError(f"Expected right-arm action width 7 or 14, got {action_width}")
    pose_slice = slice(0, 6) if action_width == 7 else slice(7, 13)
    gripper_index = 6 if action_width == 7 else 13
    for index, action in enumerate(actions):
        q, solver_ok, alpha = solve_with_retreat(
            arm, action[pose_slice][:3], action[pose_slice][3:], seed)
        q = np.asarray(q, dtype=float).reshape(-1)
        finite = q.shape == (6,) and bool(np.isfinite(q).all())
        # The vendor client clips a valid IK solution to its inner (soft) limits.
        # Keep that behavior opt-in and bounded: commands always remain inside
        # the configured soft range, and a large correction still rejects the chunk.
        clipped_q = np.clip(q, lo, hi) if finite else q
        clip_rad = float(np.max(np.abs(clipped_q - q))) if finite else float("inf")
        limits_ok = (finite and clip_rad <= max_soft_clip_rad
                     and bool(np.all(clipped_q >= joint_low)
                              and np.all(clipped_q <= joint_up)))
        step_rad = float(np.max(np.abs(clipped_q - seed))) if finite else float("inf")
        step_ok = step_rad <= MAX_JOINT_STEP_RAD
        # The vendor client uses a retreated IK solution when the exact pose
        # is unreachable. Permit this only for an explicitly bounded prefix;
        # every such waypoint must still pass the limit and step checks.
        initial_retreat = bool(solver_ok and 0.4 <= alpha < 1
                               and index < max_initial_retreat_targets)
        passed = bool(solver_ok and (alpha == 1.0 or initial_retreat)
                      and limits_ok and step_ok)
        raw_gripper_m = float(action[gripper_index])
        gripper_m = raw_gripper_m
        if (np.isfinite(raw_gripper_m)
                and -max_close_gripper_undershoot_m <= raw_gripper_m < 0.0):
            # The physical gripper cannot go below fully closed. Keep this
            # saturation opt-in and bounded; larger undershoots still reject.
            gripper_m = 0.0
        if (np.isfinite(raw_gripper_m)
                and gripper_max_m < raw_gripper_m
                <= gripper_max_m + max_open_gripper_overshoot_m):
            # A small overshoot while commanding fully open is saturated to
            # the largest open target present in the training demonstrations.
            gripper_m = min(0.080, gripper_max_m)
        gripper_ok = bool(np.isfinite(gripper_m) and 0.0 <= gripper_m <= gripper_max_m)
        if not passed or not gripper_ok:
            outside = np.flatnonzero((q < lo) | (q > hi)).tolist() if finite else []
            print(f"  [RIGHT FAIL {index}] solver_ok={bool(solver_ok)} "
                  f"alpha={alpha} limits_ok={limits_ok} outside_joint_idx={outside} "
                  f"soft_clip_rad={clip_rad:.4f} step_rad={step_rad:.4f} step_ok={step_ok} "
                  f"gripper_ok={gripper_ok} gripper_raw_m={raw_gripper_m:.4f} "
                  f"q={np.round(q, 4).tolist()}", flush=True)
            failures += 1
            continue
        if gripper_m != raw_gripper_m:
            clipped_grippers += 1
            print(f"  [RIGHT GRIPPER CLIP {index}] "
                  f"raw_m={raw_gripper_m:.4f} sent_m={gripper_m:.4f}", flush=True)
        if clip_rad > 1e-9:
            soft_clipped_targets += 1
            changed = np.flatnonzero(np.abs(clipped_q - q) > 1e-9).tolist()
            print(f"  [RIGHT SOFT CLIP {index}] joints={changed} "
                  f"max_delta_rad={clip_rad:.4f} "
                  f"q_raw={np.round(q, 4).tolist()} "
                  f"q_safe={np.round(clipped_q, 4).tolist()}", flush=True)
        if initial_retreat:
            initial_retreat_targets += 1
            print(f"  [RIGHT INITIAL RETREAT {index}] alpha={alpha} "
                  f"soft_clip_rad={clip_rad:.4f} step_rad={step_rad:.4f}",
                  flush=True)
        q = clipped_q
        seed = q
        waypoints.append([float(x) for x in q])
        grippers.append(float(gripper_m / gripper_max_m))

    print(f"[RIGHT PREFLIGHT] checked={len(actions)} failed={failures} "
          f"soft_clipped_targets={soft_clipped_targets} "
          f"initial_retreat_targets={initial_retreat_targets} "
          f"gripper_clipped={clipped_grippers}", flush=True)
    if failures:
        raise ValueError(f"Rejected right-arm chunk: {failures} target(s) failed")
    return waypoints, grippers


def main():
    args = parse_args()
    if not (0 < args.speed <= 1 and args.rate > 0 and 2 <= args.exec_steps <= 50
            and 0 <= args.margin < 0.1
            and 0 <= args.max_soft_clip_rad <= 0.05
            and args.switch_delay >= 0):
        raise SystemExit("Invalid speed/rate/exec-steps/margin/switch-delay")

    # SDK imports occur only after argument validation.
    from teleimager.image_client import ImageClient  # noqa: E402
    from xr_teleoperate2.robot_control.robot_dual_touch import DualTouchRobot  # noqa: E402

    server = args.server.rstrip("/")
    with urllib.request.urlopen(server + "/health", timeout=5) as response:
        health = json.load(response)
    if (health.get("schema") != SCHEMA or health.get("checkpoint") != args.expected_checkpoint
            or health.get("chunk_size") != 50 or health.get("action_dim") != 14
            or health.get("task") != TASK or health.get("hardware_connected") is not False):
        raise SystemExit(f"Unexpected inference service: {health}")
    print(f"[MODEL] {health['checkpoint']}", flush=True)

    duration = args.exec_steps / args.rate / args.speed
    robot = None
    image_client = None
    try:
        print("[ROBOT] DualTouchRobot initializes and homes BOTH arms", flush=True)
        robot = DualTouchRobot()
        time.sleep(2.0)
        image_client = ImageClient(host=args.img_host, auto_subscribe=False)
        image_client.start_subscriptions()

        images = {
            "observation.images.cam_high": wait_frame(image_client.get_head_frame),
            "observation.images.cam_left_wrist": wait_frame(image_client.get_left_wrist_frame),
            "observation.images.cam_right_wrist": wait_frame(image_client.get_right_wrist_frame),
        }
        left_q = np.asarray(robot.left_arm.get_joint_positions(), dtype=float).reshape(-1)
        right_q = np.asarray(robot.right_arm.get_joint_positions(), dtype=float).reshape(-1)
        state = np.concatenate([
            left_q, [float(robot.left_arm.get_gripper_distance())],
            right_q, [float(robot.right_arm.get_gripper_distance())],
        ])
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Invalid 14-dimensional robot observation")
        right_q = state[7:13]
        soft_low = JOINT_LOW + args.margin
        soft_high = JOINT_UP - args.margin
        hard_outside = np.flatnonzero((right_q < JOINT_LOW) | (right_q > JOINT_UP)).tolist()
        soft_outside = np.flatnonzero((right_q < soft_low) | (right_q > soft_high)).tolist()
        print(f"[OBSERVED RIGHT] q={np.round(right_q, 5).tolist()} "
              f"gripper={state[13]:.4f}m hard_outside={hard_outside} "
              f"soft_outside={soft_outside}", flush=True)
        observed_ee = None
        try:
            ee_pos, ee_euler = robot.right_arm.get_ee_pose_euler()
            ee_pos = np.asarray(ee_pos, dtype=float).reshape(3)
            ee_euler = np.asarray(ee_euler, dtype=float).reshape(3)
            observed_ee = {"xyz": ee_pos.tolist(), "euler": ee_euler.tolist()}
            print(f"[OBSERVED RIGHT EE] xyz={np.round(ee_pos, 5).tolist()} "
                  f"euler={np.round(ee_euler, 5).tolist()}", flush=True)
        except Exception as exc:
            print(f"[OBSERVED RIGHT EE] unavailable: {exc!r}", flush=True)
        payload = {
            "schema": SCHEMA, "task": TASK, "session_id": "right-single-chunk", "seq": 0,
            "state": state.tolist(),
            "images_jpeg_base64": {key: enc_jpeg_b64(value)
                                   for key, value in images.items()},
        }
        snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
        if snapshot_dir is not None:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            (snapshot_dir / "observation.json").write_text(
                json.dumps({"checkpoint": health["checkpoint"], "state": state.tolist(),
                            "right_ee": observed_ee, "margin": args.margin}, indent=2),
                encoding="utf-8",
            )
            for camera_key, encoded in payload["images_jpeg_base64"].items():
                filename = camera_key.removeprefix("observation.images.") + ".jpg"
                (snapshot_dir / filename).write_bytes(base64.b64decode(encoded))
            print(f"[SNAPSHOT] observation and camera images: {snapshot_dir}", flush=True)
        request = urllib.request.Request(
            server + "/predict", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if result.get("schema") != SCHEMA or result.get("seq") != 0:
            raise ValueError("SmolVLA response identity mismatch")
        actions = np.asarray(result["actions"], dtype=float)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid SmolVLA output: shape={actions.shape}")
        if snapshot_dir is not None:
            np.save(snapshot_dir / "raw_actions.npy", actions)
            print(f"[SNAPSHOT] raw actions: {snapshot_dir / 'raw_actions.npy'}", flush=True)
        if observed_ee is not None:
            target_gap_mm = float(np.linalg.norm(actions[0, 7:10] - ee_pos) * 1000)
            euler_gap = (actions[0, 10:13] - ee_euler + np.pi) % (2 * np.pi) - np.pi
            print(f"[FIRST TARGET GAP] position={target_gap_mm:.1f}mm "
                  f"euler_components_deg={np.round(np.degrees(euler_gap), 2).tolist()}",
                  flush=True)
            if args.send_right_chunk and (target_gap_mm > 20.0 or
                                          np.max(np.abs(np.degrees(euler_gap))) > 12.0):
                raise ValueError("Rejected first target: too far from measured right end effector")
        elif args.send_right_chunk:
            raise ValueError("Cannot send a chunk without measured right end-effector pose")
        for horizon in (0, 9, 14, 24, 34, 49):
            print(f"[HORIZON {horizon:02d}] "
                  f"right_xyz={np.round(actions[horizon, 7:10], 4).tolist()} "
                  f"right_gripper={actions[horizon, 13]:.4f}m", flush=True)
        first_close = np.flatnonzero(actions[:, 13] < 0.04)
        print(f"[GRIPPER 50] min={np.min(actions[:, 13]):.4f}m "
              f"max={np.max(actions[:, 13]):.4f}m "
              f"first_below_0.04={int(first_close[0]) if first_close.size else 'none'}",
              flush=True)
        segment = actions[:args.exec_steps]
        print(f"[PREDICT] shape={actions.shape} "
              f"right_xyz0={np.round(segment[0, 7:10], 4).tolist()} "
              f"duration={duration:.3f}s margin={args.margin}", flush=True)

        waypoints, grippers = check_right_targets(
            robot.right_arm, segment, args.margin, solve_with_retreat,
            JOINT_LOW, JOINT_UP, GRIPPER_MAX_M, args.max_soft_clip_rad)
        if not args.send_right_chunk:
            print("[IK ONLY] No policy command sent", flush=True)
            return

        right_before = np.asarray(
            robot.right_arm.get_joint_positions(), dtype=float).reshape(-1)
        print("[SEND] one RIGHT policy chunk; no LEFT policy command", flush=True)
        robot.right_arm.update_joint_waypoint_chunk_with_gripper(
            waypoints, grippers, time_sec=duration,
            switch_delay_sec=args.switch_delay)
        time.sleep(duration + args.switch_delay + 0.1)
        right_after = np.asarray(
            robot.right_arm.get_joint_positions(), dtype=float).reshape(-1)
        if right_after.shape == (6,) and np.isfinite(right_after).all():
            last_target = np.asarray(waypoints[-1], dtype=float)
            print(f"[RIGHT TRACK] before={np.round(right_before, 4).tolist()} "
                  f"after={np.round(right_after, 4).tolist()} "
                  f"last_target={np.round(last_target, 4).tolist()} "
                  f"max_last_error={np.max(np.abs(right_after - last_target)):.4f} rad",
                  flush=True)
        try:
            ee_position, _ = robot.right_arm.get_ee_pose_euler()
            ee_position = np.asarray(ee_position, dtype=float).reshape(3)
            print(f"[RIGHT EE] actual_xyz={np.round(ee_position, 4).tolist()} "
                  f"predicted_last_xyz={np.round(segment[-1, 7:10], 4).tolist()}",
                  flush=True)
        except Exception as exc:
            print(f"[RIGHT EE] measurement unavailable: {exc!r}", flush=True)
        print("[DONE] right chunk duration elapsed", flush=True)
    finally:
        if robot is not None:
            try:
                print("[HOME] go_home() moves BOTH arms", flush=True)
                robot.go_home()
                time.sleep(4.0)
            finally:
                try:
                    if image_client is not None:
                        image_client.close()
                finally:
                    cleanup = getattr(robot, "cleanup", None)
                    if callable(cleanup):
                        cleanup()
                    else:
                        print("[CLOSE] DualTouchRobot has no cleanup() method; "
                              "process is exiting after go_home()", flush=True)


if __name__ == "__main__":
    main()
