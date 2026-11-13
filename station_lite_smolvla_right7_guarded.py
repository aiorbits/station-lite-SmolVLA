#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guarded right7 SmolVLA rolling chunks; DualTouchRobot homes both arms.

Every proposed handoff is compared with the measured right joints and the
previously installed trajectory. A rejected handoff is never sent.
"""

import argparse
import base64
import json
import math
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path

import numpy as np

from station_lite_smolvla_single_chunk import (
    JOINT_LOW, JOINT_UP, GRIPPER_MAX_M, check_right_targets,
    enc_jpeg_b64, solve_with_retreat, wait_frame,
)

SCHEMA = "station_lite_smolvla_right7_v1"
REQUEST_SCHEMA = "station_lite_smolvla_right7_request_v1"
TASK = "Pick up the red wooden block from the white grid and place it on the brown target area."
EXPECTED_CHECKPOINT = (
    "/home/hans/cc/ckpt/smolvla_woodblock_right7_train35_v1_10000/"
    "checkpoints/010000/pretrained_model"
)


def make_initial_prefix(current, target, steps, *, max_position_m=.030):
    """Bounded single-chunk entry path; every generated target still needs IK."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != (7,) or target.shape != (7,) or not 3 <= steps <= 10:
        raise ValueError("Invalid initial bridge inputs")
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite initial bridge inputs")
    delta = target - current
    delta[3:6] = (delta[3:6] + np.pi) % (2 * np.pi) - np.pi
    pos_gap_mm = float(1000 * np.linalg.norm(delta[:3]))
    rot_gap_deg = float(np.degrees(np.max(np.abs(delta[3:6]))))
    grip_gap_mm = float(1000 * abs(delta[6]))
    if (pos_gap_mm > 1000 * max_position_m
            or rot_gap_deg > 12.0
            or grip_gap_mm > 15.0):
        raise ValueError(
            f"Initial bridge exceeds envelope: pos={pos_gap_mm:.1f}mm (max {1000*max_position_m:g}mm), "
            f"rot={rot_gap_deg:.1f}deg (max 12deg), grip={grip_gap_mm:.1f}mm (max 15mm)"
        )
    # Do not command the measured start itself: Home can be outside soft margins.
    # These intermediate targets receive the same IK/soft-limit checks as policy targets.
    return current + np.arange(1, steps)[:, None] / steps * delta


def select_initial_prefix(arm, prefix, margin):
    """Omit only leading bridge candidates outside soft but inside physical limits.

    The measured start may be outside the inset. The controller starts at that
    measured state; all commanded waypoints must pass the original guards.
    Policy targets are never removed here. The normal first-target distance and
    joint-step guards still apply to the chosen first command.
    """
    seed = np.asarray(arm.get_joint_positions(), dtype=float).reshape(6)
    for index, candidate in enumerate(prefix):
        q, ok = arm.solve_ik_euler(candidate[:3].tolist(), candidate[3:6].tolist(), seed.tolist())
        q = np.asarray(q, dtype=float).reshape(-1)
        if (not ok or q.shape != (6,) or not np.isfinite(q).all()
                or np.any(q < JOINT_LOW) or np.any(q > JOINT_UP)):
            raise ValueError("Initial bridge candidate failed direct IK or physical limits")
        if np.all(q >= JOINT_LOW+margin) and np.all(q <= JOINT_UP-margin):
            return prefix[index:].copy(), index
    return prefix[:0].copy(), len(prefix)


def initial_state_drift(observed, measured):
    """State-stability check for the single-chunk entry experiment, radians."""
    observed = np.asarray(observed, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if (observed.shape != (6,) or measured.shape != (6,)
            or not np.isfinite(observed).all() or not np.isfinite(measured).all()):
        raise ValueError("Invalid initial state stability measurement")
    return float(np.max(np.abs(measured-observed)))


def right_pose(a):
    a = np.asarray(a)
    return a[..., :6] if a.shape[-1] == 7 else a[..., 7:13]


def right_xyz(a):
    return right_pose(a)[..., :3]


def right_gripper(a):
    a = np.asarray(a)
    return a[..., 6] if a.shape[-1] == 7 else a[..., 13]

from joint_trace import JointTraceSession


def blend_head(old_next, new_head, mode="cosine"):
    overlap = int(min(len(old_next), len(new_head)))
    out = np.asarray(new_head[:overlap], dtype=float).copy()
    width = out.shape[-1]
    pose_slice = slice(0, 6) if width == 7 else slice(7, 13)
    grip_indices = (6,) if width == 7 else (6, 13)
    for index in range(overlap):
        t = (index + 1) / float(overlap)
        if mode == "linear":
            alpha = t
        elif mode == "minjerk":
            alpha = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
        else:
            alpha = 0.5 * (1.0 - math.cos(math.pi * t))
        old = np.asarray(old_next[index], dtype=float)
        new = out[index].copy()
        for dim in range(width):
            if dim in grip_indices:
                continue
            if dim in range(pose_slice.start + 3, pose_slice.stop):
                delta = (new[dim] - old[dim] + math.pi) % (2 * math.pi) - math.pi
                out[index, dim] = old[dim] + alpha * delta
            else:
                out[index, dim] = (1 - alpha) * old[dim] + alpha * new[dim]
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--img-host", default="127.0.0.1")
    parser.add_argument("--expected-checkpoint", default=EXPECTED_CHECKPOINT)
    parser.add_argument("--expected-seed-mode", choices=("sequence", "fixed"),
                        default=None,
                        help="Require the inference service to report this sampling mode")
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--initial-bridge-steps", type=int, default=0,
                        help="Opt-in 3..10 subdivisions of a <=30mm initial entry; single loop only")
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--exec-steps", type=int, default=15)
    parser.add_argument("--switch-after", type=int, default=10)
    parser.add_argument("--blend-overlap", type=int, default=5)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument(
        "--max-soft-clip-rad", type=float, default=0.0,
        help="Allow bounded clipping to inner joint limits; default 0 rejects any clip",
    )
    parser.add_argument(
        "--max-initial-retreat-targets", type=int, default=0,
        help="Only in chunk 0, permit up to this many leading retreated IK "
             "solutions after limit and joint-step checks (maximum 3)",
    )
    parser.add_argument("--switch-delay", type=float, default=0.05)
    parser.add_argument("--max-switch-gap-mm", type=float, default=20.0)
    parser.add_argument("--max-initial-gap-mm", type=float, default=25.0,
                        help="Maximum allowed initial Cartesian position gap from Home to the first commanded waypoint (mm)")
    parser.add_argument("--max-adaptive-gap-mm", type=float, default=25.0,
                        help="Candidate position-gap limit for phase-alignment diagnosis")
    parser.add_argument(
        "--early-max-adaptive-gap-mm", type=float, default=None,
        help="Optional position-gap limit when installing chunks 1-3 (the first "
             "four chunks are numbered 0-3); chunk 4 onward uses "
             "--max-adaptive-gap-mm. Capped at 35 mm",
    )
    parser.add_argument("--max-tracking-joint-rad", type=float, default=0.12)
    parser.add_argument("--max-state-drift-rad", type=float, default=0.12)
    parser.add_argument(
        "--max-observation-tracking-rad", type=float, default=0.06,
        help="Adaptive handoff: allow a 0.05-0.06 rad observation mismatch only "
             "when the fresh joint reading tracks within 0.02 rad and state "
             "changed by no more than 0.06 rad during inference, with at most "
             "three waypoint phases between readings",
    )
    parser.add_argument("--max-phase-step-error", type=int, default=3)
    parser.add_argument("--max-rotation-gap-deg", type=float, default=12.0)
    parser.add_argument("--max-gripper-gap-mm", type=float, default=15.0)
    parser.add_argument(
        "--max-open-gripper-overshoot-mm", type=float, default=0.0,
        help="Opt in to saturating open targets just above 85 mm to the "
             "training-data open target of 80 mm (maximum allowance: 2 mm)",
    )
    parser.add_argument(
        "--max-close-gripper-undershoot-mm", type=float, default=0.0,
        help="Opt in to saturating negative close targets to 0 mm "
             "(maximum allowance: 20 mm)",
    )
    parser.add_argument("--max-gripper-step-mm", type=float, default=15.0,
                        help="Largest commanded gripper change between adjacent waypoints; "
                             "opt-in up to 80 mm, as used by the ACT vendor client")
    parser.add_argument(
        "--commit-grasp-chunk", action="store_true",
        help="If a checked chunk closes after its next observation point, finish "
             "that chunk before observing again; restart only after a fresh "
             "state and a guarded first waypoint. Requires an adaptive mode",
    )
    parser.add_argument("--max-observation-age-ms", type=float, default=500.0)
    parser.add_argument("--snapshot-dir", default="",
                        help="Save the exact camera images and robot state sent to the model")
    parser.add_argument("--joint-trace-dir", default="",
                        help="Continuously record right-joint state and per-loop controller plans (1–5 loops)")
    parser.add_argument("--joint-trace-hz", type=float, default=30.0,
                        help="Read-only measured joint polling rate for --joint-trace-dir (5–60 Hz)")
    parser.add_argument(
        "--diagnose-switch-only", action="store_true",
        help="With two loops, send the first chunk, inspect the second, but do not send it",
    )
    parser.add_argument(
        "--send-right-chunks", action="store_true",
        help="Actually command right-arm policy chunks; default checks one chunk only",
    )
    parser.add_argument(
        "--probe-steps", type=int, nargs="+", default=None,
        help="Send only the first chunk; collect predictions at these nominal steps "
             "without sending more chunks (example: 25 32 39 45)",
    )
    parser.add_argument(
        "--adaptive-two-chunk", action="store_true",
        help="Execute at most two chunks; align the second to measured right joints "
             "and stop if any adaptive handoff check fails",
    )
    parser.add_argument(
        "--adaptive-five-chunk", action="store_true",
        help="Execute at most five individually guarded right-arm chunks; "
             "observe when measured joints reach the planned phase, stopping "
             "before the reserved handoff window expires",
    )
    parser.add_argument(
        "--adaptive-eight-chunk", action="store_true",
        help="Execute exactly eight guarded right-arm chunks using the same "
             "measured-phase observation and reserved handoff window as "
             "--adaptive-five-chunk",
    )
    parser.add_argument(
        "--adaptive-ten-chunk", action="store_true",
        help="Execute at most ten individually guarded right-arm chunks; "
             "schedule each next observation using the measured clock/phase gap",
    )
    parser.add_argument(
        "--adaptive-fifteen-chunk", action="store_true",
        help="Execute at most fifteen individually guarded right-arm chunks; "
             "observe later chunks at measured joint phase; stop if any "
             "IK, tracking, or position-gap guard rejects",
    )
    parser.add_argument(
        "--adaptive-refine", action="store_true",
        help="Insert intermediate Cartesian targets in later chunks when "
             "adjacent IK joint targets exceed 0.07 rad; recheck the whole chunk",
    )
    parser.add_argument(
        "--horizon-aware-alignment", action="store_true",
        help="For nonfinal adaptive chunks, search only phase matches that leave "
             "at least 17 raw actions for the next guarded handoff",
    )
    parser.add_argument(
        "--adaptive-second-preflight-only", action="store_true",
        help="Send the FIRST chunk, then validate but do not send the second chunk",
    )
    parser.add_argument(
        "--final-probe", action="store_true",
        help="After all requested chunks finish, save a fresh observation and "
             "prediction without sending another robot command",
    )
    parser.add_argument(
        "--ik-preflight-only", action="store_true",
        help="Connect only the right-arm SDK and run native right7 IK plus dry-run planning; never home or publish motion",
    )
    parser.add_argument("--right-can-interface", default="can1",
                        help="Right arm CAN interface for the no-motion preflight")
    parser.add_argument("--second-check-after-settle", action="store_true",
                        help="Finish chunk 0 and check chunk 1 from a settled state; never send chunk 1")
    parser.add_argument("--settled-chunks", action="store_true",
                        help="Execute 2..5 chunks with a settled observation at every handoff")
    parser.add_argument("--settled-bridge-max-mm", type=float, default=30.,
                        help="Explicit experimental settled bridge envelope, 30..40 mm")
    parser.add_argument("--placement-bridge-max-mm", type=float, default=50.0,
                        help="Placement bridge envelope from grasp pose to transport chunk (default 50.0mm, range 30..60mm)")
    parser.add_argument("--condition-final-grasp", action="store_true",
                        help="Opt-in bounded closure conditioning on chunk 4 only")
    parser.add_argument("--hold-grasp-pose", action="store_true",
                        help="With --condition-final-grasp, hold chunk 3's settled 6D pose while executing bounded closure")
    parser.add_argument("--place-after-grasp", action="store_true",
                        help="Continue after conditioned grasp to transport and place the block in the target area (loops 6 or 7)")
    parser.add_argument("--task", "--prompt", dest="task", default=TASK,
                        help=f"Language task instruction sent to SmolVLA policy (default: '{TASK}')")
    parser.add_argument("--target-color", choices=["auto", "red", "green", "blue", "cyan", "yellow"], default="auto",
                        help="Target block color to grasp (default: auto, inferred from task/prompt)")
    parser.add_argument("--phase-speeds", action="store_true",
                        help="Enable phase-specific speeds and durations: approach=0.20, "
                             "grasp_duration=2.5s, transport=0.25, place=0.24")
    parser.add_argument("--approach-speed", type=float, default=0.20,
                        help="Trajectory playback speed multiplier for approach chunks (default: 0.20)")
    parser.add_argument("--transport-speed", type=float, default=0.25,
                        help="Trajectory playback speed multiplier for transport chunks (default: 0.25)")
    parser.add_argument("--place-speed", type=float, default=0.24,
                        help="Trajectory playback speed multiplier for place/release chunks (default: 0.24)")
    parser.add_argument("--grasp-duration-s", type=float, default=2.5,
                        help="Execution duration in seconds for conditioned grasp chunk when hold_pose is True (default: 2.5s)")
    parser.add_argument("--smooth-waypoints", dest="smooth_waypoints", action="store_true", default=True,
                        help="Enable 3-point zero-phase moving average smoothing on joint waypoints to eliminate arm vibration (default: True)")
    parser.add_argument("--no-smooth-waypoints", dest="smooth_waypoints", action="store_false",
                        help="Disable joint waypoint smoothing")
    parser.add_argument("--waypoint-smooth-alpha", type=float, default=0.20,
                        help="Alpha weight for 3-point waypoint smoothing w[i]=alpha*w[i-1]+(1-2alpha)*w[i]+alpha*w[i+1] (default: 0.20)")
    parser.add_argument("--waypoint-smooth-passes", type=int, default=1,
                        help="Number of smoothing passes for joint waypoints (default: 1)")
    parser.add_argument("--wrist-obb-refine", dest="wrist_obb_refine", action="store_true", default=True,
                        help="Enable wrist-camera OBB visual centering refinement during approach (default: True)")
    parser.add_argument("--no-wrist-obb-refine", dest="wrist_obb_refine", action="store_false",
                        help="Disable wrist-camera OBB visual centering refinement")
    parser.add_argument("--high-cam-guidance", dest="high_cam_guidance", action="store_true", default=True,
                        help="Enable cam_high global grid target selection and coarse approach guidance (default: True)")
    parser.add_argument("--no-high-cam-guidance", dest="high_cam_guidance", action="store_false",
                        help="Disable cam_high global grid target selection")
    parser.add_argument("--wrist-align-tol-px", type=float, default=20.0,
                        help="Tracking error tolerance in pixels for wrist OBB alignment acceptance (default: 20.0)")
    parser.add_argument("--strict-wrist-alignment", dest="strict_wrist_alignment", action="store_true", default=False,
                        help="Strictly abort grasp if wrist alignment check fails (default: False, allows model closure)")
    parser.add_argument("--max-expired-replans", type=int, default=1,
                        help="Maximum times a settled chunk may re-observe and re-plan if observation_expired is the only post-plan guard failure (default: 1)")
    parser.add_argument("--max-grasp-retries", type=int, default=1,
                        help="Maximum times a confirmed empty grasp may retreat to high hover, re-observe, and retry (default: 1)")
    parser.add_argument("--hover-z-m", type=float, default=0.182,
                        help="Hover clearance height in meters during approach phase (default: 0.182, candidate: 0.200)")
    return parser.parse_args()


def read_observation(robot, image_client, wait_frame, enc_jpeg_b64, schema,
                     session_id, sequence, timing=None, task=TASK):
    images = {}
    cameras = (
        ("observation.images.cam_high", image_client.get_head_frame),
        ("observation.images.cam_right_wrist", image_client.get_right_wrist_frame),
    )
    if timing is not None:
        timing["observation_start"] = time.monotonic()
        timing["camera_received"] = {}
    for key, get_frame in cameras:
        images[key] = wait_frame(get_frame)
        if timing is not None:
            timing["camera_received"][key] = time.monotonic()
    arm = getattr(robot, "right_arm", robot)
    right_q = np.asarray(arm.get_joint_positions(), dtype=float).reshape(-1)
    state = np.concatenate([right_q, [float(arm.get_gripper_distance())]])
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("Invalid 7-dimensional right-arm observation")
    if timing is not None:
        timing["state_received"] = time.monotonic()
    return {
        "schema": REQUEST_SCHEMA, "task": task, "session_id": session_id, "seq": sequence,
        "state": state.tolist(),
        "images_jpeg_base64": {key: enc_jpeg_b64(value)
                               for key, value in images.items()},
    }


def evaluate_handoff(old_segment, old_waypoints, raw, blended, measured_q,
                     elapsed_s, step_s, switch_after, overlap, observation_age_ms,
                     args):
    """Return measured/commanded seam diagnostics and conservative rejection reasons."""
    old_q = np.asarray(old_waypoints, dtype=float)
    measured_q = np.asarray(measured_q, dtype=float)
    expected = int(np.clip(round((elapsed_s - args.switch_delay) / step_s),
                           0, len(old_q) - 1))
    q_errors = np.max(np.abs(old_q - measured_q), axis=1)
    global_nearest = int(np.argmin(q_errors))
    low = max(0, expected - args.max_phase_step_error)
    high = min(len(old_q), expected + args.max_phase_step_error + 1)
    local_nearest = low + int(np.argmin(q_errors[low:high]))
    tracking_rad = float(q_errors[local_nearest])
    old_pose = right_pose(old_segment[local_nearest])
    raw_pose = right_pose(raw[0])
    xyz_gap_mm = float(1000 * np.linalg.norm(old_pose[:3] - raw_pose[:3]))
    euler_delta = (raw_pose[3:] - old_pose[3:] + math.pi) % (2 * math.pi) - math.pi
    rotation_gap_deg = float(np.degrees(np.max(np.abs(euler_delta))))
    grip_gap_mm = float(1000 * abs(right_gripper(old_segment[local_nearest]) - right_gripper(raw[0])))
    old_tail = right_xyz(old_segment[switch_after:switch_after + overlap])
    blend_head = right_xyz(blended[:overlap])
    old_progress = old_tail[-1] - old_tail[0]
    blended_progress = blend_head[-1] - blend_head[0]
    old_distance = float(np.linalg.norm(old_progress))
    projected_blend_mm = (float(1000 * np.dot(blended_progress, old_progress)
                                / old_distance) if old_distance > 0 else 0.0)
    reasons = []
    if observation_age_ms > args.max_observation_age_ms:
        reasons.append("stale_observation")
    if abs(local_nearest - switch_after) > args.max_phase_step_error:
        reasons.append("switch_phase")
    if tracking_rad > args.max_tracking_joint_rad:
        reasons.append("joint_tracking")
    if xyz_gap_mm > args.max_switch_gap_mm:
        reasons.append("raw_position_gap")
    if rotation_gap_deg > args.max_rotation_gap_deg:
        reasons.append("raw_rotation_gap")
    if grip_gap_mm > args.max_gripper_gap_mm:
        reasons.append("raw_gripper_gap")
    if old_distance > 0.005 and projected_blend_mm < -2.0:
        reasons.append("blend_reversal")
    metrics = {
        "expected_idx": expected, "local_nearest_idx": local_nearest,
        "global_nearest_idx": global_nearest,
        "tracking_rad": tracking_rad,
        "global_tracking_rad": float(q_errors[global_nearest]),
        "raw_xyz_gap_mm": xyz_gap_mm,
        "raw_rotation_gap_deg": rotation_gap_deg,
        "raw_gripper_gap_mm": grip_gap_mm,
        "old_progress_mm": 1000 * old_distance,
        "blended_projected_progress_mm": projected_blend_mm,
        "observation_age_ms": observation_age_ms,
    }
    return metrics, reasons


def time_consistent_waypoint_index(errors, expected_index=None,
                                   window_steps=4, tie_rad=0.005):
    """Resolve repeated joint poses near the phase predicted by the clock."""
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError("Invalid waypoint tracking errors")
    if expected_index is None:
        return int(np.argmin(errors))
    center = int(np.clip(expected_index, 0, len(errors) - 1))
    start = max(0, center - window_steps)
    stop = min(len(errors), center + window_steps + 1)
    nearby = np.arange(start, stop)
    best_error = float(np.min(errors[nearby]))
    tied = nearby[errors[nearby] <= best_error + tie_rad]
    return int(min(tied, key=lambda i: (abs(int(i) - center), errors[i])))


def diagnose_phase_alignment(old_segment, old_waypoints, raw, observed_q,
                             measured_q, overlap, max_position_mm=25.0,
                             max_rotation_deg=12.0, max_tracking_rad=0.05,
                             min_remaining=15, max_gripper_mm=15.0,
                             expected_observation_index=None,
                             expected_current_index=None):
    """Find a possible phase match for review; never authorize a robot command.

    The observation and the post-inference joint read are matched separately.
    A raw action later in the prediction may fit the current robot pose better
    than action zero, but a Cartesian match alone is not enough to send it.
    """
    old_q = np.asarray(old_waypoints, dtype=float)
    old = np.asarray(old_segment, dtype=float)
    new = np.asarray(raw, dtype=float)
    obs_errors = np.max(np.abs(old_q - np.asarray(observed_q, dtype=float)), axis=1)
    now_errors = np.max(np.abs(old_q - np.asarray(measured_q, dtype=float)), axis=1)
    obs_index = time_consistent_waypoint_index(
        obs_errors, expected_observation_index)
    now_index = time_consistent_waypoint_index(
        now_errors, expected_current_index)
    available_old = len(old) - now_index
    actual_overlap = min(overlap, available_old)
    result = {
        "observation_old_index": obs_index,
        "observation_clock_index": expected_observation_index,
        "observation_joint_error_rad": float(obs_errors[obs_index]),
        "current_old_index": now_index,
        "current_clock_index": expected_current_index,
        "current_joint_error_rad": float(now_errors[now_index]),
        "old_remaining_steps": available_old,
        "overlap_steps": actual_overlap,
        "candidate_new_index": None,
        "position_gap_mm": None,
        "rotation_gap_deg": None,
        "gripper_gap_mm": None,
        "direction_cosine": None,
        "new_remaining_steps": None,
        "reason": None,
    }
    if now_errors[now_index] > max_tracking_rad:
        result["reason"] = "joint_tracking"
        return result
    if actual_overlap < 8:
        result["reason"] = "insufficient_old_overlap"
        return result

    old_pose = right_pose(old[now_index])
    old_motion = right_xyz(old[now_index + actual_overlap - 1]) - old_pose[:3]
    old_distance = np.linalg.norm(old_motion)
    best = None
    best_valid = None
    for new_index in range(len(new) - min_remaining + 1):
        if new_index + actual_overlap > len(new):
            continue
        position_mm = 1000 * np.linalg.norm(right_xyz(new[new_index]) - old_pose[:3])
        euler_delta = (right_pose(new[new_index])[3:] - old_pose[3:] + math.pi) % (2 * math.pi) - math.pi
        rotation_deg = float(np.degrees(np.max(np.abs(euler_delta))))
        gripper_mm = 1000 * abs(right_gripper(new[new_index]) - right_gripper(old[now_index]))
        new_motion = (right_xyz(new[new_index + actual_overlap - 1])
                      - right_xyz(new[new_index]))
        new_distance = np.linalg.norm(new_motion)
        direction = (float(np.dot(old_motion, new_motion) / (old_distance * new_distance))
                     if old_distance > 1e-6 and new_distance > 1e-6 else 0.0)
        candidate = (position_mm / max_position_mm
                     + rotation_deg / max_rotation_deg
                     + gripper_mm / max_gripper_mm
                     + max(0.0, 1.0 - direction), new_index,
                     position_mm, rotation_deg, gripper_mm, direction)
        if best is None or candidate < best:
            best = candidate
        if (position_mm <= max_position_mm and rotation_deg <= max_rotation_deg
                and gripper_mm <= max_gripper_mm and direction >= 0.7):
            if best_valid is None or candidate < best_valid:
                best_valid = candidate
    if best is None:
        result["reason"] = "no_new_horizon"
        return result
    # A slightly lower score must not displace a candidate that satisfies
    # every guard; otherwise we can reject a usable handoff after scoring it.
    _, new_index, position_mm, rotation_deg, gripper_mm, direction = (
        best_valid if best_valid is not None else best)
    result.update(candidate_new_index=new_index,
                  position_gap_mm=float(position_mm),
                  rotation_gap_deg=float(rotation_deg),
                  gripper_gap_mm=float(gripper_mm),
                  direction_cosine=float(direction),
                  new_remaining_steps=len(new) - new_index)
    failures = []
    if position_mm > max_position_mm:
        failures.append("position_gap")
    if rotation_deg > max_rotation_deg:
        failures.append("rotation_gap")
    if gripper_mm > max_gripper_mm:
        failures.append("gripper_gap")
    if direction < 0.7:
        failures.append("direction_change")
    result["reason"] = ",".join(failures) if failures else "geometry_only_pass"
    return result


def adaptive_position_limit(chunk_index, normal_mm, early_mm=None):
    """Use the early limit only for handoffs installing chunks 1 through 3."""
    return early_mm if early_mm is not None and 1 <= chunk_index < 4 else normal_mm


def grasp_commit_step(segment, observation_step, open_m=0.06,
                      closed_m=0.04, stable_steps=3):
    """Find a sustained open-to-close transition that normal handoff would cut short."""
    grip = right_gripper(np.asarray(segment, dtype=float))
    if not np.isfinite(grip).all() or grip[0] < open_m:
        return None
    for index in range(len(grip) - stable_steps + 1):
        if np.all(grip[index:index + stable_steps] < closed_m):
            if (index >= observation_step and grip[-1] < closed_m
                    and not np.any(grip[index:] > open_m)):
                return index
            return None
    return None


def diagnose_completed_handoff(old_segment, old_waypoints, raw, observed_q,
                               measured_q, measured_gripper, max_position_mm,
                               max_rotation_deg, max_gripper_mm,
                               max_tracking_rad=0.05, require_closed=True):
    """Check a fresh policy start after the previous trajectory has completed."""
    old = np.asarray(old_segment, dtype=float)
    old_q = np.asarray(old_waypoints, dtype=float)
    new = np.asarray(raw, dtype=float)
    observed_q = np.asarray(observed_q, dtype=float)
    measured_q = np.asarray(measured_q, dtype=float)
    if (old.ndim != 2 or old.shape[1] not in (7, 14) or old_q.shape != (len(old), 6)
            or new.ndim != 2 or new.shape[1] not in (7, 14) or not len(new)
            or observed_q.shape != (6,) or measured_q.shape != (6,)
            or not np.isfinite(old).all() or not np.isfinite(old_q).all()
            or not np.isfinite(new).all() or not np.isfinite(observed_q).all()
            or not np.isfinite(measured_q).all()
            or not np.isfinite(measured_gripper)):
        raise ValueError("Invalid completed-chunk observation or prediction")
    old_index = len(old) - 1
    observed_error = float(np.max(np.abs(old_q[-1] - observed_q)))
    current_error = float(np.max(np.abs(old_q[-1] - measured_q)))
    xyz_gap = float(1000 * np.linalg.norm(right_xyz(new[0]) - right_xyz(old[-1])))
    euler_delta = (right_pose(new[0])[3:] - right_pose(old[-1])[3:] + math.pi) % (2 * math.pi) - math.pi
    rotation_gap = float(np.degrees(np.max(np.abs(euler_delta))))
    gripper_gap = float(1000 * abs(right_gripper(new[0]) - measured_gripper))
    reasons = []
    if observed_error > max_tracking_rad or current_error > max_tracking_rad:
        reasons.append("joint_tracking")
    if require_closed and measured_gripper > 0.06:
        reasons.append("gripper_not_closed")
    if xyz_gap > max_position_mm:
        reasons.append("position_gap")
    if rotation_gap > max_rotation_deg:
        reasons.append("rotation_gap")
    if gripper_gap > max_gripper_mm:
        reasons.append("gripper_gap")
    return {
        "observation_old_index": old_index,
        "observation_joint_error_rad": observed_error,
        "current_old_index": old_index,
        "current_joint_error_rad": current_error,
        "candidate_new_index": 0,
        "overlap_steps": 0,
        "position_gap_mm": xyz_gap,
        "rotation_gap_deg": rotation_gap,
        "gripper_gap_mm": gripper_gap,
        "direction_cosine": None,
        "new_remaining_steps": len(new),
        "reason": ",".join(reasons) if reasons else "completed_chunk_pass",
    }


def validate_right7_health(server, expected_checkpoint):
    with urllib.request.urlopen(server + "/health", timeout=5) as response:
        health = json.load(response)
    if (health.get("schema") != SCHEMA
            or health.get("checkpoint") != expected_checkpoint
            or health.get("chunk_size") != 50
            or health.get("state_dim") != 7
            or health.get("action_dim") != 7
            or health.get("source_schema") != REQUEST_SCHEMA
            or not isinstance(health.get("task"), str) or not health.get("task")
            or health.get("device") != "cuda"
            or health.get("hardware_connected") is not False):
        raise RuntimeError(f"Unexpected right7 inference service: {health}")
    return health


def save_preflight_observation(snapshot, payload, actions, measured_q,
                              measured_gripper, timing, report):
    """Persist a live input before IK so rejected predictions remain diagnosable."""
    if snapshot is None:
        return
    snapshot = Path(snapshot)
    snapshot.mkdir(parents=True, exist_ok=True)
    np.save(snapshot / "right7_raw_actions.npy", np.asarray(actions, dtype=float))
    for key, encoded in payload["images_jpeg_base64"].items():
        name = key.rsplit(".", 1)[-1]
        (snapshot / f"{name}.jpg").write_bytes(base64.b64decode(encoded))
    (snapshot / "state.json").write_text(json.dumps({
        "state": payload["state"], "right_q": np.asarray(measured_q).tolist(),
        "right_gripper_m": measured_gripper, "timing": timing,
    }, indent=2) + "\n", encoding="utf-8")
    (snapshot / "preflight.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")


def cleanup_preflight_arm(arm):
    """Release SDK resources; this is not a claim that motors are disabled."""
    if arm is not None:
        arm.arm.cleanup()
        print("[SDK CLEANUP] returned; motor disabled state not verified", flush=True)


def run_ik_preflight_only(args):
    """Read one live right-arm observation and dry-run IK/planning only."""
    server = args.server.rstrip("/")
    parsed = urllib.parse.urlparse(server)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or any(ord(char) > 127 for char in server) or "<" in server or ">" in server):
        raise SystemExit("--server must be an actual right7 service URL")
    health = validate_right7_health(server, args.expected_checkpoint)

    # SingleArm attaches only the right CAN controller. Unlike DualTouchRobot,
    # its constructor does not call go_home(); the SDK constructor may still
    # enable motors, so this is a no-policy-command check, not a no-side-effect
    # hardware connection.
    from teleimager.image_client import ImageClient
    from superbot.bots.dual_touch_new import SingleArm

    image_client = ImageClient(host=args.img_host, auto_subscribe=False)
    arm = None
    try:
        image_client.start_subscriptions()
        arm = SingleArm(can_interface_=args.right_can_interface,
                        gripper=True, enable_fd_=False)
        print("[NO POLICY COMMAND] Connected the right-arm SDK for state/IK reads. "
              "This branch sends no home/move/update/waypoint command; SDK "
              "construction may enable the CAN motors.", flush=True)
        timing = {}
        session_id = "right7-ik-preflight-" + uuid.uuid4().hex[:12]
        payload = read_observation(arm, image_client, wait_frame, enc_jpeg_b64,
                                   REQUEST_SCHEMA, session_id, 0, timing, task=args.task)
        request = urllib.request.Request(
            server + "/predict", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if (result.get("schema") != SCHEMA or result.get("session_id") != session_id
                or result.get("seq") != 0 or result.get("action_dim") != 7
                or result.get("hardware_connected") is not False):
            raise RuntimeError(f"Unexpected right7 response: {result.keys()}")
        actions = np.asarray(result.get("actions"), dtype=float)
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid right7 action chunk: {actions.shape}")

        measured_q = np.asarray(arm.get_joint_positions(), dtype=float).reshape(-1)
        measured_gripper = float(arm.get_gripper_distance())
        measured_pos, measured_euler = arm.get_ee_pose_euler()
        if measured_q.shape != (6,) or not np.isfinite(measured_q).all():
            raise RuntimeError("Invalid live right-arm joint state")
        snapshot = Path(args.snapshot_dir) if args.snapshot_dir else None
        base_report = {
            "schema": SCHEMA, "checkpoint": health["checkpoint"],
            "device": health["device"], "gpu": health.get("gpu"),
            "hardware_motion_sent": False,
            "policy_waypoints_sent": False,
            "sdk_constructor_may_enable_motors": True,
            "observation_state_dim": 7, "action_shape": list(actions.shape),
            "measured_right_q": measured_q.tolist(),
            "measured_right_xyz_m": np.asarray(measured_pos, dtype=float).reshape(3).tolist(),
            "measured_right_euler_rad": np.asarray(measured_euler, dtype=float).reshape(3).tolist(),
            "measured_gripper_m": measured_gripper,
            "sampling_seed": result.get("sampling_seed"),
            "inference_ms": result.get("inference_ms"),
            "margin_rad": args.margin,
            "max_soft_clip_rad": args.max_soft_clip_rad,
            "timing": timing,
        }
        save_preflight_observation(
            snapshot, payload, actions, measured_q, measured_gripper, timing,
            {**base_report, "passed": False, "stage": "ik_check_pending"})
        try:
            waypoints, grippers = check_right_targets(
                arm, actions, args.margin, solve_with_retreat,
                JOINT_LOW, JOINT_UP, GRIPPER_MAX_M,
                max_soft_clip_rad=args.max_soft_clip_rad)
        except Exception as exc:
            # Keep the exact live input and prediction even when a guard rejects
            # the chunk; these are the most useful artifacts for diagnosis.
            import re
            match = re.search(r"(\d+) target\(s\) failed", str(exc))
            failure_report = {
                **base_report, "passed": False, "stage": "ik_check",
                "ik_checked": len(actions),
                "ik_failed": int(match.group(1)) if match else None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            save_preflight_observation(
                snapshot, payload, actions, measured_q, measured_gripper, timing,
                failure_report)
            print(json.dumps(failure_report, indent=2), flush=True)
            raise
        if getattr(args, "smooth_waypoints", True) and waypoints is not None:
            waypoints = smooth_joint_waypoints(
                waypoints,
                alpha=getattr(args, "waypoint_smooth_alpha", 0.20),
                passes=getattr(args, "waypoint_smooth_passes", 1))
        waypoint_array = np.asarray(waypoints, dtype=float)
        if waypoint_array.shape != (50, 6) or not np.isfinite(waypoint_array).all():
            raise RuntimeError("IK preflight did not produce 50 finite right-arm targets")

        # Vendor dry-run planner: computes the exact controller trajectory but
        # does not publish it to CAN or the robot.
        planner = getattr(arm.arm, "plan_joint_waypoints_with_gripper", None)
        if planner is None:
            raise RuntimeError("SDK lacks plan_joint_waypoints_with_gripper dry-run API")
        _, preflight_duration = get_phase_speed_and_duration("approach", 50, args)
        planned = planner(
            measured_q.tolist(), waypoints, grippers,
            time_sec=preflight_duration)
        planned = np.asarray(planned, dtype=float)
        if planned.ndim != 2 or planned.shape[1] < 15 or not np.isfinite(planned).all():
            raise RuntimeError(f"Invalid dry-run planner samples: {planned.shape}")
        dt = np.diff(planned[:, 0])
        if len(dt) and np.any(dt <= 0):
            raise RuntimeError("Dry-run trajectory timestamps are not strictly increasing")
        q_samples = planned[:, 1:7]
        q_speed = np.diff(q_samples, axis=0) / dt[:, None] if len(dt) else np.zeros((0, 6))
        q_accel = (np.diff(q_speed, axis=0) / ((dt[:-1] + dt[1:])[:, None] / 2)
                   if len(dt) > 1 else np.zeros((0, 6)))
        xyz = actions[:, :3]
        adj_xyz_mm = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000
        close = np.flatnonzero(actions[:, 6] < 0.04)
        report = {
            "schema": SCHEMA, "checkpoint": health["checkpoint"],
            "device": health["device"], "gpu": health.get("gpu"),
            "hardware_motion_sent": False,
            "policy_waypoints_sent": False,
            "sdk_constructor_may_enable_motors": True,
            "passed": True, "stage": "completed",
            "observation_state_dim": 7, "action_shape": list(actions.shape),
            "ik_checked": len(waypoints), "ik_failed": 0,
            "soft_clip_targets": 0, "retreated_ik_targets": 0,
            "max_adjacent_joint_step_rad": float(np.max(np.abs(np.diff(waypoint_array, axis=0)))),
            "max_adjacent_xyz_step_mm": float(np.max(adj_xyz_mm)),
            "dry_run_sample_count": int(len(planned)),
            "dry_run_duration_s": float(planned[-1, 0] - planned[0, 0]),
            "dry_run_max_joint_speed_rad_s": (
                float(np.max(np.abs(q_speed))) if q_speed.size else 0.0),
            "dry_run_max_joint_accel_rad_s2": (
                float(np.max(np.abs(q_accel))) if q_accel.size else 0.0),
            "gripper_min_m": float(np.min(actions[:, 6])),
            "first_close_index": int(close[0]) if close.size else None,
            "measured_gripper_m": measured_gripper,
            "sampling_seed": result.get("sampling_seed"),
            "inference_ms": result.get("inference_ms"),
        }
        if snapshot:
            target = snapshot
            np.save(target / "right7_waypoints.npy", waypoint_array)
            np.save(target / "right7_controller_plan.npy", planned)
            save_preflight_observation(
                target, payload, actions, measured_q, measured_gripper, timing, report)
        print(json.dumps(report, indent=2), flush=True)
        print("[NO POLICY COMMAND] IK and controller trajectory planning passed; "
              "no waypoint command was sent. SDK construction may have enabled motors.", flush=True)
        return 0
    finally:
        try:
            cleanup_preflight_arm(arm)
        finally:
            image_client.close()


def build_phase_aligned_segment(old_segment, raw, alignment, blend_head):
    """Create a shortened next chunk; the first point equals the old plan."""
    old_index = alignment["current_old_index"]
    new_index = alignment["candidate_new_index"]
    overlap = alignment["overlap_steps"]
    if (alignment["reason"] != "geometry_only_pass" or new_index is None
            or overlap < 8):
        raise ValueError(f"No usable phase match: {alignment['reason']}")
    segment = np.asarray(raw[new_index:], dtype=float).copy()
    segment[:overlap] = blend_head(
        np.asarray(old_segment[old_index:old_index + overlap], dtype=float),
        segment[:overlap], "cosine")
    return segment


def interpolate_action_midpoint(first, second):
    """Bisect a Cartesian action, taking the short path for Euler angles."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    midpoint = (first + second) / 2.0
    for start in ((3,) if first.size == 7 else (3, 10)):
        delta = (second[start:start + 3] - first[start:start + 3]
                 + math.pi) % (2 * math.pi) - math.pi
        midpoint[start:start + 3] = first[start:start + 3] + delta / 2.0
    return midpoint


def refine_joint_steps(segment, plan, limit_rad=0.07, max_insertions=4):
    """Subdivide oversized Cartesian and joint steps within one insertion budget."""
    refined = np.asarray(segment, dtype=float).copy()
    insertions = []
    for attempt in range(max_insertions + 1):
        xyz_steps = np.linalg.norm(np.diff(right_xyz(refined), axis=0), axis=1)
        if xyz_steps.size and xyz_steps.max() > .015 and attempt < max_insertions:
            index = int(xyz_steps.argmax())
            midpoint = interpolate_action_midpoint(refined[index], refined[index + 1])
            refined = np.insert(refined, index + 1, midpoint, axis=0)
            insertions.append((index, None, float(xyz_steps[index] * 1000)))
            continue
        waypoints, grippers = plan(refined)
        joint_steps = np.abs(np.diff(np.asarray(waypoints, dtype=float), axis=0))
        if not joint_steps.size:
            return refined, waypoints, grippers, insertions
        index, joint = np.unravel_index(np.argmax(joint_steps), joint_steps.shape)
        largest = float(joint_steps[index, joint])
        if largest <= limit_rad or attempt == max_insertions:
            return refined, waypoints, grippers, insertions
        midpoint = interpolate_action_midpoint(refined[index], refined[index + 1])
        refined = np.insert(refined, index + 1, midpoint, axis=0)
        insertions.append((int(index), int(joint), largest))
    raise AssertionError("unreachable refinement state")


def next_adaptive_observation_step(segment_length, desired_overlap,
                                   nominal_step, phase_reserve=8):
    """Observe early enough to leave old waypoints despite measured phase lead."""
    if segment_length < phase_reserve + 9:
        raise ValueError("Adaptive segment too short for another handoff")
    return max(1, min(nominal_step,
                      segment_length - desired_overlap - phase_reserve))


def adaptive_phase_reserve(measured_index, clock_index):
    """Leave two extra points beyond the latest observed phase lead."""
    return int(np.clip(measured_index - clock_index + 2, 2, 8))


def observation_tracking_guard(observation_error_rad, current_error_rad,
                               state_drift_rad, phase_gap_steps,
                               max_observation_error_rad):
    """Accept a small stale-observation mismatch only with fresh tracking evidence."""
    if observation_error_rad <= 0.05:
        return True, False
    bounded = (observation_error_rad <= max_observation_error_rad
               and current_error_rad <= 0.02
               and state_drift_rad <= 0.06
               and phase_gap_steps <= 3)
    return bounded, bounded


def wait_for_measured_phase(read_joints, waypoints, target_index, deadline,
                            max_tracking_rad=0.05, poll_s=0.05,
                            clock=time.monotonic, sleep=time.sleep,
                            not_before=None):
    """Wait for both nominal playback time and measured target proximity."""
    planned = np.asarray(waypoints, dtype=float)
    if planned.ndim != 2 or planned.shape[1] != 6 or not 0 <= target_index < len(planned):
        raise ValueError("Invalid measured-phase target or right-arm waypoints")
    last_index, last_error = None, None
    while clock() < deadline:
        if not_before is not None and clock() < not_before:
            sleep(min(poll_s, max(0.0, not_before - clock())))
            continue
        measured = np.asarray(read_joints(), dtype=float).reshape(-1)
        if measured.shape != (6,) or not np.isfinite(measured).all():
            raise ValueError("Invalid live right-arm joint state while waiting")
        errors = np.max(np.abs(planned - measured), axis=1)
        # A stationary or returning trajectory can match an earlier waypoint
        # as well. Only search the requested phase and its future suffix.
        last_index = target_index + int(np.argmin(errors[target_index:]))
        last_error = float(errors[last_index])
        if last_index >= target_index and last_error <= max_tracking_rad:
            return last_index, last_error
        sleep(min(poll_s, max(0.0, deadline - clock())))
    raise TimeoutError(f"measured phase did not reach {target_index}; "
                       f"last_index={last_index} last_error_rad={last_error}")


def measured_observation_deadline(install_time, waypoint_count, overlap,
                                  step_time, switch_delay):
    """Stop waiting before the remaining old trajectory loses its overlap."""
    if not 0 < overlap < waypoint_count or step_time <= 0 or switch_delay < 0:
        raise ValueError("Invalid measured-observation timing")
    return install_time + switch_delay + (waypoint_count - overlap) * step_time


def smooth_joint_waypoints(waypoints, alpha: float = 0.20, passes: int = 1):
    """Apply zero-phase 3-point moving average smoothing to joint waypoints.

    Formula for inner points (1 <= i <= N - 2):
        w'[i] = alpha * w[i-1] + (1 - 2*alpha) * w[i] + alpha * w[i+1]

    Boundary points w[0] and w[-1] are strictly preserved to maintain exact
    handoff start pose and final target pose.
    Input can be a list of lists or numpy array of shape (N, D).
    Returns the same type (list or numpy array) as the input.
    """
    if waypoints is None:
        return None
    is_list = isinstance(waypoints, list)
    w = np.array(waypoints, dtype=float, copy=True)
    if len(w) <= 2 or alpha <= 0.0 or passes <= 0:
        return waypoints
    for _ in range(passes):
        w[1:-1] = alpha * w[:-2] + (1.0 - 2.0 * alpha) * w[1:-1] + alpha * w[2:]
    return w.tolist() if is_list else w


def get_phase_speed_and_duration(phase: str, segment_len: int, args) -> tuple[float, float]:
    """Calculate speed multiplier and duration for a given execution phase.

    When args.phase_speeds is False, returns (args.speed, segment_len / args.rate / args.speed).
    When args.phase_speeds is True:
      - 'approach': (args.approach_speed, segment_len / args.rate / args.approach_speed)
      - 'grasp':
          if getattr(args, 'hold_grasp_pose', False):
              duration = float(getattr(args, 'grasp_duration_s', 2.5))
              speed = float(segment_len) / float(args.rate) / duration
              return speed, duration
          else:
              speed = float(getattr(args, 'approach_speed', 0.20))
              duration = float(segment_len) / float(args.rate) / speed
              return speed, duration
      - 'transport': (args.transport_speed, segment_len / args.rate / args.transport_speed)
      - 'place': (args.place_speed, segment_len / args.rate / args.place_speed)
    """
    if segment_len <= 0 or args.rate <= 0:
        raise ValueError(f"Invalid segment_len ({segment_len}) or rate ({args.rate})")

    if not getattr(args, "phase_speeds", False):
        speed = float(args.speed)
        duration = float(segment_len) / float(args.rate) / speed
        return speed, duration

    if phase == "approach":
        speed = float(getattr(args, "approach_speed", 0.20))
        duration = float(segment_len) / float(args.rate) / speed
    elif phase == "grasp":
        if getattr(args, "hold_grasp_pose", False):
            duration = float(getattr(args, "grasp_duration_s", 2.5))
            speed = float(segment_len) / float(args.rate) / duration
        else:
            speed = float(getattr(args, "approach_speed", 0.20))
            duration = float(segment_len) / float(args.rate) / speed
    elif phase == "transport":
        speed = float(getattr(args, "transport_speed", 0.25))
        duration = float(segment_len) / float(args.rate) / speed
    elif phase == "place":
        speed = float(getattr(args, "place_speed", 0.24))
        duration = float(segment_len) / float(args.rate) / speed
    else:
        raise ValueError(f"Unknown phase: '{phase}'")

    return speed, duration


def evaluate_grasp_deferral_and_budget(
    actions,
    index: int,
    total_loops: int,
    condition_final_grasp: bool,
    completed_conditioned_grasp: bool,
    place_after_grasp: bool = False,
    max_total_loops: int = 8,
    wrist_obb: dict | None = None,
    ee_z_m: float | None = None,
    ee_pose6=None,
) -> tuple[bool, int]:
    """Decide whether to defer grasp closure at `index` and update loop budget.

    Returns `(defer_grasp_to_next_chunk, updated_total_loops)`.
    When `defer_grasp_to_next_chunk` is True and `place_after_grasp` is enabled,
    ensures `updated_total_loops >= min(max_total_loops, index + 4)` so that after
    deferring at `index`, there is room for grasp at `index + 1` plus two full
    placement chunks (`transport` at `index + 2` and `place` release at `index + 3`).
    """
    if not condition_final_grasp or completed_conditioned_grasp:
        return False, total_loops
    # Check real wrist camera alignment if wrist_obb is provided
    if wrist_obb is not None:
        from build_grasp_replay_cases import check_wrist_grasp_alignment
        eff_z = float(ee_pose6[2]) if ee_pose6 is not None else ee_z_m
        align = check_wrist_grasp_alignment(wrist_obb, ee_z_m=eff_z, ee_pose6=ee_pose6)
        if align["aligned"]:
            # If horizontally aligned at hover height (eff_z > 0.170m),
            # defer grasp to allow an approach chunk to execute pure vertical descent to table height!
            if eff_z is not None and eff_z > 0.170 and index < max_total_loops - 1:
                updated_loops = max(total_loops, min(max_total_loops, index + (4 if place_after_grasp else 2)))
                return True, updated_loops
            return False, total_loops
        elif index < max_total_loops - 1:
            updated_loops = max(total_loops, min(max_total_loops, index + (4 if place_after_grasp else 2)))
            return True, updated_loops

    actions = np.asarray(actions, dtype=float)
    if np.any(actions[:13, 6] < 0.010) or (float(actions[0, 1]) < 0.125 and np.any(actions[:13, 6] < 0.025)):
        return False, total_loops

    # For objects at near/mid table (e.g. R5/R6/R8/R9 intersection where actions[0, 1] < 0.135m),
    # do not defer if early closure is predicted within first 25 steps
    if float(actions[0, 1]) < 0.135 and np.any(actions[:25, 6] < 0.040):
        return False, total_loops

    max_y_window = float(np.max(actions[12:21, 1]))
    wants_defer = (
        (index == 3 and max_y_window < 0.170)
        or (index == 4 and float(actions[0, 1]) >= 0.130 and max_y_window < 0.165)
    )
    if not wants_defer:
        return False, total_loops
    updated_loops = total_loops
    if place_after_grasp:
        updated_loops = max(total_loops, min(max_total_loops, index + 4))
    if index >= updated_loops - 1:
        return False, total_loops
    return True, updated_loops


def evaluate_settled_post_plan_guard(
    measured_q,
    fresh_q,
    waypoint_first,
    segment_first_grip: float,
    fresh_grip: float,
    obs_age_ms: float,
    max_gripper_gap_mm: float,
    max_observation_age_ms: float,
) -> list[str]:
    """Return granular failure reasons for post-plan settled guard check."""
    reasons = []
    if initial_state_drift(measured_q, fresh_q) > 0.01:
        reasons.append("post_plan_state_drift")
    if initial_state_drift(waypoint_first, fresh_q) > 0.05:
        reasons.append("first_waypoint_gap")
    if not np.isfinite(fresh_grip):
        reasons.append("invalid_gripper_feedback")
    elif abs(float(segment_first_grip) - float(fresh_grip)) * 1000.0 > float(max_gripper_gap_mm):
        reasons.append("gripper_gap")
    if float(obs_age_ms) > float(max_observation_age_ms):
        reasons.append("observation_expired")
    return reasons


def main():
    args = parse_args()
    if not (1 <= args.loops <= 40 and 0 < args.speed <= 1 and args.rate > 0
            and 2 <= args.exec_steps <= 50 and 0 < args.switch_after < args.exec_steps
            and 1 <= args.blend_overlap <= args.exec_steps - args.switch_after
            and 0 <= args.margin < 0.1
            and 0 <= args.max_soft_clip_rad <= 0.05
            and 0 <= args.max_initial_retreat_targets <= 3
            and args.switch_delay >= 0):
        raise SystemExit("Invalid timing, loop, blend, or margin parameter")
    if not (0 < args.approach_speed <= 1.0 and 0 < args.transport_speed <= 1.0
            and 0 < args.place_speed <= 1.0 and 2.0 <= args.grasp_duration_s <= 30.0):
        raise SystemExit("Invalid phase speed or grasp duration parameter")
    if not (0.0 <= args.waypoint_smooth_alpha <= 0.33 and 1 <= args.waypoint_smooth_passes <= 5):
        raise SystemExit("Invalid waypoint smoothing parameter")
    if (args.max_switch_gap_mm <= 0 or args.max_adaptive_gap_mm <= 0
            or args.max_initial_gap_mm <= 0
            or args.max_tracking_joint_rad <= 0
            or args.max_state_drift_rad <= 0
            or not 0.05 <= args.max_observation_tracking_rad <= 0.06
            or args.max_phase_step_error < 0 or args.max_rotation_gap_deg <= 0
            or args.max_gripper_gap_mm <= 0 or args.max_gripper_step_mm <= 0
            or not 0 <= args.max_open_gripper_overshoot_mm <= 2.0
            or not 0 <= args.max_close_gripper_undershoot_mm <= 20.0
            or args.max_observation_age_ms <= 0):
        raise SystemExit("Guard thresholds must be positive")
    if args.max_gripper_step_mm > 80.0:
        raise SystemExit("--max-gripper-step-mm must be at most 80 mm")
    if (args.early_max_adaptive_gap_mm is not None
            and (not args.adaptive_eight_chunk
                 or not args.max_adaptive_gap_mm <= args.early_max_adaptive_gap_mm <= 35.0)):
        raise SystemExit("--early-max-adaptive-gap-mm requires --adaptive-eight-chunk "
                         "and a value from the normal gap limit through 35 mm")
    if not args.send_right_chunks and args.loops != 1 and not args.ik_preflight_only:
        raise SystemExit("Without --send-right-chunks, use --loops 1 for IK-only check")
    if args.ik_preflight_only and args.send_right_chunks:
        raise SystemExit("--ik-preflight-only cannot be combined with --send-right-chunks")
    allowed_grasp_loops = (4, 5, 6, 7) if args.place_after_grasp else (4, 5)
    if args.condition_final_grasp and not (args.settled_chunks and args.loops in allowed_grasp_loops):
        raise SystemExit(f"--condition-final-grasp requires --settled-chunks --loops in {allowed_grasp_loops}")
    if args.hold_grasp_pose and not args.condition_final_grasp:
        raise SystemExit("--hold-grasp-pose requires --condition-final-grasp")
    settled_mode = args.second_check_after_settle or args.settled_chunks
    if args.second_check_after_settle and not args.adaptive_two_chunk:
        raise SystemExit("--second-check-after-settle requires adaptive two-chunk mode")
    max_settled_loops = 7 if args.place_after_grasp else 5
    if args.settled_chunks and (not args.send_right_chunks or not 2 <= args.loops <= max_settled_loops
            or not args.adaptive_refine or args.ik_preflight_only
            or args.adaptive_second_preflight_only or args.commit_grasp_chunk
            or args.probe_steps is not None or args.diagnose_switch_only
            or args.adaptive_two_chunk or args.adaptive_five_chunk
            or args.adaptive_eight_chunk or args.adaptive_ten_chunk or args.adaptive_fifteen_chunk):
        raise SystemExit(f"--settled-chunks requires 2..{max_settled_loops} loops, --send-right-chunks and --adaptive-refine, without other handoff modes")
    if not 30 <= args.settled_bridge_max_mm <= 50 or (args.settled_bridge_max_mm != 30 and not settled_mode):
        raise SystemExit("Settled bridge envelope must be 30..40 mm and requires settled mode")
    if not 30 <= args.placement_bridge_max_mm <= 60:
        raise SystemExit("--placement-bridge-max-mm must be 30..60 mm")
    second_check_only = (args.adaptive_two_chunk and args.adaptive_second_preflight_only
                         and args.loops == 2)
    settled_two_chunk = args.second_check_after_settle and args.adaptive_two_chunk
    first_bridge_allowed = second_check_only or settled_two_chunk or args.settled_chunks
    if args.initial_bridge_steps and (
            not 3 <= args.initial_bridge_steps <= 10 or (args.loops != 1 and not first_bridge_allowed)
            or args.exec_steps != 50 or args.ik_preflight_only
            or args.probe_steps is not None or args.diagnose_switch_only
            or (args.adaptive_two_chunk and not first_bridge_allowed) or args.adaptive_five_chunk
            or args.adaptive_eight_chunk or args.adaptive_ten_chunk
            or args.adaptive_fifteen_chunk):
        raise SystemExit("Initial bridge requires --exec-steps 50 and either one loop or a second-chunk check without sending it")
    max_trace_loops = 7 if args.place_after_grasp else 5
    if args.joint_trace_dir and (not args.send_right_chunks or not 1 <= args.loops <= max_trace_loops
                                 or args.probe_steps is not None
                                 or args.diagnose_switch_only
                                 or (args.adaptive_two_chunk and not first_bridge_allowed) or args.adaptive_five_chunk
                                 or args.adaptive_eight_chunk or args.adaptive_ten_chunk
                                 or args.adaptive_fifteen_chunk
                                 or not 5 <= args.joint_trace_hz <= 60):
        raise SystemExit(f"--joint-trace-dir requires --send-right-chunks --loops 1..{max_trace_loops}, "
                         "without adaptive execution/probe/diagnose mode, and "
                         "--joint-trace-hz 5..60")
    if args.diagnose_switch_only and (args.loops != 2 or not args.send_right_chunks):
        raise SystemExit("--diagnose-switch-only requires --loops 2 --send-right-chunks")
    if args.probe_steps is not None:
        if (not args.send_right_chunks or args.diagnose_switch_only
                or args.loops != len(args.probe_steps) + 1
                or not all(0 < step < args.exec_steps for step in args.probe_steps)
                or sorted(set(args.probe_steps)) != args.probe_steps):
            raise SystemExit("--probe-steps requires --send-right-chunks, "
                             "--loops = number of probes + 1, and unique increasing "
                             "steps inside the first chunk")
    if args.adaptive_two_chunk and (args.loops != 2 or not args.send_right_chunks
                                    or args.diagnose_switch_only
                                    or args.probe_steps is not None
                                    or args.adaptive_five_chunk
                                    or args.adaptive_eight_chunk
                                    or args.adaptive_ten_chunk
                                    or args.adaptive_fifteen_chunk):
        raise SystemExit("--adaptive-two-chunk requires --loops 2 "
                         "--send-right-chunks, without probe/diagnose modes")
    if args.adaptive_five_chunk and (args.loops != 5 or not args.send_right_chunks
                                     or args.diagnose_switch_only
                                     or args.probe_steps is not None
                                     or args.adaptive_eight_chunk
                                     or args.adaptive_ten_chunk
                                     or args.adaptive_fifteen_chunk):
        raise SystemExit("--adaptive-five-chunk requires --loops 5 "
                         "--send-right-chunks, without probe/diagnose modes")
    if args.adaptive_eight_chunk and (args.loops != 8 or not args.send_right_chunks
                                      or args.diagnose_switch_only
                                      or args.probe_steps is not None
                                      or args.adaptive_ten_chunk
                                      or args.adaptive_fifteen_chunk):
        raise SystemExit("--adaptive-eight-chunk requires --loops 8 "
                         "--send-right-chunks, without probe/diagnose modes")
    if args.adaptive_ten_chunk and (args.loops != 10 or not args.send_right_chunks
                                    or args.diagnose_switch_only
                                    or args.probe_steps is not None
                                    or args.adaptive_eight_chunk
                                    or args.adaptive_fifteen_chunk):
        raise SystemExit("--adaptive-ten-chunk requires --loops 10 "
                         "--send-right-chunks, without probe/diagnose modes")
    if args.adaptive_fifteen_chunk and (not 2 <= args.loops <= 15
                                        or not args.send_right_chunks
                                        or args.diagnose_switch_only
                                        or args.probe_steps is not None
                                        or args.adaptive_eight_chunk):
        raise SystemExit("--adaptive-fifteen-chunk requires --loops 2..15 "
                         "--send-right-chunks, without probe/diagnose modes")
    measured_phase_mode = args.adaptive_five_chunk or args.adaptive_eight_chunk
    adaptive_mode = (args.settled_chunks or args.adaptive_two_chunk or measured_phase_mode
                     or args.adaptive_ten_chunk or args.adaptive_fifteen_chunk)
    if args.commit_grasp_chunk and not adaptive_mode:
        raise SystemExit("--commit-grasp-chunk requires an adaptive mode")
    initial_adaptive_observation_step = None
    if adaptive_mode and args.loops > 1:
        if args.blend_overlap < 8:
            raise SystemExit(
                "Adaptive handoff requires --blend-overlap >= 8; "
                "the phase matcher rejects shorter overlaps"
            )
        try:
            initial_adaptive_observation_step = next_adaptive_observation_step(
                args.exec_steps, args.blend_overlap, args.switch_after,
                phase_reserve=8)
        except ValueError as exc:
            raise SystemExit(
                "Adaptive chunk is too short to preserve the required "
                "handoff overlap; increase --exec-steps or reduce the "
                "overlap/reserve combination"
            ) from exc
    if args.adaptive_refine and not adaptive_mode:
        raise SystemExit("Adaptive refinement/preflight-only requires "
                         "--adaptive-two-chunk, --adaptive-five-chunk, "
                         "--adaptive-eight-chunk, "
                         "--adaptive-ten-chunk, or --adaptive-fifteen-chunk")
    if args.adaptive_second_preflight_only and not args.adaptive_two_chunk:
        raise SystemExit("--adaptive-second-preflight-only requires "
                         "--adaptive-two-chunk")

    if args.ik_preflight_only:
        return run_ik_preflight_only(args)

    from teleimager.image_client import ImageClient  # noqa: E402
    from xr_teleoperate2.robot_control.robot_dual_touch import DualTouchRobot  # noqa: E402

    server = args.server.rstrip("/")
    parsed_server = urllib.parse.urlparse(server)
    if (parsed_server.scheme not in ("http", "https") or not parsed_server.hostname
            or any(ord(char) > 127 for char in server) or "<" in server or ">" in server):
        raise SystemExit(
            "--server must be an actual URL, for example http://192.168.0.109:8010; "
            "replace the workstation LAN IP placeholder before running")
    with urllib.request.urlopen(server + "/health", timeout=5) as response:
        health = json.load(response)
    if (health.get("schema") != SCHEMA or health.get("checkpoint") != args.expected_checkpoint
            or health.get("chunk_size") != 50 or health.get("state_dim") != 7
            or health.get("action_dim") != 7 or health.get("source_schema") != REQUEST_SCHEMA
            or not isinstance(health.get("task"), str) or not health.get("task")
            or health.get("device") != "cuda"
            or health.get("hardware_connected") is not False):
        raise SystemExit(f"Unexpected inference service: {health}")
    if (args.expected_seed_mode is not None
            and health.get("seed_mode") != args.expected_seed_mode):
        raise SystemExit(f"Unexpected inference seed mode: {health}")
    print(f"[MODEL] {health['checkpoint']}", flush=True)
    print(f"[TASK] {args.task}", flush=True)
    target_color = args.target_color
    if target_color == "auto":
        from build_grasp_replay_cases import parse_target_color
        target_color = parse_target_color(args.task)
    print(f"[TARGET COLOR] {target_color} (inferred from task/prompt: '{args.task}')", flush=True)

    duration = args.exec_steps / args.rate / args.speed
    switch_time = args.switch_after / args.rate / args.speed
    print(f"[TIMING] loops={args.loops} chunk={duration:.3f}s "
          f"switch_after={switch_time:.3f}s blend={args.blend_overlap} "
          f"margin={args.margin} max_soft_clip_rad={args.max_soft_clip_rad} "
          f"max_observation_tracking_rad={args.max_observation_tracking_rad}",
          flush=True)
    if args.phase_speeds:
        print(f"[PHASE CONFIG] approach={args.approach_speed:.2f} "
              f"grasp_duration={args.grasp_duration_s:.1f}s "
              f"transport={args.transport_speed:.2f} "
              f"place={args.place_speed:.2f}", flush=True)
    if getattr(args, "smooth_waypoints", True):
        print(f"[SMOOTHING] joint waypoint filter active: alpha={args.waypoint_smooth_alpha:.2f} "
              f"passes={args.waypoint_smooth_passes}", flush=True)
    if args.early_max_adaptive_gap_mm is not None:
        print(f"[GAP LIMIT] handoff into chunks 1-3: "
              f"{args.early_max_adaptive_gap_mm:.1f}mm; "
              f"chunk 4 onward: {args.max_adaptive_gap_mm:.1f}mm",
              flush=True)
    robot = None
    image_client = None
    last_install = None
    last_duration = duration
    last_chunk_speed = args.speed
    previous_segment = None
    previous_waypoints = None
    restart_after_grasp = False
    completed_conditioned_grasp = False
    grasp_chunk_index = None
    held_grip = None
    last_wrist_obb = None
    high_cam_target = None
    expired_replan_count = 0
    grasp_retry_count = 0
    sent_chunks = 0
    trace_session = None
    session_id = "right-rolling-" + uuid.uuid4().hex[:12]
    try:
        print("[ROBOT] DualTouchRobot initializes and homes BOTH arms", flush=True)
        robot = DualTouchRobot()
        time.sleep(2.0)
        image_client = ImageClient(host=args.img_host, auto_subscribe=False)
        image_client.start_subscriptions()

        total_loops = args.loops
        index = -1
        while index + 1 < total_loops:
            index += 1
            timing = {}
            payload = read_observation(
                robot, image_client, wait_frame, enc_jpeg_b64, SCHEMA,
                session_id, index, timing, task=args.task)
            if args.snapshot_dir:
                snapshot_dir = Path(args.snapshot_dir)
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                ee_read_start = time.monotonic()
                right_ee_position, right_ee_euler = robot.right_arm.get_ee_pose_euler()
                ee_read_end = time.monotonic()
                right_ee_position = np.asarray(right_ee_position, dtype=float).reshape(3)
                right_ee_euler = np.asarray(right_ee_euler, dtype=float).reshape(3)
                if (not np.isfinite(right_ee_position).all()
                        or not np.isfinite(right_ee_euler).all()):
                    raise ValueError("Invalid measured right-arm end-effector pose")
                for camera_key, encoded in payload["images_jpeg_base64"].items():
                    camera_name = camera_key.rsplit(".", 1)[-1]
                    (snapshot_dir / f"loop_{index:02d}_{camera_name}.jpg").write_bytes(
                        base64.b64decode(encoded))
                (snapshot_dir / f"loop_{index:02d}_state.json").write_text(
                    json.dumps({
                        "seq": index,
                        "state": payload["state"],
                        "camera_receive_monotonic_s": timing["camera_received"],
                        "state_monotonic_s": timing["state_received"],
                        "right_ee_position_m": right_ee_position.tolist(),
                        "right_ee_euler_rad": right_ee_euler.tolist(),
                        "right_ee_read_start_monotonic_s": ee_read_start,
                        "right_ee_read_end_monotonic_s": ee_read_end,
                        "previous_install_monotonic_s": last_install,
                        "previous_right_waypoints": previous_waypoints,
                        "previous_action_segment": (
                            previous_segment.tolist()
                            if previous_segment is not None else None),
                    }, indent=2),
                    encoding="utf-8")
                print(f"[SNAPSHOT {index}] {snapshot_dir}", flush=True)
                print(f"[MEASURED EE {index}] right_xyz_m="
                      f"{np.round(right_ee_position, 4).tolist()}", flush=True)
            request = urllib.request.Request(
                server + "/predict", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            timing["infer_start"] = time.monotonic()
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            timing["infer_end"] = time.monotonic()
            if (result.get("schema") != SCHEMA or result.get("session_id") != session_id
                    or result.get("seq") != index):
                raise ValueError("SmolVLA response identity mismatch")
            if (args.expected_seed_mode == "fixed"
                    and result.get("sampling_seed") != health.get("seed_base")):
                raise ValueError("SmolVLA fixed-seed response mismatch")
            actions = np.asarray(result["actions"], dtype=float)
            if actions.shape != (50, 7) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid SmolVLA output: shape={actions.shape}")
            initial_prefix = None
            if args.snapshot_dir:
                np.save(Path(args.snapshot_dir) / f"loop_{index:02d}_raw_actions.npy", actions)
            if index == 0:
                current_pos, current_euler = robot.right_arm.get_ee_pose_euler()
                current_pos = np.asarray(current_pos, dtype=float).reshape(3)
                current_euler = np.asarray(current_euler, dtype=float).reshape(3)
                entry = actions[0]
                if args.initial_bridge_steps:
                    current_action = np.concatenate([current_pos, current_euler,
                        [float(robot.right_arm.get_gripper_distance())]])
                    initial_prefix = make_initial_prefix(current_action, actions[0], args.initial_bridge_steps)
                    initial_prefix, omitted = select_initial_prefix(robot.right_arm, initial_prefix, args.margin)
                    entry = initial_prefix[0] if len(initial_prefix) else actions[0]
                    print(f"[INITIAL BRIDGE] raw_gap_mm={1000*np.linalg.norm(actions[0,:3]-current_pos):.1f} "
                          f"omitted_soft_boundary_candidates={omitted} inserted={len(initial_prefix)}; "
                          f"keep raw indices 0..{49-len(initial_prefix)}; all targets rechecked", flush=True)
                position_gap_mm = float(1000 * np.linalg.norm(right_xyz(actions[0]) - current_pos))
                rotation_gap_deg = float(np.degrees(np.max(np.abs(
                    (right_pose(actions[0])[3:] - current_euler + np.pi) % (2 * np.pi) - np.pi
                ))))
                gripper_gap_mm = float(1000 * abs(right_gripper(actions[0]) - float(robot.right_arm.get_gripper_distance())))
                if initial_prefix is not None:
                    position_gap_mm = float(1000*np.linalg.norm(entry[:3]-current_pos))
                    rotation_gap_deg = float(np.degrees(np.max(np.abs((entry[3:6]-current_euler+np.pi)%(2*np.pi)-np.pi))))
                    gripper_gap_mm = float(1000*abs(entry[6]-current_action[6]))
                print(f"[FIRST TARGET] position_gap_mm={position_gap_mm:.1f} "
                      f"rotation_gap_deg={rotation_gap_deg:.1f} "
                      f"gripper_gap_mm={gripper_gap_mm:.1f}", flush=True)
                if args.send_right_chunks and (
                    position_gap_mm > args.max_initial_gap_mm
                    or rotation_gap_deg > args.max_rotation_gap_deg
                    or gripper_gap_mm > args.max_gripper_gap_mm
                ):
                    raise ValueError("Rejected initial SmolVLA target gap")
            if args.snapshot_dir:
                np.save(Path(args.snapshot_dir) / f"loop_{index:02d}_raw_actions.npy",
                        actions)
            if getattr(args, "high_cam_guidance", True) and high_cam_target is None and index in (0, 1):
                from build_grasp_replay_cases import detect_high_cam_grid_target
                high_b64 = payload.get("images_jpeg_base64", {}).get(
                    "observation.images.cam_high"
                )
                if high_b64:
                    high_cam_target = detect_high_cam_grid_target(
                        base64.b64decode(high_b64),
                        target_color=target_color,
                    )
                    if high_cam_target:
                        print(
                            f"[HIGH CAM {index}] locked target_color='{target_color}' at cell={high_cam_target['cell']} "
                            f"(cx={high_cam_target['cx']:.1f}, cy={high_cam_target['cy']:.1f}, area={high_cam_target['area']:.1f}) -> "
                            f"hover_xyz={high_cam_target['hover_xyz']}",
                            flush=True,
                        )
            alignment = None
            settled_check = settled_mode and index > 0
            completed_restart = restart_after_grasp or settled_check
            restart_after_grasp = False
            conditioned_grasp = False
            is_placement_chunk = False
            if adaptive_mode and index > 0 and completed_restart:
                observed_q = np.asarray(payload["state"][:6], dtype=float)
                align_q = np.asarray(robot.right_arm.get_joint_positions(), dtype=float)
                measured_grip = float(robot.right_arm.get_gripper_distance())
                cur_pos, cur_euler = robot.right_arm.get_ee_pose_euler()
                current_ee_z = float(cur_pos[2])
                current_ee_pose6 = np.r_[np.asarray(cur_pos, dtype=float).reshape(3), np.asarray(cur_euler, dtype=float).reshape(3)]
                wrist_obb = None
                if getattr(args, "wrist_obb_refine", True) and index >= 1:
                    from build_grasp_replay_cases import detect_wrist_block_obb
                    wrist_b64 = payload.get("images_jpeg_base64", {}).get(
                        "observation.images.cam_right_wrist"
                    )
                    if wrist_b64:
                        wrist_obb = detect_wrist_block_obb(
                            base64.b64decode(wrist_b64),
                            target_color=target_color,
                            prev_obb=last_wrist_obb,
                        )
                        if wrist_obb is not None:
                            last_wrist_obb = wrist_obb
                            cross_str = ""
                            if "cross_check" in wrist_obb and wrist_obb["cross_check"].get("diff_deg") is not None:
                                cc = wrist_obb["cross_check"]
                                cross_str = f" poly_diff={cc['diff_deg']:.1f}° (verified={cc['verified']})"
                            print(
                                f"[WRIST OBB {index}] cx={wrist_obb['cx']:.1f} "
                                f"cy={wrist_obb['cy']:.1f} "
                                f"size={wrist_obb['w']:.1f}x{wrist_obb['h']:.1f} "
                                f"angle={wrist_obb['angle']:.1f} "
                                f"color={wrist_obb.get('color_class', 'unknown')} "
                                f"mode={wrist_obb.get('grasp_axis_mode', 'unknown')}"
                                f"{cross_str}",
                                flush=True,
                            )
                segment = actions.copy()
                prev_total_loops = total_loops
                defer_grasp_to_next_chunk, total_loops = evaluate_grasp_deferral_and_budget(
                    actions,
                    index=index,
                    total_loops=total_loops,
                    condition_final_grasp=args.condition_final_grasp,
                    completed_conditioned_grasp=completed_conditioned_grasp,
                    place_after_grasp=args.place_after_grasp,
                    max_total_loops=10,
                    wrist_obb=wrist_obb,
                    ee_z_m=current_ee_z,
                    ee_pose6=current_ee_pose6,
                )
                if total_loops > prev_total_loops:
                    print(f"[PLACEMENT BUDGET] extending loop budget from {prev_total_loops} "
                          f"to {total_loops} after deferred grasp at chunk {index}", flush=True)
                conditioned_grasp = (
                    args.condition_final_grasp and index >= 3 and not defer_grasp_to_next_chunk
                    and not completed_conditioned_grasp
                )
                is_placement_chunk = (
                    args.place_after_grasp and completed_conditioned_grasp
                )
                if conditioned_grasp:
                    from build_grasp_replay_cases import check_wrist_grasp_alignment, condition_grasp_actions
                    pos, euler = robot.right_arm.get_ee_pose_euler()
                    start = np.r_[np.asarray(pos).reshape(3), np.asarray(euler).reshape(3), measured_grip]
                    if not np.any(actions[:, 6] < .04):
                        print(f"[GRASP GUARD] no closure predicted; chunk {index} NOT sent", flush=True)
                        break

                    wrist_align_tol = getattr(args, "wrist_align_tol_px", 20.0)
                    alignment_info = check_wrist_grasp_alignment(
                        wrist_obb,
                        ee_z_m=float(start[2]),
                        ee_pose6=start[:6],
                        tol_u_px=wrist_align_tol,
                        tol_v_px=wrist_align_tol,
                    )
                    wrist_obb_aligned = alignment_info["aligned"]

                    if getattr(args, "wrist_obb_refine", True):
                        if not wrist_obb_aligned:
                            gross_misaligned = (
                                alignment_info["du"] is not None and abs(alignment_info["du"]) > 30.0
                            ) or (
                                alignment_info["dv"] is not None and abs(alignment_info["dv"]) > 30.0
                            )
                            if gross_misaligned or getattr(args, "strict_wrist_alignment", False):
                                print(f"[GRASP GUARD] unaligned at budget limit (reason={alignment_info['reason']} "
                                      f"du={alignment_info['du']}px dv={alignment_info['dv']}px "
                                      f"tol={alignment_info['tol_u_px']}/{alignment_info['tol_v_px']}px); "
                                      f"aborting grasp without blind closure", flush=True)
                                break
                            else:
                                print(f"[GRASP ALIGNMENT] notice: alignment check {alignment_info['reason']} "
                                      f"(du={alignment_info['du']}px dv={alignment_info['dv']}px "
                                      f"tol={alignment_info['tol_u_px']}/{alignment_info['tol_v_px']}px); "
                                      f"proceeding under grasp guard", flush=True)

                    try:
                        segment, clipped, splits = condition_grasp_actions(
                            start, actions, max_position_m=args.settled_bridge_max_mm / 1000,
                            hold_pose=args.hold_grasp_pose,
                            wrist_obb=wrist_obb,
                        )
                    except ValueError as exc:
                        print(f"[GRASP GUARD] chunk {index} NOT sent: {exc}", flush=True)
                        break
                    print(f"[GRASP CONDITION] points={len(segment)} hold_pose={args.hold_grasp_pose} "
                          f"wrist_obb_centered={wrist_obb_aligned} "
                          f"clipped_negative={clipped} subdivided={splits}", flush=True)
                    if args.snapshot_dir:
                        np.save(Path(args.snapshot_dir)/f"loop_{index:02d}_conditioned_grasp.npy", segment)
                elif is_placement_chunk:
                    from build_grasp_replay_cases import (
                        condition_placement_actions,
                        detect_wrist_target_obb,
                    )
                    if held_grip is not None and measured_grip < (held_grip - 0.004):
                        print(f"[PLACEMENT GUARD] block slippage detected (measured_grip={measured_grip*1000:.1f}mm < held_grip-4mm={(held_grip-0.004)*1000:.1f}mm); skipping placement chunk {index}", flush=True)
                        break
                    pos, euler = robot.right_arm.get_ee_pose_euler()
                    start = np.r_[np.asarray(pos).reshape(3), np.asarray(euler).reshape(3), measured_grip]
                    target_obb = None
                    if getattr(args, "wrist_obb_refine", True):
                        wrist_b64 = payload.get("images_jpeg_base64", {}).get(
                            "observation.images.cam_right_wrist"
                        )
                        if wrist_b64:
                            target_obb = detect_wrist_target_obb(base64.b64decode(wrist_b64))
                            if target_obb is not None:
                                print(
                                    f"[TARGET OBB {index}] cx={target_obb['cx']:.1f} "
                                    f"cy={target_obb['cy']:.1f} "
                                    f"size={target_obb['w']:.1f}x{target_obb['h']:.1f} "
                                    f"angle={target_obb['angle']:.1f}",
                                    flush=True,
                                )
                    try:
                        segment, clipped, splits = condition_placement_actions(
                            start, actions, held_grip=held_grip,
                            max_position_m=args.placement_bridge_max_mm / 1000,
                            target_obb=target_obb)
                    except ValueError as exc:
                        print(f"[PLACEMENT GUARD] chunk {index} NOT sent: {exc}", flush=True)
                        break
                    release_threshold = max(0.040, min(0.070, float(held_grip) + 0.010))
                    released = bool(np.any(segment[:, 6] >= release_threshold))
                    if not released and index + 1 >= total_loops and total_loops < 10:
                        prev_loops = total_loops
                        total_loops = min(10, index + 2)
                        print(f"[PLACEMENT BUDGET] extending loop budget from {prev_loops} "
                              f"to {total_loops} to reserve release chunk after transport at chunk {index}",
                              flush=True)
                    print(f"[PLACEMENT CONDITION] points={len(segment)} held_grip={held_grip:.4f}m "
                          f"released={released} clipped_negative={clipped} subdivided={splits}",
                          flush=True)
                    if args.snapshot_dir:
                        np.save(Path(args.snapshot_dir)/f"loop_{index:02d}_conditioned_placement.npy", segment)
                elif settled_check and args.condition_final_grasp:
                    from build_grasp_replay_cases import condition_approach_actions
                    pos, euler = robot.right_arm.get_ee_pose_euler()
                    start = np.r_[np.asarray(pos).reshape(3), np.asarray(euler).reshape(3), measured_grip]
                    if defer_grasp_to_next_chunk:
                        if wrist_obb is not None:
                            from build_grasp_replay_cases import check_wrist_grasp_alignment
                            al = check_wrist_grasp_alignment(
                                wrist_obb,
                                ee_z_m=float(start[2]),
                                ee_pose6=start[:6],
                            )
                            if al["aligned"] and current_ee_z is not None and current_ee_z > 0.170:
                                print(f"[APPROACH CONDITION] aligned at hover (z={current_ee_z:.4f}m); "
                                      f"deferring grasp to execute vertical descent chunk", flush=True)
                            else:
                                print(f"[APPROACH CONDITION] deferring grasp at chunk {index} for visual alignment "
                                      f"(du={al['du']}px dv={al['dv']}px tol=20.0px)", flush=True)
                        else:
                            defer_threshold = 0.1700 if index == 3 else 0.1650
                            print(f"[APPROACH CONDITION] deferring grasp at chunk {index} to next chunk "
                                  f"(max_y_m={float(np.max(actions[12:21, 1])):.4f} < {defer_threshold:.4f})",
                                  flush=True)
                    try:
                        grid_hover_pose = high_cam_target["hover_pose"] if high_cam_target else None
                        if high_cam_target and index >= 2:
                            # Beyond loop 1, coarse transit has handed over to fine wrist alignment
                            grid_hover_pose = None
                        segment, held_open_from, pose_anchored_from = condition_approach_actions(
                            start, actions, max_position_m=args.settled_bridge_max_mm / 1000,
                            wrist_obb=wrist_obb, hover_z_m=args.hover_z_m,
                            grid_hover_pose=grid_hover_pose)
                    except ValueError as exc:
                        print(f"[APPROACH GUARD] chunk {index} NOT sent: {exc}", flush=True)
                        break
                    print(f"[APPROACH CONDITION] points={len(segment)} "
                          f"held_open_from={held_open_from} pose_anchored_from={pose_anchored_from}",
                          flush=True)
                    if args.snapshot_dir:
                        np.save(Path(args.snapshot_dir)/f"loop_{index:02d}_conditioned_approach.npy", segment)
                alignment = diagnose_completed_handoff(
                    previous_segment, previous_waypoints, segment,
                    observed_q, align_q, measured_grip,
                    args.max_adaptive_gap_mm, args.max_rotation_gap_deg,
                    args.max_gripper_gap_mm, require_closed=not settled_check)
                label = "SETTLED HANDOFF" if settled_check else "GRASP RESTART"
                print(f"[{label}] gap_mm={alignment['position_gap_mm']:.1f} "
                      f"rotation_deg={alignment['rotation_gap_deg']:.1f} "
                      f"gripper_gap_mm={alignment['gripper_gap_mm']:.1f} "
                      f"observed_tracking_rad={alignment['observation_joint_error_rad']:.4f} "
                      f"current_tracking_rad={alignment['current_joint_error_rad']:.4f} "
                      f"result={alignment['reason']}", flush=True)
                if (settled_check and not conditioned_grasp and not is_placement_chunk
                        and alignment["reason"] in ("position_gap", "completed_chunk_pass")
                        and (alignment["position_gap_mm"] > 6.0 or alignment["rotation_gap_deg"] > 1.5)
                        and alignment["position_gap_mm"] <= args.settled_bridge_max_mm):
                    pos, euler = robot.right_arm.get_ee_pose_euler()
                    measured_pose = np.concatenate([
                        np.asarray(pos, dtype=float).reshape(3),
                        np.asarray(euler, dtype=float).reshape(3), [measured_grip]])
                    try:
                        prefix = make_initial_prefix(measured_pose, segment[0], 6,
                            max_position_m=args.settled_bridge_max_mm / 1000)
                    except ValueError as exc:
                        print(f"[SETTLED BRIDGE] rejected: {exc}", flush=True)
                        break
                    segment = np.concatenate([prefix, segment], axis=0)
                    alignment = diagnose_completed_handoff(
                        previous_segment, previous_waypoints, segment,
                        observed_q, align_q, measured_grip,
                        args.max_adaptive_gap_mm, args.max_rotation_gap_deg,
                        args.max_gripper_gap_mm, require_closed=False)
                    print(f"[SETTLED BRIDGE] inserted={len(prefix)} "
                          f"raw_gap_mm={1000*np.linalg.norm(actions[0,:3]-measured_pose[:3]):.1f} "
                          f"entry_gap_mm={alignment['position_gap_mm']:.1f} "
                          f"result={alignment['reason']}", flush=True)
                if alignment["reason"] != "completed_chunk_pass":
                    print(f"[ADAPTIVE GUARD] chunk {index} NOT sent", flush=True)
                    break
            elif adaptive_mode and index > 0:
                position_limit_mm = adaptive_position_limit(
                    index, args.max_adaptive_gap_mm, args.early_max_adaptive_gap_mm)
                observed_q = np.asarray(payload["state"][:6], dtype=float)
                align_q = np.asarray(robot.right_arm.get_joint_positions(), dtype=float)
                phase_step_time = 1.0 / (args.rate * args.speed)
                observation_clock_index = int(round(
                    (timing["state_received"] - last_install - args.switch_delay)
                    / phase_step_time))
                current_clock_index = int(round(
                    (time.monotonic() - last_install - args.switch_delay)
                    / phase_step_time))
                alignment = diagnose_phase_alignment(
                    previous_segment, previous_waypoints, actions,
                    observed_q, align_q, args.blend_overlap,
                    max_position_mm=position_limit_mm,
                    max_rotation_deg=args.max_rotation_gap_deg,
                    max_gripper_mm=args.max_gripper_gap_mm,
                    min_remaining=(17 if args.horizon_aware_alignment
                                   and index < total_loops - 1 else 15),
                    expected_observation_index=observation_clock_index,
                    expected_current_index=current_clock_index)
                print(f"[ADAPTIVE CANDIDATE] old_obs={alignment['observation_old_index']} "
                      f"old_now={alignment['current_old_index']} "
                      f"obs_clock={observation_clock_index} "
                      f"now_clock={current_clock_index} "
                      f"new_start={alignment['candidate_new_index']} "
                      f"overlap={alignment['overlap_steps']} "
                      f"gap_mm={alignment['position_gap_mm']} "
                      f"rotation_deg={alignment['rotation_gap_deg']} "
                      f"direction_cos={alignment['direction_cosine']} "
                      f"limit_mm={position_limit_mm:.1f} "
                      f"result={alignment['reason']}", flush=True)
                if alignment["reason"] != "geometry_only_pass":
                    print(f"[ADAPTIVE GUARD] chunk {index} NOT sent", flush=True)
                    break
                segment = build_phase_aligned_segment(
                    previous_segment, actions, alignment, blend_head)
            else:
                segment = actions[:args.exec_steps].copy()
            if initial_prefix is not None:
                segment = np.concatenate([initial_prefix, segment], axis=0)[:args.exec_steps]
                if args.snapshot_dir:
                    np.save(Path(args.snapshot_dir)/f"loop_{index:02d}_bridged_actions.npy", segment)
            if (previous_segment is not None and args.blend_overlap
                    and not adaptive_mode):
                overlap = args.blend_overlap
                segment[:overlap] = blend_head(
                    previous_segment[args.switch_after:args.switch_after + overlap],
                    segment[:overlap], "cosine")
            print(f"[LOOP {index}] right_xyz0={np.round(right_xyz(segment[0]), 4).tolist()} "
                  f"right_grip0={right_gripper(segment[0]):.4f}m", flush=True)
            raw_close = np.flatnonzero(actions[:, 6] < 0.04)
            sent_close = np.flatnonzero(segment[:, 6] < 0.04)
            print(f"[HORIZON {index}] right_xyz_at_switch="
                  f"{np.round(right_xyz(segment[min(args.switch_after - 1, len(segment) - 1)]), 4).tolist()} "
                  f"right_xyz_last={np.round(right_xyz(segment[-1]), 4).tolist()} "
                  f"raw50_grip_min={np.min(actions[:, 6]):.4f}m "
                  f"sent_grip_min={np.min(segment[:, 6]):.4f}m "
                  f"raw_first_below_0.04="
                  f"{int(raw_close[0]) if raw_close.size else 'none'} "
                  f"sent_first_below_0.04="
                  f"{int(sent_close[0]) if sent_close.size else 'none'}",
                  flush=True)

            try:
                if args.probe_steps is not None and index > 0:
                    # Later predictions are measured only. No IK plan is installed.
                    waypoints, grippers = None, None
                elif adaptive_mode and index > 0 and args.adaptive_refine:
                    def plan(candidate):
                        return check_right_targets(
                            robot.right_arm, candidate, args.margin,
                            solve_with_retreat, JOINT_LOW, JOINT_UP, GRIPPER_MAX_M,
                            args.max_soft_clip_rad,
                            max_open_gripper_overshoot_m=(
                                args.max_open_gripper_overshoot_mm / 1000.0),
                            max_close_gripper_undershoot_m=(
                                args.max_close_gripper_undershoot_mm / 1000.0))

                    segment, waypoints, grippers, insertions = refine_joint_steps(
                        segment, plan)
                    for pair, joint, prior_step in insertions:
                        detail = (f"prior_xyz_step_mm={prior_step:.2f}" if joint is None
                                  else f"joint={joint + 1} prior_step_rad={prior_step:.4f}")
                        print(f"[ADAPTIVE REFINE] inserted after point={pair} {detail}", flush=True)
                else:
                    waypoints, grippers = check_right_targets(
                        robot.right_arm, segment, args.margin, solve_with_retreat,
                        JOINT_LOW, JOINT_UP, GRIPPER_MAX_M,
                        args.max_soft_clip_rad,
                        max_initial_retreat_targets=(
                            args.max_initial_retreat_targets if index == 0 else 0),
                        max_open_gripper_overshoot_m=(
                            args.max_open_gripper_overshoot_mm / 1000.0),
                        max_close_gripper_undershoot_m=(
                            args.max_close_gripper_undershoot_mm / 1000.0))
            except ValueError as exc:
                label = "ADAPTIVE GUARD" if adaptive_mode and index > 0 else "RIGHT GUARD"
                print(f"[{label}] chunk {index} NOT sent: {exc}", flush=True)
                break
            if getattr(args, "smooth_waypoints", True) and waypoints is not None:
                waypoints = smooth_joint_waypoints(
                    waypoints,
                    alpha=getattr(args, "waypoint_smooth_alpha", 0.20),
                    passes=getattr(args, "waypoint_smooth_passes", 1))
            if conditioned_grasp:
                phase = "grasp"
            elif is_placement_chunk:
                release_threshold = max(0.040, min(0.070, float(held_grip or 0.030) + 0.010))
                phase = "place" if np.any(segment[:, 6] >= release_threshold) else "transport"
            else:
                phase = "approach"
            chunk_speed, chunk_duration = get_phase_speed_and_duration(phase, len(segment), args)
            print(f"[PHASE TIMING {index}] phase={phase} speed={chunk_speed:.2f} duration={chunk_duration:.3f}s", flush=True)
            if initial_prefix is not None:
                start_q = np.asarray(robot.right_arm.get_joint_positions(), dtype=float)
                planned = np.asarray(robot.right_arm.arm.plan_joint_waypoints_with_gripper(
                    start_q.tolist(), waypoints, grippers,
                    time_sec=chunk_duration), dtype=float)
                if (planned.ndim != 2 or len(planned) < 2 or planned.shape[1] < 15
                        or not np.isfinite(planned).all()
                        or np.any(np.diff(planned[:,0]) <= 0)):
                    raise ValueError("Invalid initial-bridge controller dry-run")
                if np.any(planned[:,1:7] < JOINT_LOW) or np.any(planned[:,1:7] > JOINT_UP):
                    raise ValueError("Initial-bridge controller samples exceed physical joint limits")
                if args.snapshot_dir:
                    np.save(Path(args.snapshot_dir)/f"loop_{index:02d}_bridge_controller_plan.npy", planned)
                print(f"[INITIAL BRIDGE DRY RUN] samples={len(planned)} duration_s={planned[-1,0]-planned[0,0]:.3f}; no publish", flush=True)
            if grippers is not None:
                # Subsequent handoff checks must compare against the gripper
                # targets actually sent, while raw model output remains in
                # loop_*_raw_actions.npy for diagnosis.
                segment[:, 6] = np.asarray(grippers) * GRIPPER_MAX_M
            if adaptive_mode and index > 0 and args.snapshot_dir:
                np.save(Path(args.snapshot_dir) / f"loop_{index:02d}_aligned_segment.npy", segment)
                np.save(Path(args.snapshot_dir) / f"loop_{index:02d}_aligned_waypoints.npy",
                        np.asarray(waypoints, dtype=float))
            timing["ik_end"] = time.monotonic()
            camera_times = list(timing["camera_received"].values())
            first_camera = min(camera_times)
            last_camera = max(camera_times)
            step_time = 1.0 / (args.rate * last_chunk_speed)
            old_index = ((timing["ik_end"] - last_install) / step_time
                         if last_install is not None else None)
            old_index_text = (f"old_index_est_at_ik_end={old_index:.2f} "
                              f"fixed_blend_index={args.switch_after} "
                              if old_index is not None else "")
            print(
                f"[CLOCK {index}] camera_receive_span_ms={1000 * (last_camera - first_camera):.1f} "
                f"state_after_last_camera_ms={1000 * (timing['state_received'] - last_camera):.1f} "
                f"infer_ms={1000 * (timing['infer_end'] - timing['infer_start']):.1f} "
                f"post_infer_to_ik_end_ms={1000 * (timing['ik_end'] - timing['infer_end']):.1f} "
                f"{old_index_text}"
                f"obs_receive_age_at_ik_end_ms={1000 * (timing['ik_end'] - first_camera):.1f} "
                f"camera_first_monotonic_s={first_camera:.3f} "
                f"state_monotonic_s={timing['state_received']:.3f} "
                f"ik_end_monotonic_s={timing['ik_end']:.3f} "
                f"last_install_monotonic_s={last_install if last_install is not None else 'none'}",
                flush=True)
            if (index == 0 and args.send_right_chunks
                    and 1000 * (timing["ik_end"] - first_camera) > args.max_observation_age_ms):
                print("[INITIAL GUARD] stale observation; first chunk NOT sent", flush=True)
                break
            if adaptive_mode and index > 0:
                measured_q = np.asarray(robot.right_arm.get_joint_positions(), dtype=float)
                measured_grip = float(robot.right_arm.get_gripper_distance())
                observed_q = np.asarray(payload["state"][:6], dtype=float)
                old_index = alignment["current_old_index"]
                clock_index = int(round((time.monotonic() - last_install
                                         - args.switch_delay) / step_time))
                first_joint_gap = float(np.max(np.abs(np.asarray(waypoints[0]) - measured_q)))
                plan_tracking = float(np.max(np.abs(
                    np.asarray(previous_waypoints[old_index]) - measured_q)))
                state_drift = float(np.max(np.abs(measured_q - observed_q)))
                max_joint_step = float(np.max(np.abs(np.diff(waypoints, axis=0))))
                max_xyz_step_mm = float(1000 * np.max(np.linalg.norm(
                    np.diff(right_xyz(segment), axis=0), axis=1)))
                max_gripper_step_mm = float(1000 * np.max(np.abs(
                    np.diff(segment[:, 6]))))
                first_gripper_gap_mm = float(1000 * abs(right_gripper(segment[0]) - measured_grip))
                observation_age_ms = 1000 * (time.monotonic() - first_camera)
                adaptive_reasons = []
                next_observation_step = None
                phase_reserve = (adaptive_phase_reserve(old_index, clock_index)
                                 if args.adaptive_ten_chunk
                                 or args.adaptive_fifteen_chunk else 8)
                if index < total_loops - 1 and not settled_check:
                    try:
                        next_observation_step = next_adaptive_observation_step(
                            len(segment), args.blend_overlap, args.switch_after,
                            phase_reserve=phase_reserve)
                    except ValueError:
                        adaptive_reasons.append("insufficient_future_horizon")
                commit_step = (grasp_commit_step(segment, next_observation_step)
                               if args.commit_grasp_chunk and next_observation_step is not None
                               and index < total_loops - 1 else None)
                max_phase_lag = 12 if args.adaptive_fifteen_chunk else 8
                if not settled_check and (old_index - clock_index > 8 or clock_index - old_index > max_phase_lag):
                    adaptive_reasons.append("phase_clock_gap")
                observation_ok, observation_tolerated = observation_tracking_guard(
                    alignment["observation_joint_error_rad"], plan_tracking,
                    state_drift,
                    abs(alignment["observation_old_index"]
                        - alignment["current_old_index"]),
                    args.max_observation_tracking_rad)
                if not observation_ok:
                    adaptive_reasons.append("observation_tracking")
                if plan_tracking > 0.05:
                    adaptive_reasons.append("current_tracking")
                if state_drift > args.max_state_drift_rad:
                    adaptive_reasons.append("state_drift")
                if observation_age_ms > args.max_observation_age_ms:
                    adaptive_reasons.append("stale_observation")
                if first_joint_gap > 0.05:
                    adaptive_reasons.append("first_joint_gap")
                if max_joint_step > 0.08:
                    adaptive_reasons.append("joint_step")
                if args.adaptive_refine and max_joint_step > 0.07:
                    adaptive_reasons.append("refine_incomplete")
                if max_xyz_step_mm > 15.0:
                    adaptive_reasons.append("cartesian_step")
                if max_gripper_step_mm > args.max_gripper_step_mm:
                    adaptive_reasons.append("gripper_step")
                if not np.isfinite(first_gripper_gap_mm) or first_gripper_gap_mm > args.max_gripper_gap_mm:
                    adaptive_reasons.append("measured_gripper_gap")
                print(f"[ADAPTIVE PREFLIGHT] points={len(segment)} "
                      f"old_now={old_index} clock_now={clock_index} "
                      f"max_phase_lag={max_phase_lag} "
                      f"first_joint_gap_rad={first_joint_gap:.4f} "
                      f"tracking_rad={plan_tracking:.4f} "
                      f"observation_tracking_rad="
                      f"{alignment['observation_joint_error_rad']:.4f} "
                      f"observation_phase_gap_steps="
                      f"{abs(alignment['observation_old_index'] - alignment['current_old_index'])} "
                      f"observation_tolerated={observation_tolerated} "
                      f"state_drift_rad={state_drift:.4f} "
                      f"max_joint_step_rad={max_joint_step:.4f} "
                      f"max_xyz_step_mm={max_xyz_step_mm:.1f} "
                      f"max_gripper_step_mm={max_gripper_step_mm:.1f} "
                      f"first_gripper_gap_mm={first_gripper_gap_mm:.1f} "
                      f"observation_age_ms={observation_age_ms:.1f} "
                      f"phase_reserve={phase_reserve} "
                      f"next_observation_step={next_observation_step} "
                      f"reject={','.join(adaptive_reasons) if adaptive_reasons else 'none'}",
                      flush=True)
                planned_dry_run = None
                if args.adaptive_second_preflight_only or settled_check:
                    if not adaptive_reasons:
                        duration_s = chunk_duration
                        planned_dry_run = np.asarray(robot.right_arm.arm.plan_joint_waypoints_with_gripper(
                            measured_q.tolist(), waypoints, grippers, time_sec=duration_s), dtype=float)
                        if (planned_dry_run.ndim != 2 or len(planned_dry_run) < 2 or planned_dry_run.shape[1] < 15
                                or not np.isfinite(planned_dry_run).all()
                                or np.any(np.diff(planned_dry_run[:, 0]) <= 0)
                                or np.any(planned_dry_run[:, 1:7] < JOINT_LOW)
                                or np.any(planned_dry_run[:, 1:7] > JOINT_UP)):
                            raise ValueError("Invalid second-chunk controller dry-run")
                        print(f"[SETTLED DRY RUN {index}] samples={len(planned_dry_run)} duration_s={planned_dry_run[-1,0]-planned_dry_run[0,0]:.3f}; no publish", flush=True)
                    if args.adaptive_second_preflight_only:
                        if planned_dry_run is not None and args.snapshot_dir:
                            np.save(Path(args.snapshot_dir) / f"loop_{index:02d}_controller_dry_run.npy", planned_dry_run)
                        print("[ADAPTIVE PREFLIGHT ONLY] second chunk NOT sent", flush=True)
                        break
                if adaptive_reasons:
                    print(f"[ADAPTIVE GUARD] chunk {index} NOT sent", flush=True)
                    break
                adaptive_duration = chunk_duration
                second_trace = None
                if settled_check and trace_session is not None:
                    second_trace = trace_session.prepare_chunk(
                        index, segment, waypoints, grippers,
                        {"nominal_duration_s": adaptive_duration, "switch_delay_s": args.switch_delay,
                         "rate_hz": args.rate, "speed": chunk_speed,
                         "gripper_plan_units": "SDK fraction of maximum opening"})
                if settled_check:
                    fresh_q = np.asarray(robot.right_arm.get_joint_positions(), dtype=float).reshape(6)
                    fresh_grip = float(robot.right_arm.get_gripper_distance())
                    obs_age_at_plan_ms = 1000.0 * (timing["ik_end"] - first_camera)
                    settled_reasons = evaluate_settled_post_plan_guard(
                        measured_q,
                        fresh_q,
                        waypoints[0],
                        right_gripper(segment[0]),
                        fresh_grip,
                        obs_age_at_plan_ms,
                        max_gripper_gap_mm=args.max_gripper_gap_mm,
                        max_observation_age_ms=args.max_observation_age_ms,
                    )
                    if settled_reasons:
                        if planned_dry_run is not None and args.snapshot_dir:
                            np.save(Path(args.snapshot_dir) / f"loop_{index:02d}_controller_dry_run.npy", planned_dry_run)
                        reason_text = ",".join(settled_reasons)
                        print(
                            f"[SETTLED GUARD] chunk={index} reasons={reason_text} "
                            f"obs_age_ms={obs_age_at_plan_ms:.1f}; no send",
                            flush=True,
                        )
                        if (
                            settled_reasons == ["observation_expired"]
                            and expired_replan_count < getattr(args, "max_expired_replans", 1)
                        ):
                            expired_replan_count += 1
                            print(
                                f"[SETTLED RECOVERY] chunk={index} arm settled and only observation expired; "
                                f"re-observing and re-planning ({expired_replan_count}/{getattr(args, 'max_expired_replans', 1)})",
                                flush=True,
                            )
                            index -= 1
                            continue
                        break
                    if planned_dry_run is not None and args.snapshot_dir:
                        np.save(Path(args.snapshot_dir) / f"loop_{index:02d}_controller_dry_run.npy", planned_dry_run)
                second_send_begin = time.monotonic_ns()
                print(f"[SEND {index}] PHASE-ALIGNED RIGHT only, "
                      f"{len(waypoints)} waypoints, {adaptive_duration:.3f}s, "
                      f"phase={phase}, speed={chunk_speed:.2f}", flush=True)
                robot.right_arm.update_joint_waypoint_chunk_with_gripper(
                    waypoints, grippers, time_sec=adaptive_duration,
                    switch_delay_sec=args.switch_delay)
                second_send_return = time.monotonic_ns()
                last_install = time.monotonic()
                last_duration = adaptive_duration
                last_chunk_speed = chunk_speed
                sent_chunks += 1
                if second_trace is not None:
                    trace_session.mark_chunk_sent(second_trace, second_send_begin, second_send_return)
                previous_segment = segment
                previous_waypoints = waypoints
                if commit_step is not None:
                    print(f"[GRASP COMMIT] chunk={index} closes from step {commit_step}; "
                          f"finish checked chunk before fresh observation", flush=True)
                    restart_after_grasp = True
                    next_wait = adaptive_duration + args.switch_delay + 0.1
                elif (conditioned_grasp and not args.place_after_grasp) or index == total_loops - 1:
                    next_wait = adaptive_duration + args.switch_delay + 0.1
                elif settled_check and index < total_loops - 1:
                    next_wait = adaptive_duration + args.switch_delay + 0.2
                    print(f"[SETTLE] chunk={index} wait {next_wait:.3f}s", flush=True)
                else:
                    next_wait = next_observation_step * step_time
                    print(f"[NEXT OBS] after chunk {index} at planned step "
                          f"{next_observation_step}/{len(segment)} "
                          f"({next_wait:.3f}s nominal after install; "
                          f"mode={'measured' if measured_phase_mode or args.adaptive_fifteen_chunk else 'clock'})",
                          flush=True)
                if ((measured_phase_mode or args.adaptive_fifteen_chunk)
                        and index < total_loops - 1 and not restart_after_grasp):
                    try:
                        deadline = (measured_observation_deadline(
                            last_install, len(waypoints), args.blend_overlap,
                            step_time, args.switch_delay)
                            if measured_phase_mode else
                            last_install + adaptive_duration + args.switch_delay + 0.1)
                        reached_index, reached_error = wait_for_measured_phase(
                            robot.right_arm.get_joint_positions, waypoints,
                            next_observation_step, deadline,
                            not_before=(last_install + args.switch_delay
                                        + next_observation_step * step_time))
                        print(f"[MEASURED OBS] chunk={index} target={next_observation_step} "
                              f"reached={reached_index} error_rad={reached_error:.4f} "
                              f"elapsed_s={time.monotonic() - last_install:.3f}",
                              flush=True)
                    except (TimeoutError, ValueError) as exc:
                        print(f"[ADAPTIVE GUARD] no next observation: {exc}", flush=True)
                        break
                else:
                    remaining = next_wait - (time.monotonic() - last_install)
                    if remaining > 0:
                        time.sleep(remaining)
                if settled_check and index < total_loops - 1 and not (conditioned_grasp and not args.place_after_grasp):
                    q_before = robot.right_arm.get_joint_positions()
                    time.sleep(.2)
                    drift = initial_state_drift(q_before, robot.right_arm.get_joint_positions())
                    print(f"[SETTLE] chunk={index} joint drift over 200 ms: {drift:.5f} rad", flush=True)
                    if drift > .01:
                        print("[ADAPTIVE GUARD] arm not settled; no next chunk", flush=True)
                        break
                print(f"[STATE {index}] right_q="
                      f"{np.round(robot.right_arm.get_joint_positions(), 4).tolist()} "
                      f"right_gripper={robot.right_arm.get_gripper_distance():.4f}m",
                      flush=True)
                if conditioned_grasp:
                    completed_conditioned_grasp = True
                    grasp_chunk_index = index
                    held_grip = float(robot.right_arm.get_gripper_distance())
                    print(f"[GRASP COMPLETED] chunk={index} measured_grip={held_grip:.4f}m", flush=True)
                    if not args.place_after_grasp:
                        break
                    if held_grip < 0.008:
                        if (
                            grasp_retry_count < getattr(args, "max_grasp_retries", 1)
                            and last_wrist_obb is not None
                        ):
                            failed_grip = held_grip
                            grasp_retry_count += 1
                            completed_conditioned_grasp = False
                            grasp_chunk_index = None
                            held_grip = None
                            total_loops = min(10, total_loops + 2)
                            print(
                                f"[EMPTY GRASP RECOVERY] chunk={index} measured_grip={failed_grip:.4f}m; "
                                f"allowing retry ({grasp_retry_count}/{getattr(args, 'max_grasp_retries', 1)}) with updated budget={total_loops}",
                                flush=True,
                            )
                            continue
                        print(f"[GRASP GUARD] empty grasp detected (measured_grip={held_grip*1000:.1f}mm < 8.0mm); skipping placement", flush=True)
                        break
                    if total_loops < index + 3:
                        prev_budget = total_loops
                        total_loops = index + 3
                        print(f"[PLACEMENT BUDGET] extending loop budget from {prev_budget} to {total_loops} to ensure transport and place release", flush=True)
                release_threshold = max(0.040, min(0.070, float(held_grip or 0.030) + 0.010))
                if is_placement_chunk and np.any(segment[:, 6] >= release_threshold):
                    from build_grasp_replay_cases import verify_placement_completion
                    post_release_grip = float(robot.right_arm.get_gripper_distance())
                    print(f"[PLACEMENT RELEASE SENT] chunk={index} gripper open command issued (measured_grip={post_release_grip:.4f}m)", flush=True)
                    verify_info = verify_placement_completion(post_release_grip, held_grip)
                    print(
                        f"[PLACEMENT CONFIRMED] chunk={index} confirmed={verify_info['confirmed']} "
                        f"gripper_opened={verify_info.get('gripper_opened_ok', verify_info.get('gripper_opened', True))} "
                        f"required_open_m={verify_info.get('required_open_m', verify_info.get('expected_open_m', 0.045)):.4f}m",
                        flush=True,
                    )
                    print(f"[PLACEMENT COMPLETED] chunk={index} block released on target area", flush=True)
                    break
                continue
            if previous_segment is not None:
                measured_q = np.asarray(robot.right_arm.get_joint_positions(), dtype=float)
                measured_grip = float(robot.right_arm.get_gripper_distance())
                if measured_q.shape != (6,) or not np.isfinite(measured_q).all():
                    raise ValueError("Invalid live right-arm joint state at handoff")
                if not np.isfinite(measured_grip):
                    raise ValueError("Invalid live right-arm gripper state at handoff")
                metrics, reasons = evaluate_handoff(
                    previous_segment, previous_waypoints, actions, segment,
                    measured_q, time.monotonic() - last_install, step_time,
                    args.switch_after, args.blend_overlap,
                    1000 * (time.monotonic() - first_camera), args)
                raw_q, raw_ok, raw_alpha = solve_with_retreat(
                    robot.right_arm, right_xyz(actions[0]), right_pose(actions[0])[3:], measured_q)
                raw_q = np.asarray(raw_q, dtype=float)
                raw_q_gap = (float(np.max(np.abs(raw_q - measured_q)))
                             if raw_q.shape == (6,) and np.isfinite(raw_q).all()
                             else float("inf"))
                observed_q = np.asarray(payload["state"][:6], dtype=float)
                state_drift_rad = float(np.max(np.abs(measured_q - observed_q)))
                if state_drift_rad > args.max_state_drift_rad:
                    reasons.append("state_drift")
                print(f"[TRACK {index}] measured_q={np.round(measured_q, 4).tolist()} "
                      f"observation_q={np.round(observed_q, 4).tolist()} "
                      f"state_drift_rad={state_drift_rad:.4f} "
                      f"measured_gripper_m={measured_grip:.4f} "
                      f"old_at_nearest_gripper_m="
                      f"{right_gripper(previous_segment[metrics['local_nearest_idx']]):.4f} "
                      f"new_raw0_gripper_m={right_gripper(actions[0]):.4f} "
                      f"expected_idx={metrics['expected_idx']} "
                      f"nearest_local_idx={metrics['local_nearest_idx']} "
                      f"nearest_global_idx={metrics['global_nearest_idx']} "
                      f"local_joint_error_rad={metrics['tracking_rad']:.4f} "
                      f"global_joint_error_rad={metrics['global_tracking_rad']:.4f} "
                      f"new_raw0_ik_ok={bool(raw_ok) and raw_alpha == 1.0} "
                      f"new_raw0_joint_gap_rad={raw_q_gap:.4f}", flush=True)
                print(f"[SEAM {index}] raw_xyz_gap_mm={metrics['raw_xyz_gap_mm']:.1f} "
                      f"raw_rotation_gap_deg={metrics['raw_rotation_gap_deg']:.1f} "
                      f"raw_gripper_gap_mm={metrics['raw_gripper_gap_mm']:.1f} "
                      f"old_progress_mm={metrics['old_progress_mm']:.1f} "
                      f"blended_projected_progress_mm="
                      f"{metrics['blended_projected_progress_mm']:.1f} "
                      f"observation_age_ms={metrics['observation_age_ms']:.1f} "
                      f"reject={','.join(reasons) if reasons else 'none'}", flush=True)
                if args.diagnose_switch_only or args.probe_steps is not None:
                    alignment = diagnose_phase_alignment(
                        previous_segment, previous_waypoints, actions,
                        observed_q, measured_q, args.blend_overlap,
                        max_position_mm=args.max_adaptive_gap_mm,
                        max_rotation_deg=args.max_rotation_gap_deg,
                        max_gripper_mm=args.max_gripper_gap_mm)
                    print(f"[PHASE ALIGN {index}] "
                          f"obs_old={alignment['observation_old_index']} "
                          f"now_old={alignment['current_old_index']} "
                          f"obs_qerr={alignment['observation_joint_error_rad']:.4f}rad "
                          f"now_qerr={alignment['current_joint_error_rad']:.4f}rad "
                          f"new_start={alignment['candidate_new_index']} "
                          f"overlap={alignment['overlap_steps']} "
                          f"gap_mm={alignment['position_gap_mm']} "
                          f"rotation_deg={alignment['rotation_gap_deg']} "
                          f"gripper_gap_mm={alignment['gripper_gap_mm']} "
                          f"direction_cos={alignment['direction_cosine']} "
                          f"remaining={alignment['new_remaining_steps']} "
                          f"limit_mm={args.max_adaptive_gap_mm:.1f} "
                          f"result={alignment['reason']} "
                          f"(diagnostic only; second chunk is NOT sent)", flush=True)
            if args.diagnose_switch_only and index == 1:
                old_at_fixed = right_xyz(previous_segment[args.switch_after])
                old_index_rounded = max(0, min(len(previous_segment) - 1,
                                               int(round(old_index))))
                old_at_estimated = right_xyz(previous_segment[old_index_rounded])
                blended_first = right_xyz(segment[0])
                raw_first = right_xyz(actions[0])
                print(
                    f"[SWITCH DIAG] old_fixed_xyz={np.round(old_at_fixed, 4).tolist()} "
                    f"old_est_index={old_index_rounded} "
                    f"old_est_xyz={np.round(old_at_estimated, 4).tolist()} "
                    f"new_raw_xyz0={np.round(raw_first, 4).tolist()} "
                    f"fixed_blended_xyz0={np.round(blended_first, 4).tolist()} "
                    f"old_est_to_blend_gap_mm="
                    f"{1000 * np.linalg.norm(old_at_estimated - blended_first):.1f} "
                    f"old_est_to_raw_gap_mm="
                    f"{1000 * np.linalg.norm(old_at_estimated - raw_first):.1f}",
                    flush=True)
                print("[SWITCH DIAG] second chunk NOT sent; waiting for first to finish",
                      flush=True)
                break
            if args.probe_steps is not None and index > 0:
                print(f"[PROBE {index}] prediction recorded; no additional chunk sent",
                      flush=True)
                if index < len(args.probe_steps):
                    target_time = (last_install + args.switch_delay
                                   + args.probe_steps[index] / args.rate / args.speed)
                    remaining = target_time - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                continue
            if previous_segment is not None and reasons:
                print(f"[GUARD {index}] rejecting second chunk before send: "
                      f"{','.join(reasons)}", flush=True)
                break
            if not args.send_right_chunks:
                print("[IK ONLY] No policy command sent", flush=True)
                break

            chunk_trace_dir = None
            if args.joint_trace_dir:
                if trace_session is None:
                    trace_session = JointTraceSession(
                        args.joint_trace_dir, robot.right_arm.get_joint_positions,
                        robot.right_arm.get_gripper_distance, args.joint_trace_hz,
                        {"checkpoint": health["checkpoint"], "server": server,
                         "session_id": session_id, "requested_loops": args.loops,
                         "rate_hz": args.rate, "speed": args.speed,
                         "exec_steps": args.exec_steps,
                         "switch_after": args.switch_after,
                         "blend_overlap": args.blend_overlap,
                         "margin": args.margin,
                         "max_soft_clip_rad": args.max_soft_clip_rad,
                         "phase_speeds": args.phase_speeds,
                         "approach_speed": args.approach_speed,
                         "transport_speed": args.transport_speed,
                         "place_speed": args.place_speed,
                         "grasp_duration_s": args.grasp_duration_s})
                    trace_session.start()
                pre_send_start_ns = time.monotonic_ns()
                pre_send_q = np.asarray(robot.right_arm.get_joint_positions(), dtype=float).reshape(-1)
                pre_send_gripper = float(robot.right_arm.get_gripper_distance())
                pre_send_end_ns = time.monotonic_ns()
                if (pre_send_q.shape != (6,) or not np.isfinite(pre_send_q).all()
                        or not np.isfinite(pre_send_gripper)):
                    raise ValueError("Invalid right-arm state before trace send")
                chunk_trace_dir = trace_session.prepare_chunk(
                    index, segment, waypoints, grippers,
                    {"checkpoint": health["checkpoint"], "server": server,
                     "session_id": session_id, "rate_hz": args.rate,
                     "speed": chunk_speed, "nominal_duration_s": chunk_duration,
                     "switch_delay_s": args.switch_delay,
                     "pre_send_read_start_monotonic_ns": pre_send_start_ns,
                     "pre_send_read_end_monotonic_ns": pre_send_end_ns,
                     "pre_send_right_q_rad": pre_send_q.tolist(),
                     "pre_send_right_gripper_m": pre_send_gripper,
                     "gripper_plan_units": "SDK fraction of maximum opening"})
                observation_age_ms = 1000 * (time.monotonic() - first_camera)
                if observation_age_ms > args.max_observation_age_ms:
                    print(f"[GUARD {index}] observation aged to "
                          f"{observation_age_ms:.1f} ms during trace preparation; "
                          "no chunk sent", flush=True)
                    break
            if initial_prefix is not None:
                current_q = robot.right_arm.get_joint_positions()
                drift = initial_state_drift(payload['state'][:6], current_q)
                age_ms = 1000*(time.monotonic()-first_camera)
                print(f"[INITIAL STABILITY] joint_drift_rad={drift:.5f} "
                      f"limit_rad=0.02000 observation_age_ms={age_ms:.1f} "
                      f"age_limit_ms={args.max_observation_age_ms:.1f}", flush=True)
                if drift > .02 or age_ms > args.max_observation_age_ms:
                    print("[INITIAL GUARD] start state changed or observation expired; no chunk sent", flush=True)
                    break
            print(f"[SEND {index}] RIGHT only, {len(waypoints)} waypoints, "
                  f"{chunk_duration:.3f}s, phase={phase}, speed={chunk_speed:.2f}", flush=True)
            send_begin_ns = time.monotonic_ns()
            robot.right_arm.update_joint_waypoint_chunk_with_gripper(
                waypoints, grippers, time_sec=chunk_duration,
                switch_delay_sec=args.switch_delay)
            send_return_ns = time.monotonic_ns()
            last_install = time.monotonic()
            last_duration = chunk_duration
            last_chunk_speed = chunk_speed
            sent_chunks += 1
            previous_segment = segment
            previous_waypoints = waypoints
            if chunk_trace_dir is not None:
                command_samples = None
                sample_read_ns = None
                sample_error = None
                try:
                    sample_read_start_ns = time.monotonic_ns()
                    command_samples = robot.right_arm.get_last_waypoint_command_samples()
                    sample_read_ns = [sample_read_start_ns, time.monotonic_ns()]
                except Exception as exc:
                    sample_error = repr(exc)
                command_metadata = trace_session.mark_chunk_sent(
                    chunk_trace_dir, send_begin_ns, send_return_ns,
                    command_samples, sample_read_ns, sample_error)
                print(f"[JOINT TRACE] loop={index} plan saved; "
                      f"controller_samples={command_metadata.get('controller_waypoint_sample_count', 0)} "
                      f"controller_sample_error={command_metadata.get('command_sample_error')} "
                      f"continuous session={args.joint_trace_dir}", flush=True)
            if args.probe_steps is not None:
                wait_time = (args.switch_delay
                             + args.probe_steps[0] / args.rate / args.speed)
            elif index == total_loops - 1:
                wait_time = chunk_duration + args.switch_delay + 0.1
            elif settled_mode and index == 0:
                wait_time = chunk_duration + args.switch_delay + 0.2
                print(f"[SETTLE] wait {wait_time:.3f}s before checking joint stability", flush=True)
            elif adaptive_mode and index == 0:
                # Start observation early enough that camera + inference + IK
                # latency still leaves the phase matcher its minimum overlap.
                commit_step = (grasp_commit_step(segment, initial_adaptive_observation_step)
                               if args.commit_grasp_chunk else None)
                if commit_step is not None:
                    print(f"[GRASP COMMIT] chunk=0 closes from step {commit_step}; "
                          f"finish checked chunk before fresh observation", flush=True)
                    restart_after_grasp = True
                    wait_time = chunk_duration + args.switch_delay + 0.1
                else:
                    wait_time = initial_adaptive_observation_step / args.rate / args.speed
                    print(f"[NEXT OBS] after chunk {index} at planned step "
                          f"{initial_adaptive_observation_step}/{len(segment)} "
                          f"({wait_time:.3f}s nominal after install; "
                          f"initial adaptive reserve=8; "
                          f"mode={'measured' if measured_phase_mode else 'clock'})",
                          flush=True)
            else:
                wait_time = switch_time
            remaining = wait_time - (time.monotonic() - last_install)
            if (measured_phase_mode and index == 0 and index < total_loops - 1
                    and not restart_after_grasp):
                try:
                    deadline = measured_observation_deadline(
                        last_install, len(waypoints), args.blend_overlap,
                        step_time, args.switch_delay)
                    reached_index, reached_error = wait_for_measured_phase(
                        robot.right_arm.get_joint_positions, waypoints,
                        initial_adaptive_observation_step, deadline,
                        not_before=(last_install + args.switch_delay
                                    + initial_adaptive_observation_step * step_time))
                    print(f"[MEASURED OBS] chunk={index} "
                          f"target={initial_adaptive_observation_step} "
                          f"reached={reached_index} error_rad={reached_error:.4f} "
                          f"elapsed_s={time.monotonic() - last_install:.3f}",
                          flush=True)
                except (TimeoutError, ValueError) as exc:
                    print(f"[ADAPTIVE GUARD] no next observation: {exc}", flush=True)
                    break
            elif remaining > 0:
                time.sleep(remaining)
            q_now = np.asarray(robot.right_arm.get_joint_positions(), dtype=float).reshape(-1)
            if settled_mode and index == 0:
                time.sleep(.2)
                drift = initial_state_drift(q_now, robot.right_arm.get_joint_positions())
                print(f"[SETTLE] joint drift over 200 ms: {drift:.5f} rad", flush=True)
                if drift > .01:
                    print("[ADAPTIVE GUARD] arm not settled; skip second observation", flush=True)
                    break
            grip_now = float(robot.right_arm.get_gripper_distance())
            print(f"[STATE {index}] right_q={np.round(q_now, 4).tolist()} "
                  f"right_gripper={grip_now:.4f}m", flush=True)
        if args.final_probe and (sent_chunks >= args.loops or completed_conditioned_grasp):
            final_timing = {}
            final_payload = read_observation(
                robot, image_client, wait_frame, enc_jpeg_b64, SCHEMA,
                session_id, sent_chunks, final_timing, task=args.task)
            if args.snapshot_dir:
                snapshot_dir = Path(args.snapshot_dir)
                ee_read_start = time.monotonic()
                right_ee_position, right_ee_euler = robot.right_arm.get_ee_pose_euler()
                ee_read_end = time.monotonic()
                right_ee_position = np.asarray(right_ee_position, dtype=float).reshape(3)
                right_ee_euler = np.asarray(right_ee_euler, dtype=float).reshape(3)
                if (not np.isfinite(right_ee_position).all()
                        or not np.isfinite(right_ee_euler).all()):
                    raise ValueError("Invalid measured right-arm end-effector pose")
                for camera_key, encoded in final_payload["images_jpeg_base64"].items():
                    camera_name = camera_key.rsplit(".", 1)[-1]
                    (snapshot_dir / f"final_{camera_name}.jpg").write_bytes(
                        base64.b64decode(encoded))
                (snapshot_dir / "final_state.json").write_text(
                    json.dumps({"state": final_payload["state"],
                                "right_ee_position_m": right_ee_position.tolist(),
                                "right_ee_euler_rad": right_ee_euler.tolist(),
                                "right_ee_read_start_monotonic_s": ee_read_start,
                                "right_ee_read_end_monotonic_s": ee_read_end,
                                "camera_receive_monotonic_s":
                                final_timing["camera_received"],
                                "state_monotonic_s": final_timing["state_received"]},
                               indent=2), encoding="utf-8")
            request = urllib.request.Request(
                server + "/predict", data=json.dumps(final_payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=30) as response:
                final_result = json.load(response)
            if (final_result.get("schema") != SCHEMA or final_result.get("session_id") != session_id
                    or final_result.get("seq") != sent_chunks):
                raise ValueError("SmolVLA final response identity mismatch")
            final_actions = np.asarray(final_result["actions"], dtype=float)
            if final_actions.shape != (50, 7) or not np.isfinite(final_actions).all():
                raise ValueError(f"Invalid final SmolVLA output: shape={final_actions.shape}")
            if args.snapshot_dir:
                np.save(Path(args.snapshot_dir) / "final_raw_actions.npy",
                        final_actions)
            close = np.flatnonzero(final_actions[:, 6] < 0.04)
            print(f"[FINAL PROBE] no robot command; right_xyz_first="
                  f"{np.round(right_xyz(final_actions[0]), 4).tolist()} "
                  f"right_xyz_last={np.round(right_xyz(final_actions[-1]), 4).tolist()} "
                  f"right_grip_min={np.min(final_actions[:, 6]):.4f}m "
                  f"first_below_0.04={int(close[0]) if close.size else 'none'} "
                  f"measured_right_grip={final_payload['state'][6]:.4f}m",
                  flush=True)
        print(f"[DONE] sent {sent_chunks} right-arm policy chunks", flush=True)
    finally:
        if robot is not None:
            # If a later inference/IK check failed while the previous chunk was
            # still running, let that installed chunk finish before homing.
            if last_install is not None:
                remaining = last_install + last_duration + args.switch_delay + 0.1 - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            try:
                if trace_session is not None:
                    try:
                        count = trace_session.stop(sent_chunks)
                        print(f"[JOINT TRACE] session stopped with {count} samples; "
                              f"saved to {args.joint_trace_dir}", flush=True)
                    except Exception as exc:
                        print(f"[JOINT TRACE] could not finalize trace: {exc}",
                              flush=True)
                if last_install is not None:
                    try:
                        print(f"[BEFORE HOME] right_gripper="
                              f"{robot.right_arm.get_gripper_distance():.4f}m "
                              f"right_q={np.round(robot.right_arm.get_joint_positions(), 4).tolist()}",
                              flush=True)
                    except Exception as exc:  # state read must not prevent homing
                        print(f"[BEFORE HOME] state read failed: {exc}", flush=True)
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
        elif trace_session is not None:
            trace_session.stop(sent_chunks)


if __name__ == "__main__":
    main()
