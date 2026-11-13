#!/usr/bin/env python3
"""Read-only timestamped joint tracing for installed action chunks.

The caller owns all robot commands. This module only saves a validated plan
and polls measured state after the caller has submitted that plan.
"""

import csv
import json
import threading
import time
from pathlib import Path

import numpy as np


TRACE_SCHEMA = "station_lite_right_joint_trace_v1"
SESSION_SCHEMA = "station_lite_right_joint_trace_session_v2"
FIELDS = ("read_start_monotonic_ns", "read_end_monotonic_ns", "read_end_unix_ns",
          "q0", "q1", "q2", "q3", "q4", "q5", "gripper_m")
SESSION_FIELDS = ("sample_index", *FIELDS)


class JointTraceSession:
    """Continuously sample measured state while the caller updates chunks."""

    def __init__(self, directory, read_joints, read_gripper, sample_hz, metadata):
        if not 5 <= sample_hz <= 60:
            raise ValueError("sample_hz must be between 5 and 60")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "chunks").mkdir()
        self.read_joints = read_joints
        self.read_gripper = read_gripper
        self.sample_hz = float(sample_hz)
        self.period_ns = int(1_000_000_000 / sample_hz)
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, name="joint-trace", daemon=True)
        self.error = None
        self.sample_count = 0
        self.metadata = {"schema": SESSION_SCHEMA, "status": "recording",
                         "sample_hz_requested": self.sample_hz, **metadata}
        self._write_metadata()

    def _write_metadata(self):
        (self.directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")

    def start(self):
        self.thread.start()
        if not self.ready_event.wait(timeout=2.0):
            raise RuntimeError("joint trace sampler did not start")
        if self.error:
            raise RuntimeError(f"joint trace sampler failed: {self.error}")

    def _sample(self):
        next_ns = time.monotonic_ns()
        try:
            with (self.directory / "samples.csv").open(
                    "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=SESSION_FIELDS)
                writer.writeheader()
                stream.flush()
                self.ready_event.set()
                while not self.stop_event.is_set():
                    start_ns = time.monotonic_ns()
                    q = np.asarray(self.read_joints(), dtype=np.float64).reshape(-1)
                    grip = float(self.read_gripper())
                    finish_ns = time.monotonic_ns()
                    if q.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(grip):
                        raise ValueError("invalid measured right-arm joint or gripper state")
                    writer.writerow({"sample_index": self.sample_count,
                                     "read_start_monotonic_ns": start_ns,
                                     "read_end_monotonic_ns": finish_ns,
                                     "read_end_unix_ns": time.time_ns(),
                                     **{f"q{i}": float(q[i]) for i in range(6)},
                                     "gripper_m": grip})
                    stream.flush()
                    self.sample_count += 1
                    next_ns += self.period_ns
                    remaining_ns = next_ns - time.monotonic_ns()
                    if remaining_ns > 0:
                        self.stop_event.wait(remaining_ns / 1_000_000_000)
                    else:
                        next_ns = time.monotonic_ns()
        except Exception as exc:  # keep the sampler failure visible to the caller
            self.error = repr(exc)
            self.ready_event.set()

    def prepare_chunk(self, index, actions, waypoints, grippers, metadata):
        actions = np.asarray(actions, dtype=np.float64)
        waypoints = np.asarray(waypoints, dtype=np.float64)
        grippers = np.asarray(grippers, dtype=np.float64)
        if (actions.ndim != 2 or actions.shape[1] not in (7, 14)
                or len(actions) == 0
                or waypoints.shape != (len(actions), 6)
                or grippers.shape != (len(actions),)
                or not np.isfinite(actions).all()
                or not np.isfinite(waypoints).all()
                or not np.isfinite(grippers).all()):
            raise ValueError("trace plan must contain finite aligned action/joint/gripper arrays")
        chunk_dir = self.directory / "chunks" / f"chunk_{index:02d}"
        chunk_dir.mkdir(parents=False, exist_ok=False)
        np.save(chunk_dir / "actions_target.npy", actions)
        np.save(chunk_dir / "right_joint_waypoints.npy", waypoints)
        np.save(chunk_dir / "right_gripper_targets_fraction.npy", grippers)
        chunk_metadata = {"schema": SESSION_SCHEMA, "status": "plan_saved_before_send",
                          "loop_index": int(index), **metadata,
                          "action_dim": int(actions.shape[1]),
                          "action_representation": "right_ee_pose_gripper" if actions.shape[1] == 7 else "dual_ee_pose_gripper"}
        (chunk_dir / "metadata.json").write_text(
            json.dumps(chunk_metadata, indent=2) + "\n", encoding="utf-8")
        return chunk_dir

    def mark_chunk_sent(self, chunk_dir, send_begin_ns, send_return_ns,
                        command_samples=None, command_sample_read_ns=None,
                        command_sample_error=None):
        chunk_dir = Path(chunk_dir)
        metadata_path = chunk_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({"status": "command_sent",
                         "send_begin_monotonic_ns": int(send_begin_ns),
                         "send_return_monotonic_ns": int(send_return_ns),
                         "command_sample_read_monotonic_ns": command_sample_read_ns,
                         "command_sample_error": command_sample_error})
        if command_samples is not None:
            samples = np.asarray(command_samples, dtype=np.float64)
            if samples.ndim == 2 and samples.shape[1] == 15 and np.isfinite(samples).all():
                np.save(chunk_dir / "controller_waypoint_samples.npy", samples)
                metadata["controller_waypoint_sample_count"] = int(len(samples))
                metadata["controller_waypoint_columns"] = [
                    "t_rel", "q1", "q2", "q3", "q4", "q5", "q6",
                    "x", "y", "z", "roll", "pitch", "yaw", "gripper",
                    "planning_time_s"]
            else:
                metadata["command_sample_error"] = (
                    "unexpected sample shape or non-finite values: "
                    f"{samples.shape}")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n",
                                 encoding="utf-8")
        return metadata

    def stop(self, sent_chunks=0):
        if self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=3.0)
        if self.thread.is_alive():
            self.error = self.error or "sampler thread did not stop within 3 seconds"
        self.metadata.update({"status": "trace_complete" if not self.error else "trace_error",
                              "sample_count": self.sample_count,
                              "sent_chunks": int(sent_chunks),
                              "sampler_error": self.error,
                              "timing_note": "monotonic timestamps bracket SDK reads; "
                                             "controller command samples are saved per chunk "
                                             "only when the SDK returns valid nonempty rows"})
        self._write_metadata()
        return self.sample_count


def prepare_trace(directory, actions, waypoints, grippers, metadata):
    """Create a new trace directory before a command is sent; never overwrite one."""
    directory = Path(directory)
    actions = np.asarray(actions, dtype=np.float64)
    waypoints = np.asarray(waypoints, dtype=np.float64)
    grippers = np.asarray(grippers, dtype=np.float64)
    if (actions.ndim != 2 or actions.shape[1] not in (7, 14)
            or len(actions) == 0
            or waypoints.shape != (len(actions), 6)
            or grippers.shape != (len(actions),)
            or not np.isfinite(actions).all()
            or not np.isfinite(waypoints).all()
            or not np.isfinite(grippers).all()):
        raise ValueError("trace plan must contain finite, aligned action/joint/gripper arrays")
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "actions_target.npy", actions)
    np.save(directory / "right_joint_waypoints.npy", waypoints)
    np.save(directory / "right_gripper_targets_fraction.npy", grippers)
    (directory / "metadata.json").write_text(
        json.dumps({"schema": TRACE_SCHEMA, "status": "plan_saved_before_send",
                    **metadata, "action_dim": int(actions.shape[1]),
                    "action_representation": "right_ee_pose_gripper" if actions.shape[1] == 7 else "dual_ee_pose_gripper"}, indent=2) + "\n", encoding="utf-8")
    return directory


def record_joint_trace(directory, read_joints, read_gripper,
                       send_begin_monotonic_ns, send_return_monotonic_ns,
                       duration_s, switch_delay_s, sample_hz,
                       monotonic_ns=time.monotonic_ns, unix_ns=time.time_ns,
                       sleep=time.sleep):
    """Poll measured joints until the nominal chunk end; save actual read times.

    The two read timestamps bracket the SDK calls, so downstream analysis can
    account for read latency. The waypoint schedule is only nominal; the SDK
    does not expose controller-internal execution timestamps here.
    """
    directory = Path(directory)
    if (send_begin_monotonic_ns <= 0 or send_return_monotonic_ns < send_begin_monotonic_ns
            or duration_s <= 0 or switch_delay_s < 0 or not 5 <= sample_hz <= 60):
        raise ValueError("invalid trace timing")
    period_ns = int(1_000_000_000 / sample_hz)
    end_ns = send_begin_monotonic_ns + int((duration_s + switch_delay_s + 0.1) * 1_000_000_000)
    next_ns = monotonic_ns()
    count = 0
    with (directory / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        stream.flush()
        while True:
            start_ns = monotonic_ns()
            q = np.asarray(read_joints(), dtype=np.float64).reshape(-1)
            grip = float(read_gripper())
            finish_ns = monotonic_ns()
            if q.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(grip):
                raise ValueError("invalid measured right-arm joint or gripper state")
            row = {"read_start_monotonic_ns": start_ns,
                   "read_end_monotonic_ns": finish_ns,
                   "read_end_unix_ns": unix_ns(),
                   **{f"q{i}": float(q[i]) for i in range(6)},
                   "gripper_m": grip}
            writer.writerow(row)
            stream.flush()
            count += 1
            if finish_ns >= end_ns:
                break
            next_ns += period_ns
            remaining_ns = next_ns - monotonic_ns()
            if remaining_ns > 0:
                sleep(remaining_ns / 1_000_000_000)
            else:
                next_ns = monotonic_ns()
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({"status": "trace_complete", "send_begin_monotonic_ns": send_begin_monotonic_ns,
                     "send_return_monotonic_ns": send_return_monotonic_ns,
                     "nominal_duration_s": duration_s, "switch_delay_s": switch_delay_s,
                     "sample_hz_requested": sample_hz, "sample_count": count,
                     "nominal_end_monotonic_ns": end_ns,
                     "timing_note": "monotonic timestamps bracket SDK reads; controller internal clock unavailable"})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return count
