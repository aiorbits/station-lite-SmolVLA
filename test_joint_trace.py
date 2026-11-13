#!/usr/bin/env python3
"""Offline timing and file-format checks for the read-only joint trace helper."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from joint_trace import JointTraceSession, prepare_trace, record_joint_trace
from analyze_joint_trace import analyze


class FakeClock:
    def __init__(self):
        self.now = 1_000_000_000

    def monotonic_ns(self):
        return self.now

    def unix_ns(self):
        return 2_000_000_000 + self.now

    def sleep(self, seconds):
        self.now += int(seconds * 1_000_000_000)


class JointTraceTest(unittest.TestCase):
    def test_native_right7_and_legacy14_session_plans(self):
        for width in (7,14):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as tmp:
                session = JointTraceSession(Path(tmp)/'session', lambda:np.zeros(6), lambda:.08, 60, {})
                actions = np.zeros((50,width))
                chunk = session.prepare_chunk(0, actions, np.zeros((50,6)), np.ones(50), {})
                np.testing.assert_array_equal(np.load(chunk/'actions_target.npy'), actions)
                self.assertEqual(json.loads((chunk/'metadata.json').read_text())['action_dim'], width)
                status = session.mark_chunk_sent(chunk, 1, 2, np.zeros((2,15)))
                self.assertEqual(status['status'], 'command_sent')
                self.assertEqual(session.stop(1), 0)
                with self.assertRaises(ValueError):
                    session.prepare_chunk(1, np.zeros((50,8)), np.zeros((50,6)), np.ones(50), {})
                with self.assertRaises(ValueError):
                    session.prepare_chunk(1, actions, np.zeros((49,6)), np.ones(50), {})

    def test_timestamps_and_plan_are_saved_without_robot_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trace"
            actions = np.zeros((3, 14))
            waypoints = np.array([[0] * 6, [0.01] * 6, [0.02] * 6], dtype=float)
            grippers = np.ones(3)
            prepare_trace(directory, actions, waypoints, grippers, {"checkpoint": "test"})
            clock = FakeClock()
            count = record_joint_trace(
                directory, lambda: np.full(6, clock.now / 1e11), lambda: 0.08,
                1_000_000_000, 1_000_000_000, duration_s=0.1,
                switch_delay_s=0, sample_hz=20,
                monotonic_ns=clock.monotonic_ns, unix_ns=clock.unix_ns,
                sleep=clock.sleep)
            with (directory / "samples.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(count, len(rows))
            self.assertGreaterEqual(count, 3)
            self.assertEqual(metadata["status"], "trace_complete")
            self.assertEqual(metadata["sample_count"], count)
            self.assertEqual(np.load(directory / "right_joint_waypoints.npy").shape, (3, 6))
            self.assertTrue(all(int(row["read_end_monotonic_ns"]) >=
                                int(row["read_start_monotonic_ns"]) for row in rows))
            self.assertEqual(int(rows[-1]["read_end_monotonic_ns"]), 1_200_000_000)
            summary = analyze(directory)
            self.assertEqual(summary["sample_count"], count)
            self.assertEqual(summary["planned_waypoint_count"], 3)
            self.assertAlmostEqual(summary["sample_interval_median_ms"], 50.0)
            with self.assertRaises(FileExistsError):
                prepare_trace(directory, actions, waypoints, grippers, {})


if __name__ == "__main__":
    unittest.main()
