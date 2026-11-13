"""Regression checks for repeated poses in measured-phase scheduling."""

import unittest

import numpy as np

from station_lite_smolvla_right_guarded import (
    time_consistent_waypoint_index,
    wait_for_measured_phase,
)


class SimulatedClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class AdaptivePhaseTimingTests(unittest.TestCase):
    def test_repeated_joint_poses_cannot_finish_three_second_wait_immediately(self):
        clock = SimulatedClock()
        waypoints = np.zeros((20, 6))
        reached, error = wait_for_measured_phase(
            lambda: np.zeros(6), waypoints, target_index=9,
            deadline=6.0, not_before=3.0,
            clock=clock, sleep=clock.sleep)
        self.assertEqual(reached, 9)
        self.assertEqual(error, 0.0)
        self.assertGreaterEqual(clock.now, 3.0)

    def test_waits_for_actual_target_after_time_gate(self):
        clock = SimulatedClock()
        waypoints = np.zeros((20, 6))
        waypoints[9:, 0] = 0.2

        def measured():
            return np.array([0.2 if clock.now >= 4.0 else 0.0, 0, 0, 0, 0, 0])

        reached, error = wait_for_measured_phase(
            measured, waypoints, target_index=9,
            deadline=6.0, not_before=3.0,
            clock=clock, sleep=clock.sleep)
        self.assertEqual(reached, 9)
        self.assertEqual(error, 0.0)
        self.assertGreaterEqual(clock.now, 4.0)

    def test_ambiguous_joint_positions_use_clock_phase(self):
        errors = np.full(30, 0.008)
        errors[0] = 0.007
        errors[18] = 0.009
        self.assertEqual(time_consistent_waypoint_index(errors), 0)
        self.assertEqual(time_consistent_waypoint_index(errors, 20), 20)

    def test_large_clock_tracking_mismatch_is_not_hidden(self):
        errors = np.full(30, 0.12)
        errors[0] = 0.001
        selected = time_consistent_waypoint_index(errors, 20)
        self.assertGreater(errors[selected], 0.05)


if __name__ == "__main__":
    unittest.main()
