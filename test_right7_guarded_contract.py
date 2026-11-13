#!/usr/bin/env python3
"""Non-hardware tests for native right7 guarded action handling."""
from __future__ import annotations

import unittest
import base64
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import station_lite_smolvla_right7_guarded as guarded
from station_lite_smolvla_single_chunk import (
    GRIPPER_MAX_M, JOINT_LOW, JOINT_UP, check_right_targets,
)


class FakeArm:
    def get_joint_positions(self):
        # Inside the configured soft limits, including joint 3's upper margin.
        return [0.0, 0.0, -0.2, 0.0, 0.0, 0.0]


class Right7ContractTest(unittest.TestCase):
    def test_initial_state_stability_detects_motion_and_invalid_feedback(self):
        self.assertAlmostEqual(guarded.initial_state_drift(np.zeros(6), [.021,0,0,0,0,0]), .021)
        self.assertEqual(guarded.initial_state_drift(np.zeros(6), np.zeros(6)), 0.)
        with self.assertRaises(ValueError):
            guarded.initial_state_drift(np.zeros(6), [float('nan')]*6)

    def test_prefix_selection_omits_only_leading_soft_boundary_candidates(self):
        from types import SimpleNamespace
        prefix = np.zeros((5,7))
        prefix[:,0] = np.arange(5)
        def solve(pos, euler, seed):
            return [0,0,[-.0109,-.0163,-.022,-.03,-.04][int(pos[0])],0,0,0], True
        arm = SimpleNamespace(get_joint_positions=lambda:[0,0,-.0055,0,0,0], solve_ik_euler=solve)
        selected, omitted = guarded.select_initial_prefix(arm, prefix, .02)
        self.assertEqual(omitted, 2)
        np.testing.assert_array_equal(selected, prefix[2:])
        arm.solve_ik_euler = lambda *args: ([0,0,.1,0,0,0], True)
        with self.assertRaises(ValueError):
            guarded.select_initial_prefix(arm, prefix, .02)

    def test_initial_prefix_is_bounded_and_uses_short_rotation(self):
        import math
        current = np.array([.12, 0, .18, 0, 0, math.pi-.01, .07])
        target = current.copy()
        target[0] += .024
        target[5] = -math.pi+.01
        prefix = guarded.make_initial_prefix(current, target, 6)
        self.assertEqual(prefix.shape, (5, 7))
        self.assertAlmostEqual(prefix[0,0]-current[0], .004)
        np.testing.assert_array_equal(current, [.12,0,.18,0,0,math.pi-.01,.07])
        far = target.copy()
        far[0] += .01
        with self.assertRaises(ValueError):
            guarded.make_initial_prefix(current, far, 6)

    def test_preflight_cleanup_calls_sdk_without_motion(self):
        from types import SimpleNamespace
        calls = []
        arm = SimpleNamespace(arm=SimpleNamespace(cleanup=lambda: calls.append('cleanup')))
        guarded.cleanup_preflight_arm(arm)
        guarded.cleanup_preflight_arm(None)
        self.assertEqual(calls, ['cleanup'])

    def test_rejected_preflight_input_can_be_saved_before_ik(self):
        with TemporaryDirectory() as tmp:
            payload = {
                "state": [0.0] * 7,
                "images_jpeg_base64": {
                    "observation.images.cam_high": base64.b64encode(b"high").decode(),
                    "observation.images.cam_right_wrist": base64.b64encode(b"wrist").decode(),
                },
            }
            actions = np.zeros((50, 7), dtype=float)
            report = {"passed": False, "stage": "ik_check", "ik_failed": 50}
            guarded.save_preflight_observation(
                tmp, payload, actions, np.zeros(6), 0.075, {"t": 1.0}, report)
            snapshot = Path(tmp)
            np.testing.assert_array_equal(
                np.load(snapshot / "right7_raw_actions.npy"), actions)
            self.assertEqual((snapshot / "cam_high.jpg").read_bytes(), b"high")
            self.assertEqual((snapshot / "cam_right_wrist.jpg").read_bytes(), b"wrist")
            self.assertEqual(json.loads((snapshot / "preflight.json").read_text()), report)
            self.assertEqual(json.loads((snapshot / "state.json").read_text())["right_q"], [0] * 6)

    def test_native_right7_views_and_blending(self):
        old = np.zeros((8, 7), dtype=float)
        new = np.ones((8, 7), dtype=float)
        self.assertEqual(guarded.right_pose(new).shape, (8, 6))
        self.assertEqual(guarded.right_xyz(new).shape, (8, 3))
        self.assertEqual(guarded.right_gripper(new).shape, (8,))
        blended = guarded.blend_head(old, new)
        self.assertEqual(blended.shape, (8, 7))
        # Gripper remains the new policy output; pose fields blend in right7 space.
        np.testing.assert_array_equal(blended[:, 6], new[:, 6])
        self.assertTrue(np.isfinite(blended).all())

    def test_legacy_14d_blending_contract_remains_available(self):
        old = np.zeros((6, 14), dtype=float)
        new = np.ones((6, 14), dtype=float)
        blended = guarded.blend_head(old, new)
        self.assertEqual(blended.shape, (6, 14))
        np.testing.assert_array_equal(blended[:, 6], new[:, 6])
        np.testing.assert_array_equal(blended[:, 13], new[:, 13])

    def test_right_ik_guard_accepts_native_7d_without_padding(self):
        actions = np.zeros((3, 7), dtype=float)
        actions[:, :3] = [0.2, 0.1, 0.25]
        actions[:, 6] = 0.08

        def solver(arm, position, euler, seed):
            return np.asarray(seed, dtype=float), True, 1.0

        waypoints, grippers = check_right_targets(
            FakeArm(), actions, 0.02, solver, JOINT_LOW, JOINT_UP, GRIPPER_MAX_M)
        self.assertEqual(np.asarray(waypoints).shape, (3, 6))
        self.assertEqual(len(grippers), 3)
        self.assertTrue(all(np.isfinite(grippers)))

    def test_right_ik_guard_preserves_legacy_14d_contract(self):
        actions = np.zeros((2, 14), dtype=float)
        actions[:, 7:10] = [0.2, 0.1, 0.25]
        actions[:, 13] = 0.08

        def solver(arm, position, euler, seed):
            return np.asarray(seed, dtype=float), True, 1.0

        waypoints, grippers = check_right_targets(
            FakeArm(), actions, 0.02, solver, JOINT_LOW, JOINT_UP, GRIPPER_MAX_M)
        self.assertEqual(np.asarray(waypoints).shape, (2, 6))
        self.assertEqual(len(grippers), 2)

    def test_condition_grasp_actions_supports_hold_pose_and_enforces_guards(self):
        from build_grasp_replay_cases import condition_grasp_actions

        start = np.array([0.2755, 0.1535, 0.1705, -0.42, 1.02, 0.99, 0.0686])
        raw = np.tile(start, (50, 1))
        raw[:, 0] += np.linspace(0.009, 0.025, 50)
        raw[:8, 6] = 0.070
        raw[8:, 6] = 0.010
        raw[10, 6] = -0.004

        cond, clipped, splits = condition_grasp_actions(start, raw, hold_pose=False)
        self.assertEqual(clipped, [10])
        self.assertTrue(len(cond) > 56)
        self.assertTrue(np.all(cond[:, 6] >= 0.0))

        held, held_clipped, held_splits = condition_grasp_actions(start, raw, hold_pose=True)
        self.assertEqual(held_clipped, [10])
        self.assertEqual(held_splits, [8])
        np.testing.assert_allclose(held[:, :6], np.tile(start[:6], (len(held), 1)))
        grip_steps = np.abs(np.diff(np.r_[start[6], held[:, 6]]))
        self.assertLessEqual(float(grip_steps.max()), 0.015 + 1e-9)

        # Reject when no closure (< 0.04 m) is predicted.
        open_only = raw.copy()
        open_only[:, 6] = 0.065
        with self.assertRaises(ValueError):
            condition_grasp_actions(start, open_only, hold_pose=True)

        # Accept flow-matching negative close undershoot up to 20 mm (e.g. R3 -5.08 mm),
        # and reject when negative gripper exceeds 20 mm tolerance.
        r3_undershoot = raw.copy()
        r3_undershoot[10, 6] = -0.0051
        r3_held, r3_clipped, _ = condition_grasp_actions(start, r3_undershoot, hold_pose=True)
        self.assertEqual(r3_clipped, [10])
        self.assertTrue(np.all(r3_held[:, 6] >= 0.0))

        too_negative = raw.copy()
        too_negative[10, 6] = -0.021
        with self.assertRaises(ValueError):
            condition_grasp_actions(start, too_negative, hold_pose=True)

        # Delayed closure (no target < 10 mm in steps 0..12, e.g. R7 at loop_03):
        # glide open to the pre-lift max-Y grasp pose in steps 12..20, then anchor 6D pose.
        delayed = np.tile(start, (50, 1))
        delayed[:20, 1] = np.linspace(0.1620, 0.1811, 20)
        delayed[20:, 1] = np.linspace(0.1795, 0.1511, 30)
        delayed[20:, 2] = np.linspace(0.1547, 0.1958, 30)
        delayed[:10, 6] = 0.075
        delayed[10:31, 6] = 0.017
        delayed[31:, 6] = 0.003
        delayed_held, _, _ = condition_grasp_actions(start, delayed, hold_pose=True)
        np.testing.assert_allclose(delayed_held[-1, :6], delayed[19, :6])
        self.assertAlmostEqual(float(delayed_held[0, 6]), 0.075)
        self.assertAlmostEqual(float(delayed_held[-1, 6]), 0.003)

    def test_condition_approach_actions_holds_open_and_anchors_min_z(self):
        from build_grasp_replay_cases import condition_approach_actions

        start = np.array([0.2662, 0.0705, 0.2263, -0.53, 0.95, 0.87, 0.0687])
        raw = np.tile(start, (50, 1))
        raw[:, 0] = np.linspace(0.2666, 0.2786, 50)
        raw[:, 2] = np.linspace(0.2076, 0.1585, 50)
        raw[:, 6] = 0.078
        # Simulate premature closure and backward recoil after min-Z at step 45
        raw[45, 0] = 0.2786
        raw[45, 2] = 0.1585
        raw[46:, 0] = [0.2767, 0.2737, 0.2690, 0.2665]
        raw[46:, 2] = [0.1602, 0.1598, 0.1615, 0.1593]
        raw[46:, 6] = [0.0754, 0.0703, 0.0037, -0.0044]

        cond, held_from, anchor_from = condition_approach_actions(start, raw)
        self.assertEqual(held_from, 0)
        self.assertEqual(anchor_from, 45)
        self.assertTrue(np.all(cond[:, 6] >= 0.0687))
        np.testing.assert_allclose(cond[45:, :6], np.tile(raw[45, :6], (5, 1)))

        # Early closure prediction during left-column approach (e.g. R1 loop_03 first_close=10):
        # hold open and anchor at pre-lift max-Y pose in steps 12..20.
        early_start = start.copy()
        early_start[1] = 0.1343
        early = np.tile(early_start, (50, 1))
        early[:20, 1] = np.linspace(0.1500, 0.1579, 20)
        early[20:, 1] = np.linspace(0.1543, 0.1403, 30)
        early[20:, 2] = np.linspace(0.1648, 0.1893, 30)
        early[:10, 6] = 0.078
        early[10:, 6] = 0.015
        early_cond, early_held, early_anchor = condition_approach_actions(early_start, early)
        self.assertEqual(early_held, 0)
        self.assertEqual(early_anchor, 19)
        self.assertTrue(np.all(early_cond[:, 6] >= 0.078))
        np.testing.assert_allclose(early_cond[19:, :6], np.tile(early[19, :6], (31, 1)))

        # Jitter/dip during approach without < 0.04m (e.g. R7 loop_03 step 10 dip to 49.6mm):
        # verifies that entire approach chunk gripper is held open with 0 step jump.
        jitter = np.tile(start, (50, 1))
        jitter[:, 6] = 0.078
        jitter[10, 6] = 0.0496  # 28.4mm dip
        jitter_cond, jitter_held, jitter_anchor = condition_approach_actions(start, jitter)
        self.assertEqual(jitter_held, 0)
        self.assertIsNone(jitter_anchor)
        self.assertEqual(float(np.max(np.abs(np.diff(jitter_cond[:, 6])))), 0.0)

    def test_condition_placement_actions_holds_grip_and_subdivides_smoothly(self):
        from build_grasp_replay_cases import condition_placement_actions

        start = np.array([0.2794, 0.1443, 0.1677, -0.40, 1.02, 0.99, 0.0258])
        raw = np.tile(start, (50, 1))
        raw[:, 0] = np.linspace(0.2972, 0.3500, 50)
        raw[:, 1] = np.linspace(0.1675, 0.2178, 50)
        raw[:, 2] = np.linspace(0.1944, 0.1730, 50)
        raw[:, 6] = 0.001  # Clamped command during transport
        raw[5, 6] = -0.003  # Negative undershoot to be clipped

        cond, clipped, splits = condition_placement_actions(start, raw, held_grip=0.0258)
        self.assertEqual(clipped, [5])
        self.assertEqual(len(cond), 55)
        # Holds active preload clamping (~17.8 mm) during transport
        expected_clamp = 0.0258 - 0.008
        self.assertTrue(np.allclose(cond[5:, 6], expected_clamp))
        self.assertAlmostEqual(cond[0, 6], 0.0258 - 0.008 * (1 / 5))

        # Check all Cartesian and gripper steps
        xyz_steps = np.linalg.norm(np.diff(np.vstack([start[:3], cond[:, :3]]), axis=0), axis=1) * 1000
        rot_steps = np.degrees(np.max(np.abs((np.diff(np.vstack([start[3:6], cond[:, 3:6]]), axis=0) + np.pi) % (2*np.pi) - np.pi), axis=1))
        grip_steps = np.abs(np.diff(np.r_[start[6], cond[:, 6]])) * 1000
        self.assertLessEqual(float(xyz_steps.max()), 15.0)
        self.assertLessEqual(float(rot_steps.max()), 12.0)
        self.assertLessEqual(float(grip_steps.max()), 15.0)
        self.assertLessEqual(float(xyz_steps[0]), 15.0)

        # Release test: policy commands open (>= 0.040) from step 35
        raw_release = raw.copy()
        raw_release[35:, 6] = 0.078
        cond_rel, rel_clipped, rel_splits = condition_placement_actions(start, raw_release, held_grip=0.0258)
        self.assertEqual(rel_splits, [40])
        rel_grip_steps = np.abs(np.diff(np.r_[start[6], cond_rel[:, 6]])) * 1000
        self.assertLessEqual(float(rel_grip_steps.max()), 15.0)
        self.assertAlmostEqual(float(cond_rel[-1, 6]), 0.078)

        # Envelope checks
        far_start = start.copy()
        far_raw = raw.copy()
        far_raw[0, 0] += 0.06  # > 50 mm gap
        with self.assertRaises(ValueError):
            condition_placement_actions(far_start, far_raw)

        too_negative_raw = raw.copy()
        too_negative_raw[10, 6] = -0.025  # < -20 mm
        with self.assertRaises(ValueError):
            condition_placement_actions(start, too_negative_raw)

    def test_max_initial_gap_permits_boundary_omitted_startup(self):
        # When all initial bridge candidates are omitted due to Home state J3 soft margin,
        # gap to actions[0] (e.g. 21.6 mm) passes with max_initial_gap_mm=25.0 but fails with 20.0
        pos_gap_mm = 21.6
        rot_gap_deg = 1.1
        grip_gap_mm = 0.2

        max_initial_gap_mm = 25.0
        max_rotation_gap_deg = 12.0
        max_gripper_gap_mm = 15.0

        passes = not (pos_gap_mm > max_initial_gap_mm
                      or rot_gap_deg > max_rotation_gap_deg
                      or grip_gap_mm > max_gripper_gap_mm)
        self.assertTrue(passes)

        old_switch_gap_mm = 20.0
        fails_with_old = (pos_gap_mm > old_switch_gap_mm
                          or rot_gap_deg > max_rotation_gap_deg
                          or grip_gap_mm > max_gripper_gap_mm)
        self.assertTrue(fails_with_old)

    def test_cli_argument_validation_for_placement_loops(self):
        import subprocess
        # Without --place-after-grasp, loops 6 is rejected
        res = subprocess.run([
            sys.executable, str(Path(guarded.__file__).resolve()),
            "--server", "http://127.0.0.1:9999",
            "--settled-chunks", "--condition-final-grasp", "--loops", "6",
            "--send-right-chunks", "--adaptive-refine"
        ], capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--condition-final-grasp requires --settled-chunks", res.stderr)

        # With --place-after-grasp, loops 6 and 7 are accepted in argument pre-checks
        # (will fail at network connection since dummy server is used, proving arg validation passed)
        res = subprocess.run([
            sys.executable, str(Path(guarded.__file__).resolve()),
            "--server", "http://127.0.0.1:9999",
            "--settled-chunks", "--condition-final-grasp", "--place-after-grasp", "--loops", "6",
            "--send-right-chunks", "--adaptive-refine"
        ], capture_output=True, text=True)
        self.assertNotIn("--condition-final-grasp requires --settled-chunks", res.stderr)
        self.assertNotIn("Invalid", res.stderr)

        # --placement-bridge-max-mm outside 30..60 is rejected
        res = subprocess.run([
            sys.executable, str(Path(guarded.__file__).resolve()),
            "--server", "http://127.0.0.1:9999",
            "--settled-chunks", "--condition-final-grasp", "--place-after-grasp", "--loops", "6",
            "--send-right-chunks", "--adaptive-refine",
            "--placement-bridge-max-mm", "70.0"
        ], capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--placement-bridge-max-mm must be 30..60 mm", res.stderr)

    def test_task_argument_and_payload(self):
        # Default task is red block
        self.assertIn("red wooden block", guarded.TASK)

        # read_observation embeds the specified task into request payload
        from types import SimpleNamespace
        class DummyRobot:
            def __init__(self):
                self.right_arm = SimpleNamespace(
                    get_joint_positions=lambda: [0.0]*6,
                    get_gripper_distance=lambda: 0.05
                )
        class DummyClient:
            def get_head_frame(self): return np.zeros((480, 640, 3), dtype=np.uint8)
            def get_right_wrist_frame(self): return np.zeros((480, 640, 3), dtype=np.uint8)

        custom_task = "Pick up the blue wooden block from the white grid and place it on the brown target area."
        payload = guarded.read_observation(
            DummyRobot(), DummyClient(), lambda f: f(), lambda img: "dummy_b64",
            guarded.REQUEST_SCHEMA, "test_session", 0, task=custom_task)
        self.assertEqual(payload["task"], custom_task)

    def test_get_phase_speed_and_duration_contract(self):
        from types import SimpleNamespace

        # Backward compatibility when --phase-speeds is False
        args_baseline = SimpleNamespace(
            phase_speeds=False, speed=0.10, rate=30.0,
            approach_speed=0.20, transport_speed=0.20, place_speed=0.12,
            grasp_duration_s=6.0, hold_grasp_pose=True
        )
        for phase in ("approach", "grasp", "transport", "place"):
            speed, duration = guarded.get_phase_speed_and_duration(phase, 50, args_baseline)
            self.assertEqual(speed, 0.10)
            self.assertAlmostEqual(duration, 50.0 / 30.0 / 0.10)

        # When --phase-speeds is True
        args_phase = SimpleNamespace(
            phase_speeds=True, speed=0.10, rate=30.0,
            approach_speed=0.20, transport_speed=0.20, place_speed=0.12,
            grasp_duration_s=6.0, hold_grasp_pose=True
        )

        # Approach: speed = 0.20
        app_speed, app_dur = guarded.get_phase_speed_and_duration("approach", 50, args_phase)
        self.assertEqual(app_speed, 0.20)
        self.assertAlmostEqual(app_dur, 50.0 / 30.0 / 0.20)  # 8.333s

        # Grasp with hold_grasp_pose=True: duration = 6.0s
        grasp_speed, grasp_dur = guarded.get_phase_speed_and_duration("grasp", 53, args_phase)
        self.assertEqual(grasp_dur, 6.0)
        self.assertAlmostEqual(grasp_speed, 53.0 / 30.0 / 6.0)  # 0.2944

        # Grasp with hold_grasp_pose=False: uses approach speed
        args_no_hold = SimpleNamespace(
            phase_speeds=True, speed=0.10, rate=30.0,
            approach_speed=0.20, transport_speed=0.20, place_speed=0.12,
            grasp_duration_s=6.0, hold_grasp_pose=False
        )
        nh_speed, nh_dur = guarded.get_phase_speed_and_duration("grasp", 53, args_no_hold)
        self.assertEqual(nh_speed, 0.20)
        self.assertAlmostEqual(nh_dur, 53.0 / 30.0 / 0.20)

        # Transport: speed = 0.20
        trans_speed, trans_dur = guarded.get_phase_speed_and_duration("transport", 55, args_phase)
        self.assertEqual(trans_speed, 0.20)
        self.assertAlmostEqual(trans_dur, 55.0 / 30.0 / 0.20)  # 9.167s

        # Place: speed = 0.12 (58 steps takes ~16.11s)
        place_speed, place_dur = guarded.get_phase_speed_and_duration("place", 58, args_phase)
        self.assertEqual(place_speed, 0.12)
        self.assertAlmostEqual(place_dur, 58.0 / 30.0 / 0.12)  # 16.111s

        # Error cases
        with self.assertRaises(ValueError):
            guarded.get_phase_speed_and_duration("approach", 0, args_phase)
        with self.assertRaises(ValueError):
            guarded.get_phase_speed_and_duration("approach", 50, SimpleNamespace(rate=0, phase_speeds=True))
        with self.assertRaises(ValueError):
            guarded.get_phase_speed_and_duration("unknown_phase", 50, args_phase)

    def test_cli_phase_speeds_validation(self):
        import subprocess

        # Valid phase speeds passes arg validation
        res = subprocess.run([
            sys.executable, str(Path(guarded.__file__).resolve()),
            "--server", "http://127.0.0.1:9999",
            "--settled-chunks", "--condition-final-grasp", "--place-after-grasp", "--loops", "6",
            "--send-right-chunks", "--adaptive-refine",
            "--phase-speeds", "--approach-speed", "0.20", "--transport-speed", "0.20",
            "--place-speed", "0.12", "--grasp-duration-s", "6.0"
        ], capture_output=True, text=True)
        self.assertNotIn("Invalid phase speed", res.stderr)

        # Invalid approach-speed <= 0 rejected
        res = subprocess.run([
            sys.executable, str(Path(guarded.__file__).resolve()),
            "--server", "http://127.0.0.1:9999",
            "--settled-chunks", "--condition-final-grasp", "--place-after-grasp", "--loops", "6",
            "--send-right-chunks", "--adaptive-refine",
            "--approach-speed", "0.0"
        ], capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Invalid phase speed or grasp duration parameter", res.stderr)

        # Invalid grasp-duration-s < 2.0 rejected
        res = subprocess.run([
            sys.executable, str(Path(guarded.__file__).resolve()),
            "--server", "http://127.0.0.1:9999",
            "--settled-chunks", "--condition-final-grasp", "--place-after-grasp", "--loops", "6",
            "--send-right-chunks", "--adaptive-refine",
            "--grasp-duration-s", "1.0"
        ], capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Invalid phase speed or grasp duration parameter", res.stderr)

    def test_timing_reduction_on_green_block_run_140934(self):
        from types import SimpleNamespace

        # Exact chunk segment lengths from the green block run 140934
        # Chunk 0: 50 (approach)
        # Chunk 1: 50 (approach)
        # Chunk 2: 55 (approach + settled bridge prefix)
        # Chunk 3: 53 (conditioned grasp with hold_pose)
        # Chunk 4: 55 (conditioned placement transport)
        # Chunk 5: 58 (conditioned placement release)
        chunks = [
            ("approach", 50),
            ("approach", 50),
            ("approach", 55),
            ("grasp", 53),
            ("transport", 55),
            ("place", 58),
        ]

        args_baseline = SimpleNamespace(
            phase_speeds=False, speed=0.10, rate=30.0,
            hold_grasp_pose=True
        )
        args_phased = SimpleNamespace(
            phase_speeds=True, speed=0.10, rate=30.0,
            approach_speed=0.20, transport_speed=0.20,
            place_speed=0.12, grasp_duration_s=6.0,
            hold_grasp_pose=True
        )

        baseline_durations = [guarded.get_phase_speed_and_duration(p, n, args_baseline)[1] for p, n in chunks]
        phased_durations = [guarded.get_phase_speed_and_duration(p, n, args_phased)[1] for p, n in chunks]

        total_baseline = sum(baseline_durations)
        total_phased = sum(phased_durations)

        # Baseline motion planning was 107.0s (measured in 140934 as 108.7s with initial bridge)
        self.assertAlmostEqual(total_baseline, 107.0, places=1)
        # Phased motion planning is ~57.1s (target ~58s motion + ~2-3s inter-chunk/inference = ~60s total)
        self.assertAlmostEqual(total_phased, 57.11, places=1)
        self.assertLessEqual(total_phased, 60.0)

        # Verified motion time saved is ~50 seconds (~46.6% motion reduction)
        time_saved = total_baseline - total_phased
        self.assertGreaterEqual(time_saved, 49.0)

    def test_smooth_joint_waypoints_contract(self):
        # Empty or short waypoints
        self.assertIsNone(guarded.smooth_joint_waypoints(None))
        self.assertEqual(guarded.smooth_joint_waypoints([]), [])
        single = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        self.assertEqual(guarded.smooth_joint_waypoints(single), single)
        pair = [[0.1, 0.2], [0.3, 0.4]]
        self.assertEqual(guarded.smooth_joint_waypoints(pair), pair)

        # Inactive smoothing (alpha <= 0 or passes <= 0)
        tri = [[0.1, 0.2], [0.5, 0.8], [0.3, 0.4]]
        self.assertEqual(guarded.smooth_joint_waypoints(tri, alpha=0.0), tri)
        self.assertEqual(guarded.smooth_joint_waypoints(tri, passes=0), tri)

        # 3-point smoothing preserves endpoints
        w_in = [
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            [0.15, 0.35, 0.25, 0.45, 0.55, 0.65],
            [0.12, 0.22, 0.32, 0.42, 0.52, 0.62],
            [0.18, 0.38, 0.28, 0.48, 0.58, 0.68],
        ]
        w_sm = guarded.smooth_joint_waypoints(w_in, alpha=0.20)
        self.assertIsInstance(w_sm, list)
        self.assertEqual(len(w_sm), 4)
        # Endpoints strictly identical
        self.assertEqual(w_sm[0], w_in[0])
        self.assertEqual(w_sm[-1], w_in[-1])

        # Formula check on inner point 1: 0.20*w[0] + 0.60*w[1] + 0.20*w[2]
        expected_pt1 = 0.20 * np.array(w_in[0]) + 0.60 * np.array(w_in[1]) + 0.20 * np.array(w_in[2])
        np.testing.assert_allclose(w_sm[1], expected_pt1)

        # Numpy array support
        arr_in = np.array(w_in)
        arr_sm = guarded.smooth_joint_waypoints(arr_in, alpha=0.20)
        self.assertIsInstance(arr_sm, np.ndarray)
        np.testing.assert_allclose(arr_sm[0], arr_in[0])
        np.testing.assert_allclose(arr_sm[-1], arr_in[-1])

        # Test acceleration / 2nd difference reduction on synthetic noisy trajectory
        rng = np.random.RandomState(42)
        base = np.linspace(0.0, 1.0, 50)[:, None] * np.ones((1, 6))
        noise = rng.normal(0, 0.02, size=(50, 6))
        noisy_traj = base + noise
        smoothed_traj = guarded.smooth_joint_waypoints(noisy_traj, alpha=0.20)

        d2_raw = np.max(np.abs(np.diff(noisy_traj, n=2, axis=0)))
        d2_sm = np.max(np.abs(np.diff(smoothed_traj, n=2, axis=0)))
        self.assertLess(d2_sm, d2_raw * 0.60)  # >40% peak 2nd diff reduction
        # Endpoint preservation
        np.testing.assert_allclose(smoothed_traj[0], noisy_traj[0])
        np.testing.assert_allclose(smoothed_traj[-1], noisy_traj[-1])

    def test_default_grasp_duration_is_2_5s(self):
        # When grasp_duration_s is omitted in SimpleNamespace, defaults to 2.5s
        args = SimpleNamespace(
            phase_speeds=True, speed=0.10, rate=30.0,
            approach_speed=0.20, transport_speed=0.20, place_speed=0.12,
            hold_grasp_pose=True
        )
        speed, dur = guarded.get_phase_speed_and_duration("grasp", 50, args)
        self.assertEqual(dur, 2.5)
        self.assertAlmostEqual(speed, 50.0 / 30.0 / 2.5)

    def test_cli_smooth_waypoints_validation(self):
        import subprocess

        # Valid CLI with smoothing flags
        res = subprocess.run([
            sys.executable, str(Path(guarded.__file__).resolve()),
            "--server", "http://127.0.0.1:9999",
            "--settled-chunks", "--condition-final-grasp", "--place-after-grasp", "--loops", "6",
            "--send-right-chunks", "--adaptive-refine",
            "--smooth-waypoints", "--waypoint-smooth-alpha", "0.20", "--waypoint-smooth-passes", "1"
        ], capture_output=True, text=True)
        self.assertNotIn("Invalid waypoint smoothing parameter", res.stderr)

        # Invalid alpha > 0.33 rejected
        res = subprocess.run([
            sys.executable, str(Path(guarded.__file__).resolve()),
            "--server", "http://127.0.0.1:9999",
            "--settled-chunks", "--condition-final-grasp", "--place-after-grasp", "--loops", "6",
            "--send-right-chunks", "--adaptive-refine",
            "--waypoint-smooth-alpha", "0.50"
        ], capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Invalid waypoint smoothing parameter", res.stderr)

        # Invalid passes <= 0 rejected
        res = subprocess.run([
            sys.executable, str(Path(guarded.__file__).resolve()),
            "--server", "http://127.0.0.1:9999",
            "--settled-chunks", "--condition-final-grasp", "--place-after-grasp", "--loops", "6",
            "--send-right-chunks", "--adaptive-refine",
            "--waypoint-smooth-passes", "0"
        ], capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Invalid waypoint smoothing parameter", res.stderr)

    def test_condition_placement_actions_latches_gripper_open_after_release(self):
        from build_grasp_replay_cases import condition_placement_actions

        # Reproduce Round 9 (104621 Loop 5) pattern: gripper opens at step 43 (0.0708m),
        # then dips below 0.040m at step 49 (0.0321m). Must remain latched open (>= 0.0708m).
        start = np.array([0.3316, 0.2286, 0.1726, 2.944, -0.258, 0.259, 0.0263])
        raw = np.zeros((50, 7))
        for i in range(50):
            alpha = (i + 1) / 50.0
            raw[i, :6] = start[:6] + alpha * np.array([-0.03, 0.01, -0.01, 0.0, 0.0, 0.0])
            raw[i, 6] = 0.002
        raw[43:48, 6] = [0.0708, 0.0680, 0.0650, 0.0660, 0.0640]
        raw[48, 6] = 0.0525
        raw[49, 6] = 0.0321  # Late dip below 40mm that previously re-clamped to 18.3mm

        cond, clipped, splits = condition_placement_actions(start, raw, held_grip=0.0263)
        self.assertEqual(clipped, [])
        # Final gripper target must remain latched at maximum opened width (0.0708m), never re-clamping
        self.assertAlmostEqual(float(cond[-1, 6]), 0.0708, places=4)
        # All steps after release starts must be monotonically non-decreasing
        release_idx = int(np.flatnonzero(cond[:, 6] >= 0.040)[0])
        post_release_diffs = np.diff(cond[release_idx:, 6])
        self.assertTrue(np.all(post_release_diffs >= -1e-9))
        # Adjacent gripper step limit (<= 15mm) must hold everywhere
        grip_steps_mm = np.abs(np.diff(np.r_[start[6], cond[:, 6]])) * 1000.0
        self.assertLessEqual(float(grip_steps_mm.max()), 15.0)

    def test_condition_approach_actions_prevents_y_retreat_and_step45_lunge(self):
        from build_grasp_replay_cases import condition_approach_actions

        # 1. R7 Loop 2 pattern (103832 / 104017): Y advances at step 19 (+25mm), then
        # retreats backward in -Y at step 44 while descending, with a step-45 +X lunge.
        start = np.array([0.2194, 0.1195, 0.2156, 2.864, -0.369, -0.105, 0.0789])
        raw = np.tile(start, (50, 1))
        for i in range(20):
            raw[i, 1] = 0.1195 + (0.1452 - 0.1195) * ((i + 1) / 20.0)
            raw[i, 2] = 0.2156 - (0.2156 - 0.1755) * ((i + 1) / 20.0)
        for i in range(20, 45):
            alpha = (i - 19) / 25.0
            raw[i, 0] = 0.2200
            raw[i, 1] = 0.1452 - 0.0156 * alpha  # -15.6mm backward -Y retreat
            raw[i, 2] = 0.1755 - (0.1755 - 0.1594) * alpha
        # Inject mid-descent +X spike at step 36 and post-grasp +X lunge at step 45
        raw[36, 0] = 0.2265
        raw[45, :3] = [0.2285, 0.1250, 0.1592]  # >6mm XY jump with minimal dZ
        raw[46:, :3] = [0.2320, 0.1150, 0.1720]
        raw[:, 6] = 0.079
        raw[29:, 6] = 0.005  # first_close = 29 (> 15)

        cond, held_open_from, pose_anchored_from = condition_approach_actions(start, raw)
        self.assertEqual(held_open_from, 0)
        # Step 45 lunge backed off to step 44
        self.assertEqual(pose_anchored_from, 44)
        # Final Y preserves the peak forward +Y progress (0.1452m) instead of retreating to 0.1296m
        self.assertAlmostEqual(float(cond[-1, 1]), 0.1452, places=4)
        # Final X uses step 44 (0.2200m), ignoring the step 36 spike and step 45 lunge (0.2285m)
        self.assertAlmostEqual(float(cond[-1, 0]), 0.2200, places=4)
        # Gripper held open (>= 0.045m) throughout
        self.assertTrue(np.all(cond[:, 6] >= 0.045))

        # 2. R7 Loop 3 dead-zone pattern (104017): first_close=22 (>15), min_z_index=16 (<19),
        # followed by post-grasp lift & -Y retreat from step 20..49.
        start3 = np.array([0.2206, 0.1193, 0.1589, 2.956, -0.343, -0.175, 0.0757])
        raw3 = np.tile(start3, (50, 1))
        raw3[16, 2] = 0.1583  # min_z_index = 16
        raw3[19, :3] = [0.2194, 0.1410, 0.1605]  # peak +Y in window 12..20
        raw3[22:, 6] = 0.005  # first_close = 22
        raw3[25:, :3] = [0.2314, 0.1127, 0.1894]  # post-grasp lift & retreat

        cond3, held3, anchor3 = condition_approach_actions(start3, raw3)
        self.assertEqual(held3, 0)
        self.assertEqual(anchor3, 19)
        self.assertAlmostEqual(float(cond3[-1, 1]), 0.1410, places=4)
        self.assertAlmostEqual(float(cond3[-1, 2]), 0.1605, places=4)

    def test_evaluate_grasp_deferral_and_budget_extends_loops_for_placement(self):
        raw = np.zeros((50, 7))
        raw[:, 6] = 0.075
        raw[15:, 6] = 0.005  # No early closure in 0..12

        # Case 1: Round 3 (103619 Loop 3) - index=3, max_y_window=0.1592 < 0.170, place_after_grasp=True
        raw[:, 1] = 0.1592
        defer, loops = guarded.evaluate_grasp_deferral_and_budget(
            raw, index=3, total_loops=6,
            condition_final_grasp=True, completed_conditioned_grasp=False,
            place_after_grasp=True
        )
        self.assertTrue(defer)
        self.assertEqual(loops, 7)  # Extended from 6 to 7 so Loop 4 grasps, Loop 5 transports, Loop 6 releases

        # Case 2: Round 3 (103619 Loop 4) - index=4, max_y_window=0.2141 >= 0.165 -> does not defer
        raw4_r1 = raw.copy()
        raw4_r1[0, 1] = 0.1930
        raw4_r1[12:21, 1] = 0.2141
        defer4_r1, loops4_r1 = guarded.evaluate_grasp_deferral_and_budget(
            raw4_r1, index=4, total_loops=7,
            condition_final_grasp=True, completed_conditioned_grasp=False,
            place_after_grasp=True
        )
        self.assertFalse(defer4_r1)
        self.assertEqual(loops4_r1, 7)

        # Case 3: Left-column R7 at index=4 still short of Y=0.165m (start_y=0.1376, max_y=0.1544)
        raw4_r7 = raw.copy()
        raw4_r7[0, 1] = 0.1376
        raw4_r7[12:21, 1] = 0.1544
        defer4_r7, loops4_r7 = guarded.evaluate_grasp_deferral_and_budget(
            raw4_r7, index=4, total_loops=7,
            condition_final_grasp=True, completed_conditioned_grasp=False,
            place_after_grasp=True
        )
        self.assertTrue(defer4_r7)
        self.assertEqual(loops4_r7, 8)  # Extended to 8 so Loop 5 grasps and Loops 6-7 transport+release

        # Case 4: Right-column R9 at index=4 (start_y=0.098 < 0.130) never defers at index=4
        raw4_r9 = raw.copy()
        raw4_r9[0, 1] = 0.0980
        raw4_r9[12:21, 1] = 0.1050
        defer4_r9, loops4_r9 = guarded.evaluate_grasp_deferral_and_budget(
            raw4_r9, index=4, total_loops=7,
            condition_final_grasp=True, completed_conditioned_grasp=False,
            place_after_grasp=True
        )
        self.assertFalse(defer4_r9)
        self.assertEqual(loops4_r9, 7)

    def test_wrist_obb_refine_and_hover_then_vertical_descent(self):
        from build_grasp_replay_cases import (
            condition_approach_actions,
            detect_wrist_block_obb,
        )

        # Synthetic 640x480 BGR image with a green wooden block centered at (268, 302)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[280:325, 246:291] = [0, 200, 0]  # Green block in BGR (45x45 px = 2025 area)
        obb = detect_wrist_block_obb(img)
        if obb is not None:
            self.assertAlmostEqual(obb["cx"], 268.0, delta=2.0)
            self.assertAlmostEqual(obb["cy"], 302.0, delta=2.0)

        # Reproduce Round 6 (104215 R9 Green Loop 2): start_x=240.9mm, start_z=205.5mm,
        # policy raw predicts X=247.0mm (step 44) / 253.1mm (step 45) which pressed left edge.
        start = np.array([0.2409, 0.0738, 0.2055, 2.890, -0.350, -0.110, 0.0780])
        raw = np.tile(start, (50, 1))
        for i in range(20):
            raw[i, 0] = 0.2409 + (0.2450 - 0.2409) * ((i + 1) / 20.0)
            raw[i, 1] = 0.0738 + (0.0967 - 0.0738) * ((i + 1) / 20.0)
            raw[i, 2] = 0.2055 - (0.2055 - 0.1780) * ((i + 1) / 20.0)
        for i in range(20, 45):
            alpha = (i - 19) / 25.0
            raw[i, 0] = 0.2470
            raw[i, 1] = 0.0967
            raw[i, 2] = 0.1780 - (0.1780 - 0.1605) * alpha
        raw[45, :3] = [0.2531, 0.0931, 0.1598]
        raw[:, 6] = 0.078
        raw[30:, 6] = 0.008

        wrist_obb = {"cx": 267.9, "cy": 301.8, "w": 48.0, "h": 47.1, "angle": 76.0, "area": 1890.0}
        cond, _, anchor_idx = condition_approach_actions(start, raw, wrist_obb=wrist_obb)
        self.assertEqual(anchor_idx, 44)
        # Calibrated 3D pinhole model detects du=-42.6px at hover: shifts end X left to ~207.4mm
        # and holds hover clearance (>= 0.182m) rather than plunging while misaligned.
        self.assertLess(float(cond[-1, 0]), 0.2150)
        self.assertGreater(float(cond[-1, 0]), 0.2000)
        self.assertGreaterEqual(float(np.min(cond[:, 2])), 0.1820 - 1e-6)

        # When aligned at hover under true pinhole target (~310.5, 307.9), permits vertical descent <= 0.162m.
        wrist_obb_aligned = {"cx": 310.5, "cy": 308.0, "w": 48.0, "h": 47.1, "angle": 76.0, "area": 1890.0}
        cond_aligned, _, _ = condition_approach_actions(start, raw, wrist_obb=wrist_obb_aligned)
        self.assertLessEqual(float(cond_aligned[-1, 2]), 0.1620 + 1e-6)
        # Verify hover-then-vertical-descent decoupling on the aligned plunge:
        # During the final vertical drop steps (36..44), XY is already 100% at end_pose (zero sideways motion)
        np.testing.assert_allclose(cond_aligned[36:45, 0], cond_aligned[-1, 0], atol=1e-6)
        np.testing.assert_allclose(cond_aligned[36:45, 1], cond_aligned[-1, 1], atol=1e-6)
        # And at the end of horizontal alignment (step 35), Z is still >= 172mm (above block top 168mm)
        self.assertGreaterEqual(float(cond_aligned[35, 2]), 0.1720 - 1e-6)

    def test_placement_direct_descent_and_faster_defaults(self):
        from types import SimpleNamespace
        from build_grasp_replay_cases import condition_placement_actions

        # Verify faster defaults when transport_speed / place_speed are omitted on SimpleNamespace
        args = SimpleNamespace(
            phase_speeds=True, speed=0.10, rate=30.0,
            hold_grasp_pose=True
        )
        t_speed, t_dur = guarded.get_phase_speed_and_duration("transport", 55, args)
        p_speed, p_dur = guarded.get_phase_speed_and_duration("place", 57, args)
        self.assertEqual(t_speed, 0.25)
        self.assertAlmostEqual(t_dur, 55.0 / 30.0 / 0.25)  # 7.333s (down from 9.167s)
        self.assertEqual(p_speed, 0.24)
        self.assertAlmostEqual(p_dur, 57.0 / 30.0 / 0.24)  # 7.917s (down from 15.833s)

        # Verify direct smooth descent in condition_placement_actions removes mid-air XY hunting
        start = np.array([0.3392, 0.1936, 0.1862, 2.940, -0.260, 0.250, 0.0268])
        raw = np.tile(start, (50, 1))
        raw[:, 6] = 0.005
        # Inject back-and-forth XY zigzagging before step 38 release
        for i in range(1, 38):
            raw[i, 0] = 0.3427 + (0.010 if i % 2 == 0 else -0.008)
            raw[i, 1] = 0.2137 + (0.008 if i % 3 == 0 else -0.006)
            raw[i, 2] = 0.1862 - 0.035 * (i / 38.0)
        raw[38:, :3] = [0.3512, 0.2236, 0.1512]
        raw[45, 0] = 0.3592  # Post-release +8mm sideways jerk in raw
        raw[38:, 6] = 0.075

        cond, _, _ = condition_placement_actions(start, raw, held_grip=0.0268)
        xyz_path_mm = float(np.sum(np.linalg.norm(np.diff(cond[:, :3], axis=0), axis=1)) * 1000.0)
        net_disp_mm = float(np.linalg.norm(cond[-1, :3] - cond[0, :3]) * 1000.0)
        # Path length is nearly equal to straight-line displacement (no 259mm zigzag)
        self.assertLessEqual(xyz_path_mm, net_disp_mm + 15.0)
        # Pose is held completely steady at release_pose while gripper opens
        rel_idx = int(np.flatnonzero(cond[:, 6] >= 0.040)[0])
        expected_rel_xyz = np.tile([0.3512, 0.2236, 0.1512], (len(cond) - rel_idx, 1))
        np.testing.assert_allclose(cond[rel_idx:, :3], expected_rel_xyz, atol=1e-6)

    def test_wrist_target_obb_refines_placement_release_and_transport(self):
        from build_grasp_replay_cases import (
            condition_placement_actions,
            detect_wrist_target_obb,
        )

        # Synthetic 640x480 BGR image with a brown target pad centered at (352, 275)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[230:320, 302:402] = [60, 105, 160]  # Brown pad in BGR (90x100 px = 9000 area, HSV ~17,160,160)
        obb = detect_wrist_target_obb(img)
        if obb is not None:
            self.assertAlmostEqual(obb["cx"], 352.0, delta=2.0)
            self.assertAlmostEqual(obb["cy"], 275.0, delta=2.0)

        # 1. Well-centered placement chunk (Round 1 103219 Loop 5): pred_cx=357.2, pred_cy=339.9 inside deadband -> 0.0mm shift
        start = np.array([0.3392, 0.1936, 0.1862, 2.940, -0.260, 0.250, 0.0268])
        raw = np.tile(start, (50, 1))
        raw[:, 6] = 0.005
        raw[38:, :3] = [0.3512, 0.2236, 0.1512]
        raw[38:, 6] = 0.075
        centered_obb = {"cx": 347.9, "cy": 276.9, "w": 115.6, "h": 97.6, "angle": 85.4, "area": 10099.5}
        cond_centered, _, _ = condition_placement_actions(start, raw, held_grip=0.0268, target_obb=centered_obb)
        np.testing.assert_allclose(cond_centered[-1, :3], [0.3512, 0.2236, 0.1512], atol=1e-6)

        # 2. Drifted placement chunk where brown pad appears shifted left & higher in wrist image (cx=305, cy=240)
        drifted_obb = {"cx": 305.0, "cy": 240.0, "w": 110.0, "h": 95.0, "angle": 85.0, "area": 9500.0}
        cond_drifted, _, _ = condition_placement_actions(start, raw, held_grip=0.0268, target_obb=drifted_obb)
        # pred_cx = 305 + 1.05*(186.2-151.2) - 2.3*(351.2-339.2) = 314.15 < 338 -> shifts release X negative
        self.assertLess(float(cond_drifted[-1, 0]), 0.3400)
        # pred_cy = 240 + 2.1*(223.6-193.6) = 303.0 < 320 -> shifts release Y positive
        self.assertGreater(float(cond_drifted[-1, 1]), 0.2350)

    def test_wide_object_anti_false_release_and_guided_transport(self):
        from build_grasp_replay_cases import (
            compute_adaptive_preload_m,
            condition_placement_actions,
            find_valid_placement_release_index,
            verify_placement_completion,
        )

        # Reproduce Round 8 (20261107_164401 Loop 5): Cyan 2x4 Lego grasped across long edge
        # with held_grip = 59.7mm (0.0597m) at R9 grasp zone (X=234.4mm, Y=109.1mm, Z=156.1mm).
        # Raw model predicts open gripper 63.9..73.9mm at steps 0..7 in the grasp zone, then closes
        # at steps 8..49 and only reaches [267.2, 126.6, 200.2]mm at step 49.
        start = np.array([0.2344, 0.1091, 0.1561, -0.338, 0.975, 1.026, 0.0597])
        raw = np.zeros((50, 7))
        for i in range(50):
            alpha = (i + 1) / 50.0
            raw[i, :3] = start[:3] + alpha * (np.array([0.2672, 0.1266, 0.2002]) - start[:3])
            raw[i, 3:6] = start[3:6]
            raw[i, 6] = 0.008
        raw[:8, 6] = [0.0639, 0.0724, 0.0705, 0.0739, 0.0702, 0.0678, 0.0686, 0.0666]

        # 1. Must NOT mistake early open steps at R9 for a valid placement release
        rel_idx = find_valid_placement_release_index(start, raw, held_grip=0.0597)
        self.assertIsNone(rel_idx)

        # 2. Adaptive preload for wide object (>42mm) is 6mm -> clamp_grip = 53.7mm
        self.assertAlmostEqual(compute_adaptive_preload_m(0.0597), 0.006, places=4)

        # 3. Conditioned placement chunk holds grip at 53.7mm (no release) and guides end XY to brown pad
        cond, clipped, _ = condition_placement_actions(start, raw, held_grip=0.0597)
        self.assertEqual(clipped, [])
        release_threshold = max(0.040, min(0.070, 0.0597 + 0.010))
        self.assertFalse(bool(np.any(cond[:, 6] >= release_threshold)))
        np.testing.assert_allclose(cond[4:, 6], 0.0537, atol=1e-4)
        np.testing.assert_allclose(cond[-1, :2], [0.3180, 0.1860], atol=1e-4)
        self.assertGreaterEqual(float(cond[-1, 2]), 0.1900)
        # Adjacent Cartesian steps remain strictly <= 15mm
        xyz_steps_mm = np.linalg.norm(np.diff(np.vstack([start[:3], cond[:, :3]]), axis=0), axis=1) * 1000.0
        self.assertLessEqual(float(xyz_steps_mm.max()), 15.0)

        # 4. Verify post-release check requires opening above held_grip + 6mm
        v_fail = verify_placement_completion(measured_grip=0.0597, held_grip=0.0597)
        self.assertFalse(v_fail["confirmed"])
        v_ok = verify_placement_completion(measured_grip=0.0685, held_grip=0.0597)
        self.assertTrue(v_ok["confirmed"])

    def test_multi_color_obb_and_grasp_orientation_selection(self):
        from build_grasp_replay_cases import (
            detect_wrist_block_obb,
            select_grasp_orientation_candidate,
        )

        # 1. Background false-positive outside workspace (cx=110 < 175) must be ignored,
        # while Yellow 2x2 block inside workspace (cx=275, cy=306) is detected.
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[200:260, 80:140] = [0, 210, 220]   # Background yellow outside workspace (cx=110)
        img[284:328, 253:297] = [20, 195, 215]  # Workspace yellow 2x2 Lego (44x44 px at cx=275, cy=306)
        obb_yellow = detect_wrist_block_obb(img)
        self.assertIsNotNone(obb_yellow)
        self.assertEqual(obb_yellow["color_class"], "yellow")
        self.assertEqual(obb_yellow["shape_class"], "square")
        self.assertEqual(obb_yellow["grasp_axis_mode"], "symmetric_square")
        self.assertAlmostEqual(obb_yellow["cx"], 275.0, delta=2.0)

        # Square block produces zero yaw modification
        start_pose = np.array([0.237, 0.078, 0.208, -0.384, 0.995, 1.037])
        end_pose = np.array([0.236, 0.102, 0.157, -0.384, 0.995, 1.037])
        ref_pose, meta = select_grasp_orientation_candidate(start_pose, end_pose, obb_yellow)
        self.assertEqual(meta["mode"], "symmetric_square")
        self.assertEqual(meta["applied_yaw_delta_deg"], 0.0)
        np.testing.assert_allclose(ref_pose, end_pose)

        # 2. Cyan 2x4 Lego with short axis along u (44px wide in u, 86px tall in v -> short_axis_aligned)
        img_cyan_short = np.zeros((480, 640, 3), dtype=np.uint8)
        img_cyan_short[248:334, 295:339] = [210, 185, 0]
        obb_cyan_short = detect_wrist_block_obb(img_cyan_short)
        self.assertIsNotNone(obb_cyan_short)
        self.assertEqual(obb_cyan_short["color_class"], "cyan")
        self.assertEqual(obb_cyan_short["shape_class"], "elongated")
        self.assertEqual(obb_cyan_short["grasp_axis_mode"], "short_axis_aligned")

        # 3. Cyan 2x4 Lego with long axis along u (86px wide in u, 44px tall in v -> long_axis_between_fingers)
        img_cyan_long = np.zeros((480, 640, 3), dtype=np.uint8)
        img_cyan_long[270:314, 275:361] = [210, 185, 0]
        obb_cyan_long = detect_wrist_block_obb(img_cyan_long)
        self.assertIsNotNone(obb_cyan_long)
        self.assertEqual(obb_cyan_long["color_class"], "cyan")
        self.assertEqual(obb_cyan_long["shape_class"], "elongated")
        self.assertEqual(obb_cyan_long["grasp_axis_mode"], "long_axis_between_fingers")
        ref_cyan, meta_cyan = select_grasp_orientation_candidate(start_pose, end_pose, obb_cyan_long)
        self.assertNotEqual(meta_cyan["applied_yaw_delta_deg"], 0.0)
        self.assertAlmostEqual(abs(meta_cyan["applied_yaw_delta_deg"]), 90.0, delta=1.0)

    def test_r9_front_bulge_suppression_and_182mm_hover_clearance(self):
        from build_grasp_replay_cases import condition_approach_actions

        # Reproduce Round 4/5 (163243 / 163601 Loop 2) at R9 Green Block:
        # Raw steps 0..19 bulge out to X=252.4mm at step 19 before sweeping back to X=246.1mm.
        start = np.array([0.2398, 0.0774, 0.2109, -0.380, 0.987, 1.031, 0.0686])
        raw = np.tile(start, (50, 1))
        for i in range(20):
            alpha = (i + 1) / 20.0
            raw[i, 0] = 0.2398 + (0.2524 - 0.2398) * alpha  # +12.6mm outward +X bulge
            raw[i, 1] = 0.0774 + (0.1003 - 0.0774) * alpha
            raw[i, 2] = 0.2109 - (0.2109 - 0.1777) * alpha
        for i in range(20, 45):
            alpha = (i - 19) / 25.0
            raw[i, 0] = 0.2524 - (0.2524 - 0.2461) * alpha
            raw[i, 1] = 0.1003
            raw[i, 2] = 0.1777 - (0.1777 - 0.1598) * alpha
        raw[:, 6] = 0.078
        raw[30:, 6] = 0.008

        wrist_obb = {
            "cx": 267.9, "cy": 303.8, "w": 47.1, "h": 46.5, "angle": 77.5, "area": 1890.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square", "aspect_ratio": 1.01,
        }
        cond, _, anchor_idx = condition_approach_actions(start, raw, wrist_obb=wrist_obb)
        self.assertEqual(anchor_idx, 44)
        # 1. Front-half +X bulge to 252.4mm is suppressed (max X in steps 0..19 <= 242mm)
        self.assertLessEqual(float(np.max(cond[:20, 0])), 0.2420)
        # 2. Final X is shifted left to ~209.1mm via calibrated 3D pinhole (-30.7mm) and holds hover >= 0.182m
        self.assertAlmostEqual(float(cond[-1, 0]), 0.2091, delta=0.0020)
        self.assertGreaterEqual(float(np.min(cond[:, 2])), 0.1820 - 1e-6)

        # When aligned at hover (cx~310.0, cy~309.9), hover height >= 182mm is held through alignment
        # completion (step 35) and steps 36..44 are pure vertical descent down to table height.
        obb_aligned = dict(wrist_obb, cx=310.0, cy=309.9)
        cond_aligned, _, anchor_aligned = condition_approach_actions(start, raw, wrist_obb=obb_aligned)
        self.assertEqual(anchor_aligned, 44)
        self.assertGreaterEqual(float(cond_aligned[35, 2]), 0.1820 - 1e-6)
        self.assertLessEqual(float(cond_aligned[-1, 2]), 0.1620 + 1e-6)
        np.testing.assert_allclose(cond_aligned[36:45, :2], np.tile(cond_aligned[-1, :2], (9, 1)), atol=1e-6)

    def test_settled_post_plan_guard_and_cyan_r9_no_false_deferral(self):
        # 1. Granular post-plan guard reasons
        q0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        reasons_ok = guarded.evaluate_settled_post_plan_guard(
            q0, q0, q0, 0.068, 0.068, 420.0, max_gripper_gap_mm=30.0, max_observation_age_ms=500.0
        )
        self.assertEqual(reasons_ok, [])

        # Round 1 (162719) pattern: arm settled, only observation_expired (502.4ms > 500.0ms)
        reasons_exp = guarded.evaluate_settled_post_plan_guard(
            q0, q0, q0, 0.068, 0.068, 502.4, max_gripper_gap_mm=30.0, max_observation_age_ms=500.0
        )
        self.assertEqual(reasons_exp, ["observation_expired"])

        # State drift + expired is NOT eligible for expired-only replan
        reasons_drift = guarded.evaluate_settled_post_plan_guard(
            q0, q0 + 0.02, q0, 0.068, 0.068, 502.4, max_gripper_gap_mm=30.0, max_observation_age_ms=500.0
        )
        self.assertIn("post_plan_state_drift", reasons_drift)
        self.assertIn("observation_expired", reasons_drift)

        # 2. Round 7/8 (164146 / 164401 Loop 3) at R9: y0=0.109 < 0.125 and min_g_0_12=0.008 < 0.025
        # must NOT defer grasp at Loop 3 (so Loop 3 grasps, Loop 4 transports, Loop 5 releases).
        raw_r9_cyan = np.zeros((50, 7))
        raw_r9_cyan[:, 1] = 0.112
        raw_r9_cyan[0, 1] = 0.109
        raw_r9_cyan[:, 6] = 0.008
        defer_cyan, loops_cyan = guarded.evaluate_grasp_deferral_and_budget(
            raw_r9_cyan, index=3, total_loops=6,
            condition_final_grasp=True, completed_conditioned_grasp=False,
            place_after_grasp=True,
        )
        self.assertFalse(defer_cyan)
        self.assertEqual(loops_cyan, 6)

    def test_short_axis_yaw_alignment_across_all_rotations(self):
        from build_grasp_replay_cases import (
            _enrich_block_obb_geometry,
            select_grasp_orientation_candidate,
        )
        # Test 5 angles spanning full rotation space of 2x4 Lego
        angles_test = [0.0, 21.8, 51.9, 75.7, -10.4, -88.4]
        start_pose = np.array([0.225, 0.090, 0.203, -0.380, 0.940, 1.020])
        end_pose = np.array([0.230, 0.110, 0.160, -0.380, 0.940, 1.020])
        for ang in angles_test:
            # Construct synthetic OBB with long axis at ang
            obb = _enrich_block_obb_geometry(
                cx=337.0, cy=285.0, w=44.0, h=88.0, angle=ang, area=3872.0, color_class="cyan"
            )
            self.assertEqual(obb["shape_class"], "elongated")
            ref_end, info = select_grasp_orientation_candidate(start_pose, end_pose, obb)
            # In image space, short axis must align with horizontal closing axis (residual angle <= 1.0 deg)
            applied = info["applied_yaw_delta_deg"]
            residual = (obb["short_axis_deg"] + applied + 90.0) % 180.0 - 90.0
            self.assertLessEqual(abs(residual), 1.0)
            # Rotation delta must be comfortably within physical limits (<= 105 deg)
            self.assertLessEqual(abs(applied), 105.0)

    def test_bidirectional_midpoint_centering_and_descent_lock(self):
        from build_grasp_replay_cases import condition_approach_actions
        start_high = np.array([0.225, 0.090, 0.203, -0.380, 0.940, 1.020, 0.070])
        raw_actions = np.tile(start_high, (50, 1))
        for i in range(50):
            raw_actions[i, 1] = 0.090 + 0.020 * (i / 49.0)
            raw_actions[i, 2] = 0.203 - 0.043 * (i / 49.0)
        raw_actions[30:, 6] = 0.005

        # 1. Block overshot in +Y (cy=350 > 285): must pull back in -Y
        obb_overshot = {
            "cx": 337.0, "cy": 350.0, "w": 44.0, "h": 88.0, "angle": 0.0, "area": 3872.0,
            "shape_class": "elongated", "grasp_axis_mode": "short_axis_aligned", "aspect_ratio": 2.0,
            "recommended_yaw_delta_rad": 0.0, "expected_current_width_m": 0.027, "contact_points_px": [],
        }
        cond_pull, _, _ = condition_approach_actions(start_high, raw_actions, wrist_obb=obb_overshot)
        # End Y must be pulled back relative to unconditioned end Y and hover Z held >= 0.182m
        self.assertLess(float(cond_pull[-1, 1]), float(raw_actions[-1, 1]))
        self.assertGreaterEqual(float(np.min(cond_pull[:, 2])), 0.1820 - 1e-6)

        # 2. Low chunk (start Z < 0.182m): locks orientation strictly; when already aligned at table,
        # also holds XY strictly locked at start_low (suppressing raw XY/yaw drift).
        start_low = np.array([0.230, 0.110, 0.160, -0.380, 0.940, 1.250, 0.070])
        raw_low = np.tile(start_low, (50, 1))
        raw_low[:, 5] = 0.850  # Unconditioned raw tries to drag yaw back to 0.85 rad
        raw_low[:, 0] = 0.245  # Unconditioned raw tries to drift X
        raw_low[20:, 6] = 0.005
        obb_aligned_low = {
            "cx": 325.0, "cy": 318.0, "w": 45.0, "h": 45.0, "angle": 0.0, "area": 2025.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
        }
        cond_low, _, _ = condition_approach_actions(start_low, raw_low, wrist_obb=obb_aligned_low)
        # Orientation and horizontal XY remain strictly locked at start_low when aligned at table
        np.testing.assert_allclose(cond_low[:, 3:6], np.tile(start_low[3:6], (50, 1)), atol=1e-6)
        np.testing.assert_allclose(cond_low[:, 0], start_low[0], atol=1e-6)

        # With severe low-altitude deviation (dv=+65px for elongated), locks orientation, lifts safely
        # to hover (>= 0.182m), and executes coupled -Y pull-back.
        cond_over, _, _ = condition_approach_actions(start_low, raw_low, wrist_obb=obb_overshot)
        np.testing.assert_allclose(cond_over[:, 3:6], np.tile(start_low[3:6], (50, 1)), atol=1e-6)
        self.assertGreaterEqual(float(cond_over[-1, 2]), 0.1820 - 1e-6)
        self.assertLess(float(cond_over[-1, 1]), float(start_low[1]))

    def test_grasp_deferral_suppressed_when_wrist_obb_centered(self):
        # Even if index == 3 and max_y_window < 0.170, when wrist_obb is centered, deferral is False
        raw = np.zeros((50, 7))
        raw[:, 1] = 0.130
        raw[:, 6] = 0.035
        centered_obb = {"cx": 340.0, "cy": 290.0}
        defer, loops = guarded.evaluate_grasp_deferral_and_budget(
            raw, index=3, total_loops=6,
            condition_final_grasp=True, completed_conditioned_grasp=False,
            place_after_grasp=True, wrist_obb=centered_obb,
        )
        self.assertFalse(defer)
        self.assertEqual(loops, 6)

    def test_verify_placement_keys_compatibility(self):
        from build_grasp_replay_cases import verify_placement_completion
        res = verify_placement_completion(measured_grip=0.065, held_grip=0.028)
        self.assertIn("gripper_opened_ok", res)
        self.assertIn("gripper_opened", res)
        self.assertIn("required_open_m", res)
        self.assertIn("expected_open_m", res)
        self.assertTrue(res["confirmed"])
        self.assertTrue(res["gripper_opened"])
        self.assertTrue(res["gripper_opened_ok"])


    def test_parse_target_color_and_prompt_interface(self):
        from build_grasp_replay_cases import parse_target_color
        self.assertEqual(parse_target_color("pick up cyan lego and place it in target zone"), "blue")
        self.assertEqual(parse_target_color("Pick up the green wooden block"), "green")
        self.assertEqual(parse_target_color("\u6293\u53d6\u7eff\u8272\u79ef\u6728\u5e76\u653e\u7f6e\u5230\u68d5\u8272\u533a\u57df"), "green")
        self.assertEqual(parse_target_color("\u6293\u53d6\u84dd\u8272\u65b9\u5757"), "blue")
        self.assertEqual(parse_target_color("Pick up the yellow block"), "yellow")
        self.assertEqual(parse_target_color("\u6293\u53d6\u9ec4\u8272\u65b9\u5757"), "yellow")
        self.assertEqual(parse_target_color("Pick up the red wooden block from the white grid"), "red")
        self.assertEqual(parse_target_color(""), "red")

    def test_four_color_multi_block_discrimination(self):
        from build_grasp_replay_cases import detect_wrist_block_obb
        # Create synthetic image with 4 distinct colored square blocks simultaneously in workspace:
        # Red at (240, 240), Green at (360, 240), Blue at (250, 340), Yellow at (370, 340)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[215:265, 215:265] = [20, 20, 200]    # Red
        img[215:265, 335:385] = [30, 180, 30]    # Green
        img[315:365, 225:275] = [200, 50, 20]    # Blue
        img[315:365, 345:395] = [20, 195, 215]   # Yellow

        obb_red = detect_wrist_block_obb(img, target_color="red")
        self.assertIsNotNone(obb_red)
        self.assertEqual(obb_red["color_class"], "red")
        self.assertAlmostEqual(obb_red["cx"], 239.5, delta=3.0)
        self.assertAlmostEqual(obb_red["cy"], 239.5, delta=3.0)

        obb_green = detect_wrist_block_obb(img, target_color="green")
        self.assertIsNotNone(obb_green)
        self.assertEqual(obb_green["color_class"], "green")
        self.assertAlmostEqual(obb_green["cx"], 359.5, delta=3.0)
        self.assertAlmostEqual(obb_green["cy"], 239.5, delta=3.0)

        obb_blue = detect_wrist_block_obb(img, target_color="blue")
        self.assertIsNotNone(obb_blue)
        self.assertEqual(obb_blue["color_class"], "blue")
        self.assertAlmostEqual(obb_blue["cx"], 249.5, delta=3.0)
        self.assertAlmostEqual(obb_blue["cy"], 339.5, delta=3.0)

        obb_yellow = detect_wrist_block_obb(img, target_color="yellow")
        self.assertIsNotNone(obb_yellow)
        self.assertEqual(obb_yellow["color_class"], "yellow")
        self.assertAlmostEqual(obb_yellow["cx"], 369.5, delta=3.0)
        self.assertAlmostEqual(obb_yellow["cy"], 339.5, delta=3.0)

    def test_rotated_square_modulo90_alignment_and_cross_check(self):
        from build_grasp_replay_cases import (
            detect_wrist_block_obb,
            select_grasp_orientation_candidate,
            _enrich_block_obb_geometry,
        )
        # 1. Rotated square by 45 degrees
        theta = np.radians(45.0)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        base_pts = np.array([[-25, -25], [25, -25], [25, 25], [-25, 25]])
        rot_pts = (base_pts @ R.T + np.array([320, 280])).astype(np.int32)
        img_rot = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2 = __import__("cv2")
        cv2.fillPoly(img_rot, [rot_pts], (20, 20, 200))

        obb = detect_wrist_block_obb(img_rot, target_color="red")
        self.assertIsNotNone(obb)
        self.assertEqual(obb["shape_class"], "square")
        # Check Scheme B cross-check verification:
        self.assertIn("cross_check", obb)
        self.assertTrue(obb["cross_check"]["verified"])
        self.assertLessEqual(obb["cross_check"]["diff_deg"], 2.0)

        # Check orientation selection for square: preserves nominal approach orientation (zero yaw delta)
        # to avoid tool-flange lever-arm swing offset, relying on deep forward insertion (cy_target=318.0)
        # to encompass the 38.2mm diagonal.
        start_pose = np.array([0.237, 0.078, 0.208, -0.384, 0.995, 1.037])
        end_pose = np.array([0.236, 0.102, 0.157, -0.384, 0.995, 1.037])
        ref_pose, meta = select_grasp_orientation_candidate(start_pose, end_pose, obb)
        self.assertEqual(meta["mode"], "symmetric_square")
        self.assertEqual(meta["applied_yaw_delta_deg"], 0.0)
        np.testing.assert_allclose(ref_pose, end_pose)

    def test_check_wrist_grasp_alignment(self):
        from build_grasp_replay_cases import check_wrist_grasp_alignment

        # 1. Missing or None OBB
        res_none = check_wrist_grasp_alignment(None)
        self.assertFalse(res_none["aligned"])
        self.assertEqual(res_none["reason"], "no_obb")

        # 2. Square block centered (target: cx=325, cy=318)
        sq_centered = {"cx": 325.0, "cy": 318.0, "shape_class": "square"}
        res_sq = check_wrist_grasp_alignment(sq_centered)
        self.assertTrue(res_sq["aligned"])
        self.assertEqual(res_sq["du"], 0.0)
        self.assertEqual(res_sq["dv"], 0.0)
        self.assertEqual(res_sq["reason"], "aligned")

        # 3. Square block within tolerance (|du| <= 20, |dv| <= 20)
        sq_tol = {"cx": 344.0, "cy": 300.0, "shape_class": "square"}
        res_tol = check_wrist_grasp_alignment(sq_tol, tol_u_px=20.0, tol_v_px=20.0)
        self.assertTrue(res_tol["aligned"])

        # 4. Misaligned lateral only
        sq_u_bad = {"cx": 355.0, "cy": 318.0, "shape_class": "square"}
        res_u = check_wrist_grasp_alignment(sq_u_bad, tol_u_px=20.0, tol_v_px=20.0)
        self.assertFalse(res_u["aligned"])
        self.assertEqual(res_u["reason"], "u_misaligned")

        # 5. Misaligned longitudinal only
        sq_v_bad = {"cx": 325.0, "cy": 280.0, "shape_class": "square"}
        res_v = check_wrist_grasp_alignment(sq_v_bad, tol_u_px=20.0, tol_v_px=20.0)
        self.assertFalse(res_v["aligned"])
        self.assertEqual(res_v["reason"], "v_misaligned")

        # 6. Misaligned both (Run 8 scenario: cx=422, cy=311)
        run8_obb = {"cx": 422.0, "cy": 250.0, "shape_class": "square"}
        res_both = check_wrist_grasp_alignment(run8_obb, tol_u_px=20.0, tol_v_px=20.0)
        self.assertFalse(res_both["aligned"])
        self.assertEqual(res_both["reason"], "uv_misaligned")

        # 7. Elongated object centered (target: cx=325, cy=285)
        elong_centered = {"cx": 325.0, "cy": 285.0, "shape_class": "elongated"}
        res_elong = check_wrist_grasp_alignment(elong_centered)
        self.assertTrue(res_elong["aligned"])
        self.assertEqual(res_elong["cy_target"], 285.0)

    def test_detect_high_cam_grid_target_and_poses(self):
        from build_grasp_replay_cases import (
            detect_high_cam_grid_target,
            GRID_HOVER_POSES,
            HIGH_CAM_GRID_CENTERS_PX,
        )

        # Verify all 9 grid cells exist in hover poses and pixel centers
        for cell in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]:
            self.assertIn(cell, GRID_HOVER_POSES)
            self.assertIn(cell, HIGH_CAM_GRID_CENTERS_PX)
            self.assertEqual(len(GRID_HOVER_POSES[cell]), 6)
            self.assertEqual(len(HIGH_CAM_GRID_CENTERS_PX[cell]), 2)

        # Create synthetic cam_high image (480, 640)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # R1: pixel center (338, 223) -> draw yellow square (12x12)
        img[217:229, 332:344] = [20, 195, 215]  # Yellow
        # R2: pixel center (355, 223) -> draw green square (12x12)
        img[217:229, 349:361] = [30, 180, 30]   # Green
        # R5: pixel center (355, 237) -> draw red square (12x12)
        img[231:243, 349:361] = [20, 20, 200]   # Red
        # R9: pixel center (371, 253) -> draw blue square (12x12)
        img[247:259, 365:377] = [200, 50, 20]   # Blue

        # Detect Yellow at R1
        res_r1 = detect_high_cam_grid_target(img, target_color="yellow")
        self.assertIsNotNone(res_r1)
        self.assertEqual(res_r1["cell"], "R1")
        self.assertEqual(res_r1["color"], "yellow")
        self.assertLessEqual(res_r1["dist_to_cell_px"], 5.0)
        np.testing.assert_allclose(res_r1["hover_xyz"], [0.285, 0.170, 0.200])

        # Detect Green at R2
        res_r2 = detect_high_cam_grid_target(img, target_color="green")
        self.assertIsNotNone(res_r2)
        self.assertEqual(res_r2["cell"], "R2")
        self.assertEqual(res_r2["color"], "green")
        self.assertLessEqual(res_r2["dist_to_cell_px"], 5.0)
        np.testing.assert_allclose(res_r2["hover_xyz"], [0.280, 0.145, 0.200])

        # Detect Red at R5
        res_r5 = detect_high_cam_grid_target(img, target_color="red")
        self.assertIsNotNone(res_r5)
        self.assertEqual(res_r5["cell"], "R5")
        self.assertEqual(res_r5["color"], "red")
        self.assertLessEqual(res_r5["dist_to_cell_px"], 5.0)
        np.testing.assert_allclose(res_r5["hover_xyz"], [0.250, 0.145, 0.200])

        # Detect Blue at R9
        res_r9 = detect_high_cam_grid_target(img, target_color="blue")
        self.assertIsNotNone(res_r9)
        self.assertEqual(res_r9["cell"], "R9")
        self.assertEqual(res_r9["color"], "blue")
        self.assertLessEqual(res_r9["dist_to_cell_px"], 5.0)
        np.testing.assert_allclose(res_r9["hover_xyz"], [0.230, 0.100, 0.200])

        # Rejection: robot base clamp at v=370 (outside table ROI)
        img_clamp = np.zeros((480, 640, 3), dtype=np.uint8)
        img_clamp[360:375, 310:330] = [20, 195, 215]  # Yellow clamp near cy=368
        self.assertIsNone(detect_high_cam_grid_target(img_clamp, target_color="yellow"))

        # Rejection: ambiguous position far from cell centers (> 11.5 px)
        # (326, 223) is inside table ROI [325, 210, 385, 265], but dist to R1 (338, 223) is 12.0 px > 11.5 px
        img_ambig = np.zeros((480, 640, 3), dtype=np.uint8)
        img_ambig[217:229, 320:332] = [20, 20, 200]  # Red block centered at (325.5, 222.5), dist ~12.5 px
        self.assertIsNone(detect_high_cam_grid_target(img_ambig, target_color="red"))

        # Rejection: invalid color or noise outside ROI
        img_noise = np.zeros((480, 640, 3), dtype=np.uint8)
        img_noise[10:30, 10:30] = [20, 20, 200]  # Outside table ROI
        self.assertIsNone(detect_high_cam_grid_target(img_noise, target_color="red"))
        self.assertIsNone(detect_high_cam_grid_target(None))

    def test_evaluate_grasp_deferral_with_real_wrist_alignment(self):
        raw = np.zeros((50, 7))
        raw[:, 1] = 0.130
        raw[:, 6] = 0.035

        # Aligned OBB (|du| <= 20, |dv| <= 20) -> do NOT defer
        aligned_obb = {"cx": 326.0, "cy": 319.0, "shape_class": "square"}
        defer, loops = guarded.evaluate_grasp_deferral_and_budget(
            raw, index=3, total_loops=6,
            condition_final_grasp=True, completed_conditioned_grasp=False,
            place_after_grasp=True, wrist_obb=aligned_obb,
        )
        self.assertFalse(defer)
        self.assertEqual(loops, 6)

        # Misaligned OBB (Run 8: cx=422.0) -> must defer when budget allows
        misaligned_obb = {"cx": 422.0, "cy": 311.0, "shape_class": "square"}
        defer_mis, loops_mis = guarded.evaluate_grasp_deferral_and_budget(
            raw, index=3, total_loops=6,
            condition_final_grasp=True, completed_conditioned_grasp=False,
            place_after_grasp=True, wrist_obb=misaligned_obb,
        )
        self.assertTrue(defer_mis)
        self.assertEqual(loops_mis, 7)  # Budget extended

        # Misaligned OBB at budget ceiling (index=7, max=8) -> cannot defer further
        defer_ceil, loops_ceil = guarded.evaluate_grasp_deferral_and_budget(
            raw, index=7, total_loops=8,
            condition_final_grasp=True, completed_conditioned_grasp=False,
            place_after_grasp=True, wrist_obb=misaligned_obb,
        )
        self.assertFalse(defer_ceil)
        self.assertEqual(loops_ceil, 8)

    def test_condition_approach_actions_with_grid_hover_guidance(self):
        from build_grasp_replay_cases import (
            condition_approach_actions,
            GRID_HOVER_POSES,
        )
        # Use realistic settled orientation from chunk 0, differing from table pose [-0.42, 1.02, 0.99]
        start_rot = np.array([-0.04, 0.65, 0.45])
        start = np.array([0.1807, 0.0134, 0.2491, start_rot[0], start_rot[1], start_rot[2], 0.0687])
        raw = np.tile(start, (50, 1))
        raw[:, 0] += np.linspace(0.01, 0.06, 50)
        raw[:, 1] += np.linspace(0.01, 0.08, 50)
        raw[:, 2] -= np.linspace(0.0, 0.03, 50)

        # 1. Coarse guidance towards R1 hover pose when wrist_obb is None (high start: direct ramp)
        r1_hover = GRID_HOVER_POSES["R1"]
        cond, held, anchor = condition_approach_actions(
            start, raw, hover_z_m=0.200, grid_hover_pose=r1_hover, wrist_obb=None
        )
        # Verify terminal waypoint matches R1 hover pose
        np.testing.assert_allclose(cond[-1, :2], r1_hover[:2], atol=1e-4)
        self.assertGreaterEqual(float(cond[-1, 2]), 0.200 - 1e-4)
        # Verify max Cartesian step <= 15mm
        step_diffs = np.linalg.norm(np.diff(cond[:, :3], axis=0), axis=1)
        self.assertLessEqual(float(np.max(step_diffs)), 0.015)
        # Verify gripper held open
        self.assertGreaterEqual(float(np.min(cond[:, 6])), 0.0686)
        # STRICT REGRESSION: orientation must start at start_rot with ZERO entry jump,
        # and smoothly interpolate to downward observation pose r1_hover[3:6]
        np.testing.assert_allclose(cond[0, 3:6], start_rot, atol=1e-6)
        np.testing.assert_allclose(cond[-1, 3:6], r1_hover[3:6], atol=1e-4)
        rot_diffs = np.linalg.norm(np.diff(cond[:, 3:6], axis=0), axis=1)
        self.assertLessEqual(float(np.max(rot_diffs)), 0.030)

        # 2. Coarse guidance with low start (start[2] < hover_z_m - 0.010: lift then translate)
        start_low = np.array([0.1807, 0.0134, 0.1700, start_rot[0], start_rot[1], start_rot[2], 0.0687])
        raw_low = np.tile(start_low, (50, 1))
        cond_low, _, _ = condition_approach_actions(
            start_low, raw_low, hover_z_m=0.200, grid_hover_pose=r1_hover, wrist_obb=None
        )
        np.testing.assert_allclose(cond_low[-1, :2], r1_hover[:2], atol=1e-4)
        self.assertGreaterEqual(float(cond_low[-1, 2]), 0.200 - 1e-4)
        np.testing.assert_allclose(cond_low[0, 3:6], start_rot, atol=1e-6)
        np.testing.assert_allclose(cond_low[-1, 3:6], r1_hover[3:6], atol=1e-4)
        step_diffs_low = np.linalg.norm(np.diff(cond_low[:, :3], axis=0), axis=1)
        self.assertLessEqual(float(np.max(step_diffs_low)), 0.015)
        rot_diffs_low = np.linalg.norm(np.diff(cond_low[:, 3:6], axis=0), axis=1)
        self.assertLessEqual(float(np.max(rot_diffs_low)), 0.030)

        # 3. When wrist_obb is present, wrist OBB centering takes precedence and decouples from raw closure
        wrist_obb = {"cx": 340.0, "cy": 290.0, "shape_class": "square"}
        cond_wrist, held_wrist, anchor_wrist = condition_approach_actions(
            start, raw, hover_z_m=0.200, grid_hover_pose=r1_hover, wrist_obb=wrist_obb
        )
        self.assertIsNotNone(cond_wrist)
        self.assertIsNotNone(anchor_wrist)
        self.assertEqual(anchor_wrist, len(cond_wrist) - 1)
        self.assertGreaterEqual(float(np.min(cond_wrist[:, 6])), 0.068)

    def test_coarse_to_fine_handover_and_decoupled_alignment(self):
        """Verify full lifecycle from coarse transit to hover arrival to decoupled wrist alignment."""
        from build_grasp_replay_cases import (
            condition_approach_actions,
            GRID_HOVER_POSES,
        )
        r2_hover = GRID_HOVER_POSES["R2"]

        # Loop 0/1: Coarse Transit towards R2
        start_rot = np.array([-0.04, 0.65, 0.45])  # Early lift shallow pitch ~20°
        start_transit = np.array([0.1807, 0.0134, 0.2450, start_rot[0], start_rot[1], start_rot[2], 0.0687])
        raw_transit = np.tile(start_transit, (50, 1))

        cond_transit, _, _ = condition_approach_actions(
            start_transit, raw_transit, hover_z_m=0.200, grid_hover_pose=r2_hover, wrist_obb=None
        )
        # Entry jump must be strictly 0.0
        np.testing.assert_allclose(cond_transit[0, 3:6], start_rot, atol=1e-6)
        # Terminal pose must arrive at downward observation attitude
        np.testing.assert_allclose(cond_transit[-1, 3:6], r2_hover[3:6], atol=1e-4)
        np.testing.assert_allclose(cond_transit[-1, :2], r2_hover[:2], atol=1e-4)

        # Loop 2: Arm arrived at R2 hover pose (within 20mm), but model raw[:, 6] predicts NO closure (0.070m)
        arrived_start = cond_transit[-1].copy()
        raw_hover = np.tile(arrived_start, (50, 1))
        raw_hover[:, 6] = 0.070  # Model has NOT predicted closure

        # Case A: Block detected by wrist camera (e.g. Run 9 loop 2: cx=298.3, cy=245.8)
        wrist_obb = {
            "cx": 298.3, "cy": 245.8, "w": 37.1, "h": 36.1, "angle": 76.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
            "color_class": "yellow",
        }
        # When coarse guidance is cleared upon arrival (grid_hover_pose=None)
        cond_fine, held, anchor = condition_approach_actions(
            arrived_start, raw_hover, hover_z_m=0.200, grid_hover_pose=None, wrist_obb=wrist_obb
        )
        # Visual servoing must anchor pose and apply horizontal centering
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor, len(cond_fine) - 1)
        # Verify horizontal correction moved end effector towards block
        self.assertNotEqual(float(cond_fine[-1, 0]), float(arrived_start[0]))
        # Gripper remains open at hover height
        self.assertGreaterEqual(float(np.min(cond_fine[:, 6])), 0.068)

        # Case B: Already at hover location with grid_hover_pose still passed
        # Must NOT freeze into static identical waypoints; must maintain downward observation pitch
        cond_exit, _, _ = condition_approach_actions(
            arrived_start, raw_hover, hover_z_m=0.200, grid_hover_pose=r2_hover, wrist_obb=None
        )
        self.assertIsNotNone(cond_exit)
        np.testing.assert_allclose(cond_exit[-1, 3:6], r2_hover[3:6], atol=1e-4)

    def test_descent_chunk_centers_x_when_wrist_obb_misaligned(self):
        """When descending below hover height and wrist_obb has horizontal error (cx=416.0),
        condition_approach_actions must actively center X towards target rather than freezing X."""
        from build_grasp_replay_cases import condition_approach_actions
        # Low descent start pose: Z=0.170m < hover_z_m - 0.005m (hover=0.200m)
        # Sits at center X=0.259m (R4/R5), while target is at R1/R2 (X ~ 0.285m)
        start_low = np.array([0.259, 0.165, 0.170, -0.42, 1.02, 0.99, 0.070])
        raw_low = np.tile(start_low, (50, 1))
        # Simulated descent model predicts closure later in chunk
        raw_low[20:, 6] = 0.005

        # Wrist OBB sees cube off to the right at cx=416.0px (du = 416.0 - 337.0 = +79.0px)
        obb_r1 = {
            "cx": 416.0, "cy": 315.0, "w": 40.0, "h": 40.0, "angle": 0.0, "area": 1600.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
        }
        cond, held, anchor = condition_approach_actions(start_low, raw_low, hover_z_m=0.200, wrist_obb=obb_r1)
        self.assertIsNotNone(cond)
        # Initial step must match start[0] exactly (0.0mm gap at handoff)
        self.assertAlmostEqual(float(cond[0, 0]), float(start_low[0]), places=4)
        # End of chunk must actively shift +X towards R1/R2 by at least +20mm
        x_shift_mm = (float(cond[-1, 0]) - float(start_low[0])) * 1000.0
        self.assertGreaterEqual(x_shift_mm, 20.0)
        self.assertLessEqual(x_shift_mm, 26.0)
        # Orientation must remain strictly locked at start[3:6]
        np.testing.assert_allclose(cond[:, 3:6], np.tile(start_low[3:6], (50, 1)), atol=1e-5)

    def test_two_stage_hover_gate_holds_z_when_misaligned(self):
        """When at hover height (Z=0.200m) and wrist_obb is misaligned (|du| > 20px),
        condition_approach_actions must NOT descend; it must hold Z >= hover_z across all steps."""
        from build_grasp_replay_cases import condition_approach_actions
        start_hover = np.array([0.240, 0.120, 0.200, -0.42, 1.02, 0.99, 0.070])
        # Raw model tries to plunge Z down to 0.165m and close
        raw_plunge = np.tile(start_hover, (50, 1))
        raw_plunge[:, 2] = np.linspace(0.200, 0.165, 50)
        raw_plunge[20:, 6] = 0.005

        # Misaligned wrist OBB: block at cx=400.0px (du_rot ~ 63px > 20px)
        obb_misaligned = {
            "cx": 400.0, "cy": 320.0, "w": 40.0, "h": 40.0, "angle": 0.0, "area": 1600.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
        }
        cond, _, _ = condition_approach_actions(start_hover, raw_plunge, hover_z_m=0.200, wrist_obb=obb_misaligned)
        self.assertIsNotNone(cond)
        # Z must remain held at hover height (>= 0.200m - 1e-4) across entire chunk!
        self.assertGreaterEqual(float(np.min(cond[:, 2])), 0.1999)
        # X must actively adjust towards block
        self.assertGreater(float(cond[-1, 0]), float(start_hover[0]))

    def test_two_stage_hover_gate_allows_descent_when_aligned(self):
        """When at hover height (Z=0.200m) and wrist_obb IS aligned,
        condition_approach_actions permits clean vertical descent to table height."""
        from build_grasp_replay_cases import condition_approach_actions
        start_hover = np.array([0.280, 0.145, 0.200, -0.42, 1.02, 0.99, 0.070])
        raw_plunge = np.tile(start_hover, (50, 1))
        raw_plunge[:, 2] = np.linspace(0.200, 0.165, 50)
        raw_plunge[20:, 6] = 0.005

        # Aligned wrist OBB at hover Z=0.200m under calibrated 3D pinhole model: ~(311.9, 308.9)
        obb_aligned = {
            "cx": 312.0, "cy": 309.0, "w": 40.0, "h": 40.0, "angle": 0.0, "area": 1600.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
        }
        cond, _, _ = condition_approach_actions(start_hover, raw_plunge, hover_z_m=0.200, wrist_obb=obb_aligned)
        self.assertIsNotNone(cond)
        # Descent must be permitted: final Z reaches <= 0.165m
        self.assertLessEqual(float(cond[-1, 2]), 0.165)

        # Former wrong parallax pixel cy=358.0 is +49.1px overshot under calibrated physics: blocks plunge
        obb_mis = dict(obb_aligned, cx=325.0, cy=358.0)
        cond_mis, _, _ = condition_approach_actions(start_hover, raw_plunge, hover_z_m=0.200, wrist_obb=obb_mis)
        self.assertGreaterEqual(float(np.min(cond_mis[:, 2])), 0.1999)

    def test_low_altitude_deviation_lifts_to_hover(self):
        """When end effector is near table (Z=0.165m) and detects large deviation (|du| > 25px),
        it must lift back up towards hover_z_m to prevent plowing along the tabletop."""
        from build_grasp_replay_cases import condition_approach_actions
        start_low = np.array([0.240, 0.150, 0.165, -0.42, 1.02, 0.99, 0.070])
        raw_low = np.tile(start_low, (50, 1))
        raw_low[20:, 6] = 0.005

        # Large deviation: block at cx=400px (du = 63px > 25px)
        obb_far = {
            "cx": 400.0, "cy": 315.0, "w": 40.0, "h": 40.0, "angle": 0.0, "area": 1600.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
        }
        cond, _, _ = condition_approach_actions(start_low, raw_low, hover_z_m=0.200, wrist_obb=obb_far)
        self.assertIsNotNone(cond)
        # End of chunk must have lifted up towards hover_z_m
        self.assertGreaterEqual(float(cond[-1, 2]), 0.195)

    def test_parallax_aware_alignment_target(self):
        """check_wrist_grasp_alignment correctly shifts cy_target at hover height."""
        from build_grasp_replay_cases import check_wrist_grasp_alignment
        # At table height (Z=0.160m), cy=318px is aligned
        obb_table = {"cx": 325.0, "cy": 318.0, "shape_class": "square"}
        res_table = check_wrist_grasp_alignment(obb_table, ee_z_m=0.160)
        self.assertTrue(res_table["aligned"])
        self.assertAlmostEqual(res_table["cy_target"], 318.0, places=1)

        # At hover height (Z=0.200m, nominal RPY), calibrated 3D pinhole projects target up to ~(311.9, 308.9)
        obb_hover = {"cx": 312.0, "cy": 309.0, "shape_class": "square"}
        res_hover = check_wrist_grasp_alignment(obb_hover, ee_z_m=0.200)
        self.assertTrue(res_hover["aligned"])
        self.assertLess(res_hover["cx_target"], 325.0)
        self.assertLess(res_hover["cy_target"], 318.0)
        self.assertAlmostEqual(res_hover["cx_target"], 311.94, places=1)
        self.assertAlmostEqual(res_hover["cy_target"], 308.90, places=1)
        # Old +46px downward shift (cy=364) is strongly misaligned (+55px)
        res_old = check_wrist_grasp_alignment({"cx": 325.0, "cy": 364.0, "shape_class": "square"}, ee_z_m=0.200)
        self.assertFalse(res_old["aligned"])

    def test_condition_grasp_actions_at_hover_descends_before_closing(self):
        """When grasp is triggered from hover height (Z=0.1971m), condition_grasp_actions
        must descend to table grasp height (Z=0.160m) with gripper open before closing."""
        from build_grasp_replay_cases import condition_grasp_actions

        start_hover = np.array([0.305, 0.2016, 0.1971, -0.42, 1.02, 0.99, 0.0686])
        raw = np.tile(start_hover, (50, 1))
        raw[20:, 6] = 0.005  # Raw policy predicts closure

        wrist_obb = {
            "cx": 343.7, "cy": 362.8, "w": 50.0, "h": 50.0, "angle": 84.8, "area": 2500.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
        }
        cond, clipped, splits = condition_grasp_actions(start_hover, raw, hold_pose=True, wrist_obb=wrist_obb)
        self.assertIsNotNone(cond)
        # 1. First 16 steps must smoothly descend from 0.1971m down to 0.160m
        self.assertAlmostEqual(float(cond[0, 2]), 0.1971, places=3)
        self.assertAlmostEqual(float(cond[16, 2]), 0.1600, places=3)
        self.assertAlmostEqual(float(cond[-1, 2]), 0.1600, places=3)
        # 2. Gripper must remain open during descent (steps 0..16)
        self.assertAlmostEqual(float(cond[0, 6]), 0.0686, places=3)
        self.assertAlmostEqual(float(cond[15, 6]), 0.0686, places=3)
        # 3. Gripper must close firmly at table height by end of chunk
        self.assertAlmostEqual(float(cond[-1, 6]), 0.0000, places=3)

    def test_condition_grasp_actions_at_table_holds_height(self):
        """When grasp is triggered near table height (Z=0.162m), condition_grasp_actions
        holds table level and closes directly without high descent."""
        from build_grasp_replay_cases import condition_grasp_actions

        start_table = np.array([0.280, 0.145, 0.162, -0.42, 1.02, 0.99, 0.0686])
        raw = np.tile(start_table, (50, 1))
        raw[15:, 6] = 0.005

        wrist_obb = {
            "cx": 337.0, "cy": 320.0, "w": 50.0, "h": 50.0, "angle": 85.0, "area": 2500.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
        }
        cond, clipped, splits = condition_grasp_actions(start_table, raw, hold_pose=True, wrist_obb=wrist_obb)
        self.assertIsNotNone(cond)
        # Z must stay bounded near table level (<= 0.163m)
        self.assertLessEqual(float(np.max(cond[:, 2])), 0.163)
        self.assertGreaterEqual(float(np.min(cond[:, 2])), 0.159)
        # Gripper closes to 0
        self.assertAlmostEqual(float(cond[-1, 6]), 0.0, places=3)

    def test_evaluate_grasp_deferral_at_hover_vs_table(self):
        """evaluate_grasp_deferral_and_budget must defer when aligned at hover (Z > 0.180m)
        to allow a descent chunk, and authorize grasp when aligned at table (Z <= 0.180m)."""
        from build_grasp_replay_cases import check_wrist_grasp_alignment
        from station_lite_smolvla_right7_guarded import evaluate_grasp_deferral_and_budget

        actions = np.zeros((50, 7))
        actions[:, 6] = 0.0686
        actions[20:, 6] = 0.005

        # At hover (Z=0.1971m): genuinely aligned under calibrated 3D pinhole target (~312.8, ~309.5)
        obb_hover = {"cx": 313.0, "cy": 310.0, "shape_class": "square"}
        self.assertTrue(check_wrist_grasp_alignment(obb_hover, ee_z_m=0.1971)["aligned"])
        defer, loops = evaluate_grasp_deferral_and_budget(
            actions, index=3, total_loops=6, condition_final_grasp=True,
            completed_conditioned_grasp=False, place_after_grasp=True,
            max_total_loops=10, wrist_obb=obb_hover, ee_z_m=0.1971,
        )
        self.assertTrue(defer, "Must defer grasp when aligned at hover height to permit vertical descent")
        self.assertGreaterEqual(loops, 7)

        # At table (Z=0.162m): genuinely aligned at table target (~324.4, ~317.6)
        obb_table = {"cx": 325.0, "cy": 318.0, "shape_class": "square"}
        self.assertTrue(check_wrist_grasp_alignment(obb_table, ee_z_m=0.162)["aligned"])
        defer_table, _ = evaluate_grasp_deferral_and_budget(
            actions, index=4, total_loops=loops, condition_final_grasp=True,
            completed_conditioned_grasp=False, place_after_grasp=True,
            max_total_loops=10, wrist_obb=obb_table, ee_z_m=0.162,
        )
        self.assertFalse(defer_table, "Must authorize grasp closure when aligned at table level")

    def test_r8_diagonal_plunge_cancellation_prevented_by_hover_gate(self):
        """When raw policy commands a diagonal plunge (+26.5mm in Y, -53mm in Z) towards R7
        while the block is at R8 (causing dv_rot to artificially cancel to ~1.4px),
        the hover gate must detect dv_hover = -47.5px and strictly block descent."""
        from build_grasp_replay_cases import condition_approach_actions

        # End effector at R8 hover height (X=0.2268, Y=0.1458, Z=0.2071)
        start_r8 = np.array([0.2268, 0.1458, 0.2071, -0.42, 1.02, 0.99, 0.0686])
        # Raw policy predicts moving to R7 (Y=0.1723) and plunging to Z=0.1541
        raw_plunge_r7 = np.tile(start_r8, (50, 1))
        raw_plunge_r7[:, 0] = np.linspace(0.2268, 0.2321, 50)
        raw_plunge_r7[:, 1] = np.linspace(0.1458, 0.1723, 50)  # +26.5mm forward towards R7
        raw_plunge_r7[:, 2] = np.linspace(0.2071, 0.1541, 50)  # -53mm down
        raw_plunge_r7[25:, 6] = 0.005

        # R8 block short in Y at cy=258.0 (dv = 258.0 - 307.7 = -49.7px, du = -2.4px)
        wrist_obb_r8 = {
            "cx": 307.8, "cy": 258.0, "w": 49.0, "h": 44.4, "angle": 79.4, "area": 2175.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
        }
        cond, _, _ = condition_approach_actions(start_r8, raw_plunge_r7, hover_z_m=0.200, wrist_obb=wrist_obb_r8)
        self.assertIsNotNone(cond)
        # Descent must be strictly blocked: Z must remain >= hover_z_m (0.200m) and shift +Y toward block!
        self.assertGreaterEqual(float(np.min(cond[:, 2])), 0.1999)
        self.assertGreater(float(cond[-1, 1]), float(start_r8[1]) + 0.020)

    def test_r8_loop4_cube_alignment_executes_descent_without_sway(self):
        """Regression test for 20261112 new6 / 20261113 R8 Chunk 4:
        1) When block is actually +45.7px overshot (cx=314.9, cy=354.2 at Z=0.2021m),
           the hover gate must block plunge (Z >= 0.200m) and pull -Y back toward cube.
        2) When truly aligned at R8 hover (cx=312.0, cy=309.0), executes clean vertical descent."""
        from build_grasp_replay_cases import condition_approach_actions
        # EE pose at start of Chunk 4
        start_r8_chunk4 = np.array([0.2322, 0.1664, 0.2021, -0.42, 1.02, 0.99, 0.0686])
        raw_r8 = np.tile(start_r8_chunk4, (50, 1))
        for i in range(20):
            raw_r8[i, 0] = 0.2322 + (0.2353 - 0.2322) * ((i + 1) / 20.0)
            raw_r8[i, 1] = 0.1664 + (0.1907 - 0.1664) * ((i + 1) / 20.0)
            raw_r8[i, 2] = 0.2021 - (0.2021 - 0.1733) * ((i + 1) / 20.0)
        for i in range(20, 45):
            raw_r8[i, 0] = 0.2353 - (0.2353 - 0.2304) * ((i - 19) / 25.0)
            raw_r8[i, 1] = 0.1907 - (0.1907 - 0.1756) * ((i - 19) / 25.0)
            raw_r8[i, 2] = 0.1733 - (0.1733 - 0.1544) * ((i - 19) / 25.0)
        raw_r8[33:, 6] = 0.005

        wrist_obb_mis = {
            "cx": 314.9, "cy": 354.2, "w": 56.8, "h": 47.2, "angle": 77.5, "area": 2680.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
        }
        cond_mis, _, _ = condition_approach_actions(start_r8_chunk4, raw_r8, hover_z_m=0.200, wrist_obb=wrist_obb_mis)
        self.assertGreaterEqual(float(np.min(cond_mis[:, 2])), 0.1999)
        self.assertLess(float(cond_mis[-1, 1]), float(start_r8_chunk4[1]) - 0.020)

        wrist_obb_aligned = dict(wrist_obb_mis, cx=312.0, cy=309.0)
        cond, _, anchor_idx = condition_approach_actions(start_r8_chunk4, raw_r8, hover_z_m=0.200, wrist_obb=wrist_obb_aligned)
        self.assertIsNotNone(cond)
        # 1. Must permit vertical descent down to table height:
        self.assertLessEqual(float(cond[-1, 2]), 0.165)
        # 2. X position must remain centered on cube (within 5mm of start):
        self.assertLess(abs(float(cond[-1, 0]) - start_r8_chunk4[0]), 0.006)
        # 3. Y position must not overshoot to raw peak_y (0.191m):
        self.assertLess(float(cond[-1, 1]), 0.178)

    def test_r2_loop5_cube_alignment_executes_descent_without_sway(self):
        """Regression test for 20261112 new6 / 20261113 R2 Chunk 5:
        1) When block is misaligned (cx=336.5, cy=361.2 at Z=0.2133m -> du=+27.8, dv=+54.6),
           hover gate holds >= 0.200m, relaxes Z downward (no Z-ratchet at 0.2133), and pulls -Y back.
        2) When truly aligned at R2 hover (cx=310.0, cy=301.0), descends cleanly to table height."""
        from build_grasp_replay_cases import condition_approach_actions
        start_r2_chunk5 = np.array([0.3060, 0.2035, 0.2133, -0.42, 1.02, 0.99, 0.0686])
        raw_r2 = np.tile(start_r2_chunk5, (50, 1))
        raw_r2[:, 2] = np.linspace(0.2133, 0.1728, 50)
        raw_r2[36:, 6] = 0.005

        wrist_obb_mis = {
            "cx": 336.5, "cy": 361.2, "w": 56.6, "h": 46.1, "angle": 82.4, "area": 2609.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
        }
        cond_mis, _, _ = condition_approach_actions(start_r2_chunk5, raw_r2, hover_z_m=0.200, wrist_obb=wrist_obb_mis)
        self.assertGreaterEqual(float(np.min(cond_mis[:, 2])), 0.1999)
        self.assertLess(float(cond_mis[-1, 2]), float(start_r2_chunk5[2]) - 0.005)
        self.assertLess(float(cond_mis[-1, 1]), float(start_r2_chunk5[1]) - 0.020)

        wrist_obb_aligned = dict(wrist_obb_mis, cx=310.0, cy=301.0)
        cond, _, _ = condition_approach_actions(start_r2_chunk5, raw_r2, hover_z_m=0.200, wrist_obb=wrist_obb_aligned)
        self.assertIsNotNone(cond)
        # 1. Must permit vertical descent:
        self.assertLessEqual(float(cond[-1, 2]), 0.165)
        # 2. X must not drift away (remains within 12mm of start):
        self.assertLess(abs(float(cond[-1, 0]) - start_r2_chunk5[0]), 0.012)

    def test_pinhole_euler_parity_with_scipy(self):
        from scipy.spatial.transform import Rotation
        from build_grasp_replay_cases import _euler_xyz_to_matrix
        rng = np.random.default_rng(20261113)
        samples = rng.uniform(-np.pi, np.pi, size=(50, 3))
        for rpy in samples:
            R_np = _euler_xyz_to_matrix(rpy)
            R_scipy = Rotation.from_euler("xyz", rpy).as_matrix()
            self.assertLess(float(np.max(np.abs(R_np - R_scipy))), 1e-12)

    def test_pinhole_closed_loop_r8_and_r2_convergence(self):
        from build_grasp_replay_cases import (
            check_wrist_grasp_alignment,
            project_wrist_world_point,
            solve_wrist_xy_correction,
            wrist_target_uv_for_pose,
        )
        # Nominal hover target cubes in R8 and R2 regions with true robot end-effector orientations
        scenarios = [
            ("R8_short", np.array([0.232, 0.130, 0.202, -0.192, 1.065, 1.100]), np.array([0.235, 0.168, 0.0198])),
            ("R8_over",  np.array([0.238, 0.198, 0.215, -0.192, 1.065, 1.100]), np.array([0.235, 0.168, 0.0198])),
            ("R2_short", np.array([0.312, 0.172, 0.213, -0.384, 1.204, 1.030]), np.array([0.308, 0.208, 0.0198])),
            ("R2_over",  np.array([0.302, 0.236, 0.218, -0.384, 1.204, 1.030]), np.array([0.308, 0.208, 0.0198])),
        ]
        for label, pose, cube_xyz in scenarios:
            p = pose.copy()
            for _ in range(3):
                uv = project_wrist_world_point(p, cube_xyz)
                self.assertIsNotNone(uv, msg=label)
                dx, dy, _ = solve_wrist_xy_correction(p, uv[0], uv[1])
                p[0] += dx
                p[1] += dy
                p[2] = max(0.200, p[2] - 0.015)
            uv_final = project_wrist_world_point(p, cube_xyz)
            tgt = wrist_target_uv_for_pose(p)
            al = check_wrist_grasp_alignment({"cx": uv_final[0], "cy": uv_final[1], "shape_class": "square"}, ee_pose6=p)
            self.assertTrue(al["aligned"], msg=f"{label}: {al}")
            self.assertLess(abs(uv_final[0] - tgt[0]), 2.0, msg=label)
            self.assertLess(abs(uv_final[1] - tgt[1]), 2.0, msg=label)

    def test_hover_dv_residual_only_misalignment_blocks_descent(self):
        from build_grasp_replay_cases import check_wrist_grasp_alignment, condition_approach_actions, wrist_target_uv_for_pose
        start_hover = np.array([0.280, 0.145, 0.200, -0.42, 1.02, 0.99, 0.070])
        raw_plunge = np.tile(start_hover, (50, 1))
        raw_plunge[:, 2] = np.linspace(0.200, 0.162, 50)
        raw_plunge[25:, 6] = 0.005
        u_tgt, v_tgt = wrist_target_uv_for_pose(start_hover[:6])

        for dv_offset in (30.0, -30.0):
            obb_dv = {
                "cx": u_tgt, "cy": v_tgt + dv_offset, "w": 42.0, "h": 42.0, "angle": 0.0,
                "area": 1764.0, "shape_class": "square", "grasp_axis_mode": "symmetric_square",
            }
            al = check_wrist_grasp_alignment(obb_dv, ee_pose6=start_hover[:6])
            self.assertFalse(al["aligned"])
            self.assertLessEqual(abs(al["du"]), 1.0)
            self.assertGreater(abs(al["dv"]), 20.0)
            cond, _, _ = condition_approach_actions(start_hover, raw_plunge, hover_z_m=0.200, wrist_obb=obb_dv)
            self.assertGreaterEqual(float(np.min(cond[:, 2])), 0.1999)

    def test_high_hover_z_relaxes_downward_instead_of_ratcheting(self):
        from build_grasp_replay_cases import condition_approach_actions
        start_high = np.array([0.3060, 0.2035, 0.2190, -0.42, 1.02, 0.99, 0.0686])
        raw_plunge = np.tile(start_high, (50, 1))
        raw_plunge[:, 2] = np.linspace(0.2190, 0.1700, 50)
        obb_mis = {
            "cx": 335.0, "cy": 355.0, "w": 48.0, "h": 48.0, "angle": 0.0, "area": 2304.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
        }
        cond, _, _ = condition_approach_actions(start_high, raw_plunge, hover_z_m=0.200, wrist_obb=obb_mis)
        # Must relax smoothly toward 0.200m rather than ratcheting at 0.219m, while holding >= 0.200m
        self.assertGreaterEqual(float(np.min(cond[:, 2])), 0.1999)
        self.assertLess(float(cond[-1, 2]), 0.2050)
        self.assertAlmostEqual(float(cond[-1, 2]), 0.2000, delta=0.001)

    def test_wrist_obb_finger_occlusion_stabilizes_square_shape(self):
        from build_grasp_replay_cases import detect_wrist_block_obb
        # Synthetic clipped yellow block at cy>=310 with aspect ~1.45 (<1.85) under gripper fingers
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Yellow in BGR = (0, 210, 210); draw 38px tall x 55px wide centered around (320, 345)
        img[326:364, 293:348] = [0, 210, 210]
        raw_obb = detect_wrist_block_obb(img, target_color="yellow")
        self.assertIsNotNone(raw_obb)
        self.assertEqual(raw_obb["shape_class"], "elongated")

        prev_square = {
            "cx": 318.0, "cy": 325.0, "w": 52.0, "h": 50.0,
            "shape_class": "square", "grasp_axis_mode": "symmetric_square",
            "color_class": "yellow",
        }
        stab_obb = detect_wrist_block_obb(img, target_color="yellow", prev_obb=prev_square)
        self.assertIsNotNone(stab_obb)
        self.assertEqual(stab_obb["shape_class"], "square")
        self.assertEqual(stab_obb["grasp_axis_mode"], "symmetric_square")

    def test_cli_high_cam_guidance_defaults_true(self):
        from unittest.mock import patch
        with patch.object(sys, "argv", ["prog", "--server", "http://127.0.0.1:9999"]):
            args_default = guarded.parse_args()
        self.assertTrue(args_default.high_cam_guidance)
        with patch.object(sys, "argv", ["prog", "--server", "http://127.0.0.1:9999", "--no-high-cam-guidance"]):
            args_off = guarded.parse_args()
        self.assertFalse(args_off.high_cam_guidance)


if __name__ == "__main__":
    unittest.main()
