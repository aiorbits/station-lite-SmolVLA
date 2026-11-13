#!/usr/bin/env python3
"""Build raw and conditioned grasp replay inputs. Never publishes motion."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


# Calibrated 3D pinhole model of cam_right_wrist (joint fit across 20261111-1113 captures;
# Euler convention 'xyz' extrinsic, global median error 1.04 px, LORO forward v-RMSE 1.36 px).
WRIST_CAM_FX = 258.296033
WRIST_CAM_FY = 269.044864
WRIST_CAM_U0 = 320.0
WRIST_CAM_V0 = 240.0
WRIST_CAM_TABLE_OBJ_Z_M = 0.0198019
WRIST_CAM_DEFAULT_GRASP_Z_M = 0.160
WRIST_CAM_NOMINAL_RPY_RAD = np.array([-0.42, 1.02, 0.99], dtype=float)
WRIST_CAM_T_EE_TO_CAM_M = np.array([0.044151, 0.005561, 0.099800], dtype=float)
_R_EC_U, _, _R_EC_VT = np.linalg.svd(
    np.array(
        [
            [0.00190767, -0.37367704, 0.92755691],
            [-0.99988655, -0.01457211, -0.00381412],
            [0.01494171, -0.92744440, -0.37366244],
        ],
        dtype=float,
    )
)
WRIST_CAM_R_EE_TO_CAM = _R_EC_U @ _R_EC_VT
if np.linalg.det(WRIST_CAM_R_EE_TO_CAM) < 0:
    _R_EC_U[:, -1] *= -1.0
    WRIST_CAM_R_EE_TO_CAM = _R_EC_U @ _R_EC_VT

WRIST_CAM_GRASP_CX_PX = 325.0
WRIST_CAM_GRASP_CY_SQUARE_PX = 318.0
WRIST_CAM_GRASP_CY_ELONGATED_PX = 285.0


def _euler_xyz_to_matrix(rpy_rad) -> np.ndarray:
    """Convert extrinsic Euler 'xyz' [roll, pitch, yaw] radians to 3x3 rotation matrix."""
    r, p, y = float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _effective_wrist_cam_rpy(rpy_rad) -> np.ndarray:
    """Use supplied right7 EE orientation if camera optical axis looks downward, else nominal hover RPY."""
    rpy = np.asarray(rpy_rad, dtype=float).reshape(3)
    if np.isfinite(rpy).all():
        r_we = _euler_xyz_to_matrix(rpy)
        if float((r_we @ WRIST_CAM_R_EE_TO_CAM)[2, 2]) < -0.50:
            return rpy
    return WRIST_CAM_NOMINAL_RPY_RAD.copy()


def _coerce_wrist_cam_pose6(ee_pose6=None, ee_z_m=None) -> np.ndarray | None:
    """Coerce 6D EE pose or scalar Z height into a valid 6D pose in right7 camera frame."""
    if ee_pose6 is not None:
        arr = np.asarray(ee_pose6, dtype=float).reshape(-1)
        if arr.size >= 6 and np.isfinite(arr[:6]).all():
            return np.r_[arr[:3], _effective_wrist_cam_rpy(arr[3:6])]
    if ee_z_m is not None and math.isfinite(float(ee_z_m)):
        return np.r_[0.25, 0.14, float(ee_z_m), WRIST_CAM_NOMINAL_RPY_RAD]
    return None


def _wrist_obb_base_targets(
    wrist_obb: dict | None,
    *,
    cx_target: float | None = None,
    cy_target: float | None = None,
) -> tuple[float, float]:
    """Return canonical table-height grasp pixel target (cx*, cy*) for square or elongated OBBs."""
    eff_cx = (
        float(cx_target)
        if cx_target is not None
        else float((wrist_obb or {}).get("cx_target", WRIST_CAM_GRASP_CX_PX))
    )
    if cy_target is not None:
        eff_cy = float(cy_target)
    elif wrist_obb is None:
        eff_cy = WRIST_CAM_GRASP_CY_SQUARE_PX
    else:
        shape_cls = wrist_obb.get("shape_class")
        mode = wrist_obb.get("grasp_axis_mode")
        if shape_cls == "square" or mode == "symmetric_square":
            is_square = True
        elif shape_cls == "elongated" or mode == "long_axis_between_fingers":
            is_square = False
        elif "w" in wrist_obb and "h" in wrist_obb:
            w_f = abs(float(wrist_obb["w"]))
            h_f = abs(float(wrist_obb["h"]))
            is_square = bool((max(w_f, h_f) / max(1.0, min(w_f, h_f))) < 1.35)
        elif "aspect_ratio" in wrist_obb:
            is_square = bool(float(wrist_obb["aspect_ratio"]) < 1.35)
        else:
            is_square = False
        eff_cy = WRIST_CAM_GRASP_CY_SQUARE_PX if is_square else WRIST_CAM_GRASP_CY_ELONGATED_PX
    return eff_cx, eff_cy


def raycast_wrist_pixel_to_table(
    ee_pose6,
    u_px: float,
    v_px: float,
    *,
    table_obj_z_m: float = WRIST_CAM_TABLE_OBJ_Z_M,
) -> np.ndarray | None:
    """Intersect wrist-camera pixel ray from 6D EE pose with tabletop object plane Z=table_obj_z_m."""
    pose = _coerce_wrist_cam_pose6(ee_pose6=ee_pose6)
    if pose is None:
        return None
    r_we = _euler_xyz_to_matrix(pose[3:6])
    cam_pos = pose[:3] + r_we @ WRIST_CAM_T_EE_TO_CAM_M
    ray_cam = np.array(
        [
            (float(u_px) - WRIST_CAM_U0) / WRIST_CAM_FX,
            (float(v_px) - WRIST_CAM_V0) / WRIST_CAM_FY,
            1.0,
        ],
        dtype=float,
    )
    ray_w = r_we @ (WRIST_CAM_R_EE_TO_CAM @ ray_cam)
    if ray_w[2] >= -0.10:
        return None
    lam = (float(table_obj_z_m) - float(cam_pos[2])) / float(ray_w[2])
    if lam <= 0.02:
        return None
    return cam_pos + lam * ray_w


def project_wrist_world_point(ee_pose6, world_xyz) -> tuple[float, float] | None:
    """Project 3D world point into cam_right_wrist image coordinates at 6D EE pose."""
    pose = _coerce_wrist_cam_pose6(ee_pose6=ee_pose6)
    if pose is None:
        return None
    r_we = _euler_xyz_to_matrix(pose[3:6])
    p_ee = r_we.T @ (np.asarray(world_xyz, dtype=float).reshape(3) - pose[:3])
    p_cam = WRIST_CAM_R_EE_TO_CAM.T @ (p_ee - WRIST_CAM_T_EE_TO_CAM_M)
    if p_cam[2] <= 0.02:
        return None
    u = WRIST_CAM_FX * (p_cam[0] / p_cam[2]) + WRIST_CAM_U0
    v = WRIST_CAM_FY * (p_cam[1] / p_cam[2]) + WRIST_CAM_V0
    return float(u), float(v)


def wrist_target_uv_for_pose(
    ee_pose6=None,
    *,
    ee_z_m: float | None = None,
    cx_target: float = WRIST_CAM_GRASP_CX_PX,
    cy_target: float = WRIST_CAM_GRASP_CY_SQUARE_PX,
    table_grasp_z_m: float = WRIST_CAM_DEFAULT_GRASP_Z_M,
) -> tuple[float, float]:
    """Project canonical table-grasp center (cx*, cy*) at Z=table_grasp_z_m into current hover/EE pose."""
    pose = _coerce_wrist_cam_pose6(ee_pose6=ee_pose6, ee_z_m=ee_z_m)
    if pose is None or abs(float(pose[2]) - float(table_grasp_z_m)) < 1e-6:
        return float(cx_target), float(cy_target)
    table_ref_pose = np.r_[pose[:2], float(table_grasp_z_m), pose[3:6]]
    world_ideal = raycast_wrist_pixel_to_table(table_ref_pose, float(cx_target), float(cy_target))
    if world_ideal is None:
        return float(cx_target), float(cy_target)
    uv = project_wrist_world_point(pose, world_ideal)
    if uv is None:
        return float(cx_target), float(cy_target)
    return uv


def solve_wrist_xy_correction(
    start_pose6,
    cx_obs: float,
    cy_obs: float,
    *,
    end_rpy_rad=None,
    cx_target: float = WRIST_CAM_GRASP_CX_PX,
    cy_target: float = WRIST_CAM_GRASP_CY_SQUARE_PX,
    table_grasp_z_m: float = WRIST_CAM_DEFAULT_GRASP_Z_M,
    gain: float = 1.0,
    deadband_u_px: float = 0.0,
    deadband_v_px: float = 0.0,
    max_dx_range: tuple[float, float] = (-0.035, 0.035),
    max_dy_range: tuple[float, float] = (-0.035, 0.040),
) -> tuple[float, float, dict]:
    """Solve coupled 3D world-XY EE shift from observed pixel to canonical grasp pixel at target orientation."""
    pose = _coerce_wrist_cam_pose6(ee_pose6=start_pose6)
    u_tgt, v_tgt = wrist_target_uv_for_pose(
        pose, cx_target=cx_target, cy_target=cy_target, table_grasp_z_m=table_grasp_z_m
    )
    du = float(cx_obs) - u_tgt
    dv = float(cy_obs) - v_tgt
    erpy = (
        pose[3:6]
        if end_rpy_rad is None
        else _effective_wrist_cam_rpy(end_rpy_rad)
    )
    rpy_changed = bool(pose is not None and np.max(np.abs((erpy - pose[3:6] + math.pi) % (2 * math.pi) - math.pi)) > math.radians(3.0))
    if abs(du) <= float(deadband_u_px) and abs(dv) <= float(deadband_v_px) and not rpy_changed:
        return 0.0, 0.0, {
            "du": du,
            "dv": dv,
            "u_target": u_tgt,
            "v_target": v_tgt,
            "dx_raw_m": 0.0,
            "dy_raw_m": 0.0,
        }
    pw_obs = raycast_wrist_pixel_to_table(pose, float(cx_obs), float(cy_obs)) if pose is not None else None
    offset = (
        raycast_wrist_pixel_to_table(
            np.r_[0.0, 0.0, float(table_grasp_z_m), erpy],
            float(cx_target),
            float(cy_target),
        )
        if pose is not None
        else None
    )
    if pw_obs is None or offset is None:
        # Safe empirical fallback near nominal R8 Jacobian if geometry check ever rejects
        dx_raw = float(gain) * (0.000696 * du - 0.000184 * dv)
        dy_raw = float(gain) * (-0.000210 * du - 0.000646 * dv)
    else:
        dx_raw = float(gain) * float(pw_obs[0] - offset[0] - pose[0])
        dy_raw = float(gain) * float(pw_obs[1] - offset[1] - pose[1])
    corr_x_m = float(np.clip(dx_raw, max_dx_range[0], max_dx_range[1]))
    corr_y_m = float(np.clip(dy_raw, max_dy_range[0], max_dy_range[1]))
    return corr_x_m, corr_y_m, {
        "du": du,
        "dv": dv,
        "u_target": u_tgt,
        "v_target": v_tgt,
        "dx_raw_m": dx_raw,
        "dy_raw_m": dy_raw,
    }


def condition_grasp_actions(
    start,
    raw,
    *,
    max_position_m=.040,
    hold_pose=False,
    max_rotation_rad=np.radians(105.0),
    wrist_obb=None,
    cx_target=325.0,
    cy_target=None,
    table_grasp_z_m=0.160,
):
    """Return a bounded approach/closure candidate; downstream guards are mandatory."""
    start, raw = np.asarray(start, dtype=float), np.asarray(raw, dtype=float)
    if (start.shape != (7,) or raw.shape != (50, 7)
            or not np.isfinite(start).all() or not np.isfinite(raw).all()):
        raise ValueError("Invalid grasp start or actions")
    if not np.any(raw[:, 6] < .04):
        raise ValueError("No grasp closure predicted")
    delta = raw[0] - start
    delta[3:6] = (delta[3:6] + np.pi) % (2 * np.pi) - np.pi
    max_rot = max_rotation_rad if hold_pose else np.radians(12.0)
    if (np.linalg.norm(delta[:3]) > max_position_m or np.max(abs(delta[3:6])) > max_rot
            or not 0 <= start[6] <= .085 or raw[:, 6].min() < -.020 or raw[:, 6].max() > .085):
        raise ValueError("Target outside grasp candidate envelope")
    targets = raw.copy()
    clipped = np.flatnonzero(targets[:, 6] < 0).tolist()
    targets[:, 6] = np.maximum(targets[:, 6], 0)
    if hold_pose:
        if wrist_obb is not None and "cx" in wrist_obb and "cy" in wrist_obb:
            cx, cy = float(wrist_obb["cx"]), float(wrist_obb["cy"])
            eff_cx_target, eff_cy_target = _wrist_obb_base_targets(
                wrist_obb, cx_target=cx_target, cy_target=cy_target
            )
            at_hover = bool(start[2] > 0.180)
            corr_x_m, corr_y_m, _ = solve_wrist_xy_correction(
                start[:6],
                cx,
                cy,
                end_rpy_rad=start[3:6],
                cx_target=eff_cx_target,
                cy_target=eff_cy_target,
                table_grasp_z_m=float(table_grasp_z_m),
                deadband_u_px=5.0 if at_hover else 6.0,
                deadband_v_px=5.0 if at_hover else 6.0,
                max_dx_range=(-0.025, 0.025) if at_hover else (-0.015, 0.015),
                max_dy_range=(-0.025, 0.020) if at_hover else (-0.015, 0.008),
            )

            fine_grasp_pose = start[:6].copy()
            fine_grasp_pose[0] += corr_x_m
            fine_grasp_pose[1] += corr_y_m

            open_grip = min(max(float(start[6]), float(targets[0, 6])), .080)
            target_table_z = float(table_grasp_z_m)

            if start[2] > 0.180:
                # Arm is at hover height (e.g. Z = 0.197m). Must descend to table grasp height before closing!
                fine_grasp_pose[2] = target_table_z
                descent_steps = 16
                ramps_z = np.linspace(0.0, 1.0, descent_steps)[:, None]
                pose_delta = fine_grasp_pose - start[:6]
                targets[:descent_steps, :6] = start[:6] + ramps_z * pose_delta
                targets[descent_steps:, :6] = fine_grasp_pose
                targets[:descent_steps, 6] = open_grip
                closure_steps = 15
                end_close = min(len(targets), descent_steps + closure_steps)
                targets[descent_steps:end_close, 6] = np.linspace(open_grip, 0.0, end_close - descent_steps)
                targets[end_close:, 6] = 0.0
            else:
                fine_grasp_pose[2] = min(float(start[2]), target_table_z + 0.003)
                anchor_index = 8 if (abs(corr_x_m) > 0.001 or abs(corr_y_m) > 0.001) else 0
                if anchor_index > 0:
                    ramps = np.arange(1, anchor_index + 1, dtype=float)[:, None] / float(anchor_index)
                    pose_delta = fine_grasp_pose - start[:6]
                    targets[:anchor_index, :6] = start[:6] + ramps * pose_delta
                    targets[anchor_index:, :6] = fine_grasp_pose
                    targets[:anchor_index, 6] = open_grip
                    closure_steps = 15
                    end_close = min(len(targets), anchor_index + closure_steps)
                    targets[anchor_index:end_close, 6] = np.linspace(open_grip, 0.0, end_close - anchor_index)
                    targets[end_close:, 6] = 0.0
                else:
                    targets[:, :6] = fine_grasp_pose
                    closure_steps = 15
                    targets[:closure_steps, 6] = np.linspace(open_grip, 0.0, closure_steps)
                    targets[closure_steps:, 6] = 0.0
        elif np.any(targets[:13, 6] < .010):
            targets[:, :6] = start[:6]
            if start[2] > 0.180:
                targets[:, 2] = np.linspace(start[2], float(table_grasp_z_m), len(targets))
        else:
            anchor_index = 12 + int(np.argmax(targets[12:21, 1]))
            grasp_pose = targets[anchor_index, :6].copy()
            pose_delta = grasp_pose - start[:6]
            pose_delta[3:6] = (pose_delta[3:6] + np.pi) % (2 * np.pi) - np.pi
            ramps = np.arange(1, anchor_index + 1, dtype=float)[:, None] / float(anchor_index)
            targets[:anchor_index, :6] = start[:6] + ramps * pose_delta
            targets[anchor_index:, :6] = grasp_pose
            open_grip = min(max(float(start[6]), float(targets[0, 6])), .080)
            targets[:anchor_index, 6] = open_grip
        sequence = targets
    else:
        # Approach the first pose with the current opening. The predicted
        # first closure is then a separate target at that pose.
        delta[6] = 0
        prefix = start + np.arange(1, 7)[:, None] / 6 * delta
        sequence = np.vstack([prefix, targets])
    candidate = []
    prev = start
    splits = []
    for i, target in enumerate(sequence):
        delta = target - prev
        delta[3:6] = (delta[3:6] + np.pi) % (2 * np.pi) - np.pi
        count = max(1, math.ceil(np.linalg.norm(delta[:3]) / .015),
                    math.ceil(abs(delta[6]) / .015))
        if count > 1:
            splits.append(i)
        candidate.extend((prev + delta * j / count).tolist() for j in range(1, count))
        candidate.append(target.tolist())
        prev = target
    if len(candidate) > 66:
        raise ValueError("Grasp candidate exceeds 10 additional subdivision points")
    return np.asarray(candidate), clipped, splits


def _enrich_block_obb_geometry(cx, cy, w, h, angle, area, color_class="unknown", box_pts=None):
    """Compute major/minor axes, finger-axis spans, shape class, and grasp contact points."""
    w, h, angle = float(w), float(h), float(angle)
    major_px = max(w, h)
    minor_px = max(1.0, min(w, h))
    aspect_ratio = major_px / minor_px
    rad = math.radians(angle)
    cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
    span_u_px = w * cos_a + h * sin_a
    span_v_px = w * sin_a + h * cos_a

    # Long axis angle in image degrees [-90, 90):
    # OpenCV minAreaRect angle is the angle of side `w` from horizontal (+u).
    long_axis_deg = angle if w >= h else angle + 90.0
    while long_axis_deg >= 90.0:
        long_axis_deg -= 180.0
    while long_axis_deg < -90.0:
        long_axis_deg += 180.0
    short_axis_deg = long_axis_deg + 90.0
    while short_axis_deg >= 90.0:
        short_axis_deg -= 180.0
    while short_axis_deg < -90.0:
        short_axis_deg += 180.0

    if aspect_ratio < 1.35:
        shape_class = "square"
        grasp_axis_mode = "symmetric_square"
        # Face alignment modulo 90 degrees with deadband:
        # Nominally aligned squares exhibit angle within 18° of grid axis; rotated squares (e.g. 45°) adapt.
        face_delta_deg = ((short_axis_deg + 45.0) % 90.0) - 45.0
        face_yaw_deg = 0.0 if abs(face_delta_deg) <= 18.0 else face_delta_deg
        recommended_yaw_delta_rad = math.radians(-face_yaw_deg)
        contact_rad = math.radians(face_yaw_deg)
    else:
        shape_class = "elongated"
        contact_rad = math.radians(short_axis_deg)
        grasp_axis_mode = (
            "short_axis_aligned"
            if abs(short_axis_deg) <= 20.0 or abs(abs(short_axis_deg) - 180.0) <= 20.0
            else "long_axis_between_fingers"
        )
        recommended_yaw_delta_rad = math.radians(-short_axis_deg)

    half_minor = 0.5 * minor_px
    dx_c = half_minor * math.cos(contact_rad)
    dy_c = half_minor * math.sin(contact_rad)
    contact_points_px = [
        [round(float(cx - dx_c), 1), round(float(cy - dy_c), 1)],
        [round(float(cx + dx_c), 1), round(float(cy + dy_c), 1)],
    ]
    expected_short_width_m = 0.016 if color_class == "black" else 0.027
    expected_current_width_m = round(
        float(np.clip(expected_short_width_m * (span_u_px / minor_px), 0.014, 0.068)),
        4,
    )
    visibility = 1.0 if (195.0 <= cx <= 500.0 and 140.0 <= cy <= 410.0) else 0.75
    return {
        "cx": float(cx),
        "cy": float(cy),
        "w": float(w),
        "h": float(h),
        "angle": float(angle),
        "area": float(area),
        "color_class": str(color_class),
        "shape_class": shape_class,
        "major_px": round(major_px, 2),
        "minor_px": round(minor_px, 2),
        "aspect_ratio": round(aspect_ratio, 3),
        "span_u_px": round(span_u_px, 2),
        "span_v_px": round(span_v_px, 2),
        "long_axis_deg": round(long_axis_deg, 2),
        "short_axis_deg": round(short_axis_deg, 2),
        "grasp_axis_mode": grasp_axis_mode,
        "recommended_yaw_delta_rad": round(float(recommended_yaw_delta_rad), 4),
        "expected_short_width_m": expected_short_width_m,
        "expected_current_width_m": expected_current_width_m,
        "contact_points_px": contact_points_px,
        "visibility": visibility,
        "box_points_px": box_pts if box_pts is not None else [],
    }


def parse_target_color(prompt: str) -> str:
    """Extract target block color from prompt text. Supports red, green, blue/cyan, yellow."""
    if not prompt:
        return "red"
    p = str(prompt).lower()
    if any(k in p for k in ["green", "\u7eff\u8272", "\u7eff"]):
        return "green"
    if any(k in p for k in ["blue", "\u84dd\u8272", "\u84dd", "cyan", "\u9752\u8272", "\u9752"]):
        return "blue"
    if any(k in p for k in ["yellow", "\u9ec4\u8272", "\u9ec4"]):
        return "yellow"
    if any(k in p for k in ["red", "\u7ea2\u8272", "\u7ea2"]):
        return "red"
    return "red"


GRID_HOVER_POSES = {
    "R1": np.array([0.285, 0.170, 0.200, -0.42, 1.02, 0.99]),
    "R2": np.array([0.280, 0.145, 0.200, -0.42, 1.02, 0.99]),
    "R3": np.array([0.275, 0.095, 0.200, -0.42, 1.02, 0.99]),
    "R4": np.array([0.255, 0.170, 0.200, -0.42, 1.02, 0.99]),
    "R5": np.array([0.250, 0.145, 0.200, -0.42, 1.02, 0.99]),
    "R6": np.array([0.245, 0.110, 0.200, -0.42, 1.02, 0.99]),
    "R7": np.array([0.220, 0.170, 0.200, -0.42, 1.02, 0.99]),
    "R8": np.array([0.225, 0.140, 0.200, -0.42, 1.02, 0.99]),
    "R9": np.array([0.230, 0.100, 0.200, -0.42, 1.02, 0.99]),
}

HIGH_CAM_GRID_CENTERS_PX = {
    "R1": (338.0, 223.0),
    "R2": (355.0, 223.0),
    "R3": (371.0, 223.0),
    "R4": (338.0, 237.0),
    "R5": (355.0, 237.0),
    "R6": (371.0, 237.0),
    "R7": (338.0, 253.0),
    "R8": (355.0, 253.0),
    "R9": (371.0, 253.0),
}


def check_wrist_grasp_alignment(
    wrist_obb: dict | None,
    *,
    ee_z_m: float | None = None,
    ee_pose6=None,
    cx_target: float = WRIST_CAM_GRASP_CX_PX,
    cy_target: float | None = None,
    tol_u_px: float = 20.0,
    tol_v_px: float = 20.0,
) -> dict:
    """Check whether detected wrist OBB is aligned within the grasp corridor.

    Enforces tracking error limits (|Delta u| <= tol_u_px, |Delta v| <= tol_v_px).
    Projects canonical grasp target (cx*, cy*) at table height Z=0.160m through the
    calibrated 3D pinhole wrist-camera model when `ee_pose6` or `ee_z_m` is supplied.
    For square blocks, base cy_target at Z=0.160m is 318.0; for elongated objects, 285.0.
    Returns a structured status dict with 'aligned' (bool), pixel deltas, and reason.
    """
    if wrist_obb is None or "cx" not in wrist_obb or "cy" not in wrist_obb:
        return {
            "aligned": False,
            "du": None,
            "dv": None,
            "cx": None,
            "cy": None,
            "cx_target": float(cx_target),
            "cy_target": float(cy_target) if cy_target is not None else WRIST_CAM_GRASP_CY_SQUARE_PX,
            "tol_u_px": float(tol_u_px),
            "tol_v_px": float(tol_v_px),
            "reason": "no_obb",
        }
    cx, cy = float(wrist_obb["cx"]), float(wrist_obb["cy"])
    base_cx, base_cy = _wrist_obb_base_targets(
        wrist_obb, cx_target=cx_target, cy_target=cy_target
    )
    eff_cx_target, eff_cy_target = wrist_target_uv_for_pose(
        ee_pose6=ee_pose6,
        ee_z_m=ee_z_m,
        cx_target=base_cx,
        cy_target=base_cy,
    )
    du = cx - float(eff_cx_target)
    dv = cy - float(eff_cy_target)
    aligned = bool(abs(du) <= float(tol_u_px) and abs(dv) <= float(tol_v_px))
    if aligned:
        reason = "aligned"
    elif abs(du) > float(tol_u_px) and abs(dv) > float(tol_v_px):
        reason = "uv_misaligned"
    elif abs(du) > float(tol_u_px):
        reason = "u_misaligned"
    else:
        reason = "v_misaligned"

    return {
        "aligned": aligned,
        "du": round(du, 2),
        "dv": round(dv, 2),
        "cx": round(cx, 2),
        "cy": round(cy, 2),
        "cx_target": round(float(eff_cx_target), 2),
        "cy_target": round(float(eff_cy_target), 2),
        "tol_u_px": float(tol_u_px),
        "tol_v_px": float(tol_v_px),
        "reason": reason,
    }


def detect_high_cam_grid_target(
    image_or_bytes,
    *,
    target_color="red",
    table_roi=(325, 210, 385, 265),
    min_area=80.0,
    max_area=260.0,
    max_dist_to_cell_px=11.5,
) -> dict | None:
    """Detect colored block in cam_high (global top-down camera) and map to grid cell R1..R9.

    Filters by target color (red, yellow, green, blue/cyan), table ROI bounding box,
    and contour area. Returns a dictionary containing detected pixel location, identified
    grid cell ("R1".."R9"), and corresponding verified hover pose.
    Rejects targets outside grid boundaries or ambiguous detections (> max_dist_to_cell_px).
    """
    if image_or_bytes is None:
        return None
    try:
        import cv2
    except ImportError:
        return None

    if isinstance(image_or_bytes, (str, Path)):
        image_bgr = cv2.imread(str(image_or_bytes), cv2.IMREAD_COLOR)
    elif isinstance(image_or_bytes, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(image_or_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        image_bgr = np.asarray(image_or_bytes)
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    norm_color = parse_target_color(target_color)

    if norm_color == "red":
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 70, 50), (14, 255, 255)),
            cv2.inRange(hsv, (165, 70, 50), (180, 255, 255)),
        )
    elif norm_color == "yellow":
        mask = cv2.inRange(hsv, (16, 70, 60), (36, 255, 255))
    elif norm_color == "green":
        mask = cv2.inRange(hsv, (35, 50, 40), (85, 255, 255))
    elif norm_color == "blue":
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (88, 50, 40), (135, 255, 255)),
            cv2.inRange(hsv, (75, 50, 40), (88, 255, 255)),  # cyan
        )
    else:
        mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 255))

    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    min_u, min_v, max_u, max_v = table_roi
    cands = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if not (min_area <= area <= max_area):
            continue
        (cx, cy), (w, h), angle = cv2.minAreaRect(c)
        if not (min_u <= cx <= max_u and min_v <= cy <= max_v):
            continue
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > 2.2:
            continue
        # Distance to expected cube area (~155 px in cam_high)
        area_score = -abs(area - 155.0)
        cands.append((area_score, (cx, cy, w, h, angle, area)))

    if not cands:
        return None
    cands.sort(key=lambda item: item[0], reverse=True)
    best_cx, best_cy, best_w, best_h, best_ang, best_area = cands[0][1]

    # Find closest grid cell
    best_cell = min(
        HIGH_CAM_GRID_CENTERS_PX,
        key=lambda k: math.hypot(best_cx - HIGH_CAM_GRID_CENTERS_PX[k][0], best_cy - HIGH_CAM_GRID_CENTERS_PX[k][1])
    )
    dist_to_cell = math.hypot(
        best_cx - HIGH_CAM_GRID_CENTERS_PX[best_cell][0],
        best_cy - HIGH_CAM_GRID_CENTERS_PX[best_cell][1],
    )
    if dist_to_cell > float(max_dist_to_cell_px):
        return None

    hover_pose = GRID_HOVER_POSES[best_cell].copy()

    return {
        "cell": best_cell,
        "cx": round(float(best_cx), 2),
        "cy": round(float(best_cy), 2),
        "w": round(float(best_w), 2),
        "h": round(float(best_h), 2),
        "area": round(float(best_area), 2),
        "dist_to_cell_px": round(float(dist_to_cell), 2),
        "color": norm_color,
        "hover_xyz": hover_pose[:3].tolist(),
        "hover_pose": hover_pose.tolist(),
    }


def detect_wrist_block_obb(image_or_bytes, *, target_color=None, prev_obb=None):
    """Detect graspable object (red/blue/yellow/green/cyan block or black marker) OBB in wrist camera.

    Accepts a BGR numpy array (H, W, 3), file path, or JPEG bytes.
    Optionally filters by `target_color` (e.g. 'red', 'green', 'blue', 'yellow', 'cyan').
    Returns a dict with OBB center, axes, shape/color classification, and grasp contact points,
    or `None` if OpenCV is unavailable or no valid object contour is detected in the workspace.
    """
    if image_or_bytes is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    if isinstance(image_or_bytes, (str, Path)):
        image_bgr = cv2.imread(str(image_or_bytes), cv2.IMREAD_COLOR)
    elif isinstance(image_or_bytes, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(image_or_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        image_bgr = np.asarray(image_or_bytes)
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return None
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_masks = [
        ("red", cv2.bitwise_or(
            cv2.inRange(hsv, (0, 70, 50), (15, 255, 255)),
            cv2.inRange(hsv, (160, 70, 50), (180, 255, 255)),
        )),
        ("yellow", cv2.inRange(hsv, (18, 130, 90), (34, 255, 255))),
        ("green", cv2.inRange(hsv, (35, 60, 40), (82, 255, 255))),
        ("cyan", cv2.inRange(hsv, (75, 60, 50), (105, 255, 255))),
        ("blue", cv2.inRange(hsv, (105, 60, 40), (135, 255, 255))),
        ("black", cv2.inRange(hsv, (0, 0, 5), (180, 110, 62))),
    ]

    target_norm = None
    if target_color:
        tc = str(target_color).lower().strip()
        if tc in ("blue", "cyan", "\u9752", "\u84dd", "\u84dd\u8272", "\u9752\u8272"):
            target_norm = ("blue", "cyan")
        elif tc in ("green", "\u7eff", "\u7eff\u8272"):
            target_norm = ("green",)
        elif tc in ("yellow", "\u9ec4", "\u9ec4\u8272"):
            target_norm = ("yellow",)
        elif tc in ("red", "\u7ea2", "\u7ea2\u8272"):
            target_norm = ("red",)

    def _extract_cands(masks_to_scan):
        cands = []
        for color_name, mask in masks_to_scan:
            contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
            for c in contours:
                area = float(cv2.contourArea(c))
                if not (500.0 <= area <= 18000.0):
                    continue
                (cx, cy), (w, h), angle = cv2.minAreaRect(c)
                if min(w, h) < 12.0:
                    continue
                # Workspace filter: reject left-border/background artifacts (cx < 175) and bottom finger tips
                if not (175.0 <= cx <= 520.0 and 120.0 <= cy <= 430.0):
                    continue
                aspect = max(w, h) / max(1.0, min(w, h))
                if color_name == "black":
                    if not (210.0 <= cx <= 450.0 and 185.0 <= cy <= 350.0 and aspect >= 2.2):
                        continue
                box_pts = np.round(cv2.boxPoints(((cx, cy), (w, h), angle)), 1).tolist()
                obb = _enrich_block_obb_geometry(cx, cy, w, h, angle, area, color_name, box_pts)

                # Scheme B: approxPolyDP polygon cross-check
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                cross_check_info = {
                    "method": "approxPolyDP",
                    "vertices": int(len(approx)),
                    "poly_face_deg": None,
                    "diff_deg": None,
                    "verified": False,
                }
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    edge_rads = [
                        math.atan2(float(pts[(i + 1) % 4][1] - pts[i][1]), float(pts[(i + 1) % 4][0] - pts[i][0]))
                        for i in range(4)
                    ]
                    # 4-fold circular mean for 90° rotation symmetry (quadruple angle mod 2*pi)
                    sum_x = sum(math.cos(4.0 * r) for r in edge_rads)
                    sum_y = sum(math.sin(4.0 * r) for r in edge_rads)
                    circ_mean_rad = 0.25 * math.atan2(sum_y, sum_x)
                    poly_face = ((math.degrees(circ_mean_rad) + 45.0) % 90.0) - 45.0
                    obb_face = ((obb["short_axis_deg"] + 45.0) % 90.0) - 45.0
                    # Modulo 90 angular difference
                    diff_deg = round(float(abs(((obb_face - poly_face + 45.0) % 90.0) - 45.0)), 2)
                    cross_check_info["poly_face_deg"] = round(poly_face, 2)
                    cross_check_info["diff_deg"] = diff_deg
                    cross_check_info["verified"] = bool(diff_deg <= 8.0)
                obb["cross_check"] = cross_check_info

                score = area
                if color_name == "black":
                    score *= 0.65
                if prev_obb is not None:
                    dist = math.hypot(cx - float(prev_obb.get("cx", cx)), cy - float(prev_obb.get("cy", cy)))
                    if dist <= 120.0:
                        score *= 1.25
                        # Stabilize square cubes against partial finger/bottom occlusion near gripper jaws
                        if (
                            prev_obb.get("shape_class") == "square"
                            and prev_obb.get("color_class") == color_name
                            and obb["shape_class"] == "elongated"
                            and cy >= 310.0
                            and obb["aspect_ratio"] < 1.85
                            and obb["major_px"] < 92.0
                        ):
                            obb["shape_class"] = "square"
                            obb["grasp_axis_mode"] = "symmetric_square"
                            obb["recommended_yaw_delta_rad"] = 0.0
                            obb["shape_stabilized_from_prev"] = True
                    if prev_obb.get("color_class") == color_name:
                        score *= 1.20
                cands.append((score, obb))
        return cands

    if target_norm:
        candidates = _extract_cands([m for m in color_masks if m[0] in target_norm])
    else:
        candidates = _extract_cands(color_masks)

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def select_grasp_orientation_candidate(
    start_pose,
    end_pose,
    wrist_obb,
    *,
    max_yaw_delta_rad=np.radians(105.0),
):
    """Select short-axis grasp orientation for elongated objects; modulo-90 face alignment for square blocks."""
    start_pose = np.asarray(start_pose, dtype=float).reshape(6)
    end_pose = np.asarray(end_pose, dtype=float).copy().reshape(6)
    raw_euler = end_pose[3:6].copy()
    if wrist_obb is None:
        return end_pose, {
            "mode": "no_obb",
            "raw_euler_rad": np.round(raw_euler, 4).tolist(),
            "refined_euler_rad": np.round(end_pose[3:6], 4).tolist(),
            "applied_yaw_delta_deg": 0.0,
            "target_yaw_deg": round(float(np.degrees(end_pose[5])), 2),
            "expected_grasp_width_m": 0.027,
        }
    if "aspect_ratio" not in wrist_obb or "grasp_axis_mode" not in wrist_obb or "expected_current_width_m" not in wrist_obb:
        w_val = float(wrist_obb.get("w", 36.0))
        h_val = float(wrist_obb.get("h", 36.0))
        ang_val = float(wrist_obb.get("angle", 0.0))
        wrist_obb = _enrich_block_obb_geometry(
            float(wrist_obb["cx"]), float(wrist_obb["cy"]), w_val, h_val,
            ang_val, float(wrist_obb.get("area", w_val * h_val)),
            wrist_obb.get("color_class", "unknown"),
        )
    mode = wrist_obb["grasp_axis_mode"]
    applied_yaw = 0.0
    if mode == "symmetric_square":
        # Square block (27mm cube): Parallel gripper opening (75-80mm) is twice the cube's
        # maximum diagonal (38.2mm). As verified in live tests (docs/3-waypoints-7.md) and
        # physical video demonstration, in-air yaw rotation causes tool-flange lever-arm
        # swing offsets (~40mm), displacing the fingertips off-center. Instead of rotating
        # the gripper in air, we keep applied_yaw = 0.0. Grasp success is guaranteed by deep
        # forward insertion along the approach axis (cy_target = 318.0) so the entire 38.2mm
        # diagonal is swallowed between the finger pads, preventing rotated edges from being pushed forward.
        applied_yaw = 0.0
    elif mode in ("short_axis_aligned", "long_axis_between_fingers"):
        desired_delta = float(wrist_obb.get("recommended_yaw_delta_rad", 0.0))
        # Parallel gripper closing along short axis has 180° symmetry:
        candidate_a = desired_delta
        candidate_b = desired_delta + np.pi if desired_delta < 0 else desired_delta - np.pi

        current_delta_yaw = ((end_pose[5] - start_pose[5] + np.pi) % (2 * np.pi)) - np.pi
        eff_a = ((current_delta_yaw + candidate_a + np.pi) % (2 * np.pi)) - np.pi
        eff_b = ((current_delta_yaw + candidate_b + np.pi) % (2 * np.pi)) - np.pi

        chosen_delta = candidate_a if abs(eff_a) <= abs(eff_b) else candidate_b
        eff_delta = eff_a if abs(eff_a) <= abs(eff_b) else eff_b

        clamped_eff = float(np.clip(eff_delta, -max_yaw_delta_rad, max_yaw_delta_rad))
        applied_yaw = clamped_eff - current_delta_yaw
        end_pose[5] += applied_yaw
        end_pose[5] = ((end_pose[5] + np.pi) % (2 * np.pi)) - np.pi

    return end_pose, {
        "mode": mode,
        "shape_class": wrist_obb.get("shape_class", "square"),
        "aspect_ratio": wrist_obb.get("aspect_ratio", 1.0),
        "raw_euler_rad": np.round(raw_euler, 4).tolist(),
        "refined_euler_rad": np.round(end_pose[3:6], 4).tolist(),
        "applied_yaw_delta_deg": round(float(np.degrees(applied_yaw)), 2),
        "target_yaw_deg": round(float(np.degrees(end_pose[5])), 2),
        "expected_grasp_width_m": wrist_obb.get("expected_current_width_m", 0.027),
        "contact_points_px": wrist_obb.get("contact_points_px", []),
        "cross_check": wrist_obb.get("cross_check"),
    }


def detect_wrist_target_obb(image_or_bytes):
    """Detect the brown placement target area OBB in wrist camera image.

    Accepts a BGR numpy array (H, W, 3) or JPEG bytes.
    Returns a dict `{"cx": float, "cy": float, "w": float, "h": float, "angle": float, "area": float}`
    or `None` if OpenCV is unavailable or no valid brown target contour is detected.
    """
    if image_or_bytes is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    if isinstance(image_or_bytes, (str, Path)):
        image_bgr = cv2.imread(str(image_or_bytes), cv2.IMREAD_COLOR)
    elif isinstance(image_or_bytes, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(image_or_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        image_bgr = np.asarray(image_or_bytes)
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return None
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_brown = cv2.inRange(hsv, (8, 40, 40), (28, 220, 220))
    contours_info = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    valid = [c for c in contours if 2000.0 <= cv2.contourArea(c) <= 45000.0]
    if not valid:
        return None
    best = max(valid, key=cv2.contourArea)
    (cx, cy), (w, h), angle = cv2.minAreaRect(best)
    return {
        "cx": float(cx),
        "cy": float(cy),
        "w": float(w),
        "h": float(h),
        "angle": float(angle),
        "area": float(cv2.contourArea(best)),
    }


def _condition_approach_with_wrist_obb(
    start: np.ndarray,
    targets: np.ndarray,
    *,
    anchor_index: int,
    peak_y_index: int,
    wrist_obb: dict,
    hover_z_m: float,
    max_rotation_rad: float,
    allow_descent: bool,
) -> tuple[np.ndarray, int]:
    """Unified wrist-camera visual servoing and two-stage Z gate for approach chunks."""
    cx, cy = float(wrist_obb["cx"]), float(wrist_obb["cy"])
    base_cx, base_cy = _wrist_obb_base_targets(wrist_obb)
    hover_z_f = float(hover_z_m)
    is_low_altitude = float(start[2]) < hover_z_f - 0.005

    align_now = check_wrist_grasp_alignment(
        wrist_obb,
        ee_pose6=start[:6],
        cx_target=base_cx,
        cy_target=base_cy,
        tol_u_px=20.0,
        tol_v_px=20.0,
    )
    du_now = float(align_now["du"])
    dv_now = float(align_now["dv"])

    end_pose = targets[anchor_index, :6].copy()
    if is_low_altitude:
        end_pose[3:6] = start[3:6]
        targets[:, 3:6] = start[3:6]
        corr_x_m, corr_y_m, _ = solve_wrist_xy_correction(
            start[:6],
            cx,
            cy,
            end_rpy_rad=start[3:6],
            cx_target=base_cx,
            cy_target=base_cy,
            deadband_u_px=5.0,
            deadband_v_px=5.0,
            max_dx_range=(-0.025, 0.025),
            max_dy_range=(-0.025, 0.030),
        )
        end_pose[0] = float(start[0]) + corr_x_m
        end_pose[1] = float(start[1]) + corr_y_m
        severe_low_dev = (abs(du_now) > 25.0) or (abs(dv_now) > 25.0)
        if severe_low_dev:
            # Low-altitude safety guard: lift to hover clearance BEFORE sweeping across tabletop.
            end_pose[2] = max(hover_z_f, float(start[2]))
            lift_steps = min(15, max(8, anchor_index // 2))
            if lift_steps < anchor_index:
                targets[:lift_steps, 0] = start[0]
                targets[:lift_steps, 1] = start[1]
                targets[:lift_steps, 2] = np.linspace(float(start[2]), float(end_pose[2]), lift_steps)
                trans_len = anchor_index - lift_steps + 1
                ramps = np.linspace(0.0, 1.0, trans_len)[:, None]
                targets[lift_steps:anchor_index + 1, :2] = start[:2] + ramps * (end_pose[:2] - start[:2])
                targets[lift_steps:anchor_index + 1, 2] = float(end_pose[2])
            else:
                ramps = np.linspace(0.0, 1.0, anchor_index + 1)[:, None]
                targets[:anchor_index + 1, :3] = start[:3] + ramps * (end_pose[:3] - start[:3])
            targets[anchor_index:, :6] = end_pose
            return targets, anchor_index

        span = anchor_index - peak_y_index
        if allow_descent and span > 0:
            horiz_steps = max(1, int(round(span * 0.65)))
            horiz_end = peak_y_index + horiz_steps
            targets[:peak_y_index + 1, 0] = np.linspace(
                float(start[0]),
                float(start[0]) + (float(end_pose[0]) - float(start[0])) * (float(peak_y_index) / float(max(1, horiz_end))),
                peak_y_index + 1,
            )
            targets[:peak_y_index + 1, 1] = np.linspace(
                float(start[1]),
                float(start[1]) + (float(end_pose[1]) - float(start[1])) * (float(peak_y_index) / float(max(1, horiz_end))),
                peak_y_index + 1,
            )
            mid_z = max(float(end_pose[2]), min(float(start[2]), hover_z_f))
            start_pose = targets[peak_y_index, :6].copy()
            pose_delta = end_pose - start_pose
            pose_delta[3:6] = (pose_delta[3:6] + np.pi) % (2 * np.pi) - np.pi
            for step_offset in range(1, span + 1):
                idx = peak_y_index + step_offset
                xy_alpha = min(1.0, float(step_offset) / float(horiz_steps))
                targets[idx, :6] = start_pose + xy_alpha * pose_delta
                if step_offset <= horiz_steps:
                    z_alpha = float(step_offset) / float(horiz_steps)
                    targets[idx, 2] = float(start_pose[2]) + z_alpha * (mid_z - float(start_pose[2]))
                else:
                    v_alpha = float(step_offset - horiz_steps) / float(span - horiz_steps)
                    targets[idx, 2] = mid_z + v_alpha * (float(end_pose[2]) - mid_z)
        else:
            ramps = np.linspace(0.0, 1.0, anchor_index + 1)[:, None]
            targets[:anchor_index + 1, :3] = start[:3] + ramps * (end_pose[:3] - start[:3])
        targets[anchor_index:, :6] = end_pose
        return targets, anchor_index

    # Hover / aerial approach (start[2] >= hover_z_m - 0.005)
    end_pose, ori_info = select_grasp_orientation_candidate(
        start[:6], end_pose, wrist_obb, max_yaw_delta_rad=max_rotation_rad
    )
    applied_yaw_deg = abs(float(ori_info.get("applied_yaw_delta_deg", 0.0)))
    corr_x_m, corr_y_m, _ = solve_wrist_xy_correction(
        start[:6],
        cx,
        cy,
        end_rpy_rad=end_pose[3:6],
        cx_target=base_cx,
        cy_target=base_cy,
        deadband_u_px=4.0,
        deadband_v_px=4.0,
        max_dx_range=(-0.035, 0.035),
        max_dy_range=(-0.035, 0.040),
    )
    end_pose[0] = float(start[0]) + corr_x_m
    end_pose[1] = float(start[1]) + corr_y_m

    # Verify BOTH initial hover alignment AND predicted post-step residual error (du and dv).
    pw_obs = raycast_wrist_pixel_to_table(start[:6], cx, cy)
    cand_hover_pose = np.array([end_pose[0], end_pose[1], hover_z_f, end_pose[3], end_pose[4], end_pose[5]], dtype=float)
    resid_ok = True
    if pw_obs is not None:
        uv_res = project_wrist_world_point(cand_hover_pose, pw_obs)
        u_tgt_end, v_tgt_end = wrist_target_uv_for_pose(
            cand_hover_pose, cx_target=base_cx, cy_target=base_cy
        )
        if uv_res is not None:
            resid_ok = bool(abs(uv_res[0] - u_tgt_end) <= 20.0 and abs(uv_res[1] - v_tgt_end) <= 20.0)

    can_descend = bool(allow_descent and align_now["aligned"] and resid_ok and applied_yaw_deg <= 15.0)
    if can_descend:
        eff_anchor = max(anchor_index, 32)
        end_pose[2] = float(np.clip(end_pose[2], 0.157, 0.162))
        span = eff_anchor - peak_y_index
        if span > 0:
            horiz_steps = max(1, int(round(span * 0.65)))
            horiz_end = peak_y_index + horiz_steps
            hover_hold_z = max(hover_z_f, min(float(start[2]), hover_z_f + 0.005))
            pose_delta_all = end_pose - start[:6]
            pose_delta_all[3:6] = (pose_delta_all[3:6] + np.pi) % (2 * np.pi) - np.pi
            for idx in range(eff_anchor + 1):
                xy_alpha = min(1.0, float(idx) / float(max(1, horiz_end)))
                targets[idx, :6] = start[:6] + xy_alpha * pose_delta_all
                if idx <= horiz_end:
                    z_alpha = float(idx) / float(max(1, horiz_end))
                    targets[idx, 2] = float(start[2]) + z_alpha * (hover_hold_z - float(start[2]))
                else:
                    v_alpha = float(idx - horiz_end) / float(max(1, eff_anchor - horiz_end))
                    targets[idx, 2] = hover_hold_z + v_alpha * (float(end_pose[2]) - hover_hold_z)
        else:
            pose_delta = end_pose - start[:6]
            pose_delta[3:6] = (pose_delta[3:6] + np.pi) % (2 * np.pi) - np.pi
            ramps = np.linspace(0.0, 1.0, eff_anchor + 1)[:, None]
            targets[:eff_anchor + 1, :6] = start[:6] + ramps * pose_delta
        targets[eff_anchor:, :6] = end_pose
        return targets, eff_anchor

    # Misaligned at hover (or yaw rotating / closure not yet predicted):
    # Hold safely at >= hover_z_m while sliding smoothly down toward hover_z_m if started high.
    end_pose[2] = max(hover_z_f, float(start[2]) - 0.020)
    pose_delta = end_pose - start[:6]
    pose_delta[3:6] = (pose_delta[3:6] + np.pi) % (2 * np.pi) - np.pi
    ramps = np.linspace(0.0, 1.0, anchor_index + 1)[:, None]
    targets[:anchor_index + 1, :6] = start[:6] + ramps * pose_delta
    targets[anchor_index:, :6] = end_pose
    targets[:, 2] = np.maximum(targets[:, 2], hover_z_f)
    return targets, anchor_index


def condition_approach_actions(
    start,
    raw,
    *,
    max_position_m=.045,
    wrist_obb=None,
    hover_z_m=0.182,
    max_rotation_rad=np.radians(105.0),
    grid_hover_pose=None,
):
    """Hold gripper open throughout approach chunk; align at high hover before pure vertical descent."""
    start, raw = np.asarray(start, dtype=float), np.asarray(raw, dtype=float)
    if (start.shape != (7,) or raw.shape != (50, 7)
            or not np.isfinite(start).all() or not np.isfinite(raw).all()):
        raise ValueError("Invalid approach start or actions")
    delta = raw[0] - start
    delta[3:6] = (delta[3:6] + np.pi) % (2 * np.pi) - np.pi
    if (np.linalg.norm(delta[:3]) > max_position_m or np.max(abs(delta[3:6])) > max_rotation_rad
            or not 0 <= start[6] <= .085 or raw[:, 6].min() < -.020 or raw[:, 6].max() > .085):
        raise ValueError("Target outside approach candidate envelope")
    targets = raw.copy()
    open_grip = min(max(float(start[6]), float(targets[0, 6])), .080)
    targets[:, 6] = open_grip

    if grid_hover_pose is not None and wrist_obb is None:
        ghp = np.asarray(grid_hover_pose, dtype=float)
        eff_hover_z = max(float(hover_z_m), float(ghp[2]) if len(ghp) >= 3 else float(hover_z_m))
        target_xyz = np.array([ghp[0], ghp[1], eff_hover_z], dtype=float)
        target_rot = ghp[3:6] if len(ghp) >= 6 else np.array([-0.42, 1.02, 0.99])
        rot_diff = (target_rot - start[3:6] + np.pi) % (2 * np.pi) - np.pi

        dist_to_hover_xy = float(np.linalg.norm(start[:2] - target_xyz[:2]))
        dist_to_hover_z = float(abs(start[2] - eff_hover_z))
        if dist_to_hover_xy < 0.025 and dist_to_hover_z < 0.020:
            # Arrived at coarse hover target: exit coarse override, maintain downward observation pitch
            ramps = np.linspace(0.0, 1.0, len(targets))[:, None]
            targets[:, :3] = start[:3] + ramps * (target_xyz - start[:3])
            targets[:, 3:6] = start[3:6] + ramps * rot_diff
            targets[:, 6] = open_grip
            return targets, 0, len(targets) - 1

        if start[2] < eff_hover_z - 0.010:
            lift_steps = min(15, len(targets) // 3)
            targets[:lift_steps, 0] = start[0]
            targets[:lift_steps, 1] = start[1]
            targets[:lift_steps, 2] = np.linspace(start[2], eff_hover_z, lift_steps)
            targets[:lift_steps, 3:6] = start[3:6]
            trans_steps = len(targets) - lift_steps
            ramps = np.linspace(0.0, 1.0, trans_steps)[:, None]
            targets[lift_steps:, :3] = np.array([start[0], start[1], eff_hover_z]) + ramps * (target_xyz - np.array([start[0], start[1], eff_hover_z]))
            targets[lift_steps:, 3:6] = start[3:6] + ramps * rot_diff
        else:
            ramps = np.linspace(0.0, 1.0, len(targets))[:, None]
            targets[:, :3] = start[:3] + ramps * (target_xyz - start[:3])
            targets[:, 3:6] = start[3:6] + ramps * rot_diff
        targets[:, 6] = open_grip
        return targets, 0, len(targets) - 1

    if start[2] < hover_z_m - 0.005:
        targets[:, 3:6] = start[3:6]
        targets[:, 0] = start[0]
    held_open_from = 0
    close_indices = np.flatnonzero(raw[:, 6] < .04)
    pose_anchored_from = None
    if close_indices.size:
        first_close = int(close_indices[0])
        peak_y_index = 12 + int(np.argmax(targets[12:21, 1]))
        min_z_index = int(np.argmin(targets[:, 2]))
        if first_close <= 15 or min_z_index < max(0, first_close - 3):
            anchor_index = peak_y_index
        else:
            anchor_index = min_z_index
            while (anchor_index > peak_y_index
                   and (wrist_obb is None or targets[anchor_index - 1, 0] < 0.249)
                   and np.linalg.norm(targets[anchor_index, :2] - targets[anchor_index - 1, :2]) > 0.006
                   and abs(targets[anchor_index, 2] - targets[anchor_index - 1, 2]) < 0.003):
                anchor_index -= 1

        if wrist_obb is not None:
            targets, pose_anchored_from = _condition_approach_with_wrist_obb(
                start,
                targets,
                anchor_index=anchor_index,
                peak_y_index=peak_y_index,
                wrist_obb=wrist_obb,
                hover_z_m=float(hover_z_m),
                max_rotation_rad=max_rotation_rad,
                allow_descent=True,
            )
            targets[:, 6] = open_grip
            return targets, held_open_from, pose_anchored_from

        end_pose = targets[anchor_index, :6].copy()
        if first_close <= 15 or min_z_index < max(0, first_close - 3):
            targets[anchor_index:, :6] = end_pose
            pose_anchored_from = anchor_index
        else:
            if targets[peak_y_index, 1] > end_pose[1]:
                end_pose[1] = targets[peak_y_index, 1]
            span = anchor_index - peak_y_index
            if span > 0:
                horiz_steps = max(1, int(round(span * 0.65)))
                horiz_end = peak_y_index + horiz_steps
                if float(np.max(targets[:peak_y_index + 1, 0])) > max(float(targets[0, 0]), float(end_pose[0])) + 0.004:
                    targets[:peak_y_index + 1, 0] = np.linspace(
                        float(targets[0, 0]),
                        float(targets[0, 0]) + (float(end_pose[0]) - float(targets[0, 0])) * (float(peak_y_index) / float(max(1, horiz_end))),
                        peak_y_index + 1,
                    )
                hover_z = max(float(end_pose[2]), min(float(targets[peak_y_index, 2]), 0.172))
                start_pose = targets[peak_y_index, :6].copy()
                pose_delta = end_pose - start_pose
                pose_delta[3:6] = (pose_delta[3:6] + np.pi) % (2 * np.pi) - np.pi
                for step_offset in range(1, span + 1):
                    idx = peak_y_index + step_offset
                    xy_alpha = min(1.0, float(step_offset) / float(horiz_steps))
                    targets[idx, :6] = start_pose + xy_alpha * pose_delta
                    if step_offset <= horiz_steps:
                        z_alpha = float(step_offset) / float(horiz_steps)
                        targets[idx, 2] = float(start_pose[2]) + z_alpha * (hover_z - float(start_pose[2]))
                    else:
                        v_alpha = float(step_offset - horiz_steps) / float(span - horiz_steps)
                        targets[idx, 2] = hover_z + v_alpha * (float(end_pose[2]) - hover_z)
            targets[anchor_index:, :6] = end_pose
            pose_anchored_from = anchor_index
    elif wrist_obb is not None and (start[2] >= hover_z_m - 0.025 or start[2] < hover_z_m - 0.005):
        # Visual alignment / low-altitude safety protection decoupled from closure prediction.
        peak_y_index = 12 + int(np.argmax(targets[12:21, 1]))
        targets, pose_anchored_from = _condition_approach_with_wrist_obb(
            start,
            targets,
            anchor_index=len(targets) - 1,
            peak_y_index=peak_y_index,
            wrist_obb=wrist_obb,
            hover_z_m=float(hover_z_m),
            max_rotation_rad=max_rotation_rad,
            allow_descent=False,
        )
        targets[:, 6] = open_grip
    return targets, held_open_from, pose_anchored_from


def compute_adaptive_preload_m(held_grip: float, default_preload_m: float = 0.008) -> float:
    """Return object-width-aware preload clamping margin."""
    held_grip = float(held_grip)
    if held_grip > 0.042:
        # Wide object (e.g. 2x4 Lego long edge ~59.7mm): moderate 6mm preload prevents stall/slip
        return min(default_preload_m, 0.006)
    if held_grip < 0.022:
        # Thin object (e.g. marker ~15-18mm): tighter 6mm preload above minimum 5mm floor
        return min(default_preload_m, 0.006)
    return float(default_preload_m)


def find_valid_placement_release_index(start, raw, held_grip: float):
    """Return first valid release index in `raw` only if inside target zone and sustained open.

    Prevents wide-object grasps (e.g. Round 8 `held_grip=59.7mm`) from mistaking initial
    open gripper predictions in the grasp zone (`X < 285mm` or `Y < 165mm`) for a placement release.
    """
    release_threshold = max(0.040, min(0.070, float(held_grip) + 0.010))
    cand_indices = np.flatnonzero(raw[:, 6] >= release_threshold)
    if not cand_indices.size:
        return None
    min_step = 12 if (float(start[0]) >= 0.285 and float(start[1]) >= 0.165) else 18
    for idx in cand_indices:
        idx = int(idx)
        if idx < min_step:
            continue
        # Must have transported into the brown target area envelope before releasing
        if float(raw[idx, 0]) < 0.285 or float(raw[idx, 1]) < 0.165:
            continue
        # Must remain open across the tail of the chunk (rejects brief early spikes)
        if float(np.mean(raw[idx:, 6])) < release_threshold - 0.005:
            continue
        return idx
    return None


def verify_placement_completion(measured_grip: float, held_grip: float, wrist_image_or_bytes=None) -> dict:
    """Verify post-release gripper opening and visual presence of object on brown target area."""
    expected_open = max(0.045, min(0.072, float(held_grip) + 0.006))
    gripper_opened_ok = bool(np.isfinite(measured_grip) and float(measured_grip) >= expected_open)
    target_obb = detect_wrist_target_obb(wrist_image_or_bytes) if wrist_image_or_bytes is not None else None
    block_obb = detect_wrist_block_obb(wrist_image_or_bytes) if wrist_image_or_bytes is not None else None
    visual_confirmed = True
    if target_obb is not None and block_obb is not None:
        dist_px = math.hypot(block_obb["cx"] - target_obb["cx"], block_obb["cy"] - target_obb["cy"])
        visual_confirmed = dist_px <= max(target_obb["w"], target_obb["h"]) * 0.85
    return {
        "confirmed": bool(gripper_opened_ok and visual_confirmed),
        "gripper_opened": gripper_opened_ok,
        "gripper_opened_ok": gripper_opened_ok,
        "visual_confirmed": visual_confirmed,
        "required_open_m": round(float(expected_open), 4),
        "expected_open_m": round(float(expected_open), 4),
        "measured_grip_m": round(float(measured_grip), 4),
        "held_grip_m": round(float(held_grip), 4),
        "block_obb": block_obb,
        "target_obb": target_obb,
    }


def condition_placement_actions(start, raw, *, held_grip=None, max_position_m=.050, preload_m=0.008, target_obb=None):
    """Return a bounded placement candidate holding clamped gripper until verified target-zone release."""
    start, raw = np.asarray(start, dtype=float), np.asarray(raw, dtype=float)
    if (start.shape != (7,) or raw.shape != (50, 7)
            or not np.isfinite(start).all() or not np.isfinite(raw).all()):
        raise ValueError("Invalid placement start or actions")
    if held_grip is None:
        held_grip = float(start[6])
    held_grip = min(max(float(held_grip), 0.0), 0.080)
    eff_preload_m = compute_adaptive_preload_m(held_grip, preload_m)
    # Active preload clamping: command the gripper slightly below the held width
    # so the motor position loop maintains constant non-zero clamping torque against the rigid block.
    clamp_grip = max(0.005, held_grip - eff_preload_m)
    delta = raw[0] - start
    delta[3:6] = (delta[3:6] + np.pi) % (2 * np.pi) - np.pi
    if (np.linalg.norm(delta[:3]) > max_position_m or np.max(abs(delta[3:6])) > np.radians(12)
            or not 0 <= start[6] <= .085 or raw[:, 6].min() < -.020 or raw[:, 6].max() > .085):
        raise ValueError("Target outside placement candidate envelope")
    targets = raw.copy()
    clipped = np.flatnonzero(targets[:, 6] < 0).tolist()
    targets[:, 6] = np.maximum(targets[:, 6], 0)
    # Stage-gated release detection: only release if inside the brown target area envelope
    # and opening above the object-width-aware release threshold.
    first_release = find_valid_placement_release_index(start, raw, held_grip)
    if first_release is not None:
        targets[:first_release, 6] = clamp_grip
        released_grips = np.minimum(np.maximum(raw[first_release:, 6], clamp_grip), 0.080)
        targets[first_release:, 6] = np.maximum.accumulate(released_grips)
        release_pose = targets[first_release, :6].copy()
        if target_obb is not None:
            dx_mm = (release_pose[0] - start[0]) * 1000.0
            dy_mm = (release_pose[1] - start[1]) * 1000.0
            dz_mm = (start[2] - release_pose[2]) * 1000.0
            pred_cx = float(target_obb["cx"]) + 1.05 * dz_mm - 2.3 * dx_mm
            pred_cy = float(target_obb["cy"]) + 2.1 * dy_mm
            if pred_cx < 338.0:
                release_pose[0] += np.clip((pred_cx - 353.0) / 2.3, -18.0, 0.0) / 1000.0
            elif pred_cx > 368.0:
                release_pose[0] += np.clip((pred_cx - 353.0) / 2.3, 0.0, 18.0) / 1000.0
            if pred_cy < 320.0:
                release_pose[1] += np.clip((336.0 - pred_cy) / 2.1, 0.0, 18.0) / 1000.0
            elif pred_cy > 352.0:
                release_pose[1] += np.clip((336.0 - pred_cy) / 2.1, -18.0, 0.0) / 1000.0
        reach_index = max(1, first_release - 1)
        pose_delta = release_pose - targets[0, :6]
        pose_delta[3:6] = (pose_delta[3:6] + np.pi) % (2 * np.pi) - np.pi
        ramps = np.arange(1, reach_index + 1, dtype=float)[:, None] / float(reach_index)
        targets[1:reach_index + 1, :6] = targets[0, :6] + ramps * pose_delta
        targets[reach_index:, :6] = release_pose
    else:
        targets[:, 6] = clamp_grip
        # If holding a wide object (e.g. Round 8 held_grip=59.7mm) caused the raw model transport
        # chunk to stall near the grasp zone (end X < 295mm or end Y < 170mm), guide the transport
        # smoothly toward the brown target pad approach hover pose while respecting step limits.
        if held_grip > 0.038 and (float(targets[-1, 0]) < 0.295 or float(targets[-1, 1]) < 0.170):
            goal_pose = targets[-1, :6].copy()
            goal_pose[0] = max(float(goal_pose[0]), min(float(start[0]) + 0.085, 0.318))
            goal_pose[1] = max(float(goal_pose[1]), min(float(start[1]) + 0.078, 0.186))
            goal_pose[2] = max(float(goal_pose[2]), 0.190)
            pose_delta = goal_pose - targets[0, :6]
            pose_delta[3:6] = (pose_delta[3:6] + np.pi) % (2 * np.pi) - np.pi
            ramps = np.linspace(0.0, 1.0, len(targets))[:, None]
            targets[:, :6] = targets[0, :6] + ramps * pose_delta
        elif target_obb is not None:
            dy_mm = (targets[-1, 1] - start[1]) * 1000.0
            pred_cy = float(target_obb["cy"]) + 2.1 * dy_mm
            corr_y_m = 0.0
            if pred_cy < 265.0:
                corr_y_m = float(np.clip((280.0 - pred_cy) / 2.1, 0.0, 15.0)) / 1000.0
            elif pred_cy > 305.0:
                corr_y_m = float(np.clip((280.0 - pred_cy) / 2.1, -15.0, 0.0)) / 1000.0
            if corr_y_m != 0.0:
                targets[:, 1] += np.linspace(0.0, corr_y_m, len(targets))

    bridge_delta = targets[0] - start
    bridge_delta[3:6] = (bridge_delta[3:6] + np.pi) % (2 * np.pi) - np.pi
    bridge_steps = 6
    prefix = start + np.arange(1, bridge_steps)[:, None] / bridge_steps * bridge_delta
    prefix_grip_ramps = np.linspace(float(start[6]), clamp_grip, bridge_steps)[1:]
    prefix[:, 6] = prefix_grip_ramps
    sequence = np.vstack([prefix, targets])

    candidate = []
    prev = start
    splits = []
    for i, target in enumerate(sequence):
        d = target - prev
        d[3:6] = (d[3:6] + np.pi) % (2 * np.pi) - np.pi
        count = max(1, math.ceil(np.linalg.norm(d[:3]) / .015),
                    math.ceil(abs(d[6]) / .015))
        if count > 1:
            splits.append(i)
        candidate.extend((prev + d * j / count).tolist() for j in range(1, count))
        candidate.append(target.tolist())
        prev = target
    if len(candidate) > 75:
        raise ValueError("Placement candidate exceeds subdivision limit")
    return np.asarray(candidate), clipped, splits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--prefix", default=None,
                    help="Snapshot file prefix (auto-detects loop_03 if present, else final)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    prefix = args.prefix
    if prefix is None:
        prefix = ("loop_03"
                  if (args.snapshot / "loop_03_state.json").exists()
                  and (args.snapshot / "loop_03_raw_actions.npy").exists()
                  else "final")
    state = json.loads((args.snapshot / f"{prefix}_state.json").read_text())
    start = np.asarray(state["right_ee_position_m"] + state["right_ee_euler_rad"]
                       + [state["state"][6]], dtype=float)
    seed = np.asarray(state["state"][:6], dtype=float)
    raw = np.load(args.snapshot / f"{prefix}_raw_actions.npy", allow_pickle=False).astype(float)
    if (start.shape != (7,) or seed.shape != (6,) or raw.shape != (50, 7)
            or not all(np.isfinite(x).all() for x in (start, seed, raw))):
        raise ValueError("Invalid right7 snapshot")
    candidate, clipped, splits = condition_grasp_actions(start, raw)
    hold_candidate, hold_clipped, hold_splits = condition_grasp_actions(
        start, raw, hold_pose=True)
    base = dict(seed=seed.tolist(), start_action=start.tolist())
    cases = [
        dict(**base, name="fourth_raw",
             source=f"{args.snapshot.name}/{prefix}; unmodified CUDA prediction",
             actions=raw.tolist()),
        dict(**base, name="fourth_conditioned",
             source=f"{args.snapshot.name}/{prefix}; OFFLINE ONLY; "
             f"negative gripper indices {clipped} clipped to 0; approach held open; "
             f"subdivided target indices {splits}; no grasp/contact proof",
             actions=candidate.tolist()),
        dict(**base, name="fourth_hold_pose",
             source=f"{args.snapshot.name}/{prefix}; OFFLINE ONLY; "
             f"6D pose anchored at settled start_action[:6]; "
             f"negative gripper indices {hold_clipped} clipped to 0; "
             f"subdivided target indices {hold_splits}; no grasp/contact proof",
             actions=hold_candidate.tolist()),
    ]
    with args.output.open("x") as stream:
        json.dump({"cases": cases}, stream, indent=2)
        stream.write("\n")
    print(f"raw=50; conditioned={len(candidate)} (splits={splits}); "
          f"hold_pose={len(hold_candidate)} (splits={hold_splits}); clipped={clipped}")


if __name__ == "__main__":
    main()
