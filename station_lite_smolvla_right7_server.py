#!/usr/bin/env python3
"""CUDA-only inference service for the single-right-arm 7D SmolVLA policy.

This is a separate protocol from station_lite_smolvla_server.py. It accepts
only right-arm state plus top/right-wrist cameras and returns native (50, 7) actions.
It never connects to robot hardware.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCHEMA = "station_lite_smolvla_right7_v1"
SOURCE_SCHEMA = "station_lite_smolvla_right7_request_v1"
TASK = "Pick up the red wooden block from the white grid and place it on the brown target area."
CAMERAS = {
    "observation.images.cam_high": "observation.images.camera1",
    "observation.images.cam_right_wrist": "observation.images.camera2",
}
IMAGE_SIZE = (640, 480)
MAX_REQUEST_BYTES = 64 * 1024 * 1024


class PolicyEngine:
    def __init__(self, checkpoint: Path, seed: int = 42,
                 seed_mode: str = "sequence"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required; refusing CPU inference")
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.checkpoint = checkpoint.resolve()
        self.device = "cuda"
        self.seed = seed
        if seed_mode not in ("sequence", "fixed"):
            raise ValueError("seed_mode must be sequence or fixed")
        self.seed_mode = seed_mode
        self.policy = SmolVLAPolicy.from_pretrained(str(self.checkpoint)).to("cuda").eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        if self.policy.config.action_feature.shape != (7,) or self.policy.config.chunk_size != 50:
            raise ValueError("expected native right7 policy output shape (50, 7)")
        self.lock = threading.Lock()

    @torch.inference_mode()
    def predict(self, payload: dict) -> tuple[np.ndarray, float, int]:
        if payload.get("schema") != SOURCE_SCHEMA:
            raise ValueError(f"input schema must be {SOURCE_SCHEMA}")
        task = payload.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        seq = payload.get("seq")
        if type(seq) is not int or seq < 0 or seq > 1_000_000:
            raise ValueError("seq must be a nonnegative integer")
        state = np.asarray(payload.get("state"), dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("right-arm state must contain 7 finite values")
        encoded = payload.get("images_jpeg_base64")
        if not isinstance(encoded, dict):
            raise ValueError("images_jpeg_base64 must be an object")
        observation = {"observation.state": torch.from_numpy(state.copy()), "task": task}
        for original, mapped in CAMERAS.items():
            value = encoded.get(original)
            if not isinstance(value, str):
                raise ValueError(f"missing image: {original}")
            try:
                raw = base64.b64decode(value, validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    image = image.convert("RGB")
                    if image.width * IMAGE_SIZE[1] != image.height * IMAGE_SIZE[0]:
                        raise ValueError(f"wrong aspect ratio for {original}: {image.size}")
                    if image.size != IMAGE_SIZE:
                        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                    rgb = np.asarray(image, dtype=np.uint8).copy()
            except (binascii.Error, OSError, ValueError) as exc:
                raise ValueError(f"invalid image for {original}: {exc}") from exc
            observation[mapped] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        start = time.perf_counter()
        sampling_seed = self.seed if self.seed_mode == "fixed" else self.seed + seq
        with self.lock:
            torch.manual_seed(sampling_seed)
            actions = self.post(self.policy.predict_action_chunk(self.pre(observation)))
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        result = actions[0].detach().float().cpu().numpy()
        if result.shape != (50, 7) or not np.isfinite(result).all():
            raise RuntimeError(f"invalid right7 model output: {result.shape}")
        return result, elapsed_ms, sampling_seed


def handler_for(engine: PolicyEngine):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code: int, obj: dict):
            body = json.dumps(obj, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self.send_json(404, {"error": "not found"})
            self.send_json(200, {
                "schema": SCHEMA, "source_schema": SOURCE_SCHEMA,
                "checkpoint": str(engine.checkpoint), "chunk_size": 50,
                "state_dim": 7, "action_dim": 7,
                "state_contract": "right_joints_6_plus_gripper",
                "cameras": list(CAMERAS), "task": TASK,
                "device": "cuda", "gpu": torch.cuda.get_device_name(0),
                "seed_base": engine.seed, "seed_mode": engine.seed_mode,
                "hardware_connected": False,
            })

        def do_POST(self):
            if self.path != "/predict":
                return self.send_json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("invalid request length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                actions, elapsed_ms, sampling_seed = engine.predict(payload)
                self.send_json(200, {
                    "schema": SCHEMA, "source_schema": SOURCE_SCHEMA,
                    "session_id": payload.get("session_id"),
                    "seq": payload.get("seq"), "actions": actions.tolist(),
                    "action_dim": 7, "inference_ms": round(elapsed_ms, 3),
                    "sampling_seed": sampling_seed, "hardware_connected": False,
                })
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                print(f"[predict error] {type(exc).__name__}: {exc}", flush=True)
                self.send_json(500, {"error": f"inference failed: {type(exc).__name__}"})

        def log_message(self, fmt, *args):
            print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8010, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--seed-mode", choices=("sequence", "fixed"), default="sequence")
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error("checkpoint directory does not exist")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required; no CPU fallback is available")
    engine = PolicyEngine(args.checkpoint, args.seed, args.seed_mode)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(engine))
    print(f"[ready] {SCHEMA} checkpoint={engine.checkpoint} "
          f"device=cuda gpu={torch.cuda.get_device_name(0)} "
          f"bind={args.host}:{args.port} hardware_connected=false", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
