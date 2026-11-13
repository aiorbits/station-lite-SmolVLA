# Quick deployment: approved Station Lite baseline

Use this on the calibrated reference station. A different station needs [calibration and staged validation](SMOLVLA_BEST_PRACTICES.md). Read the [environment notes](ENVIRONMENT_REFERENCE.md) first.

## 1. Transfer the correct source

On **Hans22**, from the project parent:

```bash
python SmolVLA/prepare_shared_repository.py --profile public \
  --output /tmp/SmolVLA_release_20261113
scp -r /tmp/SmolVLA_release_20261113 lumos@192.168.0.101:/home/lumos/replay_work/
```

Use a new export name if the destination exists. Transfer from Hans22, not from a robot shell back to the same robot IP. Copy only this curated bundle, not the working project with weights/caches. Check current host identity and LAN addresses; `.109`/`.101` are reference addresses.

On another host, replace both `CKPT` and `EXPECTED_CKPT` below with the same intended host-side absolute checkpoint path. Checkpoint contents must still match the manifest.

## 2. Terminal 1 — workstation: CUDA service

```bash
conda activate station_act
cd '/media/hans/Bridge/Lumos Station Lite/SmolVLA'
export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CKPT='/home/hans/cc/ckpt/smolvla_woodblock_right7_train35_v1_10000/checkpoints/010000/pretrained_model'
python tools/verify_baseline.py --role host --checkpoint "$CKPT" --gpu
```

**Continue only if verification succeeds.** This documentation session could not access CUDA; no fresh policy startup was tested. Never replace the gate with CPU inference. If port 8010 is occupied, identify the existing service before starting another.

```bash
python station_lite_smolvla_right7_server.py \
  --checkpoint "$CKPT" --host 0.0.0.0 --port 8010 \
  --seed 42 --seed-mode fixed
```

Offline startup requires the separate VLM/processor cache to be populated already; see the environment notes.

## 3. Terminal 2 — robot: cameras and software checks

```bash
ssh lumos@192.168.0.101
conda activate lumosteleop
/home/lumos/lumosteleop/lumosteleop status
```

If image-server is stopped, start it:

```bash
/home/lumos/lumosteleop/lumosteleop image-server
```

Then check the port and deployed files:

```bash
ss -ltn | grep ':60000'
cd /home/lumos/replay_work/SmolVLA_release_20261113
export PYTHONPATH="/home/lumos/startouch_sdk/interface_py:/home/lumos/lumosteleop/.app/runtime${PYTHONPATH:+:$PYTHONPATH}"
python tools/verify_baseline.py --role robot
python -m unittest test_right7_guarded_contract test_adaptive_phase_timing test_joint_trace
```

Tests require NumPy, OpenCV, Pillow and **SciPy**, but not pytest. Review result: 66/66 passed under the system Python with these dependencies; `station_act` lacked SciPy and returned one test error. Install missing test dependencies in the intended test environment instead of omitting tests.

## 4. First-deployment gate: IK and planning

Run with an operator present. SDK construction may enable motors. This mode sends no Home/policy waypoints and is **not** a no-side-effect hardware operation.

```bash
ssh lumos@192.168.0.101
conda activate lumosteleop
export PYTHONPATH="/home/lumos/startouch_sdk/interface_py:/home/lumos/lumosteleop/.app/runtime${PYTHONPATH:+:$PYTHONPATH}"
CODE='/home/lumos/replay_work/SmolVLA_release_20261113'
SRV='http://192.168.0.109:8010'
EXPECTED_CKPT='/home/hans/cc/ckpt/smolvla_woodblock_right7_train35_v1_10000/checkpoints/010000/pretrained_model'
curl -fsS --max-time 5 "$SRV/health"
mkdir -p "$CODE/logs"
cd /home/lumos/startouch_sdk
python "$CODE/station_lite_smolvla_right7_guarded.py" \
  --ik-preflight-only --server "$SRV" --expected-checkpoint "$EXPECTED_CKPT" \
  --expected-seed-mode fixed --right-can-interface can1 \
  --speed 0.1 --rate 30 --margin 0.02 \
  --snapshot-dir "$CODE/logs/first_ik_preflight_$(date +%Y%m%d_%H%M%S)"
```

The SDK working directory avoids an unverified initialization location. A failed gate means diagnose and stop, not enlarge limits. An unfamiliar start pose may need the established vendor Home procedure and fresh observation before retesting.

## 5. Terminal 3 — supervised reference trial

After gates pass, the normal client initializes and homes **both arms**. Clear both workspaces, use the calibrated grid/target and keep stop controls accessible.

```bash
cd /home/lumos/replay_work/SmolVLA_release_20261113
python tools/verify_baseline.py --role robot
SRV='http://192.168.0.109:8010'
EXPECTED_CKPT='/home/hans/cc/ckpt/smolvla_woodblock_right7_train35_v1_10000/checkpoints/010000/pretrained_model'
curl -fsS --max-time 5 "$SRV/health"
RUN="right7_R8_yellow_0deg_single_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
set -o pipefail
python station_lite_smolvla_right7_guarded.py \
  --server "$SRV" --expected-checkpoint "$EXPECTED_CKPT" \
  --prompt "Pick up the yellow wooden block from the white grid and place it on the brown target area." \
  --hover-z-m 0.200 --expected-seed-mode fixed --send-right-chunks --loops 6 \
  --settled-chunks --adaptive-refine --condition-final-grasp --hold-grasp-pose --place-after-grasp \
  --phase-speeds --smooth-waypoints --grasp-duration-s 2.5 --high-cam-guidance \
  --placement-bridge-max-mm 50.0 --max-initial-gap-mm 25.0 --settled-bridge-max-mm 40.0 \
  --initial-bridge-steps 6 --max-observation-age-ms 800 --speed 0.1 --rate 30 \
  --exec-steps 50 --switch-after 20 --blend-overlap 10 --margin 0.02 \
  --joint-trace-hz 60 --joint-trace-dir "logs/${RUN}_trace" \
  --snapshot-dir "logs/${RUN}_snapshot" --final-probe \
  2>&1 | tee "logs/$RUN.log"
```

These choices match `new9`; explicit high-camera/checkpoint flags avoid relying on defaults. Six loops are an initial budget, not six trials; the client can extend it. Phase speeds override the general speed for several stages. Start with one block and repeat before adding distractors/rotation. Use one requested color per prompt; blue/cyan share a parser category.

## 6. Archive the run

Keep the `.log`, `_snapshot`, `_trace`, source hashes, checkpoint identity, scene description and operator outcome. Score **correct target / aligned / lifted / placed** separately. An open-gripper log alone does not verify placement.

Copy from a workstation terminal, replacing the timestamp:

```bash
mkdir -p 'SmolVLA/diagnostics/reference_runs'
scp -r 'lumos@192.168.0.101:/home/lumos/replay_work/SmolVLA_release_20261113/logs/right7_R8_yellow_0deg_single_RUN_TIMESTAMP*' \
  'SmolVLA/diagnostics/reference_runs/'
```

Keep raw images, arrays, SDK files and weights out of Git. Share summaries and separately authorized media.
