# Engineering best practices

These lessons come from the Station Lite experiments culminating in `new9`. Their purpose is to reproduce a reliable pipeline, not copy every historical patch.

## 1. Start with the complete interface

Freeze state/action ordering, units, image roles, frame/TCP, Euler convention, nominal timing and task text before training. A seven-dimensional joint state does not imply seven joint-angle actions. Keep original recordings and version every derived training view.

## 2. Use few demonstrations with a narrow task definition

Use consistent successful approaches, closure and vertical lift, with enough repeats across the intended positions. Audit action quality before adding quantity. The reference used 35 red-block episodes for fine-tuning; this does not establish 35 as a universal sufficient count. Pretrained knowledge, position coverage and the engineered execution layer matter.

## 3. Deploy the checkpoint and processors as one artifact

Keep model weights, feature configuration, tokenizer and normalization together. Verify the exact checkpoint, source hashes, seed mode and cache revision. Set a consistent HF cache and prepare it before enabling offline mode. A missing `num2words` error is a processor dependency issue, not a reason to retrain.

## 4. Calibrate vision rather than accumulating pixel rules

The baseline uses a wrist pinhole model, an EEF-to-camera transform, full EEF pose and a coupled XY image Jacobian. This replaced fixed pixel/mm rules whose sign and pose assumptions caused drift and oscillation. A successful local linear rule is not automatically valid after a height, pitch or yaw change.

Reference constants are in `build_grasp_replay_cases.py`: intrinsics, extrinsics, object plane, canonical grasp pixel target, grasp height, top-grid ROI/centers and `GRID_HOVER_POSES`. On another station, recalibrate all affected quantities, validate reprojection on held-out poses, then validate IK and tool clearance. EEF reference-point Z is not fingertip clearance, and object-plane Z is not interchangeable with EEF grasp Z.

Keep object identity and physical shape separate from a partially occluded 2D contour. Do not let a square become a different grasp task merely because its projection is elongated. OBB angle is an image measurement, not a directly executable wrist-joint rotation. A nonvertical tool needs full 3D geometry and swept-clearance checks.

## 5. Separate approach, descent and closure

Use fresh observations to distinguish target acquisition, hover alignment, open-gripper descent, grasp-pose confirmation, closure, lift and placement. High-level alignment alone does not establish contact height. A budget limit does not make an unaligned grasp ready.

After reaching the correct grasp pose, holding that pose during closure can prevent late forward overshoot. Holding it too early causes high-air empty grasps. If low-level alignment is lost, regain clearance before horizontal repositioning. Check the complete conditioned trajectory, not just its first/last point.

## 6. Keep planning and timing guards in the control loop

Retain direct IK/retreat diagnostics, physical and soft limits, first-target gaps, adjacent changes, tracking, observation freshness and vendor trajectory planning. A rejected chunk is evidence to diagnose, not permission to widen limits. Joint smoothing can change a Cartesian path; validate after each transformation.

The baseline deliberately waits for chunk completion and measured settling. Use actual planned duration plus measured state; do not assume 50 steps at nominal 30Hz last 1.67 seconds when phase speed scales time. A requested 60Hz trace rate is not proof of achieved 60Hz sampling.

## 7. Verify what logs actually mean

`GRASP COMPLETED` means a closure block finished. Gap feedback can support empty/held classification but does not prove lift. `PLACEMENT CONFIRMED` primarily checks gripper opening, not visual occupancy of the brown mat. Score the operator/video-observed outcomes separately.

The first controller command-sample query in `new9` returned an empty array. This did not stop execution, but the controller timeline is not fully captured for that query. Retain planned targets and measured trace, and record missing telemetry rather than silently treating it as available.

## 8. Version both ends of remote deployment

Use a reviewed bundle, copy workstation → robot, and verify checksums on both hosts. One historical regression was old robot code copied back to itself. A file existing at the right path is not evidence of the right version.

Archive each run's code/weight/calibration identity, actual task, scene layout, noise seed, observations, raw and sent targets, tracking and outcome. Never infer cell/color from a reused RUN name. One person/agent changes the active baseline at a time.

## 9. Attribute system-level generalization correctly

The prompt is sent to SmolVLA and also parsed by the client. The supported color keyword selects an HSV mask for top/wrist target detection. Thus prompt-guided target selection is real system behavior, but this experiment does not isolate learned language grounding in SmolVLA. Use a single requested color; multiple color words, negation and unsupported instructions are not a general planner.

Evaluate rotation, color, position and distractors separately and then in combination. Report the inspected diagnostic holdout honestly. New real-world trials should include repeated conditions, unseen within-cell offsets and outcome evidence.

## 10. Grow scope after freezing a working baseline

Preserve the `new9` source/parameter/checkpoint combination before optimization. Future recovery must actually reopen/retreat, reacquire the target and revalidate before retrying; a loop-counter reset is not recovery. Test retry limits and interruption handling separately. Guided RTC, speed optimization, complex shapes, dual-arm sorting and LeRobot v0.7 belong in independent experiments with baseline regressions.
