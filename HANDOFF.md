# Engineering handoff: English publication edition

Updated 13 November 2026. This is the public reference edition of the project handoff. It preserves the current operating contract, results, known limitations and next actions without reproducing the full chronological investigation log. Original field notes remain in the private local workspace.

## Current milestone

New9 contains eight operator-reported successful pick-and-place trials. Cases cover yellow, red and blue/cyan blocks, rotations of 30/45/60 degrees, an intermediate grid location and scenes with two differently colored distractors. Each textual log includes a grasp and release; none records empty-grasp recovery or a guard-rejected block. This is complete-system evidence, not an isolated measurement of learned language grounding or statistical coverage of every condition.

Local fine-tuning used 35 red-block episodes, 13,231 frames. Ten fixed diagnostic holdout episodes contain 3,705 frames and have already been inspected. Standard training orientation comes from the collection protocol; no per-frame yaw audit was recorded. The pretrained base contains experience beyond these local demonstrations.

## Approved runtime

- right7 checkpoint: smolvla_woodblock_right7_train35_v1_10000/checkpoints/010000/pretrained_model.
- State: right joint angles 1–6 in radians plus measured gripper opening in metres.
- Action: absolute right EEF XYZ in metres, extrinsic xyz Euler angles in radians, and gripper target in metres.
- Cameras: top and right wrist, 640×480 RGB. The policy runs on a CUDA host; the robot performs acquisition, geometry, IK and guarded execution.
- Prompt color parsing and HSV masks select the requested object. Top-grid guidance provides coarse approach; calibrated pinhole projection and coupled XY correction provide local alignment. Grasp/placement conditioning and vendor planning contribute to success.
- Use settled chunks, phase speeds, smoothing and the quickstart parameters. This baseline is not the historical Guided RTC experiment.

See releases/right7_20261113/BASELINE_MANIFEST.json, runtime_profile.json, dataset_provenance.json, environment_snapshot.json and new9_results.json. English presentation changes are recorded separately from the original live source hashes. Retest any functional modification; do not reinterpret translated bytes as a new physical validation.

## Validation completed

The reviewed reference software passed 66 contract tests under system Python. Host source/checkpoint integrity passed. The public exporter and source verification were checked, including rejection of changed deployment code and refusal to overwrite an export. No new training, inference, SDK connection or motion was performed during documentation preparation.

The original new9 per-run snapshot/trace/log archives were not yet present in the local diagnostics directory. Current evidence is the operator record and embedded logs. Gripper opening confirms neither stable lift nor correct placement by itself; retain independent observed outcomes.

## Environment limits

The observed station_act environment lacks test-only SciPy. Its 66-test run returned one missing-dependency error. The system Python with SciPy passed all tests. The documentation-session CUDA probe returned unavailable, so fresh GPU startup was NOT RUN. This does not by itself diagnose the NVIDIA driver or invalidate earlier GPU experiments.

Observed Transformers 5.22.2 is outside the pinned LeRobot extra declaration >=5.4,<5.6. The environment snapshot is not a portable lockfile. Prepare a separate candidate environment and validate CUDA loading, processors, one training update and reload before certifying installation. Do not upgrade the working environment in place.

## Reusable failure lessons

Earlier versions encountered state/action mismatch, time alignment errors, invalid IK due to initialization context, soft-boundary entry problems, early descent, low-level chasing/pushing, high-air closure and stale-code deployment. Resolve these at their first processing stage rather than increasing loops or retraining without evidence.

The wrist-view investigation identified incorrect depth-parallax assumptions, omitted pose/XY coupling, occlusion-dependent shape switches and inconsistent pixel targets. Later geometry and phase corrections, plus correct workstation-to-robot transfer, led to new9. A source checksum on both computers is essential: an earlier robot-side scp copied obsolete code back to itself.

## Next actions

1. Archive the eight original successful runs, source versions, calibration identity and outcome evidence.
2. Validate a clean reproduction environment and resolve the recorded dependency discrepancy.
3. Freeze the working baseline before testing green targets, repeated nine-cell coverage, within-cell offsets and same-color distractors.
4. Score correct target, alignment, pushing/contact, stable lift, placement and time separately.
5. Keep speed optimization, recovery, complex shapes, dual-arm sorting, RTC and LeRobot v0.7 in isolated experiments with baseline regression checks.

The current milestone does not certify arbitrary clutter, autonomous repeated sorting, interruption recovery or all four colors at every cell. The long-term goal remains dual-arm Lego sorting with explicit task selection, result verification and bounded recovery.

SDK construction can enable motors; the normal client homes both arms. Confirm operator authorization and physical state before motion. Preserve all limits, continuity, freshness and planning guards.


## README presentation update, 13 November 2026

By project-owner request, the README introduction now highlights demonstrated system capabilities. Detailed attribution and untested conditions remain in docs/RESULTS_NEW9_20261113.md. No experimental claims, results or runtime code changed.
