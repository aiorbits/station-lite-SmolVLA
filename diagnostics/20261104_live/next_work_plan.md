# Next work plan

Current reference: new9, 13 November 2026. Eight documented system trials succeeded. Earlier plateau/IK/high-air closure reports are historical investigation evidence, not the current operating state.

1. Preserve the approved source/checkpoint/calibration combination and archive the eight original successful runs.
2. Validate a clean reproduction environment. Resolve the recorded Transformers-extra discrepancy and missing test-only SciPy. Require CUDA for policy computations; never use CPU fallback.
3. Follow the English quickstart for source transfer, checksum verification, cameras, supervised preflight and controlled execution.
4. Repeat single-object baselines before expanding position, color, yaw and distractors. Test green targets and same-color distractors explicitly. Score selection, contact/pushing, lift, placement and duration separately.
5. Keep the local 35/10 split unchanged. New cleaning rules or subsets require new dataset and model versions. The inspected holdout is diagnostic.
6. Isolate recovery, faster execution, RTC, complex shapes, LeRobot v0.7 and dual-arm sorting. Regress each against the frozen baseline.

Completed during publication preparation: document organization, source/checkpoint integrity, exporter checks and 66 software tests under an environment with complete dependencies. Fresh installation, new GPU service startup, new SDK planning and additional motion: NOT RUN.

The complete chronological plan remains in the private local investigation archive. Project entry points are ../../docs/QUICKSTART_STATION_LITE.md and ../../docs/RESOURCE_MAP.md.
