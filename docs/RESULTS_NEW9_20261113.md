# Results: reference Station Lite, 13 November 2026

## What the run demonstrates

The operator reported **8 successful pick-and-place trials out of 8**, approximately 12:25–12:42 local time. Each trial has a matching textual runtime log with a grasp block, a release confirmation, zero empty-grasp recovery events and no guard-rejected block. These are eight demonstrated system successes, not a population success-rate estimate or exhaustive coverage certificate.

| Trial / RUN time | Operator-described target and scene | Sent chunks | Grip after grasp |
|---|---|---:|---:|
| 1 / 122503 | R8 yellow, single block | 6 | 28.2mm |
| 2 / 122628 | R2 yellow, single block | 9 | 28.2mm |
| 3 / 122818 | R1 yellow, single block | 10 | 27.7mm |
| 4 / 123030 | R1 yellow, 45° | 9 | 35.2mm |
| 5 / 123253 | R1 yellow, 45°, red + blue distractors | 10 | 32.5mm |
| 6 / 123510 | R9 yellow, red + blue distractors | 6 | 28.0mm |
| 7 / 123819 | Red, 30°, between R2/R3/R5/R6, yellow + blue distractors | 7 | 27.2mm |
| 8 / 124040 | R7 blue/cyan, 60°, red + yellow distractors | 8 | 28.2mm |

All outcomes above are successful grasp **and placement** according to the operator. Trial 8's RUN name incorrectly says R5; operator notes and top-camera detection identify R7. Its actual prompt says `cyan`; the color parser normalizes blue/cyan into one category. Trial 7's coarse detector selects R3 for an object at an inter-cell location, followed by wrist refinement.

The release log primarily verifies opening of the gripper. Successful physical placement is supported by the operator's outcome record, not inferred solely from that log or from gap feedback. No independent video scoring was performed in this documentation review.

## Data and model

- 35 local training episodes, 13,231 frames; 10 fixed diagnostic holdout episodes, 3,705 frames.
- Local demonstrations used the red block with standard orientation, per the collection protocol/operator. They covered grid positions and within-cell offsets. There is no per-frame yaw audit proving every recorded angle was identical.
- Base: pretrained `lerobot/smolvla_base`; right7 fine-tuning for 10,000 updates, batch size 1, seed 42.
- Two cameras; right-joint/gripper state; absolute EEF/gripper actions. Weights were not retrained for the new target colors in this live milestone.

The pretrained base has prior experience beyond these local recordings. “35 demonstrations” describes local fine-tuning, not the entirety of the model's training history.

## Contribution attribution

| Component | Role in the demonstrated system |
|---|---|
| SmolVLA | Pretrained visual/state/language features and action chunks adapted to Station Lite |
| Prompt parser + HSV filters | Explicit requested-color selection in top/wrist images |
| Top-camera grid map | Station-specific coarse approach target |
| Calibrated wrist projection and coupled XY correction | Local geometric alignment across pose/height changes |
| Grasp/placement conditioning and SDK | Open approach, controlled descent/closure, lift, transfer, release and trajectory execution |
| Guarded client | IK, limits, continuity, freshness, tracking and rejection before publishing |

This is a **SmolVLA-based hybrid manipulation system**. No ablation here isolates how much of the new-color/rotation/distractor performance comes from the policy versus the visual/controller layers. Prompt-driven color selection is supported; arbitrary semantic reasoning, negation, new object categories and general scene planning were not tested.

## Suitable public description

> We fine-tuned pretrained SmolVLA using only 35 local demonstrations of a red-block pick-and-place task. Combined with calibrated visual target selection, geometric alignment and guarded execution on Lumos Station Lite, the system completed all eight documented live trials, including additional colors, rotated blocks and scenes with differently colored distractors. These results demonstrate transfer at the complete-system level; they do not isolate the policy's contribution or establish reliability across all conditions.

## Remaining validation

Green targets, repeated full nine-cell coverage, same-color distractors, heavy occlusion, changed lighting/cameras, arbitrary clutter, interruption recovery, autonomous repeated sorting and complex shapes are not established by this batch. One test scene has three colored blocks; avoid describing that as unrestricted clutter robustness.

For the next evaluation, freeze this version, repeat a single-block baseline, then vary cell, color, angle and distractors with a scene manifest. Score target selection, collision/pushing, stable lift, correct placement and total time separately. Use a newly designed live test set; the historical holdout is already diagnostic.

## Evidence and reproducibility status

Source: local `docs/new9.md`; full RUN IDs and parsed events are in `releases/right7_20261113/new9_results.json`. The manifest preserves original live source hashes separately from the English presentation edition. Per-run robot-side hash certificates were not included in the text log.

At review time, the eight independent `_snapshot`/`_trace`/`.log` archives were not found under local `diagnostics/20261113_live`. Archive them before claiming that the evidence package is complete. Neither raw images/arrays nor weights are included in the public export.

Current software tests: 66/66 pass under system Python. `station_act` returns one missing-SciPy error. No new model training, inference, SDK connection or robot motion was performed during preparation of this project pipeline.
