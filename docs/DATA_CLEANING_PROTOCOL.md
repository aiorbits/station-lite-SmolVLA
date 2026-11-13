# Review and clean right-arm grasp demonstrations

## Goal

Our live tests suggest that the 45 demonstrations collected on 27 October vary in quality. Review them **before any further training**. The red wooden block is the object to grasp; the brown area is the placement target. We want to know which demonstrations show a clean grasp and whether any show the same problem as the live policy: the gripper passes the block before closing.

## Data and scope

- Review the existing **35 training episodes** first. The other **10 episodes are a fixed holdout**: do not use them to choose cleaning rules or move them into training.
- Use the overhead and right-wrist views, gripper **command**, gripper **measured opening** when available, and robot state. Do not collect new data or train a model for this task.
- **Never edit or delete raw episodes.** Record recommendations in a separate table.

## How to review

1. Two reviewers independently review the same **five training episodes** and agree on the labels below. Include different grid positions and one difficult example if available.
2. Review all 35 training episodes. Watch each full episode, then inspect the grasp interval slowly in both camera views.
3. For each grasp attempt, record four original frame numbers. Use `UNKNOWN` when an event cannot be seen or measured; do not guess.

| Label | What to find |
|---|---|
| **A: aligned** | The red block is between the jaws at a suitable grasp height and direction. If only image overlap is visible, mark height `UNKNOWN`. |
| **C: close command** | The gripper target opening begins a sustained decrease. |
| **M: measured closing** | Gripper feedback begins a sustained decrease, if feedback exists. Command and measurement are different. |
| **L: lifted** | The block visibly leaves the table with the gripper. A closed gripper alone is not proof. |

4. Note whether the gripper moves past the block between A and closing, closes on an edge, repeatedly opens and closes, pushes the block on the table, or lifts it cleanly. Record grasp and placement success separately. For retries, make one row per attempt.
5. Mark each **episode** `KEEP`, `REVIEW`, or `EXCLUDE`:
   - **KEEP:** clear, usable demonstration; the block is grasped and lifted. Small safe corrections are acceptable.
   - **REVIEW:** occlusion, unclear alignment, retry, or conflicting evidence. A second reviewer must inspect it.
   - **EXCLUDE:** clear failed grasp labeled successful, human completion, or missing/mismatched critical data. A second reviewer must confirm the reason. Exclusion applies only to a proposed new training selection; retain the original.

Do not invent a distance threshold from one example or remove a successful demonstration just because its path differs. Historical frame rate is nominal; frame differences are not precise camera-to-robot latency measurements.

## What to submit

Submit one CSV row **per grasp attempt** with these columns:

```text
episode_id,attempt_id,grid_position,A_frame,C_frame,M_frame,L_frame,grasp_success,placement_success,issue,episode_decision,reason,evidence_path,reviewer,second_reviewer
```

Also submit a short summary with: (1) counts of KEEP / REVIEW / EXCLUDE **episodes**; (2) the proposed exclusions and evidence for each; (3) which grid positions would lose coverage; and (4) whether training demonstrations commonly move past the block before closing. Include `UNKNOWN` cases and reviewer disagreements. Another engineer must be able to find every cited frame.

**Stop after the audit.** The project owner will decide whether to create a new dataset version or collect additional demonstrations. Do not change the fixed holdout or start retraining as part of this assignment.
