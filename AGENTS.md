# Shared project rules

Read HANDOFF.md, HANDOFF_20261104.md and diagnostics/20261104_live/next_work_plan.md before checking files and process state. Only one agent may change code or run experiments at a time. Before handing off, stop new operations and record which services remain running.

- SmolVLA training, numerical inference and comparisons require CUDA. Stop if CUDA is unavailable; never substitute CPU policy predictions. Ordinary file processing is exempt.
- right7 state is six measured right joints plus gripper feedback. Actions are six-dimensional absolute end-effector poses plus gripper targets, not joint commands.
- The red block is the demonstrated grasp object; the brown region is the placement target. Preserve the fixed 35-training/10-holdout split. The holdout has already been inspected for diagnosis.
- Current reference: new9 on 13 November 2026, eight documented successful system trials using right7 10k, top-grid guidance, calibrated wrist geometry and the guarded client. See releases/right7_20261113/BASELINE_MANIFEST.json. The historical 50/50 IK rejection is not the current state. New stations still require lifecycle, FK/IK, calibration and planning checks.
- SDK constructors may enable motors. No policy commands does not mean no hardware side effects. Hardware work requires the scope authorized by the on-site operator; a handoff does not confirm physical state.
- Do not bypass rejection by widening physical/soft limits, removing guards or forcing closure.
- Preserve original demonstrations and failure evidence. Record inputs, checkpoint, processors, device, seed, source hashes and RUN identity.
- Commit/upload only explicitly listed files. Never add credentials, datasets, weights, caches or raw camera images to Git.
- Update HANDOFF.md after changes: completed work, evidence, outstanding work, next actions and unknown state. Mark unexecuted validation NOT RUN.
- Synchronize reusable milestones and reviewed resources to the local sharing copy. Keep original experimental assets in the local archive.
- The public export uses English publication editions of legacy notes. Original investigation records remain in the full local workspace and are not part of that export.
