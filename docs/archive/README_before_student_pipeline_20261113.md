# Historical README: English publication summary

The original README described the first 14D Station Lite experiment in early November 2026. It is retained verbatim in the private local workspace; this English edition provides historical context without presenting obsolete commands as current instructions.

The original dataset contained 45 right-arm episodes with three RGB cameras at nominal 30Hz. The fixed split selected 35 training episodes and 10 diagnostic holdout episodes. Initial SmolVLA experiments retained both-arm state/actions, used a local pretrained base and progressed from short smoke tests through 2k/4k/10k/20k checkpoints. ACT supplied a working comparison. Offline losses and action errors did not establish safe or successful physical execution.

The original data contract required finite joint feedback, EEF action poses, gripper channels, relative image paths and stable task text. Episode-level splitting and versioned manifests were mandatory. The old target identifier could encode source layout rather than destination, so later metadata separated object identity, source region and placement target. Raw data was never to be overwritten.

The early README recommended auditing data, checking the GPU/environment, validating processors/features, running short training, evaluating fixed diagnostic episodes and checking observations/IK/limits before motion. It documented that public SO100/SO101 data could illustrate format and training workflows but could not supply Station Lite control labels.

The initial 14D workflow and its three-camera mapping have been superseded for the current demonstration by native right7, two physical cameras, calibrated geometry and guarded execution. Use [the current training pipeline](../TRAINING_PIPELINE.md) and [quickstart](../QUICKSTART_STATION_LITE.md). Historical loss comparisons and incomplete preflight states must not be read as the current deployment status.
