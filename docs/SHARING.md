# Sharing and publication

Reference remote: https://github.com/aiorbits/station-lite-SmolVLA . It is currently private. Repository/app authorization is separate from preparing a local export.

## English project release

Read the [README](../README.md), [quickstart](QUICKSTART_STATION_LITE.md), [training pipeline](TRAINING_PIPELINE.md) and [best practices](SMOLVLA_BEST_PRACTICES.md).

From the directory containing SmolVLA/:

```bash
python SmolVLA/prepare_shared_repository.py --profile public --output /tmp/SmolVLA_release_20261113
```

The output must not exist. RELEASE_FILES.txt selects the concise source/documentation release. English publication editions replace legacy handoff/archival notes in this profile; the original investigation history stays in the full local workspace. The exporter rejects Chinese characters in release files. It neither initializes Git nor commits, uploads or starts a robot.

EXPORT_MANIFEST.json records every exported file's size and SHA256. tools/verify_baseline.py separately verifies the approved source/checkpoint identity. Documentation translation changes byte hashes; the baseline manifest preserves the original pre-translation source records and records the English presentation edition. Runtime color compatibility is preserved through Unicode escapes.

Only the English public export is intended as the concise GitHub publication. The full profile uses SHARED_FILES.txt and retains historical material in its original language; it is not an English-only publication. A release-only checkout cannot export historical files it does not contain.

## External assets

Supply weights, recordings, images, caches and the proprietary SDK separately. Do not add credentials or raw data to Git. New9 original captures still require archiving. Green targets, repeated nine-cell coverage and same-color distractors are not established by its eight trials. A fresh-machine installation has not been certified.

The repository root represents the SmolVLA directory. Clone into a directory named SmolVLA when using the parent-relative historical scripts. Some scripts contain reference-host paths: consult LOCAL_RESOURCES.md before adapting them.

Upstream software, weights and the vendor SDK retain their respective licenses. This project has not selected an open-source license. Local preparation and synchronization do not publish to GitHub.
