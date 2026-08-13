# ST-OMR Synthetic Curriculum v1

## Status

This is an auxiliary synthetic-data package built on the already-closed Stage 1-6 pipeline. It does not reopen Stage 6 and it does not authorize real-data training. Stage 8-3A remains the active real-data gate and is simply parked while this synthetic corpus is prepared.

## Purpose

The accepted Stage 7-C baseline used only 64 synthetic families. Synthetic Curriculum v1 expands the same validated V1 symbolic surface into a larger, deterministic corpus before returning to real-data admission.

The corpus is intended for controlled synthetic pretraining/diagnostics only. A later training run requires its own frozen run profile and evidence gate.

## Frozen profile

```text
dataset name          st-omr-synthetic-curriculum-v1
profile version       st-synthetic-curriculum-v1
families              512
seed range            100000..100511
split seed            8001
measures/family       8
raster width          1000
family profiles       mixed, note-only, rest-only, chord-only,
                      time-2-4, time-3-4, time-4-4, no-accidentals
degradation profiles  clean, light, medium
```

The 512 families divide deterministically into:

```text
train       410 families
validation   51 families
test         51 families
```

Each of the eight symbolic family profiles receives exactly 64 families before split assignment.

## Invariants

- Canonical symbolic ST music remains ground truth.
- MusicXML is generated only through the existing deterministic writer and validators.
- Rendering uses the pinned Verovio boundary.
- Degradation is appearance-only and never changes the symbolic target.
- All derivatives from one family stay in one split.
- The independent Stage 5 manifest validator remains authoritative.
- Duplicate target/image artifacts fail closed.
- Bulk PNG/MusicXML corpus bytes stay outside normal Git content.
- No real/user data, PrIMuS data, ScoreMosaic uploads, teacher corrections, checkpoint loading, optimizer steps, or Stage 9 test access occur in this package.

## Persistence

The existing Stage 6 hash-addressed writer is used without modification:

```text
<dataset-root>/
├── manifest.json
├── manifest.sha256
├── build.json
├── targets/
│   └── <musicxml_sha256>.musicxml
└── images/
    └── <png_sha256>.png
```

The output directory must be fresh and outside the Git repository.

## Acceptance gate

Before this corpus is used by any training run:

1. exact profile fingerprint must match;
2. the deterministic 512-family plan must match;
3. complete generation must pass the existing independent validators;
4. persisted bytes must pass the Stage 6 hash-addressed writer verification;
5. corpus identity (`build_id`, manifest SHA-256, config fingerprint) must be recorded;
6. training must remain a separate package with a separate frozen run profile.
