# ST-OMR Training — Current Architecture Overlay

This file records the current active lane without replacing the repository's long-form closed-stage architecture history in `ARCHITECTURE.md`.

## Active pipeline

```text
Canonical ST Music
        ↓
Deterministic MusicXML writer + independent validators
        ↓
Verovio renderer
        ↓
Controlled degradation
        ↓
Stage 5 manifest + independent dataset validator
        ↓
Stage 6 hash-addressed persistence
        ↓
Synthetic Curriculum v1 export
        ↓
Stage 7-D0 canonical export-evidence gate       ✅ CLOSED
        ↓
Stage 7-D1 archive/corpus byte acceptance       ✅ CLOSED
        ↓
Stage 7-D2 full synthetic train + validation    ✅ CLOSED
        ↓
Stage 7-D3 validation error diagnostics         🔄 ACTIVE
        ↓
Targeted model/data improvement package         🔒 choose from D3 evidence
        ↓
Stage 9 sealed test benchmark/candidate gate    🔒 TEST SEALED
```

## Accepted D2 model

D2 completed 40 epochs / 12,320 optimizer steps on 1,230 TRAIN images and selected epoch 20 from 153 VALIDATION images. The accepted checkpoint is hash-bound outside Git.

The result proves that the training/evidence pipeline works and that validation loss improved, but exact-sequence accuracy remains `0.0` and token error rate remains approximately `0.8474`. Semantic and MusicXML regeneration validity are `1.0`. This is therefore a diagnostic baseline, not a production OMR model.

## D3 diagnostic boundary

```text
Accepted frozen Drive archive/corpus
        ↓
D1 re-verification (integrity only)
        ├── transport + manifest + build identities
        ├── every persisted artifact SHA-256
        └── TEST may be read only here for storage integrity
        ↓
D3 validation loader
        ├── TRAIN row → skip before artifact path/byte derivation
        ├── TEST row  → skip before artifact path/byte derivation
        └── VALIDATION → 51 families / 153 images
        ↓
Exact accepted D2 checkpoint + verification gate
        ↓
Frozen D2 greedy constrained decoder
        ↓
Per-validation-sample semantic comparison
        ├── token / exact sequence
        ├── measure / meter
        ├── event type / onset / duration
        ├── pitch / accidental
        ├── rest recognition
        └── chord size
        ↓
Feature buckets
        ├── meter
        ├── note/rest/chord
        ├── duration
        ├── chord size
        ├── accidental presence
        └── clean/light/medium degradation
        ↓
hash-addressed diagnostics + verification evidence
```

D3 performs **zero optimizer steps**. It cannot modify the checkpoint and cannot use TRAIN or TEST artifact bytes after D1.

## Split boundary

- Train: 410 families / 1,230 images — not exposed to D3 diagnostics after D1.
- Validation: 51 families / 153 images — D3 diagnostic surface.
- Test: 51 families / 153 images — sealed until Stage 9.

D1 may hash TEST bytes only for complete archive integrity and returns no test sample data. After D1 returns, D3 skips TRAIN and TEST before deriving any artifact path or reading any artifact byte.

## Decision boundary after D3

D3 does not itself choose or train a new model. Its accepted error map must identify the dominant failure mode before the next package is opened. Examples:

- pitch-dominant failure → visual encoder/feature extraction investigation;
- rhythm/duration-dominant failure → sequence/decoder or target-balance investigation;
- chord/rest-specific failure → targeted curriculum balancing;
- degradation-sensitive failure → image preprocessing/augmentation investigation;
- broad failure across all categories → architecture-capacity review before adding more data.

Only one small improvement axis should be changed at a time so that its effect can be measured on the same validation surface.

## Real-data lane

The existing Stage 8 real-data rights/provenance/privacy/intake/preparation architecture remains preserved and parked during D3. The 50-pair real pilot contract remains 40 train + 10 validation after admission; no real data is admitted merely by being present.

ScoreMosaic uploads and teacher corrections are not automatic training data. Online/automatic learning remains prohibited.
