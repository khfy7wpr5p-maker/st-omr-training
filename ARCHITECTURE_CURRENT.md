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
Stage 7-D2 full synthetic train + validation    🔄 ACTIVE
        ↓
Stage 9 sealed test benchmark/candidate gate    🔒 TEST SEALED
```

## D2 execution boundary

```text
Accepted frozen Drive archive/corpus
        ↓
D1 re-verification (integrity only)
        ├── transport + manifest + build identities
        ├── every persisted artifact SHA-256
        └── TEST may be read only here for storage integrity
        ↓
D2 development loader
        ├── TEST row? → skip before artifact path/byte derivation
        ├── TRAIN → 410 families / 1230 images
        └── VALIDATION → 51 families / 153 images
        ↓
Frozen Stage 7-C model/trainer/tokenizer/preprocess
        ↓
40 epochs / batch 4 / 12,320 optimizer steps
        ↓
validation-selected best checkpoint
        ↓
validation decoding + semantic/MusicXML gates
        ↓
hash-addressed checkpoint + canonical metrics + strict reload verification
```

The model/trainer/preprocessing surface is intentionally held equal to Stage 7-C. D2 isolates the effect of the larger balanced Synthetic Curriculum v1 instead of mixing a data change with an architecture change.

## Split boundary

- Train: 410 families / 1,230 images — parameter updates only.
- Validation: 51 families / 153 images — checkpoint selection and development metrics only.
- Test: 51 families / 153 images — sealed until Stage 9.

D1 may hash TEST bytes only for complete archive integrity and returns no test sample data. After D1 returns, the D2 loader skips TEST before deriving any test artifact path or reading any test artifact byte.

## Evidence boundary

D2 output may retain externally:

- one best checkpoint;
- canonical metrics JSON;
- canonical authoritative verification JSON;
- small hashes/metrics suitable for durable repository documentation after independent verification.

Large checkpoint/corpus bytes do not enter normal Git.

## Real-data lane

The existing Stage 8 real-data rights/provenance/privacy/intake/preparation architecture is preserved and parked. It is not expanded by Stage 7-D2. The existing 50-pair pilot contract remains 40 train + 10 validation after admission; no real data is admitted merely by being present.

ScoreMosaic uploads and teacher corrections are not automatic training data. Online/automatic learning remains prohibited.
