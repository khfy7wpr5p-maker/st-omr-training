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
Stage 7-D1 archive/corpus byte acceptance       🔄 ACTIVE
        ↓
Stage 7-D2 train + validation only              🔒 LOCKED
        ↓
Stage 9 sealed test benchmark/candidate gate    🔒 TEST SEALED
```

## Current D1 boundary

```text
Frozen Drive archive
        ├── transport SHA-256 + size
        ↓
External fresh extraction
        ↓
manifest.json + manifest.sha256 + build.json
        ├── frozen build/config/manifest identities
        ├── family-exclusive split/count checks
        ↓
images/<png_sha256>.png + targets/<sha256>.musicxml
        ├── exact filename set
        ├── every artifact byte SHA-256
        ├── missing/extra/symlink veto
        ↓
small canonical hash-only D1 receipt
```

D1 has no model, tokenizer, trainer, decoder, optimizer, checkpoint, or metric dependency. It cannot make the corpus training-eligible by itself; it only proves that the copied bytes match the frozen corpus contract.

## Split boundary

- Train: 410 families / 1,230 images — future D2 parameter updates only.
- Validation: 51 families / 153 images — future D2 checkpoint selection/development metrics only.
- Test: 51 families / 153 images — sealed until the later benchmark decision.

D1 may hash all three splits solely to prove archive integrity. It does not return test artifacts to development code.

## Real-data lane

The existing Stage 8 real-data rights/provenance/privacy/intake/preparation architecture is preserved and parked. It is not expanded by Stage 7-D1/D2. The existing 50-pair pilot contract remains 40 train + 10 validation after admission; no real data is admitted merely by being present.

ScoreMosaic uploads and teacher corrections are not automatic training data. Online/automatic learning remains prohibited.
