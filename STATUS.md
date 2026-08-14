# ST-OMR Training Lab Status

This file is the current stage-status source for this repository. Detailed closed-stage architecture remains in `ARCHITECTURE.md`; the current active-lane overlay is in `ARCHITECTURE_CURRENT.md`.

## Current repository phase

Verified baseline before Stage 7-D6 work:

- `main`: `b1e0b5b67544627222b10bbe64a7c9ae3aebbb61`
- PR #42 — Stage 7-D5 StaffSet + StructureSet deterministic geometry pilot: MERGED
- D5 post-merge main CI: run #160 (`31806509396`) — SUCCESS
- D5 post-merge regression: 514/514 PASS
- Stage 7-D0 through Stage 7-D5: CLOSED

The current active lane is **Stage 7-D6 — TRAIN/VALIDATION StaffSet + StructureSet specialist derivative gate**. D6 performs no model training. It creates small canonical JSON labels that reference the existing frozen corpus PNGs by SHA-256; it does not duplicate the image corpus. TEST rows are skipped before specialist field/path/label derivation.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0–6 | Deterministic music → validated synthetic dataset pipeline | ✅ Closed / main CI verified |
| 7-A | Training contract freeze | ✅ Closed / main CI verified |
| 7-B | Tokenizer/data/model/trainer implementation | ✅ Closed / main CI verified |
| 7-C | Bounded baseline training + evidence | ✅ Closed / non-production baseline |
| 7-D0 | Synthetic Curriculum v1 export-evidence identity gate | ✅ Closed |
| 7-D1 | Synthetic corpus transport/byte/manifest acceptance | ✅ Closed |
| 7-D2 | Synthetic V1 train/validation execution | ✅ Closed / non-production baseline |
| 7-D3 | Validation-only semantic error diagnostics | ✅ Closed / specialist decomposition selected |
| 7-D4 | Specialist OMR architecture + GT/fusion contract | ✅ Closed |
| 7-D5 | StaffSet + StructureSet deterministic geometry pilot | ✅ Closed / PR #42 / main CI #160 PASS |
| 7-D6 | TRAIN/VALIDATION StaffSet + StructureSet derivatives | 🔄 Active — no training / TEST sealed |
| 8-0 | Real-data rights/provenance/fine-tuning contract | ✅ Closed / preserved |
| 8-1 | Real-data quarantine/intake + byte validation | ✅ Closed / preserved |
| 8-2 | Paired experiment profile | ✅ Closed / preserved |
| 8-3A | Real pilot preparation/admission components | ⏸ Parked during specialist synthetic work |
| 8-3B | Paired real train/validation execution | 🔒 Not started |
| 9 | Sealed benchmark and candidate decision | 🔒 Not started — TEST sealed |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Frozen Synthetic Curriculum v1

```text
source commit       adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90
config fingerprint  154bf1c3e6dfe4e6db096f8b668f29df0623cfd38352b89a04d295764c7458cb
build id            d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352
manifest SHA-256     44a963cd7dbc612fa29c2953ea8b2c8776d89ce470074e8f8b3fe25c6e165f34
transport SHA-256    4a9f3bb337ef99386081dff29c4c1fc3047dc3ada4db13c93b6254e680918e2b
families             512 = 410 train + 51 validation + 51 test
images               1536 = 1230 train + 153 validation + 153 test
targets              512 MusicXML
```

## Accepted Stage 7-D2 / D3 finding

D2 proved the monolithic execution path but not usable OMR recognition: best validation loss `0.9379074645594616`, exact sequence accuracy `0.0`, TER `0.847364947676063`, semantic/MusicXML validity `1.0`, TEST development exposure `0`.

D3 then diagnosed the accepted D2 checkpoint on 153 validation images with zero optimizer steps, zero TRAIN diagnostic exposure and zero TEST diagnostic exposure. Key results included pitch identity `0.0`, duration accuracy about `0.1842`, chord-size accuracy `0.0`, rest recognition about `0.6563`, and TER about `0.8474`. The error map therefore rejected a simple “more epochs” response and selected **specialist musical-task decomposition**.

## Accepted specialist architecture

D4 froze the V1 task graph:

```text
StaffSet      -> staff geometry
StructureSet  -> system / measure / barline / G2 / meter
NoteHeadSet   -> note-head center / bbox / fill
RestSet       -> supported rest glyph + duration
AccidentalSet -> sharp / flat / natural glyphs
RhythmSet     -> stem / beam / flag / duration
PitchSet      -> staff position only
ChordSet      -> 2–4 note vertical grouping
ContextSet    -> deterministic musical validation
```

Absolute pitch is not authoritative learned output in V1. It is resolved deterministically from `G2 + staff position + accidental state`.

Synthetic symbolic GT comes from canonical music. Synthetic spatial GT comes from pinned renderer geometry plus exact deterministic coordinate replay. Real spatial GT later requires independently human-verified annotation; an image+MusicXML pair alone is insufficient.

## Accepted Stage 7-D5 geometry proof

D5 established the synthetic spatial GT boundary using the same pinned Verovio 6.2.1 layout plus separately fingerprinted invisible bbox instrumentation. It extracts graphical staff instances, five staff-line segments, staff spacing, systems, measures, trailing barline segments, visible G2/meter boxes and canonical measure/meter binding.

D5 also corrected two geometry assumptions before closure:

- scalar `barline_x` was superseded by `barline_segment`, because final-PNG rotation can slant a barline;
- pinned Verovio drawing coordinates are resolved through the nested `class="definition-scale"` space plus ancestor transforms, rather than treating the outer SVG viewBox as drawing coordinates.

All D5 live golden/raster-equivalence and clean/light/medium coordinate-mapping tests passed before merge; PR #42 then merged and post-merge CI #160 passed 514/514 tests.

## Stage 7-D6 boundary

D6 materializes the accepted D5 geometry only for the development splits:

```text
TRAIN       410 families / 1,230 PNG → 1,230 label sidecars
VALIDATION   51 families /   153 PNG →   153 label sidecars
TEST         51 families /   153 PNG →     0 specialist labels
```

Each sidecar is canonical hash-addressed JSON and binds the exact source PNG SHA-256, source MusicXML/SVG lineage, renderer/degradation fingerprints and final-PNG-coordinate StaffSet/StructureSet geometry. PNGs and MusicXML files are not copied into the derivative set.

The builder reruns D1 whole-corpus integrity first. D1 may hash TEST artifacts only as frozen storage-integrity evidence. Once the D6 manifest is read, TEST is skipped immediately after `split`; no TEST specialist artifact path, image hash, geometry or label is derived.

Persisted D6 artifacts are independently reparsed and gated for canonical bytes, source/provenance binding, exact counts, family-exclusive split inheritance, no forbidden split, label hashes, five-line StaffSet structure, system/staff/measure cross-reference consistency and finite in-bounds final-PNG geometry.

See `STAGE7D6_SPECIALIST_DERIVATIVES.md` for the exact derivative contract.

## Safety boundaries

- No direct commits to `main`; changes use branch/PR packages.
- Large corpus/checkpoint artifacts stay outside normal Git.
- D6 contains no model trainer, optimizer, checkpoint loader or TEST evaluator.
- TEST specialist labels are not derived during development and TEST remains sealed until Stage 9.
- Existing Stage 8 rights/provenance/privacy/duplicate/leakage controls remain preserved and parked.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Real geometry labels require human-verified annotation and explicit admission.
- No online or automatic learning path is allowed.
- Deterministic validators retain veto authority over learned specialist candidates.

## Next gate

Complete D6 exact-head CI and independent review, then run the authoritative derivative build against the frozen Synthetic Curriculum v1 outside Git. D6 closes only if the persisted gate proves exactly **1,383 development labels / 461 families / TEST specialist records = 0**. No StaffSet/StructureSet model training begins before that gate closes.