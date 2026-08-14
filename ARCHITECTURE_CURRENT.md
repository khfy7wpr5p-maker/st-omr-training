# ST-OMR Training — Current Architecture Overlay

This file records the current active lane without replacing the repository's long-form closed-stage architecture history in `ARCHITECTURE.md`.

## Active pipeline

```text
Canonical ST Music
        ↓
Deterministic MusicXML writer + independent validators
        ↓
Pinned Verovio renderer
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
Stage 7-D3 validation error diagnostics         ✅ CLOSED
        ↓
Stage 7-D4 specialist architecture contract     🔄 ACTIVE
        ↓
StaffSet + StructureSet geometry pilot          🔒 NEXT
        ↓
NoteHead / Rest / Accidental specialists        🔒
        ↓
Rhythm / StaffPosition / Chord specialists      🔒
        ↓
Deterministic fusion + ContextSet validation    🔒
        ↓
Stage 9 sealed test benchmark/candidate gate    🔒 TEST SEALED
```

## Accepted D2/D3 evidence

D2 completed 40 epochs / 12,320 optimizer steps on 1,230 TRAIN images and selected epoch 20 from 153 VALIDATION images. The accepted checkpoint reduced validation loss but exact-sequence accuracy remained `0.0` and token error rate remained about `0.8474`; it is not a production OMR candidate.

D3 then diagnosed the accepted checkpoint on the same 153 validation images with zero optimizer steps, zero TRAIN diagnostic exposure and zero TEST diagnostic exposure. The error map showed broad failure across pitch, rhythm, event type, chord size and event completeness. Clean/light/medium variants were diagnostically identical family-by-family. D3 therefore selected specialist task decomposition rather than further monolithic training.

## Stage 7-D4 specialist architecture

```text
PDF / score image
        ↓
page/system preparation
        ↓
StaffSet / staff_geometry
        ↓
StructureSet / systems + measures + barlines + G2 + meter
        ↓
┌───────────────────────────────────────────────┐
│ NoteHeadSet      note-head center/bbox/fill  │
│ RestSet          supported rest glyphs       │
│ AccidentalSet    sharp/flat/natural glyphs   │
└───────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────┐
│ RhythmSet        stem/beam/flag/duration     │
│ PitchSet         discrete staff position     │
│ ChordSet         2–4 note vertical grouping │
└───────────────────────────────────────────────┘
        ↓
Deterministic association / pitch resolution
        ↓
ContextSet + hard musical validators
        ↓
Canonical candidate music
        ↓
Existing deterministic MusicXML writer
        ↓
Independent MusicXML validation / round-trip
        ↓
Candidate + confidence / veto
```

## Ground-truth authority

Synthetic labels are split into two classes:

- **symbolic** labels come from the existing canonical music object;
- **spatial** labels must come from pinned Verovio renderer geometry and be mapped through the exact raster/degradation geometry transform.

MusicXML alone is not a valid source of pixel geometry. If renderer elements cannot be linked reliably to canonical events, that specialist sample fails closed.

For real images, an admitted image+MusicXML pair is not sufficient for spatial specialist training. Staff lines, measure boxes, note-head centers, accidental boxes, stem/beam geometry and chord spatial grouping require independently human-verified annotation under a later explicit admission contract.

## Pitch authority

The learned V1 pitch specialist is spatial only:

```text
note-head + staff geometry
        ↓
staff_position + confidence
        ↓
DETERMINISTIC
G2 + staff position + accidental state
        ↓
canonical pitch
```

A learned absolute pitch prediction cannot override the deterministic resolver.

## Fusion authority

V1 fusion is deterministic, not learned. It must enforce at minimum:

- supported V1 G2/key/meter surface;
- exact measure duration totals;
- deterministic accidental scope;
- chord size 2–4;
- common chord onset and duration;
- explicit conflicts/low-confidence regions;
- veto on unsupported or ambiguous hard-rule violations;
- existing deterministic MusicXML writer and independent validation/round-trip.

## V1/deferred boundary

V1 remains single part, single staff, single voice, treble G2, key 0, meters `2/4|3/4|4/4`, whole/half/quarter/eighth notes, half/quarter/eighth rests, 2–4-note chords, and controlled sharp/flat/natural.

Still deferred: multiple voices, grand staff, multiple instruments, cross-staff, tuplets, ties, slurs, dotted values, full-measure/multi-measure rests, and non-zero key signatures.

## Split boundary

- Train: 410 families / 1,230 images — specialist development may derive labels only here.
- Validation: 51 families / 153 images — specialist selection/diagnostics may derive labels here.
- Test: 51 families / 153 images — sealed until Stage 9; no specialist label derivation during development.

Every specialist derivative inherits its source family's split. D1 whole-corpus hashing remains storage-integrity only and does not open TEST for model development.

## Real-data lane

The existing Stage 8 rights/provenance/privacy/intake/preparation architecture remains preserved and parked during specialist synthetic architecture work. Existing real image+MusicXML admission does not automatically satisfy specialist spatial-label requirements.

ScoreMosaic uploads and teacher corrections are not automatic training data. Any future correction must pass explicit permission/licensing/privacy/provenance/quality/split admission before it may become training data. Online/automatic learning remains prohibited.