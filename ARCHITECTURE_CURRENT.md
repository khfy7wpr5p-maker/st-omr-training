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
Stage 7-D4 specialist architecture contract     ✅ CLOSED
        ↓
Stage 7-D5 StaffSet + StructureSet geometry     🔄 ACTIVE
        ↓
TRAIN/VALIDATION specialist derivative gate     🔒 NEXT
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

## Stage 7-D5 StaffSet + StructureSet geometry

D5 operationalizes the first spatial ground-truth boundary without training a model.

```text
validated canonical MusicXML
        │
        ├──────────────► frozen normal Verovio render ─► training PNG
        │
        └──────────────► same pinned layout
                         + invisible bbox instrumentation
                                  ↓
                         Verovio definition-scale SVG geometry
                                  ↓
                         system / measure / five staff lines
                         barline / clef / meter geometry
                                  ↓
                         canonical measure+meter binding
                                  ↓
                         deterministic coordinate replay
                         ├── SVG ancestor transforms
                         ├── CairoSVG uniform raster scale
                         ├── Pillow expand rotation
                         └── photometric transforms = no geometry change
                                  ↓
                         final PNG pixel coordinates
```

The secondary geometry render is separately fingerprinted. It adds only pinned Verovio `svgBoundingBoxes` and `svgContentBoundingBoxes`; it does not change the frozen base renderer configuration. Live tests require the normal render and bbox-instrumented render to rasterize to exactly the same clean image bytes.

Pinned Verovio 6.2.1 expresses drawing paths below a nested SVG with the **class token** `definition-scale`; the page-margin translation and any supported ancestor transform are replayed explicitly. D5 fails closed on missing/ambiguous geometry, non-five-line staffs, hash/provenance mismatch, unsupported transforms, canonical/SVG measure-count disagreement, or coordinate-space drift.

### D5 label surface

`StaffSet`:

- graphical staff instance id;
- owning system id;
- exactly five staff-line segments;
- staff-instance bounding box;
- staff spacing.

`StructureSet` pilot:

- system id + bounding box;
- measure id + canonical measure number + bounding box;
- trailing barline segment;
- visible G2 clef bounding box when present;
- visible meter bounding box when present;
- canonical meter class `2/4|3/4|4/4`.

D4's scalar `barline_x` is superseded operationally by `barline_segment`. This is a versioned D5 correction rather than a rewrite of the historical D4 fingerprint: rotation in final PNG space can make the barline slanted, so one scalar x-coordinate cannot be authoritative.

The pilot has no dataset split loader. Repository golden fixtures prove geometry extraction only; a later small derivative-builder package will apply the accepted geometry contract to TRAIN and VALIDATION while continuing to skip TEST before label/path derivation.

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

V1 remains single part, one logical staff, single voice, treble G2, key 0, meters `2/4|3/4|4/4`, whole/half/quarter/eighth notes, half/quarter/eighth rests, 2–4-note chords, and controlled sharp/flat/natural. A rendered page may contain multiple graphical instances of that one logical staff when the notation wraps across systems.

Still deferred: multiple voices, grand staff, multiple instruments, cross-staff, tuplets, ties, slurs, dotted values, full-measure/multi-measure rests, and non-zero key signatures.

## Split boundary

- Train: 410 families / 1,230 images — specialist development may derive labels only here.
- Validation: 51 families / 153 images — specialist selection/diagnostics may derive labels here.
- Test: 51 families / 153 images — sealed until Stage 9; no specialist label derivation during development.

Every specialist derivative inherits its source family's split. D1 whole-corpus hashing remains storage-integrity only and does not open TEST for model development.

D5 itself does not enumerate the frozen corpus and has no split loader. The next derivative-data package must explicitly reject/skip TEST before deriving any TEST path, SVG geometry, label, or image byte for specialist development.

## Real-data lane

The existing Stage 8 rights/provenance/privacy/intake/preparation architecture remains preserved and parked during specialist synthetic architecture work. Existing real image+MusicXML admission does not automatically satisfy specialist spatial-label requirements.

ScoreMosaic uploads and teacher corrections are not automatic training data. Any future correction must pass explicit permission/licensing/privacy/provenance/quality/split admission before it may become training data. Online/automatic learning remains prohibited.