# ST-OMR Training — Current Architecture Overlay

This file records the current active lane without replacing the repository's long-form closed-stage history in `ARCHITECTURE.md`.

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
Stage 7-D2 monolithic train + validation        ✅ CLOSED / diagnostic baseline
        ↓
Stage 7-D3 validation error diagnostics         ✅ CLOSED
        ↓
Stage 7-D4 specialist architecture contract     ✅ CLOSED
        ↓
Stage 7-D5 StaffSet + StructureSet geometry     ✅ CLOSED
        ↓
Stage 7-D6 TRAIN/VAL specialist derivatives     🔄 ACTIVE / TEST SEALED
        ↓
StaffSet + StructureSet specialist training     🔒
        ↓
NoteHead / Rest / Accidental specialists        🔒
        ↓
Rhythm / StaffPosition / Chord specialists      🔒
        ↓
Deterministic fusion + ContextSet validation    🔒
        ↓
Stage 9 sealed test benchmark/candidate gate    🔒 TEST SEALED
```

## Why the architecture changed

The accepted D2 monolithic baseline learned enough to produce structurally valid semantic sequences, but recognition remained poor: exact-sequence accuracy `0.0` and TER about `0.8474`. D3 localized broad failures across pitch, duration, event type, chord size and event completeness, while clean/light/medium derivatives were diagnostically identical family-by-family. This rejected “more epochs” as the primary response and selected small musical tasks with specialist models.

## Specialist OMR graph

```text
PDF / score image
        ↓
page / system preparation
        ↓
StaffSet
        ↓
StructureSet
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

Synthetic labels have two authorities:

- **symbolic GT** comes from the deterministic canonical music/MusicXML path;
- **spatial GT** comes from pinned Verovio geometry and the exact accepted coordinate transform into the final PNG.

MusicXML alone is not a pixel-geometry source. AI never manufactures GT.

For real images, a valid image+MusicXML pair is not enough for spatial specialist training. Staff lines, measure regions, note-head centers, accidental boxes, stem/beam geometry and spatial grouping later require independently human-verified annotation plus the existing rights/provenance/privacy admission gate.

## Closed D5 spatial boundary

D5 uses two renders of the same validated MusicXML under the same pinned Verovio 6.2.1 layout:

```text
validated MusicXML
      ├── normal frozen render ─────────────► training PNG
      └── same layout + invisible bbox instrumentation
                                      ↓
                             Verovio geometry
                                      ↓
                      staff/system/measure/barline
                         visible G2/meter boxes
                                      ↓
                      exact coordinate replay
                  SVG transforms → raster scale
                    → Pillow expand rotation
                                      ↓
                       final PNG pixel geometry
```

The geometry render is separately fingerprinted and does not change the frozen Stage-3 renderer configuration. D5 live tests prove the normal and instrumented renders preserve visible raster output.

Pinned Verovio drawing geometry is resolved through the nested `class="definition-scale"` coordinate space and supported ancestor transforms. D5 fails closed on ambiguous/missing geometry, non-five-line staffs, unsupported transforms, hash/provenance mismatch, canonical/SVG disagreement or coordinate drift.

A rotated final PNG can slant a barline, so D5 operationally supersedes D4's scalar `barline_x` with a two-point `barline_segment`.

## Active D6 derivative layer

D6 applies the closed D5 geometry contract to the frozen **development** surface only:

```text
Frozen source manifest
        ↓
read split first
        ├── TEST → immediate skip
        ├── TRAIN 1,230 images / 410 families
        └── VALIDATION 153 images / 51 families
                    ↓
            render geometry once per family
                    ↓
      replay each derivative's exact PNG transform
                    ↓
       canonical hash-addressed label sidecar
```

D6 output contains no copied PNGs or MusicXML files. Each of the 1,383 development sidecars references its already-frozen PNG by SHA-256 and binds:

- source sample/family/split/page identity;
- source PNG SHA-256 and dimensions;
- source MusicXML and normal SVG SHA-256 lineage;
- renderer/degradation fingerprints;
- D5 geometry/transform versions;
- final-PNG StaffSet and StructureSet geometry.

The independent D6 gate reparses persisted JSON and verifies exact development cardinality, family-exclusive split inheritance, canonical bytes, hash/provenance bindings, label-file exactness, five staff lines, system/staff/measure references, supported meter classes and finite in-bounds coordinates.

D1 remains allowed to hash the sealed TEST split only as whole-corpus storage integrity. D6 creates **zero TEST specialist records** and does not derive TEST image/path/geometry/label data after the split boundary.

## StaffSet / StructureSet surface

`StaffSet`:

- one or more graphical staff instances belonging to the one V1 logical staff;
- exactly five staff-line segments per graphical instance;
- staff bounding box;
- staff spacing;
- owning system id.

`StructureSet`:

- system id + bounding box;
- measure id + canonical measure number + content bounding box;
- trailing barline segment;
- visible G2 clef box when present;
- visible meter box when present;
- canonical `2/4|3/4|4/4` meter class.

## Pitch authority

The learned V1 pitch specialist remains spatial only:

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

A learned absolute pitch prediction cannot override this resolver.

## Fusion authority

V1 fusion is deterministic. It must enforce at minimum supported G2/key/meter scope, exact measure duration totals, accidental scope, chord size 2–4, common chord onset/duration, explicit low-confidence/conflict handling, existing deterministic MusicXML writing and independent validation/round-trip. Hard-rule violations retain veto authority.

## V1 / deferred boundary

V1 remains single part, one logical staff, single voice, treble G2, key 0, meters `2/4|3/4|4/4`, whole/half/quarter/eighth notes, half/quarter/eighth rests, 2–4-note chords, and controlled sharp/flat/natural. A rendered page may contain multiple graphical instances of the one logical staff when notation wraps across systems.

Still deferred: multiple voices, grand staff, multiple instruments, cross-staff, tuplets, ties, slurs, dotted values, full-measure/multi-measure rests and non-zero key signatures.

## Split boundary

- TRAIN: 410 families / 1,230 images — D6 may derive specialist labels.
- VALIDATION: 51 families / 153 images — D6 may derive specialist labels.
- TEST: 51 families / 153 images — sealed until Stage 9; D6 derives zero specialist labels.

Every specialist derivative inherits its source family's split. No specialist model training starts until D6's authoritative frozen-corpus build and independent gate are accepted.

## Real-data lane

Stage 8 rights/provenance/privacy/intake/preparation architecture remains preserved and parked while the synthetic specialist lane is established. ScoreMosaic uploads and teacher corrections are not automatic training data. Online/automatic learning remains prohibited.

See `STAGE7D6_SPECIALIST_DERIVATIVES.md` for the active derivative contract.