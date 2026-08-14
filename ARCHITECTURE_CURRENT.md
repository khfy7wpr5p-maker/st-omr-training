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
Stage 7-D6 TRAIN/VAL specialist derivatives     ✅ CLOSED / TEST SEALED
        ↓
Stage 7-D7 StaffSet + StructureSet training     🔄 ACTIVE / TEST SEALED
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

The accepted D2 monolithic baseline learned enough to produce structurally valid semantic sequences, but recognition remained poor: exact-sequence accuracy `0.0` and TER about `0.8474`. D3 localized broad failures across pitch, duration, event type, chord size and event completeness. This rejected “more epochs” as the primary response and selected small musical tasks with specialist models.

## Specialist OMR graph

```text
PDF / score image
        ↓
page / system preparation
        ↓
StaffSet → Staff specialist
        ↓
StructureSet → Structure specialist
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

- **symbolic GT** comes from deterministic canonical music/MusicXML;
- **spatial GT** comes from pinned Verovio geometry and the exact accepted coordinate transform into final PNG pixels.

MusicXML alone is not a pixel-geometry source. AI never manufactures ground truth. Real spatial GT later requires independently human-verified annotation plus the existing rights/provenance/privacy admission gate.

## Closed D5 spatial boundary

D5 established deterministic StaffSet/StructureSet geometry with pinned Verovio 6.2.1 and exact final-PNG coordinate replay. It fails closed on ambiguous/missing geometry, unsupported transforms, non-five-line staffs, provenance mismatch or coordinate drift.

The final accepted geometry contract is `stage7d5-staff-structure-geometry-v2`. It excludes post-barline courtesy/anticipatory `meterSig` geometry from the current measure while retaining fail-closed behavior for ambiguous pre-barline meters. Rotated final PNGs use a two-point `barline_segment`, not a scalar x-coordinate.

## Closed D6 derivative layer

D6 applied the closed D5 geometry contract only to the frozen development surface:

```text
Frozen source manifest
        ↓
read split first
        ├── TEST → immediate skip
        ├── TRAIN       1,230 images / 410 families
        └── VALIDATION    153 images /  51 families
                    ↓
         canonical hash-addressed label sidecars
```

Accepted D6 evidence:

```text
derivative build ID      0faafe229f3497b1147cf0f0ac0ce4b7efe6fa31f360a6a33a3b82c986c8c519
manifest SHA-256          e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9
artifact binding SHA-256  3b7558f0f927ad47a61ed5afb5faa8584dca8647cf8683d4043686eb7b077ea1
receipt SHA-256           8fe85747b77f2282be3662f0c3d180a440c88028638bf1df7ddadfbb7650fff2
labels                     1,383
families                     461
TEST specialist records        0
```

D6 output contains no copied PNGs/MusicXML. Each sidecar references the frozen source PNG by SHA-256 and binds source lineage, renderer/degradation fingerprints, D5 geometry/transform versions and final-PNG StaffSet/StructureSet geometry.

## Active D7 specialist training

D7 is the first specialist model-training stage. It trains **two independent models** from the same accepted D6 development dataset; they do not share weights or an optimizer.

```text
D6 TRAIN sidecars + frozen PNGs
        ├── Staff specialist
        │      targets: staff_lines + staff_region
        │
        └── Structure specialist
               targets: system_region
                        measure_region
                        barline
                        clef_g2
                        meter_2_4 / meter_3_4 / meter_4_4
```

D7 rasterizes variable-length D6 geometry deterministically to fixed `96 × 512` dense targets. The frozen profile uses batch size 6, 8 epochs per specialist, AdamW, learning rate `0.0007`, weight decay `0.0001`, gradient clip `1.0`, and BCE-with-logits + soft Dice. Best checkpoints are selected independently by minimum validation loss.

Split authority is strict:

- TRAIN 1,230 images / 410 families: optimizer allowed;
- VALIDATION 153 images / 51 families: read-only model selection/metrics;
- TEST: no D7 record/path/byte access; sealed until Stage 9.

Authoritative D7 output stays outside normal Git and is hash-addressed (`checkpoint`, `metrics`, `verification`, `COMPLETE`). The checkpoint must reload with `weights_only=True` and reproduce exact Staff/Structure state hashes before a run can close.

## StaffSet / StructureSet surface

`StaffSet`:

- one or more graphical staff instances belonging to the one V1 logical staff;
- exactly five staff-line segments per graphical instance;
- staff bounding box and spacing;
- owning system id.

`StructureSet`:

- system bounding boxes;
- measure bounding boxes and canonical measure numbers;
- trailing barline segments;
- visible G2 clef boxes;
- visible meter boxes;
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

A learned absolute pitch prediction cannot override the deterministic resolver.

## Fusion authority

V1 fusion is deterministic. It must enforce supported G2/key/meter scope, exact measure duration totals, accidental scope, chord size 2–4, common chord onset/duration, explicit low-confidence/conflict handling, deterministic MusicXML writing and independent validation/round-trip. Hard-rule violations retain veto authority.

## V1 / deferred boundary

V1 remains single part, one logical staff, single voice, treble G2, key 0, meters `2/4|3/4|4/4`, whole/half/quarter/eighth notes, half/quarter/eighth rests, 2–4-note chords, and controlled sharp/flat/natural.

Deferred: multiple voices, grand staff, multiple instruments, cross-staff, tuplets, ties, slurs, dotted values, full-measure/multi-measure rests and non-zero key signatures.

## Real-data lane

Stage 8 rights/provenance/privacy/intake/preparation architecture remains preserved and parked while the synthetic specialist lane is established. ScoreMosaic uploads and teacher corrections are not automatic training data. Online/automatic learning remains prohibited.

See `STAGE7D7_STAFF_STRUCTURE_TRAINING.md` for the active D7 contract.
