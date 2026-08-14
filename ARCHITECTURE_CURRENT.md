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
Stage 7-D7 StaffSet + StructureSet training     ✅ CLOSED / TEST SEALED
        ↓
Stage 7-D8 Structure validation diagnostics     🔄 ACTIVE / optimizer 0 / TEST SEALED
        ↓
Structure refinement decision                   🔒 evidence-dependent
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

D7 then proved that specialist learning works, but it also showed why specialist results must be interpreted per task rather than accepted from one aggregate loss. Staff geometry learned strongly; Structure learned system/measure/clef well while barline and meter channels remained weak. D8 therefore diagnoses those exact channels before any refinement is selected.

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

D5 established deterministic StaffSet/StructureSet geometry with pinned Verovio 6.2.1 and exact final-PNG coordinate replay. The accepted contract is `stage7d5-staff-structure-geometry-v2`.

It excludes post-barline courtesy/anticipatory `meterSig` geometry from the current measure, retains fail-closed behavior for ambiguous pre-barline meters, and uses a two-point `barline_segment` for rotation-safe geometry.

## Closed D6 derivative layer

D6 applied the closed D5 geometry contract only to the frozen development surface:

```text
TRAIN       1,230 images / 410 families
VALIDATION    153 images /  51 families
TEST            0 specialist labels
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

## Closed D7 specialist training

D7 trained two independent dense-geometry models; they share neither weights nor optimizer state.

```text
Staff specialist
  staff_lines
  staff_region

Structure specialist
  system_region
  measure_region
  barline
  clef_g2
  meter_2_4
  meter_3_4
  meter_4_4
```

Frozen D7 profile:

```text
input                  96 × 512 grayscale
batch                   6
epochs                  8 per specialist
optimizer               AdamW
learning rate           0.0007
weight decay            0.0001
grad clip               1.0
objective               BCE-with-logits + soft Dice
checkpoint selection    minimum validation loss per specialist
TRAIN                   1,230 / optimizer allowed
VALIDATION                153 / read-only
TEST                        0 records
```

Authoritative D7 external evidence:

```text
run ID                 4ce2903206c7965471bb9569d379d8d9d1022d9248d80886638acfe0bd822598
checkpoint SHA-256     5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc
metrics SHA-256        43cd98a75c2db740b4af6ee3c8826122fa387347820d2e7d2c639ac2fe30f792
verification SHA-256   cdc0733af1bd6c7336f5bd2a0cb12fcae269120d8b5a9a564f08db860ee21a0a
TEST opened            false
```

Staff result:

```text
best validation loss  0.11952157418888348
staff_lines Dice      0.9216719705324906
staff_region Dice     0.9126690970017359
```

Structure result:

```text
best validation loss  0.49127569106908947
system_region Dice    0.93046746804164
measure_region Dice   0.8445145579484793
clef_g2 Dice          0.8228637140530807
barline Dice          0.2667824041384917
meter_2_4 Dice        0.34398488560691476
meter_3_4 Dice        0.34151152062874574
meter_4_4 Dice        0.3092358358777486
```

D7 is closed because the first specialist training package and evidence path succeeded; the weak Structure channels are explicitly carried forward as unresolved diagnostic questions, not silently accepted as production quality.

## Active D8 diagnostic boundary

D8 performs **no training**. It binds the exact D7 checkpoint/metrics/verification bundle, safely reloads the checkpoint with `weights_only=True`, reproduces the accepted Structure validation loss and all seven channel Dice values, and only then performs extra diagnostics on VALIDATION.

```text
TRAIN tensors          0
VALIDATION tensors   153 / 51 families
TEST records           0
optimizer steps         0
model mutation      false
```

D8 measures:

1. global probability threshold sweep from `0.05` to `0.95`;
2. exact threshold-`0.50` precision/recall/Dice;
3. deterministic per-channel best diagnostic threshold;
4. positive-record and positive-pixel prevalence;
5. mean probability separation on positive vs negative target pixels;
6. 1-pixel and 2-pixel tolerant localization F1 at threshold `0.50` and at each diagnostic best threshold.

This evidence is used to decide whether barline/meter weakness is primarily:

- threshold/calibration;
- thin-object near-miss localization;
- sparse-target/representation limitation;
- or a combination.

D8 itself does not change the accepted D7 loss, target masks, threshold, crop policy, model, epochs, or architecture.

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

See `STAGE7D8_STRUCTURE_DIAGNOSTICS.md` for the active D8 contract.
