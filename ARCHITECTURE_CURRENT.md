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
Stage 7-D8 Structure validation diagnostics     ✅ CLOSED / optimizer 0 / TEST SEALED
        ↓
Stage 7-D9 Structure refinement contract        🔄 ACTIVE / no training / TEST SEALED
        ↓
Local Structure derivative/training package     🔒 contract-dependent
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

D7 proved that specialist learning works, but also showed that one aggregate validation loss is not enough. Staff geometry learned strongly. Structure learned large/medium visual targets well—system regions, measure regions and G2 clefs—while barlines and meter glyphs remained weak.

D8 then reproduced the accepted D7 Structure baseline with zero optimizer steps and diagnosed the seven Structure channels on VALIDATION only. Threshold sweeps and 1–2 pixel tolerant metrics showed that barline/meter weakness is not primarily a threshold-calibration problem and is not explained by a small spatial offset. The weak targets occupy only about `0.00066–0.00104` of page pixels, pointing to sparse-object / representation pressure inside the shared low-resolution whole-page Structure segmentation model.

D9 therefore preserves the strong accepted D7 Structure core and decomposes only the weak sparse channels into local, higher-resolution specialists before any new optimizer run.

## Specialist OMR graph

```text
PDF / score image
        ↓
page / system preparation
        ↓
StaffSet → Staff specialist
        ↓
StructureSet → Structure internal decomposition
        │
        ├─ frozen Structure core → system / measure / G2
        ├─ barline refiner       → local measure-end ROI
        ├─ meter refiner         → local measure-start ROI
        └─ deterministic fusion  → unchanged StructureSet outputs
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

D7 is closed because the first specialist training package and evidence path succeeded; weak Structure channels were carried forward as unresolved development targets rather than silently accepted as production quality.

## Closed D8 diagnostic boundary

D8 performed **no training**. It bound the exact D7 checkpoint/metrics/verification bundle, safely reloaded the checkpoint with `weights_only=True`, reproduced the accepted Structure validation loss and all seven channel Dice values, and then performed additional diagnostics on VALIDATION.

Accepted D8 evidence:

```text
repository head           e0e721bf5a6d13025546fdf5eeb755647eef383f
report SHA-256            46de5f6766f78bb567f70794a364ccd44835d09af94ef29c3f1eab5cd13ce968
baseline validation loss  0.49127569106908947
TRAIN tensors             0
VALIDATION tensors        153 / 51 families
TEST records              0
optimizer steps           0
model mutation            false
TEST opened               false
```

Key D8 diagnostic results:

| channel | Dice@0.50 | best threshold | best Dice | threshold gain | tolerant F1 @2px |
|---|---:|---:|---:|---:|---:|
| system_region | 0.9304226398 | 0.25 | 0.9339324257 | 0.0035097860 | 0.9898584521 |
| measure_region | 0.8449312699 | 0.45 | 0.8455369242 | 0.0006056543 | 0.9793445289 |
| barline | 0.2736204205 | 0.35 | 0.2749698120 | 0.0013493914 | 0.3670878904 |
| clef_g2 | 0.8286431574 | 0.30 | 0.8298937531 | 0.0012505957 | 0.9441274383 |
| meter_2_4 | 0.3481060606 | 0.50 | 0.3481060606 | 0.0 | 0.3892495018 |
| meter_3_4 | 0.3528485803 | 0.70 | 0.3557100829 | 0.0028615027 | 0.4025582667 |
| meter_4_4 | 0.3103351169 | 0.55 | 0.3109830753 | 0.0006479584 | 0.3547076746 |

D8 rejected threshold-only correction and selected a local-specialist refinement path.

## Active D9 Structure refinement contract

D9 keeps the external D4 `StructureSet` inference boundary unchanged. The internal implementation is frozen as:

```text
accepted D7 Structure core
  ├─ system_region  → frozen
  ├─ measure_region → frozen
  └─ clef_g2        → frozen

barline_refiner
  ├─ measure-end crop
  ├─ conditioned on five staff lines + measure bbox
  └─ barline_segment + confidence

meter_refiner
  ├─ measure-start crop
  ├─ conditioned on five staff lines + measure bbox
  ├─ none | 2/4 | 3/4 | 4/4
  └─ meter_bbox + confidence

structure_fusion
  └─ deterministic fail-closed fusion
       ↓
unchanged external StructureSet outputs
```

### Frozen barline ROI

```text
anchor                     measure end
x before                    5.0 staff spacings
x after                     1.5 staff spacings
vertical margin             3.0 staff spacings above/below
model surface               192 × 128
resize                      fit/pad preserving aspect ratio
new trainable params cap    500,000
```

### Frozen meter ROI

```text
anchor                     measure start
x before                    0.5 staff spacing
x after                    12.0 staff spacings
vertical margin             3.0 staff spacings above/below
model surface               192 × 256
resize                      fit/pad preserving aspect ratio
classes                     none | 2/4 | 3/4 | 4/4
new trainable params cap    750,000
```

The explicit meter `none` class is required because a measure may carry an active meter semantically without displaying a new meter glyph. Courtesy meter geometry remains governed by the accepted D6 binding and must not be rebound to the wrong current measure.

### Frozen D9 pre-training gates

These thresholds are frozen before any future refiner optimizer run:

```text
TEST records                                  0
accepted D7 Structure core mutation           forbidden
new trainable parameters total                <= 1,250,000
barline strict Dice                           >= 0.500
barline tolerant F1 @2px                      >= 0.700
meter none|2/4|3/4|4/4 macro F1              >= 0.800
meter positive localization tolerant F1 @2px  >= 0.600
```

A failed refinement run cannot open TEST or change these gates after seeing results. It requires another development package.

D9 itself is declarative only. It contains no model, optimizer, trainer, checkpoint loader, dataloader or TEST evaluation path and authorizes no new training run.

## StaffSet / StructureSet surface

`StaffSet`:

- one or more graphical staff instances belonging to the one V1 logical staff;
- exactly five staff-line segments per graphical instance;
- staff bounding box and spacing;
- owning system id.

`StructureSet` external surface remains:

- system bounding boxes;
- measure bounding boxes and canonical measure numbers;
- trailing barline positions/segments;
- visible G2 clef candidate;
- visible meter candidate/class;
- confidence.

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

## Next gate

Close D9 contract/review/CI on one exact branch head. Only after D9 merge may the next package implement deterministic local ROI derivatives and bounded barline/meter refiner training under this frozen contract. TEST remains sealed.

See `STAGE7D9_STRUCTURE_REFINEMENT_CONTRACT.md` for the active D9 contract.
