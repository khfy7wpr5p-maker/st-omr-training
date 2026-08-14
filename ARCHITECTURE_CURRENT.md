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
Stage 7-D9 Structure refinement contract        ✅ CLOSED / no training / TEST SEALED
        ↓
Stage 7-D10 local ROI derivatives               🔄 ACTIVE / optimizer 0 / TEST SEALED
        ↓
Barline + meter refiner training                🔒 D10 evidence-dependent
        ↓
NoteHead / Rest / Accidental specialists        🔒
        ↓
Rhythm / StaffPosition / Chord specialists      🔒
        ↓
Deterministic fusion + ContextSet validation    🔒
        ↓
Stage 9 sealed test benchmark/candidate gate    🔒 TEST SEALED
```

## Why Structure was decomposed

The accepted D2 monolithic baseline produced valid semantic sequences but recognition remained poor. D3 rejected a simple more-epochs response and selected specialist tasks.

D7 proved specialist learning could work: Staff geometry became strong and Structure learned `system_region`, `measure_region` and `clef_g2` well. The same whole-page Structure model remained weak on barlines and meter glyphs.

D8 reproduced that checkpoint on VALIDATION with zero optimizer steps. Threshold sweeps and 1–2 pixel tolerant metrics did not materially repair barline/meter performance. Those targets occupy only a tiny fraction of page pixels, identifying sparse-object / representation pressure rather than a simple threshold error.

D9 therefore froze a local higher-resolution refinement architecture before any new training. D10 now materializes that local data surface deterministically.

## Current Structure graph

```text
score image
    ↓
Staff specialist
    ↓
accepted D7 Structure core (FROZEN)
    ├─ system_region
    ├─ measure_region
    └─ clef_g2
    │
    ├─────────────── measure geometry + staff geometry ───────────────┐
    │                                                                 │
    ↓                                                                 ↓
D10 measure-end ROI                                           D10 measure-start ROI
192 x 128                                                     192 x 256
    ↓                                                                 ↓
future barline_refiner                                        future meter_refiner
barline_segment                                               none|2/4|3/4|4/4 + bbox
    └───────────────────────┬─────────────────────────────────────────┘
                            ↓
                 deterministic fail-closed fusion
                            ↓
                 unchanged external StructureSet
```

D10 creates only the two ROI derivative families shown above. The future refiner models do not exist in D10 and no optimizer is authorized by this stage.

## Ground-truth authority

Synthetic ground truth remains split into two deterministic authorities:

- **symbolic GT**: canonical ST music / deterministic MusicXML;
- **spatial GT**: pinned Verovio geometry replayed through the exact accepted final-PNG transform.

D10 does not manufacture new ground truth. It crops accepted D6 final-PNG spatial GT into the frozen D9 local coordinate systems.

Real geometry later still requires separately admitted, human-verified annotation. ScoreMosaic uploads and teacher corrections remain excluded from automatic training.

## Closed D6 source surface

D10 inherits the accepted D6 development surface through the verified D7 loader:

```text
TRAIN        1,230 images / 410 families
VALIDATION     153 images /  51 families
TEST             0 D6 specialist labels exposed to D10
```

Accepted D6 identity:

```text
derivative build ID      0faafe229f3497b1147cf0f0ac0ce4b7efe6fa31f360a6a33a3b82c986c8c519
manifest SHA-256          e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9
artifact binding SHA-256  3b7558f0f927ad47a61ed5afb5faa8584dca8647cf8683d4043686eb7b077ea1
receipt SHA-256           8fe85747b77f2282be3662f0c3d180a440c88028638bf1df7ddadfbb7650fff2
```

## Closed D7/D8 evidence that selected D9

D7 Structure result:

```text
system_region Dice   0.9304674680
measure_region Dice  0.8445145579
clef_g2 Dice         0.8228637141
barline Dice         0.2667824041
meter_2_4 Dice       0.3439848856
meter_3_4 Dice       0.3415115206
meter_4_4 Dice       0.3092358359
```

Accepted D8 report:

```text
repository head       e0e721bf5a6d13025546fdf5eeb755647eef383f
report SHA-256         46de5f6766f78bb567f70794a364ccd44835d09af94ef29c3f1eab5cd13ce968
barline Dice@0.50      0.2736204205
barline tolerant@2px   0.3670878904
meter tolerant@2px     about 0.35–0.40
optimizer steps        0
TEST records           0
```

## Closed D9 local refinement contract

### Barline ROI

```text
anchor                     measure end
x before                    5.0 staff spacings
x after                     1.5 staff spacings
vertical margin             3.0 staff spacings above/below
output                      192 x 128
resize                      aspect-preserving fit/pad
future parameter cap        500,000
```

### Meter ROI

```text
anchor                     measure start
x before                    0.5 staff spacing
x after                    12.0 staff spacings
vertical margin             3.0 staff spacings above/below
output                      192 x 256
resize                      aspect-preserving fit/pad
classes                     none | 2/4 | 3/4 | 4/4
future parameter cap        750,000
```

The explicit `none` class is essential: an active meter may continue semantically while no current-measure meter glyph is displayed. Accepted D6 courtesy-meter geometry remains authoritative.

Frozen future validation gates:

```text
TEST records                                  0
accepted D7 Structure core mutation           forbidden
new trainable params total                    <= 1,250,000
barline strict Dice                           >= 0.500
barline tolerant F1 @2px                      >= 0.700
meter 4-class macro F1                        >= 0.800
meter positive localization tolerant F1 @2px  >= 0.600
```

## Active D10 derivative gate

For every accepted source measure D10 deterministically derives:

```text
1 barline record
1 meter record
```

The source PNG and D6 sidecar are re-hashed before use. ROI transforms are computed in the native final-PNG coordinate system from accepted measure/staff geometry and staff spacing. The crop is fit/padded to the frozen local size, and the identical transform is replayed on target coordinates.

Each record identity binds:

```text
D10 version
D9 contract fingerprint
source sample id
source PNG SHA-256
accepted D6 label SHA-256
split
kind
measure number
ROI policy id
```

The authoritative runner rejects any source surface other than exactly `1230 TRAIN + 153 VALIDATION` with `410 + 51` family-exclusive groups.

### Persisted-output authority

D10 output remains outside Git and is not trusted merely because writes succeeded. Before `COMPLETE`, an independent verifier reopens the bundle and validates:

- manifest/receipt canonical bytes and hashes;
- exact source/family split cardinality;
- TEST=0 and optimizer=0;
- every ROI PNG SHA/mode/dimensions;
- every label SHA/schema/identity/source binding;
- record-id provenance recomputation;
- target and transform bounds;
- exactly one barline + one meter record per source measure;
- explicit meter class/bbox coherence;
- artifact-binding SHA;
- absence of unexpected persisted image/label files.

Only after that pass is `COMPLETE` emitted and the completed bundle verified again.

## Specialist OMR graph after Structure closes

```text
PDF / score image
        ↓
StaffSet
        ↓
StructureSet
        ↓
NoteHeadSet / RestSet / AccidentalSet
        ↓
RhythmSet / PitchSet(staff position) / ChordSet
        ↓
Deterministic association + pitch resolution
        ↓
ContextSet + hard musical validators
        ↓
Canonical candidate music
        ↓
Deterministic MusicXML writer + round-trip validation
```

Absolute pitch is still resolved deterministically from `G2 + staff position + accidental state`; a learned absolute pitch may not override that resolver.

## V1 boundary

V1 remains single part, one logical staff, single voice, treble G2, key 0, meters `2/4|3/4|4/4`, whole/half/quarter/eighth notes, half/quarter/eighth rests, 2–4-note chords, and controlled sharp/flat/natural.

Deferred: multiple voices, grand staff, multiple instruments, cross-staff, tuplets, ties, slurs, dotted values, full-measure/multi-measure rests and non-zero key signatures.

## Next gate

Finish D10 exact-head review/CI, then materialize the authoritative external D10 bundle from the accepted D6 development surface. Record its manifest/artifact-binding/receipt evidence with TEST=0 and optimizer=0. Only after D10 closes may a separate package implement and train the bounded barline/meter refiners.

See `STAGE7D10_LOCAL_ROI_DERIVATIVES.md` for the active D10 contract.