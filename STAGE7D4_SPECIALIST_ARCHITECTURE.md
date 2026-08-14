# Stage 7-D4 — Specialist OMR Architecture Contract

Status: **active — architecture contract only; no training**.

Stage 7-D4 replaces the next-step assumption of a larger monolithic image-to-sequence model with a frozen V1 decomposition into small musical perception tasks, deterministic assembly, and independent musical validation.

D4 does **not** train a model, change the frozen Synthetic Curriculum v1 bytes, derive TEST specialist labels, ingest real data, or integrate anything into ScoreMosaic.

## Why D4 exists

The accepted Stage 7-D3 validation error map showed broad failure of the monolithic baseline rather than one narrow degradation problem:

```text
exact sequence accuracy      0.0000
token error rate             0.8473649477
measure exact accuracy       0.0049019608
meter accuracy               0.3014705882
event type accuracy          0.1799660441
onset accuracy               0.1494057725
duration accuracy            0.1842105263
pitch identity accuracy      0.0000
display accidental accuracy  0.0000
chord-size accuracy          0.0000
rest recognition accuracy    0.6563467492
missing events               1200
extra events                 267
```

Clean/light/medium derivatives were diagnostically identical family-by-family. The accepted direction is therefore **specialist musical-task decomposition**, not simply more epochs on the same monolithic model.

## Frozen V1 specialist pipeline

```text
PDF / score image
        ↓
page/system preparation
        ↓
┌───────────────────────────────────────────────────┐
│ VISUAL SPECIALISTS                                │
│                                                   │
│ StaffSet      → graphical staff instances         │
│ StructureSet  → system/measure/barline/G2/meter  │
│ NoteHeadSet   → note-head center/bbox/fill       │
│ RestSet       → supported rest glyph/duration    │
│ AccidentalSet → sharp/flat/natural candidates    │
│ RhythmSet     → stem/beam/flag/duration          │
│ PitchSet      → discrete staff position only     │
│ ChordSet      → 2–4 note vertical grouping       │
└───────────────────────────────────────────────────┘
        ↓
Deterministic association + pitch resolution
        ↓
ContextSet / deterministic musical validation
        ├── measure duration exactly filled?
        ├── G2 + staff position + accidental state → pitch?
        ├── accidental scope coherent?
        ├── chord members share onset/duration?
        ├── supported V1 meter/clef/key only?
        ├── unsupported/ambiguous region?
        └── low-confidence specialist conflict?
        ↓
Canonical candidate music
        ↓
Existing deterministic MusicXML writer
        ↓
Existing independent MusicXML validation + round-trip
        ↓
Candidate + confidence / veto
```

## Logical staff versus graphical staff instances

V1 contains **one logical staff per part**, but a rendered page may contain that same logical staff more than once because notation can wrap into multiple systems. Therefore StaffSet does not assume one staff rectangle per page.

```text
one logical V1 staff
        ↓
render layout
        ↓
one or more graphical staff instances
        ↓
StructureSet groups instances into systems/measures
```

Every graphical staff instance must have its own bounded identity, bbox, five line geometries, spacing, and coordinate lineage.

## Important pitch boundary

The V1 **Pitch specialist does not authoritatively predict C4/D5/etc.** Its learned responsibility is spatial:

```text
note-head center + owning staff instance
        ↓
staff_position specialist
        ↓
discrete staff position + confidence
        ↓
DETERMINISTIC resolver
G2 clef + staff position + measure accidental state
        ↓
canonical pitch
```

This prevents a learned model from silently overriding deterministic notation rules. The canonical pitch stored in `PitchSet` is an audit/reference label, not the authoritative output of the specialist.

## Ground-truth policy

Ground truth is never created by AI.

### Synthetic specialist sets

Two different ground-truth classes are required.

**Symbolic labels** come from the existing canonical music object, for example:

- meter;
- event type;
- onset;
- duration;
- displayed accidental;
- canonical pitch audit;
- chord membership and size;
- measure capacity/context rules.

**Spatial labels** cannot be guessed from MusicXML. They must come from the pinned renderer geometry and be transformed deterministically into the exact raster/degraded image coordinate system.

```text
Canonical music
      ↓
Deterministic MusicXML
      ↓
Pinned Verovio SVG
      ↓
fail-closed geometry extractor
      ↓
SVG-space labels
      ↓
exact raster scale / rotation-expand transform
      ↓
PNG-space labels
```

If a rendered glyph cannot be linked reliably to its canonical event where an event link is required, the specialist training sample is rejected. No heuristic or learned model may fabricate the missing label.

Photometric degradation (blur, brightness, contrast, noise, JPEG round-trip) does not change geometry. Rotation with expansion does change coordinates and therefore must have an explicit replayable geometry transform bound to the derivative identity.

### Machine-readable geometry policy

The D4 contract freezes these geometry rules:

- synthetic source coordinate space: pinned Verovio SVG;
- training coordinate space: final PNG pixels;
- clean raster mapping must use the exact SVG viewBox → CairoSVG raster dimensions;
- rotation must replay the exact Pillow `rotate(..., expand=True)` geometry;
- blur/brightness/contrast/noise/JPEG do not rewrite coordinates;
- every geometry transform must be fingerprinted;
- ambiguous or unlinked required renderer geometry rejects that specialist sample.

### Real specialist sets

An admitted real image + MusicXML pair is **not sufficient by itself** for spatial specialist training. Real spatial labels require independently human-verified annotation.

Current Stage 8 admission controls remain necessary but do not automatically create:

- staff-line coordinates;
- system/measure boxes;
- note-head boxes/centers;
- accidental boxes;
- stem/beam geometry;
- note-to-chord spatial grouping.

Those labels require a later explicit annotation/admission contract before real specialist fine-tuning.

## Dataset contracts

### StaffSet

Purpose: graphical staff geometry foundation for the one logical V1 staff across one or more rendered systems.

V1 labels per graphical staff instance:

- staff instance identity;
- staff instance bounding box;
- five staff lines;
- staff spacing.

### StructureSet

Purpose: page/system/measure segmentation and supported notation header structure.

V1 labels:

- system identity and bounding box;
- measure identity and bounding box;
- barline positions;
- G2 clef bounding box/presence;
- meter bounding box and class (`2/4`, `3/4`, `4/4`).

### NoteHeadSet

Purpose: note-head detection independent of pitch.

V1 labels:

- note-head bbox;
- note-head center;
- open/filled appearance;
- canonical event identity for audit.

### RhythmSet

Purpose: supported note duration perception.

V1 labels:

- note-head identity;
- stem presence/direction;
- beam count;
- flag count;
- duration class.

### RestSet

Purpose: supported V1 rest recognition.

V1 labels:

- rest bbox;
- rest class;
- duration class;
- canonical event identity.

### AccidentalSet

Purpose: visible accidental glyph recognition.

V1 labels:

- accidental bbox;
- `sharp | flat | natural`;
- canonical event identity for audit.

The specialist detects a glyph; deterministic association and accidental-scope validation decide its musical effect.

### PitchSet

Purpose: note-head-to-owning-staff spatial position.

V1 labels:

- note-head identity;
- discrete staff position;
- canonical pitch audit value.

The trainable output is staff position, not authoritative absolute pitch.

### ChordSet

Purpose: vertical grouping for supported V1 2–4-note chords.

V1 labels:

- note-head identity;
- chord group identity;
- chord size;
- shared-onset audit value.

### ContextSet

Purpose: deterministic whole-measure/context validation, not a trainable V1 context model.

V1 labels/rules:

- measure capacity;
- event onsets/durations;
- accidental scope;
- chord onsets;
- MusicXML validity.

## Dependency order

```text
staff_geometry
    ↓
structure
    ├──────────────┬──────────────┐
    ↓              ↓              ↓
notehead          rest          header/meter
    ├──────────────┬──────────────┐
    ↓              ↓              ↓
rhythm       staff_position   accidental
    └──────────────┬──────────────┘
                   ↓
             chord_grouping
                   ↓
          context_validation
```

The machine-readable contract performs an explicit cycle check.

## V1 musical scope remains frozen

Supported in this architecture contract:

- one part;
- one **logical** staff per part, with one or more graphical staff instances across rendered systems;
- one voice;
- treble clef G2;
- key signature 0;
- `2/4`, `3/4`, `4/4`;
- whole/half/quarter/eighth notes;
- half/quarter/eighth rests;
- 2–4-note chords;
- controlled sharp/flat/natural.

Still deferred and forbidden from silently entering V1 specialist targets:

- multiple voices;
- grand staff;
- multiple instruments;
- cross-staff;
- tuplets;
- ties;
- slurs;
- dotted values;
- full-measure/multi-measure rests;
- non-zero key signatures.

## Split safety

Every specialist derivative inherits the source music family's existing split.

```text
TRAIN       410 families → specialist training development
VALIDATION   51 families → specialist selection/diagnostics
TEST         51 families → SEALED until Stage 9
```

During development, D4 forbids derivation of TEST specialist labels. Whole-corpus byte hashing by the existing D1 storage-integrity gate remains a separate integrity operation and does not make TEST available to specialist development.

## Fusion policy

V1 fusion remains deterministic.

It must:

1. associate specialist candidates using bounded geometry and IDs;
2. resolve pitch from G2 + staff position + accidental state;
3. enforce exact measure capacity;
4. enforce chord size `2..4` and common onset/duration;
5. reject unsupported V1 structures;
6. expose conflicts and low-confidence regions instead of inventing certainty;
7. produce canonical candidate music only after all hard validators pass;
8. use the existing deterministic MusicXML writer and independent validation/round-trip layers.

A future learned ranking model may only rank already-valid candidates under a separately approved contract; it may not replace hard validity rules.

## Teacher correction / ScoreMosaic boundary

ScoreMosaic uploads and teacher corrections are not automatic training data.

A correction can enter a future specialist training corpus only after a separate explicit admission path verifies, as applicable:

- permission;
- licensing/rights;
- privacy;
- provenance;
- annotation quality;
- family/split leakage controls;
- specialist label completeness.

No online or automatic learning is introduced.

## D4 implementation boundary

D4 adds only:

- this architecture document;
- a machine-readable frozen specialist contract;
- D3→D4 decision provenance binding;
- contract invariant tests;
- current status/architecture synchronization.

D4 adds no model weights, optimizer, trainer, specialist dataset bytes, checkpoint, TEST evaluation, or ScoreMosaic integration.

## Closure gate

Stage 7-D4 may close only after:

1. the specialist task list and V1/deferred surface are machine-readable and frozen;
2. synthetic versus real ground-truth provenance rules are explicit;
3. logical-staff versus graphical-staff-instance semantics are explicit;
4. deterministic raster/degradation geometry authority is explicit;
5. deterministic pitch/fusion authority is explicit;
6. split/test/teacher-data boundaries are preserved;
7. contract-focused tests pass;
8. full repository regression + compile checks pass on the exact PR head;
9. independent review finds no P1/P2 contract defect;
10. explicit user merge approval is obtained;
11. post-merge exact-main CI succeeds.

## Next implementation package after D4

The first specialist implementation package should be **StaffSet + StructureSet ground-truth extraction contract/pilot**, because every later visual specialist depends on reliable staff and measure geometry.

That package must prove that pinned Verovio SVG elements can be linked to canonical music and transformed into clean/light/medium raster coordinates deterministically before any specialist model is trained.