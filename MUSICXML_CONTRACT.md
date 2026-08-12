# ST-OMR MusicXML Contract

## Status

This document freezes the Stage 2-A MusicXML boundary for ST-OMR Training Lab before any MusicXML writer implementation begins.

Stage 1 has produced and independently validated the canonical in-memory ST music model. MusicXML is a deterministic serialization of that model; it is not the primary musical source of truth.

Changing this contract later requires an explicit architecture decision and regression review.

## 1. Pinned MusicXML version

ST-OMR V1 targets **MusicXML 4.0**, the current published final MusicXML specification at the time of this contract.

MusicXML 4.1 is under active roadmap work but is not a final release at this point. ST-OMR must not auto-upgrade when a later MusicXML version appears. Any version change requires a separate compatibility package and explicit approval.

V1 root format:

```xml
<score-partwise version="4.0">
```

`score-timewise` is out of scope.

V1 initially emits uncompressed XML (`.musicxml` / XML bytes). Compressed `.mxl` packaging is deferred.

MusicXML 4.0's XSD has no MusicXML default namespace. The V1 writer must not invent one.

## 2. Canonical document shape

V1 contains one part only.

```text
score-partwise version="4.0"
├── part-list
│   └── score-part id="P1"
│       └── part-name
└── part id="P1"
    ├── measure number="1"
    ├── measure number="2"
    └── ...
```

The `score-part` ID and `part` IDREF must match exactly.

The V1 canonical part ID is `P1`.

The V1 canonical part name is `ST-OMR Synthetic`.

No MIDI device, playback, layout, credit, tempo, fingering, Guitar TAB, or application-specific extension elements are emitted in V1.

## 3. Measure attributes

The first measure must emit an `<attributes>` element containing:

```text
divisions
key/fifths = 0
time/beats + beat-type
clef/sign = G
clef/line = 2
```

The key signature and clef are fixed for V1 and do not need to be repeated in later measures.

If the time signature changes between generated measures, the new `<time>` value must be emitted at the start of the measure in which the change becomes active.

V1 does not use pickup/anacrusis measures, hidden measure durations, multiple staves, transposition, or mid-measure attribute changes.

## 4. Exact divisions policy

The canonical ST model stores duration in exact whole-note fractions. MusicXML `<divisions>` is measured **per quarter note**.

The writer must never use floating-point arithmetic to convert duration.

For each distinct event duration `d` represented as a reduced whole-note `Fraction`, compute:

```text
quarter_units = 4 * d
```

Reduce `quarter_units` and collect its denominator. The score-wide MusicXML divisions value is:

```text
divisions = LCM(all quarter_units denominators)
```

This is the smallest positive integer that makes every supported V1 event duration an exact integer MusicXML `<duration>` value.

For the current V1 duration set (whole, half, quarter, eighth), the resulting value is normally:

```text
divisions = 2
```

The writer must compute the value from the score rather than hard-code `2`, so the rule remains auditable if the supported duration set is deliberately extended later.

A single divisions value is used for the entire V1 score and is emitted in the first measure. Changing divisions mid-score is out of scope.

Because V1 has one contiguous voice and explicit rests, event onsets are represented by sequential note/rest durations. `<backup>` and `<forward>` are not emitted.

## 5. Note mapping

Every canonical `NoteEvent` becomes one MusicXML `<note>`.

Pitch mapping:

```text
Pitch.step   -> pitch/step
Pitch.alter  -> pitch/alter when non-zero
Pitch.octave -> pitch/octave
```

For an unaltered pitch (`alter == 0`), canonical V1 serialization omits `<alter>`.

Every normal V1 note also emits:

```text
duration
voice = 1
type = whole | half | quarter | eighth
staff = 1
```

Element ordering must follow the MusicXML 4.0 XSD order.

Dotted notes, tuplets, ties, slurs, beams, grace notes, cue notes, microtones, lyrics, stems, notehead overrides, and performance attack/release offsets are out of scope.

## 6. Rest mapping

Every canonical `RestEvent` becomes a MusicXML `<note>` containing `<rest/>` rather than `<pitch>`.

It emits the same exact duration, voice, note type, and staff fields as a pitched event.

V1 rhythmic rests are eighth, quarter, and half rests only.

A whole-duration `RestEvent` must not be used as a proxy for a full-measure rest. Full-measure-rest notation remains deferred.

## 7. Chord mapping

A canonical `ChordEvent` remains a first-class internal event, but MusicXML serializes it as consecutive `<note>` elements.

For a chord with N member notes:

- the first member is emitted as a normal pitched `<note>` without `<chord/>`;
- each subsequent member includes `<chord/>` before its `<pitch>` element;
- every member carries the same duration, voice, type, and staff values;
- all members preserve their own pitch spelling and accidental display intent.

The duration of the first member advances MusicXML musical position. Chord-member notes containing `<chord/>` do not advance the position.

The internal chord member ordering is preserved exactly; the writer must not reorder chord tones for cosmetic reasons.

## 8. Accidental and spelling mapping

Pitch semantics and visible accidental semantics remain separate.

Canonical mapping:

```text
Pitch alter +1 -> <alter>1</alter>
Pitch alter -1 -> <alter>-1</alter>
Pitch alter  0 -> omit <alter>
```

Visible notation intent maps independently:

```text
display_accidental NONE    -> omit <accidental>
display_accidental SHARP   -> <accidental>sharp</accidental>
display_accidental FLAT    -> <accidental>flat</accidental>
display_accidental NATURAL -> <accidental>natural</accidental>
```

An explicit natural therefore may have no `<alter>` element while still carrying `<accidental>natural</accidental>`.

The writer must never enharmonically normalize spelling. For example, C-sharp and D-flat remain different canonical targets.

The Stage 1 independent validator remains authoritative for accidental/alter coherence before serialization.

## 9. XML construction and deterministic bytes

Stage 2-B must construct XML with Python's standard-library `xml.etree.ElementTree` API.

Raw XML string concatenation is prohibited.

V1 serialization rules:

- UTF-8 output;
- XML declaration included;
- no pretty-printing dependency;
- stable element creation order;
- stable attribute insertion order;
- no runtime timestamp or environment-dependent metadata;
- no network access;
- no external entity expansion;
- no DTD dependency.

For the same canonical `Score` and the same writer version, the produced MusicXML bytes and SHA-256 digest must be identical within the supported runtime contract.

The MusicXML hash is a derived-artifact hash. It does not replace the canonical model/content identity.

## 10. Schema validation strategy

The schema authority for V1 is the official **W3C MusicXML 4.0 XSD set**, including `musicxml.xsd` and its required local schema resources/catalog.

Validation must be offline and reproducible:

- tests and validation must not download schemas from the network;
- the exact schema asset set must be version-pinned and SHA-256 recorded;
- schema imports must resolve only to approved local assets;
- remote DTD/entity resolution is prohibited.

Stage 2-A adds no dependency and vendors no schema file yet.

Stage 2-C will introduce the independent schema-validation adapter. The intended validation engine is an XSD-capable parser such as `lxml.etree.XMLSchema`, but adding and pinning that dependency requires its own explicit dependency/license/security review before it enters the repository.

If the approved XSD validation engine is unavailable, Stage 2-C must fail closed. XML well-formedness alone must never be reported as MusicXML schema validation.

Schema validity is necessary but not sufficient. A separate ST MusicXML semantic validator must also verify the V1 project rules independently from the writer.

## 11. Independent MusicXML semantic validation

Stage 2-C must independently verify at least:

- root is `score-partwise` and version is `4.0`;
- exactly one `score-part` / one `part`, with matching `P1` identity;
- sequential measure numbers;
- legal 2/4, 3/4, and 4/4 time signatures;
- key signature `fifths = 0`;
- treble clef G2;
- positive integer divisions;
- every duration maps exactly to the supported canonical rational value;
- voice 1 and staff 1 only;
- supported note/rest types only;
- no unsupported V1 elements silently accepted;
- chord continuation semantics are valid;
- pitch step/alter/octave is valid;
- explicit accidental display remains coherent with pitch alteration;
- total musical duration exactly fills each measure.

The validator must parse and recompute these facts. It must not trust values produced by the writer merely because the writer emitted them.

## 12. Golden MusicXML fixtures

Stage 2-B/2-C verification must include small human-reviewable golden outputs for at least:

- quarter note;
- half rest;
- 2/4 measure;
- 3/4 measure;
- 4/4 measure;
- 2-note chord;
- 3-note chord;
- 4-note chord;
- sharp;
- flat;
- context-controlled natural;
- time-signature change across consecutive measures.

Golden fixtures must be synthetic and small. They are test artifacts, not training data.

## 13. Round-trip boundary

Stage 2-D is a **supported-V1 semantic round-trip verifier**, not a general-purpose MusicXML importer.

Target verification path:

```text
Canonical Score
    ↓ Stage 2-B writer
MusicXML 4.0 bytes
    ↓ Stage 2-C schema + semantic validation
Validated MusicXML
    ↓ Stage 2-D limited parser
Canonical V1 semantic projection
    ↓ compare
Original canonical semantic projection
```

The round-trip comparison must include:

- part/measure/voice/staff structure;
- measure numbering;
- time signatures;
- key signature;
- clef;
- exact event onset and duration;
- event type (note/rest/chord);
- chord membership and member order;
- pitch step, alter, and octave;
- visible accidental intent.

V1 round-trip equivalence does **not** require reconstruction of generator-only provenance such as `seed`, config fingerprint, generator version, source ID, or score ID unless a later contract explicitly serializes those fields.

The parser must reject unsupported constructs rather than silently normalize them into V1.

Stage 3 renderer integration must not start until the Stage 2 writer, independent validation, golden fixtures, and supported-V1 semantic round-trip gate are complete.

## 14. Frozen Stage 2 decomposition

```text
Stage 2-A  MusicXML contract freeze                 ← this package
Stage 2-B  Deterministic MusicXML 4.0 writer
Stage 2-C  Offline XSD + independent semantic validator
Stage 2-D  Supported-V1 semantic round-trip verifier
Stage 3    Renderer integration
```

Stage 2-B, 2-C, and 2-D require separate implementation packages. This contract package contains no MusicXML implementation code, schema dependency, renderer, dataset, or model-training work.
