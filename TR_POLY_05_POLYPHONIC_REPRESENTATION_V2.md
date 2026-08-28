# TR-POLY-05 — Polyphonic Representation V2

Status: structured representation contract

## Purpose

TR-POLY-05 freezes the semantic target surface that later parsers, tokenizers and model families must share.

The frozen Stage 7 V1 tokenizer remains untouched. V2 is additive.

## Time model

All timing uses exact rational fractions of a whole note:

- whole = `1/1`
- half = `1/2`
- quarter = `1/4`
- eighth = `1/8`

Onset is explicit. It is not reconstructed from output order.

This is required for real polyphony because multiple independent voices may legitimately begin at exactly the same onset.

## Chord versus polyphony

V2 makes this distinction structural:

- a **chord** is one `PolyEvent`, one logical voice, one onset, with two or more `NoteAtom` noteheads;
- **polyphony** is two or more independently voiced `PolyEvent` values, which may share the same onset;
- serialization order never decides whether notes form a chord.

This prevents the V1 failure mode where simultaneous structure is flattened into a single sequence without explicit voice identity.

## Staff and cross-staff semantics

Every event has both:

- `voice` — logical musical voice;
- `staff` — current visual staff placement.

A voice can move from staff 1 to staff 2 across events without changing voice identity.

Each notehead can additionally carry `staff_override`. This supports cross-staff chords in which individual noteheads are rendered on a different staff while remaining members of one logical chord event.

## Preserved semantic surface

The required V2 surface includes:

- measure identity;
- staff;
- voice;
- pitch spelling;
- exact duration;
- exact onset;
- rests;
- explicit chord grouping;
- clef;
- key signature;
- time signature, including additive meters;
- display accidental;
- beams;
- ties;
- tuplets;
- grace notes;
- barlines;
- cross-staff placement.

## Pitch and visible accidental

`PitchSpelling` stores sounding/spelled pitch fields separately from visible accidental intent. The V2 surface supports natural, sharp, flat, double-sharp and double-flat display values.

Each notehead owns its own pitch, accidental and tie state, including noteheads inside chords.

## Rhythm and notation

A pitched event keeps both:

- exact semantic duration as an `ExactRational`;
- visible note type (`whole`, `half`, `quarter`, `eighth`, through MusicXML-style 1024th values, plus breve/long/maxima).

Tuplet ratio/boundary and beam level/state are explicit metadata.

Grace events use zero semantic duration in this contract. Their visual note type and slash state remain explicit. This avoids inventing playback or steal-time semantics that are not visually proven.

## Measure contract

A measure carries:

- sequential canonical `measure_index`;
- original/source measure number text;
- time signature;
- key signature;
- per-staff clef assignments;
- canonical events;
- barline evidence.

Non-grace events may not overrun the exact time-signature capacity.

Events must already be in canonical order:

`(onset, voice, staff, event_id)`

The representation rejects rather than silently reorders non-canonical input. This keeps transformations observable and reproducible.

## Identity and reproducibility

Event ids are globally unique within a score.

Note-atom ids are globally unique within a score.

`PolyScore.canonical_json()` produces deterministic canonical JSON and `canonical_sha256()` provides a representation fingerprint suitable for later parser/tokenizer/checkpoint binding.

## Fail-closed boundaries

The contract rejects:

- unsupported representation versions;
- invalid staff/voice indices;
- negative rational time;
- zero duration on non-grace events;
- positive semantic duration on grace events;
- note events without exactly one notehead;
- chord events with fewer than two noteheads;
- rests containing noteheads;
- duplicate event/notehead identities;
- duplicate beam levels or tuplet numbers on one event;
- invalid staff references;
- non-canonical event ordering;
- events beyond measure capacity.

## Explicit non-goals

TR-POLY-05 does not:

- parse MusicXML;
- parse or copy OLiMPiC LMX;
- serialize V2 into training tokens;
- change the V1 tokenizer or its vocabulary;
- train a model;
- create a checkpoint;
- open sealed TEST data;
- change ScoreMosaic behavior.

Those boundaries keep this package a representation decision rather than an accidental architecture/model migration.

## Relationship to OLiMPiC LMX

OLiMPiC LMX is an important reference: it preserves MusicXML-oriented voice/staff changes, backups/forwards, beams, ties, tuplets and grace notation in a compact sequence.

ST-OMR V2 intentionally differs at the semantic layer by keeping onset, voice and staff explicit in the canonical structured object. This makes the representation independent of source-specific MusicXML element ordering and provides one common target for:

- image-to-sequence models;
- detection + relation graph models;
- specialist ensemble + deterministic composer models.

TR-POLY-06 will define deterministic V2 parsing/serialization and roundtrip behavior on top of this frozen object model.
