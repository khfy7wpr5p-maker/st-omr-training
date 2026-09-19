# TR-POLY-10 — ScoreGraph V2 Additive Contract

Status: initial contract

Integration base: `main` (PR #175 already merged).

## Purpose

ScoreGraph V2 provides the parser-facing graph envelope required for real polyphonic MusicXML without replacing the already-frozen Polyphonic Representation V2 (PolyScore) or changing the V1 tokenizer/model surface.

The contract is additive:

- PolyScore remains the canonical musical core and keeps its existing canonical SHA-256 unchanged.
- ScoreGraph derives explicit Staff and Voice nodes from the core rather than duplicating them in a second mutable score model.
- source-order MusicXML cursor operations (backup / forward) are preserved as bounded navigation evidence.
- semantic relationships that are not frozen into PolyScore, such as slur and repeat-ending spans, can be represented as typed relations.
- adapter capability results are explicit: SUPPORTED, REVIEW_REQUIRED, or BLOCKED.

## Core node surface

The underlying PolyScore already carries Part, Measure, Event, explicit onset/duration, voice/staff identity, noteheads, pitch spelling, chord grouping, clef, key/time signatures, tie state, stem, beam, tuplet, grace-note evidence, cross-staff placement, and barline/repeat direction.

ScoreGraph deterministically derives Staff and Voice node views from that canonical core. This avoids creating a competing notation model or changing the existing V2 representation fingerprint.

## Source-navigation evidence

MusicXML backup and forward elements are parser-order operations rather than sounding score events. ScoreGraph preserves them in a separate ordered SourceNavigationEvent stream with kind, part id, measure index, source sequence index, and exact rational duration.

They may influence MusicXML-to-ScoreGraph reconstruction but do not mutate the canonical semantic onset stored in PolyScore.

## Relations

ScoreGraphRelation uses typed anchors and supports tie, slur, beam, tuplet, chord-membership, cross-staff, and repeat-ending relation families.

The first contract does not require adapters to emit duplicate relations for semantics already represented losslessly in PolyScore; it defines a stable place to carry relation evidence when a parser or later model needs it.

## Capability outcomes

- SUPPORTED — semantics are preserved by the active adapter/contract.
- REVIEW_REQUIRED — input is valid and bounded, but a semantic feature is not yet safely preserved. This is not a global lock.
- BLOCKED — reserved for safety, unparseable input, or boundedness failure.

BLOCKED therefore requires one of the explicit block reason classes: safety, unparseable, or boundedness. Unsupported musical semantics alone must not be promoted to BLOCKED.

## Canonical identity

ScoreGraphV2.canonical_sha256() fingerprints the additive envelope, including the core payload, navigation evidence, relations and capability findings.

ScoreGraphV2.core_sha256 remains exactly the frozen PolyScore.canonical_sha256(). An additive graph change therefore cannot silently rewrite the underlying V2 musical target.

## Fail-closed validation

The initial contract rejects unknown envelope versions, non-PolyScore cores, noncanonical source-navigation ordering, missing part/measure navigation references, duplicate navigation identities, missing relation anchors, duplicate/noncanonical relations, duplicate/noncanonical capability findings, and BLOCKED findings without an allowed block reason.

## Explicit non-goals

TR-POLY-10 does not yet parse MusicXML into ScoreGraph, normalize backup/forward into events, change V1 parsing/tokenization/checkpoints/metrics, change the frozen Polyphonic Representation V2 fingerprint, assign TRAIN/VALIDATION/TEST, read sealed TEST artifacts, train a model, or grant ScoreMosaic production authority.

The next package is the MusicXML -> ScoreGraph V2 adapter, driven by the diagnostic evidence from PR #175.
