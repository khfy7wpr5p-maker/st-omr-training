# Current Architecture Delta — Stage 8-3A

> Historical/frozen stage delta. This document preserves the Stage 8-3A architecture as it was defined for that package. It is no longer the repository's current operational-status authority. For current merged + shadow/experimental state, use `ARCHITECTURE_CURRENT.md`; for current stage summary, use `STATUS.md`.

Status: **historical Stage 8-3A architecture snapshot; contract semantics preserved**.

This document is a narrow stage-specific supplement to `ARCHITECTURE.md`. Its Stage 8 status labels describe the original Stage 8-3A package state and must not override the current architecture overlay.

## Current gate map

```text
Stage 0–8-2  closed / exact-main CI verified
        ↓
Stage 8-3A    ACTIVE — pilot preparation + admission only
        ↓
Stage 8-3B    LOCKED — paired real train/validation execution
        ↓
Stage 9       LOCKED — sealed benchmark / candidate decision
        ↓
Stage 10      LOCKED — ScoreMosaic candidate integration
```

Exact Stage 8-3A starting `main` is `99a32ef917ba4ba5c72ef6a537c24c90a1b0c47f`, the Stage 8-2 closure synchronization from PR #32. Exact-main run #104 (`31712294207`) succeeded.

## Stage 8-3A architecture

```text
external exact source PNG bytes
        ↓
immutable source identity / SHA-256
        ↓
Stage 8-3A deterministic source-PNG preparation
        ├── source mode only 1 / L / P
        ├── single frame
        ├── no transparency / alpha
        ├── no crop
        ├── no resize
        ├── no rotation / orientation rewrite
        ├── pinned Pillow 12.3.0
        ├── deterministic conversion to mode L
        ├── metadata stripped by new L image construction
        └── fixed PNG encoding + independent output verification
        ↓
8-bit grayscale Stage 8-1 training PNG bytes
        +
hash-only preparation evidence
        +
supported-V1 ground-truth MusicXML
        +
rights / provenance / pairing evidence
        ↓
Stage 8-0 admission contract
        ↓
Stage 8-1 exact-byte + semantic/token validation
        ↓
hash-only receipt
        ↓
family / exact duplicate / near-duplicate / leakage vetoes
        ↓
exactly 50 admitted development pairs
        ↓
family-exclusive 40 train / 10 validation handoff
        ↓
Stage 8-3B may become eligible
```

## Auxiliary symbolic package boundary

External PrIMuS-style packages may contain PNG, MEI, semantic, agnostic, MIDI, and PAE files. Stage 8-3A may use MEI/semantic/agnostic only for bounded **triage evidence** such as parseability, header coherence, and obvious V1 exclusion checks.

They are not automatically trusted labels. In particular:

- MEI is not silently promoted to MusicXML;
- semantic/agnostic annotations do not establish admission;
- MIDI/PAE are not first-pilot target formats;
- parser success is not pairing approval;
- no auxiliary file bypasses rights/provenance review;
- unsupported notation is rejected rather than approximated or silently dropped.

A future supported-V1 auxiliary adapter, if added, must terminate in the existing canonical ST music model, pass the independent canonical validator, serialize through the deterministic MusicXML writer, and then pass the complete Stage 8-1 MusicXML/XSD/semantic/token round-trip gate.

## Frozen V1 boundary remains unchanged

Current V1 remains:

- one part / one staff;
- treble clef G2;
- one voice;
- key signature 0;
- meter 2/4, 3/4, or 4/4;
- whole/half/quarter/eighth notes where metrically valid;
- half/quarter/eighth rests;
- single notes and 2–4-note chords;
- frozen sharp/flat/natural accidental intent.

Deferred features remain outside this pilot, including explicit beams, ties/slurs, tuplets, multiple voices, piano grand staff, cross-staff, Guitar TAB, full/multi-measure rest semantics, unsupported meters/clefs/key signatures, and sixteenth-or-shorter durations.

## Hard safety boundaries

Stage 8-3A:

- commits no real score/image/symbolic corpus bytes to Git;
- performs no model/checkpoint loading;
- performs no optimizer step or training/fine-tuning;
- does not open or enumerate either sealed test split;
- does not integrate with ScoreMosaic;
- does not enable online or automatic learning;
- does not change the frozen Stage 8-2 50-pair / 40+10 paired experiment profile.

Stage 8-3A remains open until exactly 50 pairs pass the full Stage 8-0/8-1 admission boundary and the exact family-exclusive 40/10 hash-bound handoff is proven. The current implementation package freezes preparation and triage only; it does not itself close Stage 8-3A.
