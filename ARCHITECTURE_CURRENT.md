# ST-OMR Training — Current Architecture Overlay

Updated: 2026-09-14

This file records the active architecture lane. `ARCHITECTURE.md` remains the long-form historical record; `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` contains the detailed roadmap.

## Current baseline

- protected `main`: `58c27b5403301fe478c9b6c8351682c8f8bf1624`
- latest merged package: PR #155 — TR-POLY-09B1 bounded free-running V2 inference
- active package: TR-POLY-09B2 deterministic V2 metric/adaptor support
- TEST: sealed
- ScoreMosaic / production authority: not granted

## Active pipeline

```text
Polyphonic Representation V2                         ✅ FROZEN
        ↓
V2 parser / tokenizer / lossless roundtrip          ✅
        ↓
Tiny 2D Transformer + bounded trainer               ✅ RESEARCH
        ↓
Exact checkpoint persistence/reload                  ✅
        ↓
Native explicit Polyphonic V2 TRAIN/VALIDATION      ✅ TR-POLY-09A
        ↓
BOS-only free-running greedy inference               ✅ TR-POLY-09B1
        ↓
Strict V2 parse / explicit invalid-abstain evidence ✅ TR-POLY-09B1
        ↓
Deterministic metric/adaptor surface                 ✅ IMPLEMENTED — TR-POLY-09B2
        ↓
Common VALIDATION aggregation/execution              🔄 NEXT — TR-POLY-09B3
        ↓
Exact missing-metric admission                       🔒
        ↓
Evidence-driven refinement                           🔒
        ↓
Candidate freeze + sealed TEST decision              🔒 TEST SEALED
        ↓
Separate ScoreMosaic shadow/integration              🔒
```

## B1 completed

TR-POLY-09B1 closed the teacher-forcing gap. A verified image/checkpoint can now produce tokens from BOS alone, reuse one 2D visual memory during decoding, and terminate as canonical V2 or explicit invalid/abstain evidence. PR #155 passed exact-head CI and merged at `58c27b5403301fe478c9b6c8351682c8f8bf1624`.

## B2 exact metric surface

B2 maps one canonical V2 reference and one B1 free-running prediction into every frozen TR-POLY-02 metric ID without inventing missing evidence.

### 11 available numeric metrics

```text
parse_success
ter
normalized_edit_distance
exact_sequence_accuracy
pitch_accuracy
duration_accuracy
onset_accuracy
voice_accuracy
staff_accuracy
accidental_note_f1
note_staff_f1
```

### 5 explicit unsupported metrics

```text
musicxml_validity  -> V2→MusicXML benchmark adapter not admitted
tedn               -> exact TEDn implementation not admitted
notehead_stem_f1   -> explicit notehead↔stem relation not represented in V2
beam_relation_f1   -> explicit cross-event beam relation not represented in V2
tie_relation_f1    -> explicit cross-event tie relation not represented in V2
```

V2 does contain stem direction, beam marks and tie START/STOP state, but B2 does **not** silently redefine those states as the stronger frozen relation metrics.

## Scoring behavior

- only VALIDATION descriptors are admitted;
- TRAIN and TEST descriptors are rejected;
- invalid/abstain B1 outputs remain in evaluation;
- invalid output receives parse success 0, sequence error from its actual generated tokens, and zero credit on available semantic/relation metrics;
- unsupported metrics remain unsupported rather than becoming zero;
- sequence edit distance is deterministic unit-cost Levenshtein;
- semantic fields use versioned deterministic event alignment;
- `accidental_note_f1` and `note_staff_f1` use exact relation identities represented in V2;
- report identity binds benchmark, inference evidence, reference/prediction identity, voice stratum, robustness bucket and adapter versions.

## Benchmark interpretation boundary

B2/B3 may provide useful partial common VALIDATION evidence, but a full TR-POLY-02 result is still closed.

```text
partial diagnostic benchmark evidence   ✅ permitted
11 numeric frozen metrics                ✅ available
5 explicit unsupported metrics           ⚠️ visible
complete TR-POLY-02 metric record         🔒 unavailable
winner / promotion claim                  🔒 prohibited
sealed TEST                               🔒 closed
```

## Recommended order

```text
TR-POLY-09B2 exact-head CI + merge
        ↓
TR-POLY-09B3 deterministic VALIDATION aggregation
        ↓
independent admission of missing metric surfaces
        ↓
TR-POLY-02-complete comparable evidence
        ↓
P09C evidence-driven TRAIN/VALIDATION refinement
        ↓
P09D candidate/evaluation recipe freeze
        ↓
Stage 9 one-shot sealed TEST decision
        ↓
Stage 10 separate ScoreMosaic shadow gate
```

## Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN alone may update parameters.
- VALIDATION evaluation is read-only.
- Invalid/abstain outputs must not be removed from denominators.
- Unsupported metrics must not receive proxy or placeholder numbers under frozen metric IDs.
- External data still requires rights/license/install-pin admission.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Candidate artifacts and metric reports remain hash/provenance bound.
- Deterministic musical validators retain veto authority.

## Next gate

Merge TR-POLY-09B2 only after exact-head CI is green. Then implement TR-POLY-09B3 common VALIDATION aggregation/reporting by voice stratum and robustness bucket. Do not open TEST or declare a complete benchmark winner while any frozen metric remains unsupported.
