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
Evidence-driven refinement                           🔒
        ↓
Candidate freeze + sealed TEST decision              🔒 TEST SEALED
        ↓
Separate ScoreMosaic shadow/integration              🔒
```

## TR-POLY-09B1 result

B1 closed the teacher-forcing gap. A verified image/checkpoint can now produce a prediction from BOS alone, reuse one 2D visual memory across the autoregressive loop, and terminate as canonical V2 or explicit failure/abstention. No gold target prefix is supplied and model state must not mutate.

PR #155 passed exact-head CI and merged to protected `main` at `58c27b5403301fe478c9b6c8351682c8f8bf1624`.

## TR-POLY-09B2 architecture delta

B2 maps one canonical V2 reference and one B1 free-running prediction into the frozen TR-POLY-02 metric vocabulary.

Implemented numeric metrics:

- `parse_success`;
- `ter`, `normalized_edit_distance`, `exact_sequence_accuracy`;
- `pitch_accuracy`, `duration_accuracy`, `onset_accuracy`, `voice_accuracy`, `staff_accuracy`;
- `notehead_stem_f1`, `beam_relation_f1`, `tie_relation_f1`, `accidental_note_f1`, `note_staff_f1`.

The adapter uses deterministic unit-cost sequence/event alignment, retains invalid/abstain predictions in evaluation, and binds every report to the benchmark identity, inference evidence, reference/prediction identity, voice stratum, robustness bucket and adapter versions.

### Explicitly unsupported metrics

Two frozen TR-POLY-02 metrics remain deliberately non-numeric:

- `musicxml_validity`: no admitted Polyphonic V2 → MusicXML benchmark export/validation adapter yet;
- `tedn`: no exact reviewed TEDn implementation is admitted yet.

They remain visible as `UNSUPPORTED` with explicit reasons. B2 therefore provides useful common metric evidence but **cannot produce a TR-POLY-02-complete metric record or winner claim**.

This fail-closed distinction is intentional. No surrogate TEDn or fabricated MusicXML validity score is permitted.

## Current benchmark readiness

After B2 is green on protected main, B3 may aggregate VALIDATION evidence by:

- `1_voice`;
- `2_voice`;
- `3_voice`;
- `4_plus_voice`;
- robustness bucket where admitted.

B3 must preserve invalid/abstain samples in denominators and expose metric coverage. A partial metric report may diagnose the model, but a complete winner/promotion decision remains closed until all required metric surfaces are admitted.

## Recommended order

```text
TR-POLY-09B2 metric/adaptor package
        ✅ implementation / CI+merge gate
        ↓
TR-POLY-09B3 common VALIDATION aggregation harness
        🔄 NEXT
        ↓
MusicXML-validity / TEDn admission packages as needed
        ↓
TR-POLY-02-complete comparable evidence
        ↓
P09C evidence-driven TRAIN/VALIDATION refinement
        ↓
P09D final candidate/evaluation freeze
        ↓
Stage 9 sealed TEST decision
        ↓
Stage 10 separate ScoreMosaic shadow gate
```

## Safety boundaries

- TEST remains sealed until Stage 9.
- TRAIN alone may update model parameters.
- VALIDATION evaluation must not mutate model state.
- Invalid/abstain outputs are evidence and must not be dropped from benchmark denominators.
- Unsupported metrics must not be guessed, approximated under the same metric ID, or filled with placeholder numbers.
- External datasets still require rights/license/install-pin admission.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Candidate artifacts and reports remain exact hash/provenance bound.
- Deterministic musical validators retain veto authority.

## Next gate

Merge TR-POLY-09B2 only after exact-head CI is green. Then implement TR-POLY-09B3 common VALIDATION aggregation/reporting. Do not open TEST and do not claim a complete benchmark winner while `musicxml_validity` or `tedn` remain unsupported.
