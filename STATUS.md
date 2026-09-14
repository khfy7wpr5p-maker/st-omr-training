# ST-OMR Training Lab Status

Updated: 2026-09-14

This is the current stage-status source. `ARCHITECTURE.md` preserves historical detail; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- protected `main`: `58c27b5403301fe478c9b6c8351682c8f8bf1624`
- latest merged package: PR #155 — TR-POLY-09B1 bounded free-running V2 inference
- active package: TR-POLY-09B2 deterministic V2 metric/adaptor support
- next package: TR-POLY-09B3 common VALIDATION aggregation/reporting
- TEST: sealed
- ScoreMosaic / production authority: not granted

## Current stage status

| Stage / package | Description | Status |
|---|---|---|
| 0–6 | Deterministic symbolic → rendered → validated synthetic dataset | ✅ Closed / preserved |
| 7-A/B/C | Baseline model/training evidence | ✅ Historical baseline |
| 7-D | Specialist architecture/evidence | ✅ Historical evidence |
| 8-0/1/2 | Real-data rights/intake/run contracts | ✅ Preserved |
| 8-3 | Real pilot execution | ⏸ Not current priority |
| TR-POLY-02 | Evaluation taxonomy + benchmark identity | ✅ Closed / frozen |
| TR-POLY-03/04 | External-data registry + benchmark harness | ✅ Closed |
| TR-POLY-05/06 | V2 representation + tokenizer/roundtrip | ✅ Closed / frozen |
| TR-POLY-07/08/08A/08B | Registry + 2D model + trainer + checkpoint | ✅ Research implementation |
| TR-POLY-08C | Stage 6 V1→V2 execution | ✅ Single-voice evidence only |
| TR-POLY-09A | Native explicit V2 dataset/materialization | ✅ Merged |
| TR-POLY-09B1 | Free-running greedy inference | ✅ Merged / CI green |
| TR-POLY-09B2 | Deterministic V2 metric/adaptor layer | ✅ Implemented / merge gate pending |
| TR-POLY-09B3 | Common VALIDATION aggregation/report | 🔄 Next |
| Missing metric admission | 5 frozen metrics remain unsupported | 🔒 Separate packages required |
| P09C | Evidence-driven refinement | 🔒 After VALIDATION evidence |
| Stage 9 | Final sealed TEST decision | 🔒 TEST sealed |
| Stage 10 | ScoreMosaic shadow/integration | 🔒 Not started |

## B1 completed

PR #155 passed exact-head CI and merged at `58c27b5403301fe478c9b6c8351682c8f8bf1624`. The 2D candidate can now produce strict free-running V2 prediction evidence without gold-prefix teacher forcing.

## B2 metric coverage

### Numeric and admitted in B2 — 11

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

### Explicitly unsupported — 5

```text
musicxml_validity
  v2_musicxml_export_adapter_not_admitted

tedn
  tedn_implementation_not_admitted

notehead_stem_f1
  explicit_notehead_stem_relation_not_represented_in_v2

beam_relation_f1
  explicit_cross_event_beam_relation_not_represented_in_v2

tie_relation_f1
  explicit_cross_event_tie_relation_not_represented_in_v2
```

B2 does not reinterpret event-level stem direction, beam state, or tie START/STOP state as the stronger explicit relation metrics.

## B2 safety behavior

- VALIDATION descriptors only;
- TRAIN/TEST descriptors rejected;
- invalid/abstain predictions remain in reports;
- actual generated tokens determine sequence error;
- available semantic/relation metrics receive zero credit for invalid output;
- unsupported metrics remain non-numeric;
- deterministic report fingerprint binds benchmark identity and B1 inference evidence;
- full TR-POLY-02 result gate refuses admission while any metric is unsupported.

## Current interpretation

B2 and B3 can diagnose sequence, pitch, duration, onset, voice, staff, accidental association and staff association quality. They cannot yet support a complete frozen-benchmark winner/promotion claim.

## Next gate

1. exact-head CI and merge for B2;
2. B3 aggregation by 1/2/3/4+ voice strata and robustness bucket;
3. preserve unsupported coverage and invalid outputs explicitly;
4. independently admit the five missing metric surfaces before a complete TR-POLY-02 winner claim;
5. keep TEST sealed throughout.
