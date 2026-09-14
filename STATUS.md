# ST-OMR Training Lab Status

Updated: 2026-09-14

This is the current stage-status source. `ARCHITECTURE.md` preserves historical detail; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- protected `main`: `58c27b5403301fe478c9b6c8351682c8f8bf1624`
- latest merged package: PR #155 — TR-POLY-09B1 bounded free-running V2 inference
- active package: TR-POLY-09B2 deterministic V2 metric/adaptor support
- next execution package: TR-POLY-09B3 common VALIDATION aggregation/reporting
- TEST: sealed
- ScoreMosaic / production authority: not granted

## Current stage status

| Stage / package | Description | Status |
|---|---|---|
| 0–6 | Deterministic symbolic → rendered → validated synthetic dataset pipeline | ✅ Closed / preserved |
| 7-A/B/C | Baseline tokenizer/model/trainer + bounded baseline evidence | ✅ Closed / historical baseline |
| 7-D specialist lane | Staff/structure/local specialist evidence | ✅ Historical evidence |
| 8-0/1/2 | Real-data rights, intake and paired-run contracts | ✅ Closed / preserved |
| 8-3 | Real pilot execution | ⏸ Not current priority |
| TR-POLY-02 | Evaluation taxonomy + benchmark identity | ✅ Closed / frozen contract |
| TR-POLY-03 | External dataset/license registry | ✅ Closed |
| TR-POLY-04 | Deterministic external benchmark harness | ✅ Closed |
| TR-POLY-05 | Polyphonic Representation V2 | ✅ Closed / frozen |
| TR-POLY-06 | V2 parser/tokenizer/lossless roundtrip | ✅ Closed |
| TR-POLY-07 | Research model registry | ✅ Closed |
| TR-POLY-08/08A/08B | 2D Transformer + bounded training + checkpoint contract | ✅ Implemented / research |
| TR-POLY-08C | Exact Stage 6 V1→V2 execution | ✅ Closed / single-voice evidence only |
| TR-POLY-09A | Native explicit V2 dataset/materialization | ✅ Merged |
| TR-POLY-09B1 | Free-running greedy inference + strict semantic validation | ✅ Merged / CI green |
| TR-POLY-09B2 | Deterministic V2 metric/adaptor support | ✅ Implemented in active package / merge gate pending |
| TR-POLY-09B3 | Common VALIDATION aggregation/report | 🔄 Next after B2 merge |
| Full TR-POLY-02 metric completeness | Includes MusicXML validity + TEDn | 🔒 Blocked until exact adapters admitted |
| P09C | Evidence-driven TRAIN/VALIDATION refinement | 🔒 After benchmark evidence |
| Stage 9 | Final sealed TEST decision | 🔒 TEST sealed |
| Stage 10 | ScoreMosaic shadow/integration | 🔒 Not started |

## B1 completed

PR #155 passed exact-head CI and merged at:

`58c27b5403301fe478c9b6c8351682c8f8bf1624`

The current candidate can now generate V2 tokens from BOS alone without gold-prefix teacher forcing, strictly parse EOS-complete output, expose invalid/abstain states and bind inference to exact model/checkpoint/provenance identity.

## B2 implemented metric surface

B2 currently implements deterministic numeric evidence for:

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
notehead_stem_f1
beam_relation_f1
tie_relation_f1
accidental_note_f1
note_staff_f1
```

Every report also carries all frozen TR-POLY-02 metric IDs. The two not yet safely implemented are retained as explicit unsupported observations:

```text
musicxml_validity  -> v2_musicxml_export_adapter_not_admitted
tedn               -> tedn_implementation_not_admitted
```

No zero, one, proxy, or placeholder value is substituted for either missing metric.

## B2 scoring behavior

- only VALIDATION descriptors are accepted;
- TEST descriptors are rejected;
- invalid/abstain B1 predictions are retained rather than filtered out;
- invalid output receives parse success 0 and zero semantic/relation credit while sequence metrics use the actual generated tokens;
- sequence edit distance is deterministic unit-cost Levenshtein;
- semantic fields use versioned deterministic event alignment;
- relation metrics use exact multiset F1 inside aligned events;
- report identity binds benchmark identity and inference evidence;
- a full required-metric record fails closed while any metric is unsupported.

## Current interpretation boundary

A B2/B3 report may answer questions such as:

- does the candidate parse successfully?;
- how large is token edit error?;
- are pitch, duration, onset, voice and staff correct?;
- which relation families fail?;
- which voice stratum is weakest?;

It may **not** yet claim a complete TR-POLY-02 benchmark winner because MusicXML validity and TEDn are not admitted.

## Next gate

1. exact-head CI and merge for TR-POLY-09B2;
2. implement TR-POLY-09B3 deterministic VALIDATION aggregation/reporting by voice stratum and robustness bucket;
3. preserve unsupported metric coverage explicitly;
4. separately admit MusicXML validity and TEDn before any complete winner/promotion claim;
5. keep TEST sealed throughout.
