# ST-OMR Training Lab Status

Updated: 2026-09-14

This is the current stage-status source. `ARCHITECTURE.md` preserves historical detail; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- latest merged package: PR #160 — TR-POLY-09B6 native multi-batch quality execution
- latest verified pre-B7 `main`: `429e64d790d758c8f99ad143e0a2a04c878c04bc`
- active package: TR-POLY-09B7 hash-bound quality VALIDATION execution
- frozen ≤2-step smoke trainer/checkpoint remain unchanged
- TEST: sealed
- ScoreMosaic / production authority: not granted

## Current stage status

| Stage / package | Description | Status |
|---|---|---|
| 0–6 | Deterministic symbolic → rendered → validated synthetic dataset | ✅ Closed / preserved |
| 7-A/B/C | Baseline model/training evidence | ✅ Historical baseline |
| 7-D | Specialist architecture/evidence | ✅ Historical evidence |
| TR-POLY-02 | Evaluation taxonomy + benchmark identity | ✅ Frozen |
| TR-POLY-03/04 | External-data registry + benchmark harness | ✅ Closed |
| TR-POLY-05/06 | V2 representation + tokenizer/roundtrip | ✅ Frozen |
| TR-POLY-07/08 | Registry + tiny 2D Transformer | ✅ Research implementation |
| TR-POLY-08A | ≤2-step deterministic smoke trainer | ✅ Frozen / preserved |
| TR-POLY-08B | ≤2-step exact research checkpoint | ✅ Frozen / preserved |
| TR-POLY-08C | Stage 6 V1→V2 execution | ✅ Single-voice evidence only |
| TR-POLY-09A | Native explicit V2 dataset/materialization | ✅ Merged |
| TR-POLY-09B1 | Free-running greedy inference | ✅ Merged |
| TR-POLY-09B2 | Deterministic V2 metric/adaptor layer | ✅ Merged |
| TR-POLY-09B3 | VALIDATION aggregation by voice/robustness | ✅ Merged |
| TR-POLY-09B4 | Multi-epoch TRAIN-only quality-training regime | ✅ Merged |
| TR-POLY-09B5 | Separate verified quality-checkpoint artifact + B1 bridge | ✅ Merged |
| TR-POLY-09B6 | Full native TRAIN/VALIDATION multi-batch quality execution | ✅ Merged / CI green |
| TR-POLY-09B7 | Exact descriptor-bound quality VALIDATION B1→B2→B3 execution | 🔄 Active package |
| First measured quality baseline | Real admitted B6 checkpoint + B7 descriptor manifest/run | 🔒 After B7 |
| Missing metric admission | Five frozen metrics remain unsupported | 🔒 Separate work |
| P09C | Evidence-driven refinement | 🔒 After measured quality evidence |
| P09D | Final candidate/evaluation freeze | 🔒 |
| Stage 9 | One-shot sealed TEST decision | 🔒 TEST sealed |
| Stage 10 | ScoreMosaic shadow/integration | 🔒 Not started |

## Executable quality path

```text
Native V2 TRAIN + VALIDATION build
        ↓
B6 deterministic full-population batching
        ↓
B4 multi-epoch TRAIN-only optimization
        ↓
B5 verified selected-state checkpoint
        ↓
B7 exact VALIDATION descriptor/artifact binding
        ↓
B1 free-running inference
        ↓
B2 per-sample metrics
        ↓
B3 overall + voice + robustness aggregation
```

B6 removed the old single-batch execution limitation without changing the frozen smoke path. Every model batch remains bounded to at most eight samples and semantic target truncation remains forbidden.

## B7 active contract

B7 adds the missing execution bridge between an actual B5 quality checkpoint and the already-frozen B1/B2/B3 measurement stack.

The benchmark population is the complete native V2 VALIDATION split. There is no benchmark `max_samples` or random/prefix sample selection.

Complexity and robustness labels are explicit `BenchmarkSampleDescriptor` metadata. B7 does not infer absent density values and does not default an unlabeled image to `clean`.

The split-manifest identity binds descriptor metadata to exact:

- sample/family identity;
- target and canonical representation SHA-256;
- image SHA-256;
- image dimensions;
- target token count;
- dataset manifest and build identity.

Changing metadata or bytes changes the benchmark identity.

## Checkpoint/evaluation safety

B7 requires:

- verified B5 checkpoint load before inference;
- checkpoint dataset identity == benchmark dataset identity;
- checkpoint preprocessing/materialization identity == current native V2 materialization;
- exact loaded model profile;
- one explicit bounded B1 decode limit for the whole run;
- one checkpoint/candidate identity across all sample reports;
- invalid/abstain predictions retained in B2/B3 denominators;
- TEST never accessed;
- production authority always false.

## Current metric surface

Numeric B2/B3 metrics:

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

Explicitly unsupported:

```text
musicxml_validity
tedn
notehead_stem_f1
beam_relation_f1
tie_relation_f1
```

No unsupported metric receives a proxy number. A B7 run may provide genuine partial VALIDATION evidence while the complete common-comparison/promotion gate remains closed.

## Required order from here

1. finish B7 tests/docs and exact-head CI;
2. merge B7 only if green;
3. freeze an admitted real/native V2 TRAIN/VALIDATION build and explicit descriptor manifest;
4. execute B6 to create the first real quality checkpoint;
5. execute B7 over the complete VALIDATION population;
6. inspect B2/B3 failures by voice/robustness and choose P09C from evidence;
7. admit the five missing metric implementations without proxies;
8. freeze P09D candidate/evaluation identity;
9. open sealed TEST once at Stage 9;
10. only then consider Stage 10 ScoreMosaic shadow integration.

## Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN alone changes model parameters.
- VALIDATION is read-only.
- Smoke contracts remain immutable.
- No semantic truncation is allowed.
- Invalid/abstain predictions stay visible.
- Unsupported metrics stay unsupported.
- External data still requires rights/license/install-pin admission.
- Teacher corrections and ScoreMosaic uploads are not automatic training data.
- No training or VALIDATION benchmark package grants production authority.
