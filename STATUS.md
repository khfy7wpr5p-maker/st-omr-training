# ST-OMR Training Lab Status

This file is the current stage-status source for this repository. Detailed closed-stage architecture remains in `ARCHITECTURE.md`; the current active-lane overlay is in `ARCHITECTURE_CURRENT.md`.

## Current repository phase

Verified baseline before Stage 7-D3 work:

- `main`: `9843d86bbb0599a938336c823d73a0ca53efa8d3`
- PR #39 — Stage 7-D2 synthetic V1 train/validation execution: MERGED
- post-merge main CI: run #142 (`31793136296`) — SUCCESS
- post-merge regression: 471/471 PASS
- Stage 7-D0: CLOSED
- Stage 7-D1: CLOSED
- Stage 7-D2: CLOSED

The current active lane is **Stage 7-D3 — Validation Error Diagnostics**. D3 performs no training. It analyzes the already-used 153-sample validation split to identify where the accepted D2 model is wrong while keeping TEST sealed.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0–6 | Deterministic music → validated synthetic dataset pipeline | ✅ Closed / main CI verified |
| 7-A | Training contract freeze | ✅ Closed / main CI verified |
| 7-B | Tokenizer/data/model/trainer implementation | ✅ Closed / main CI verified |
| 7-C | Bounded baseline training + evidence | ✅ Closed / non-production baseline |
| 7-D0 | Synthetic Curriculum v1 export-evidence identity gate | ✅ Closed |
| 7-D1 | Synthetic corpus transport/byte/manifest acceptance | ✅ Closed |
| 7-D2 | Synthetic V1 train/validation execution | ✅ Closed / PR #39 merged / main CI #142 PASS |
| 7-D3 | Validation-only semantic error diagnostics | 🔄 Active — no optimizer steps / TEST sealed |
| 8-0 | Real-data rights/provenance/fine-tuning contract | ✅ Closed / preserved |
| 8-1 | Real-data quarantine/intake + byte validation | ✅ Closed / preserved |
| 8-2 | Paired experiment profile | ✅ Closed / preserved |
| 8-3A | Real pilot preparation/admission components | ⏸ Parked during synthetic quality diagnosis |
| 8-3B | Paired real train/validation execution | 🔒 Not started |
| 9 | Sealed benchmark and candidate decision | 🔒 Not started — TEST sealed |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Frozen Synthetic Curriculum v1

```text
source commit       adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90
config fingerprint  154bf1c3e6dfe4e6db096f8b668f29df0623cfd38352b89a04d295764c7458cb
build id            d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352
manifest SHA-256     44a963cd7dbc612fa29c2953ea8b2c8776d89ce470074e8f8b3fe25c6e165f34
transport SHA-256    4a9f3bb337ef99386081dff29c4c1fc3047dc3ada4db13c93b6254e680918e2b
families             512 = 410 train + 51 validation + 51 test
images               1536 = 1230 train + 153 validation + 153 test
targets              512 MusicXML
```

## Accepted Stage 7-D2 result

```text
run id                 14d63841254c03463ad76bbed83df95045742c23f71ad91d7b0c5dc19495a373
checkpoint SHA-256     239cf3dbdf80235bfc7e4a68fe5fecc03e8cd6fefc8a9ff6e27a2ca879ed5291
checkpoint state SHA   466cefcd40887cb0578b7bbc87c6a1b5f676dc0272ab5eee1142e45e7da8e17d
metrics SHA-256        e80b8aed13cc8c7aafae283f4306f1f60821fbf75faaaf568ddff7b132c318bd
verification SHA-256   6743425d42da77dfacef50388e879d45aa01f01b740cfd2deb381a55436500c3
best epoch             20
untrained val loss     3.5657773256519163
best val loss          0.9379074645594616
exact sequence         0.0
token error rate       0.847364947676063
semantic validity      1.0
MusicXML validity      1.0
TEST development use   0
```

The model learned enough to reduce validation loss and to emit structurally valid supported-V1 music, but its exact reading accuracy remains poor. It is not a production OMR candidate.

## Stage 7-D3 boundary

D3 uses the accepted D2 checkpoint and the same 153 validation images only. After the D1 whole-corpus integrity recheck:

```text
TRAIN       -> skip before D3 artifact path/byte derivation
VALIDATION  -> 51 families / 153 images -> diagnostics only
TEST        -> skip before D3 artifact path/byte derivation
optimizer   -> 0 steps
```

D3 measures token, measure, meter, event type, onset, duration, pitch, accidental, rest and chord-size errors and groups them by musical feature and clean/light/medium degradation profile.

## Safety boundaries

- No direct commits to `main`; changes use small branch/PR packages.
- Large corpus/checkpoint artifacts stay outside normal Git.
- D3 cannot update parameters or optimizer state.
- D1 may hash TEST bytes only for whole-corpus storage integrity; after D1, D3 derives no TRAIN/TEST artifact path or byte.
- TEST remains sealed until Stage 9.
- Existing Stage 8 rights/provenance/privacy/duplicate/leakage controls remain intact and parked.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- No online or automatic learning path is allowed.

## Next gate

Pass focused/full CI for Stage 7-D3, then run the validation-only diagnostic CLI against the accepted Drive corpus and exact D2 checkpoint. The resulting error map determines the next model/data improvement package. Do not retrain or open TEST before that diagnosis is accepted.
