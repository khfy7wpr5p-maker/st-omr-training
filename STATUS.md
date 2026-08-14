# ST-OMR Training Lab Status

This file is the current stage-status source for this repository. Detailed closed-stage architecture remains in `ARCHITECTURE.md`; the current active-lane overlay is in `ARCHITECTURE_CURRENT.md`.

## Current repository phase

Verified baseline before Stage 7-D2 work:

- `main`: `46c7cfe85e734dbbe2e4c0a9ee33dc0f9c5d0e1a`
- PR #38 — Stage 7-D1 corpus byte acceptance: MERGED
- post-merge main CI: run #136 (`31779546692`) — SUCCESS
- post-merge regression: 465/465 PASS
- Stage 7-D0: CLOSED
- Stage 7-D1: CLOSED

The current active lane is **Stage 7-D2 — Synthetic V1 Train / Validation Execution** on draft PR #39. D2 training has not yet been accepted or merged. The exact external run must occur from a clean PR checkout after PR-head CI is green.

The Stage 7-C model remains the comparison baseline: best validation loss approximately `0.99924`, exact sequence accuracy `0%`, token error rate approximately `0.805`, and 21/21 validation predictions passed semantic/MusicXML regeneration checks.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0–6 | Deterministic music → validated synthetic dataset pipeline | ✅ Closed / main CI verified |
| 7-A | Training contract freeze | ✅ Closed / main CI verified |
| 7-B | Tokenizer/data/model/trainer implementation | ✅ Closed / main CI verified |
| 7-C | Bounded baseline training + evidence | ✅ Closed / main CI verified; non-production baseline |
| 7-D0 | Synthetic Curriculum v1 export-evidence identity gate | ✅ Closed / main CI verified |
| 7-D1 | Synthetic corpus transport/byte/manifest acceptance | ✅ Closed / PR #38 merged / main CI #136 PASS |
| 7-D2 | Synthetic V1 train/validation execution | 🔄 Active — PR #39; authoritative run pending |
| 8-0 | Real-data rights/provenance/fine-tuning contract | ✅ Closed / preserved |
| 8-1 | Real-data quarantine/intake + byte validation | ✅ Closed / preserved |
| 8-2 | Paired experiment profile | ✅ Closed / preserved |
| 8-3A | Real pilot preparation/admission components | ⏸ Parked during Synthetic V1 lane |
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
archive bytes        494006801
families             512 = 410 train + 51 validation + 51 test
images               1536 = 1230 train + 153 validation + 153 test
targets              512 MusicXML
```

## Stage 7-D1 accepted receipt

```text
receipt file SHA-256       9a86da742035a7a1644ffd8874587cdc479087539d6320596495a2bd6f7399d0
artifact binding SHA-256   e603b945c6dc60cf7e618ae28a7734dee97cf0e05a81891479107b18a87af540
target bytes total         3506839
image bytes total          494937881
```

D1 was executed against the real Drive archive, independently checked, merged through PR #38, and verified again by exact-main CI #136.

## Stage 7-D2 frozen development boundary

D2 holds the Stage 7-C model, trainer, tokenizer and preprocessing policies fixed and changes only the accepted curriculum size.

```text
TRAIN       410 families / 1230 images -> parameter updates
VALIDATION   51 families /  153 images -> checkpoint selection + development metrics
TEST         51 families /  153 images -> sealed until Stage 9
```

D2 profile: 40 epochs, batch size 4, 1 retained best checkpoint, 1536 max decode tokens, 8-measure decode constraint. Expected optimizer steps: 12,320.

Before training, D2 re-runs the D1 byte gate. After D1 returns, TEST rows are skipped before any D2 test artifact path or byte is derived. Only TRAIN and VALIDATION references may enter model-development code.

## D2 closure gate

Stage 7-D2 may close only after all of the following:

1. exact PR-head CI succeeds;
2. the accepted Drive corpus passes D1 re-verification from the clean PR checkout;
3. all 1,230 TRAIN images and 153 VALIDATION images are admitted with exact frozen identities;
4. the 40-epoch run completes without TEST development access;
5. validation loss improves over the deterministic untrained model;
6. at least one validation prediction crosses the semantic/MusicXML gate;
7. checkpoint, metrics and authoritative verification hashes are independently checked;
8. the exact run result is recorded as small evidence;
9. explicit merge approval is obtained;
10. exact-main post-merge CI succeeds.

## Safety boundaries

- No direct commits to `main`; changes use small branch/PR packages.
- Large corpus/checkpoint artifacts stay outside normal Git.
- Parameter updates use TRAIN only; VALIDATION may select the checkpoint; TEST stays sealed until Stage 9.
- D1 may read TEST artifact bytes only for whole-corpus storage integrity and returns no TEST sample data.
- Existing Stage 8 rights/provenance/privacy/duplicate/leakage controls remain intact and parked.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- No online or automatic learning path is allowed.

## Next gate

Finish PR #39 CI, then execute `stage7d2_execution` in Colab against the already accepted Drive corpus from the exact clean PR head. Do not merge PR #39 before authoritative D2 evidence is independently verified.
