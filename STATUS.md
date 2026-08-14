# ST-OMR Training Lab Status

This file is the current stage-status source for this repository. Detailed closed-stage architecture remains in `ARCHITECTURE.md`; the current active-lane overlay is in `ARCHITECTURE_CURRENT.md`.

## Current repository phase

The GitHub repository is private and GitHub Actions CI is active.

Verified baseline before Stage 7-D1 work:

- `main`: `a3e8412df4dd8d84b0c69aac58361c597883e12c`
- PR #37 — Stage 7-D export evidence gate: merged
- post-merge main CI: run #126 (`31749693208`) — SUCCESS
- Stage 7-D0: CLOSED

The current active lane is **Stage 7-D1 — Synthetic Corpus Byte / Manifest Acceptance**. Model training is not authorized in D1.

The accepted Stage 7-C model remains non-production: best validation loss was approximately `0.99924`, exact sequence accuracy was `0%`, token error rate was approximately `0.805`, and 21/21 validation predictions passed semantic/MusicXML regeneration checks.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0–6 | Deterministic music → validated synthetic dataset pipeline | ✅ Closed / main CI verified |
| 7-A | Training contract freeze | ✅ Closed / main CI verified |
| 7-B | Tokenizer/data/model/trainer implementation | ✅ Closed / main CI verified |
| 7-C | Bounded baseline training + evidence | ✅ Closed / main CI verified; non-production baseline |
| 7-D0 | Synthetic Curriculum v1 export-evidence identity gate | ✅ Closed / main CI verified |
| 7-D1 | Synthetic corpus transport/byte/manifest acceptance | 🔄 Active — no training |
| 7-D2 | Synthetic V1 train/validation execution | 🔒 Not started |
| 8-0 | Real-data rights/provenance/fine-tuning contract | ✅ Closed / preserved |
| 8-1 | Real-data quarantine/intake + byte validation | ✅ Closed / preserved |
| 8-2 | Paired experiment profile | ✅ Closed / preserved |
| 8-3A | Real pilot preparation/admission components | ⏸ Parked — do not expand during Synthetic V1 lane |
| 8-3B | Paired real train/validation execution | 🔒 Not started |
| 9 | Sealed benchmark and candidate decision | 🔒 Not started — test sealed |
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

Drive export contains the expected archive plus small export evidence, `build.json`, and `manifest.sha256`. The small files match the frozen identities. The 494 MB archive exceeds the connected Drive download limit in this environment, so D1 must not claim byte acceptance until the new verifier is run against the real archive/corpus in Colab or another local workspace and produces its small PASS receipt.

## Stage 7-D1 closure gate

D1 must independently verify:

1. exact transport archive filename, size, and SHA-256;
2. manifest SHA-256 and canonical manifest structure;
3. build ID and config fingerprint;
4. exact sample/image/target and family/split counts;
5. family-exclusive split integrity and three derivatives per family;
6. exact hash-addressed filename sets;
7. SHA-256 of every persisted PNG and MusicXML artifact;
8. no missing/extra/symlink artifacts;
9. a small canonical hash-only acceptance receipt.

Passing repository tests alone does not close D1. The actual external corpus-byte receipt is mandatory.

## Safety boundaries

- No direct commits to `main`; changes use small branch/PR packages.
- No stage closes without focused tests, full regression, and relevant CI evidence.
- Large dataset/checkpoint artifacts stay outside normal Git.
- Training may update parameters only from train; validation may select checkpoints; synthetic test remains sealed until Stage 9.
- D1 may hash test artifacts only for complete archive-integrity verification and does not expose them to training/evaluation logic.
- Existing Stage 8 rights/provenance/privacy/duplicate/leakage controls remain intact.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- No online or automatic learning path is allowed.

## Next gate

Complete Stage 7-D1 verifier implementation on a branch, pass focused tests and full GitHub CI, then run the exact verifier against the frozen Drive archive/corpus in Colab/local workspace. Only when both code/CI and external byte evidence pass can D1 be considered merge-ready. Stage 7-D2 training remains locked until D1 is closed.
