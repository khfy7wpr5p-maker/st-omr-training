# ST-OMR Training Lab Status

This file is the current stage-status source for this repository. Detailed closed-stage architecture remains in `ARCHITECTURE.md`; the current active-lane overlay is in `ARCHITECTURE_CURRENT.md`.

## Current repository phase

The GitHub repository is public and GitHub Actions CI is active.

Verified baseline before Stage 7-D1 work:

- `main`: `a3e8412df4dd8d84b0c69aac58361c597883e12c`
- PR #37 — Stage 7-D export evidence gate: merged
- post-merge main CI: run #126 (`31749693208`) — SUCCESS
- Stage 7-D0: CLOSED

The current active lane is **Stage 7-D1 — Synthetic Corpus Byte / Manifest Acceptance**. The external frozen corpus acceptance run has passed; D1 remains open until PR #38 is merged and exact-main post-merge CI succeeds. Model training is not authorized in D1.

The accepted Stage 7-C model remains non-production: best validation loss was approximately `0.99924`, exact sequence accuracy was `0%`, token error rate was approximately `0.805`, and 21/21 validation predictions passed semantic/MusicXML regeneration checks.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0–6 | Deterministic music → validated synthetic dataset pipeline | ✅ Closed / main CI verified |
| 7-A | Training contract freeze | ✅ Closed / main CI verified |
| 7-B | Tokenizer/data/model/trainer implementation | ✅ Closed / main CI verified |
| 7-C | Bounded baseline training + evidence | ✅ Closed / main CI verified; non-production baseline |
| 7-D0 | Synthetic Curriculum v1 export-evidence identity gate | ✅ Closed / main CI verified |
| 7-D1 | Synthetic corpus transport/byte/manifest acceptance | 🔄 External PASS; merge + exact-main CI pending — no training |
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

## Stage 7-D1 external acceptance evidence

The exact verifier from PR #38 was run in Colab against the real frozen Drive archive and a fresh extraction after independent transport SHA-256 verification. The run returned code `0` and emitted the canonical D1 receipt.

```text
receipt file SHA-256       9a86da742035a7a1644ffd8874587cdc479087539d6320596495a2bd6f7399d0
artifact binding SHA-256   e603b945c6dc60cf7e618ae28a7734dee97cf0e05a81891479107b18a87af540
archive bytes              494006801
target bytes total         3506839
image bytes total          494937881
families                   test 51 / train 410 / validation 51
samples                    test 153 / train 1230 / validation 153
images                     1536
targets                    512
```

The receipt is canonical ASCII JSON and independently matches the frozen source commit, build ID, config fingerprint, manifest SHA-256, transport archive name, transport SHA-256, counts, and split contract. Test artifacts were read only for storage-integrity hashing; they remain sealed from training/validation.

During the external run, a P2 archive-name contract typo in PR #38 was exposed fail-closed. The PR branch was corrected to the actual D0/Drive frozen archive name `st-omr-synthetic-curriculum-v1-d9320e362f162cd2.tar.gz`, and a regression test now requires D0 and D1 to share that exact name. PR CI run #133 (`31778806224`) then passed 465/465 tests, pinned dependency checks, `pip check`, and `compileall` on the corrected head before this status synchronization commit.

## Stage 7-D1 closure gate

D1 independently verifies:

1. exact transport archive filename, size, and SHA-256;
2. manifest SHA-256 and canonical manifest structure;
3. build ID and config fingerprint;
4. exact sample/image/target and family/split counts;
5. family-exclusive split integrity and three derivatives per family;
6. exact hash-addressed filename sets;
7. SHA-256 of every persisted PNG and MusicXML artifact;
8. no missing/extra/symlink artifacts;
9. a small canonical hash-only acceptance receipt.

The external byte gate is now PASS. D1 is not closed until the final PR head passes CI, PR #38 is explicitly approved for merge, the PR is merged, and exact-main post-merge CI succeeds.

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

Wait for CI on the final PR #38 head after this evidence/status synchronization. If it passes and no material review finding remains, mark PR #38 ready for review and request explicit merge approval. Only after merge plus exact-main post-merge CI success can Stage 7-D1 close and Stage 7-D2 become eligible as a separate later package.
