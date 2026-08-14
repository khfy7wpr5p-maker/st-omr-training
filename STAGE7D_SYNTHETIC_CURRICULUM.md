# Stage 7-D — Synthetic Curriculum V1

Stage 7-D extends the closed Stage 7-C baseline with the larger frozen synthetic curriculum before real-data work resumes.

## Frozen corpus identity

- source commit: `adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90`
- config fingerprint: `154bf1c3e6dfe4e6db096f8b668f29df0623cfd38352b89a04d295764c7458cb`
- build id: `d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352`
- manifest SHA-256: `44a963cd7dbc612fa29c2953ea8b2c8776d89ce470074e8f8b3fe25c6e165f34`
- transport SHA-256: `4a9f3bb337ef99386081dff29c4c1fc3047dc3ada4db13c93b6254e680918e2b`
- transport size: `494006801` bytes
- 512 families / 1,536 images / 512 MusicXML targets
- 410 train / 51 validation / 51 test families
- 1,230 train / 153 validation / 153 test images

## Gate status

| Gate | Purpose | Status |
|---|---|---|
| 7-D0 | Canonical export-evidence identity gate | ✅ Closed / main CI verified |
| 7-D1 | Transport + persisted corpus byte/manifest acceptance | 🔄 Active |
| 7-D2 | Synthetic V1 train/validation execution | 🔒 Locked until D1 closes |

### 7-D0

D0 validates the small canonical Colab export evidence object against the frozen source, profile, build, manifest, runtime-package, count, and transport identities. It does not inspect the 494 MB archive or every persisted artifact byte.

### 7-D1

D1 independently verifies the copied transport archive and the persisted Stage 6 corpus layout before any loader may use the curriculum. The gate must fail closed on archive drift, manifest/build drift, count drift, family/split leakage, filename/hash mismatches, missing/extra artifacts, symlinks, or changed PNG/MusicXML bytes.

D1 may read the sealed test artifacts only for storage-integrity hashing. It must not tokenize, decode, score, batch, train from, validate on, or otherwise expose the test split to model-development code.

The D1 verifier emits only a small canonical hash/count receipt. Bulk corpus bytes remain outside normal Git content.

See [STAGE7D1_CORPUS_ACCEPTANCE.md](STAGE7D1_CORPUS_ACCEPTANCE.md).

### 7-D2

D2 remains locked. When D1 is closed, D2 may use only:

- TRAIN: 410 families / 1,230 images for parameter updates;
- VALIDATION: 51 families / 153 images for checkpoint selection and development metrics.

TEST: 51 families / 153 images remains sealed until the later benchmark/candidate decision.

## Preserved boundaries

This lane does not change Stage 8 real-data rights, provenance, privacy, admission, duplicate/leakage, or family-exclusive split requirements. Existing Stage 8 components are preserved but parked while Synthetic V1 is strengthened.

ScoreMosaic uploads and teacher corrections are never automatic training data, and no online/automatic learning path is introduced.

This lane also does not add multivoice, piano/grand-staff, orchestral, multi-instrument, cross-staff, tie/slur, tuplet, full-measure/multi-rest, or nonzero-key-signature support.
