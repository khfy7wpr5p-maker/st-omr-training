# Stage 8-0 — Real Data & Fine-Tuning Contract Freeze

Status: **contract package only — no real-data intake, no training, no test opening**.

This contract defines the boundary that must exist before ST-OMR may use any real score image for Stage 8 development. It does not authorize Stage 8-1 intake, fine-tuning, a sealed-test benchmark, model publication, or ScoreMosaic integration.

## Verified Stage 7-C baseline

Stage 8-0 starts from the closed Stage 7-C evidence:

- accepted Stage 7-C source head: `7a993304218fa19609ea512665148dac3eea503a`;
- Stage 7-C main merge: `2c2c478eb361fa90a3bccd819b623680eb12de0b`;
- checkpoint SHA-256: `75c33cefeb970305f5f9171b3274dbe8b785cbcf1e8d6851de7848e66d24efa4`;
- checkpoint model-state SHA-256: `79d354f2582f3f7cc106564b40f07a6027b62b2a74c9efe13a7b2437a6c3f7a0`;
- GitHub Actions artifact: `9177923796`, scheduled to expire `2026-09-12T10:44:21Z`;
- exact sequence accuracy: `0.0`;
- token error rate: `0.8049901510177282`.

The Stage 7-C model is therefore a reproducible trainability baseline, **not a production candidate**. Artifact availability is not model identity: a future Stage 8 fine-tuning run may use Candidate A only if the exact checkpoint bytes are available and match the frozen checkpoint and model-state hashes. This contract does not move, republish, or preserve the artifact.

## Stage decomposition

```text
Stage 7-C  Synthetic baseline + accepted evidence          CLOSED
        ↓
Stage 8-0  Real-data and fine-tuning contract freeze      THIS PACKAGE
        ↓
Stage 8-1  Quarantine/intake + byte-level validators      LOCKED
        ↓
Stage 8-2  Frozen paired experiment run profile           LOCKED
        ↓
Stage 8-3  Train/validation experiments + evidence        LOCKED
        ↓
Stage 9    Sealed benchmark / candidate decision          LOCKED
        ↓
Stage 10   ScoreMosaic integration decision               LOCKED
```

The substage names after Stage 8-0 are planning labels only. They do not authorize implementation.

## Real-data admission principle

Real data is untrusted until independently admitted. A path, filename, uploader claim, license URL, consent checkbox, existing MusicXML file, or successful parser result is not sufficient by itself.

Every future train/validation-eligible real sample must have an immutable metadata record that binds:

- one broad leakage `family_id`;
- one immutable source-document SHA-256;
- one exact training-image SHA-256;
- one exact MusicXML SHA-256;
- one supported-V1 semantic fingerprint;
- source/provenance evidence SHA-256;
- rights/license/permission evidence SHA-256;
- image–MusicXML pairing-review evidence SHA-256;
- split assignment;
- quarantine/admission review state.

The Stage 8 development manifest contains **train and validation metadata only** and records only the SHA-256 commitment of the separately sealed real-test manifest. Test sample identities, labels, hashes, or evidence records must not be copied into the development manifest.

The Stage 8-0 validator in `st_omr_training/real_data_contract.py` validates only this metadata boundary. It deliberately performs no file ingestion. Stage 8-1 must separately implement byte-level source, image, MusicXML, and pairing validation before any real sample becomes train/validation accessible.

## Rights, permission, provenance, and privacy

Before quarantine may be cleared:

1. The source must have traceable provenance evidence.
2. The training-use basis must be one of the frozen classes: public domain, open license, explicit permission, or copyright-owner authorization.
3. Rights review must be explicitly approved and bound to evidence by SHA-256.
4. User-derived material requires **separate explicit training permission**. Service use or upload alone is not training permission.
5. User-derived material also requires a privacy review record before admission.
6. Ambiguous, missing, conflicting, revoked, or rights-unclear evidence fails closed.

The validator can prove that required review metadata and hashes are present and internally consistent. It cannot make a legal determination that a license or permission is substantively valid. Human/organizational rights review remains an independent admission gate.

## ScoreMosaic and teacher-correction boundary

The following are binding:

- ScoreMosaic user uploads are **not automatically training data**.
- Teacher corrections are **not automatically training data**.
- A ScoreMosaic upload, user submission, or teacher correction cannot become training-eligible unless a separate explicit training-permission evidence record and privacy-review evidence exist and the sample passes the full real-data quarantine/admission pipeline.
- No production request, upload, correction, editor action, telemetry event, or inference result may trigger parameter updates.
- No online learning, background learning, implicit retraining, or automatic feedback loop is permitted.
- No Stage 8 output may be integrated into ScoreMosaic before the separate Stage 9 candidate-quality decision and later Stage 10 integration gate.

## Image–MusicXML ground-truth contract

Real-data labels must stay inside the already-frozen supported-V1 semantic surface unless a later contract explicitly expands that surface.

Before admission, the future Stage 8-1 byte-level gate must prove at minimum:

1. image bytes match the recorded image SHA-256;
2. MusicXML bytes match the recorded MusicXML SHA-256;
3. MusicXML passes the existing offline XSD and ST semantic validators;
4. MusicXML passes the existing supported-V1 semantic round-trip gate;
5. tokenization/detokenization reproduces the same supported-V1 semantic projection;
6. a pairing review establishes that the image page and MusicXML target describe the same musical content and page/excerpt;
7. the resulting semantic fingerprint matches the recorded fingerprint;
8. unsupported notation, ambiguous page alignment, incomplete labels, or unresolved transcription disagreement stays quarantined.

A model prediction, Audiveris/Scan2Notes output, LLM output, OCR/OMR teacher output, or other automated transcription may not be treated as ground truth merely because it parses successfully.

## Quarantine and admission state

Real material enters a later Stage 8 intake only as **quarantined**. Quarantine must not expose records to a training loader.

A sample becomes admitted only after all required provenance, rights, permission/privacy where applicable, target/pairing, hash, format, and semantic checks pass. Rejected material remains rejected; it must not be silently reclassified or copied under a new identity.

The Stage 8-0 metadata validator distinguishes quarantine records from admitted training-manifest records. A quarantined record is structurally representable but is rejected by the training-eligible manifest validator.

## Identity, duplicate, and leakage controls

`sample_id` is derived from immutable content identity and deliberately excludes split and review state. Moving a sample between train/validation/test or changing approval state cannot manufacture a fresh identity.

An admitted real-data manifest must reject at minimum:

- duplicate sample identities;
- duplicate exact image hashes;
- duplicate exact image/MusicXML pairs;
- one family appearing in more than one split;
- one source document appearing under multiple families or splits;
- one exact MusicXML target appearing under multiple families or splits;
- one exact supported-V1 semantic fingerprint appearing under multiple families or splits.

`family_id` must be assigned at the broadest practical leakage boundary: alternate scans/photos/renders/crops/transcriptions/corrections of the same page or excerpt belong to one family. A later intake package must add perceptual/near-duplicate review; Stage 8-0 exact-hash checks do not claim to solve visual near-duplicate detection.

## Split policy and sealed test

Real data uses exactly three family-exclusive splits, but Stage 8 development must not place all three into one inspectable manifest:

- `train`: may update model parameters in a later authorized Stage 8 training package;
- `validation`: may select checkpoints and report Stage 8 development metrics;
- `test`: stored as a separately sealed manifest and reserved for Stage 9.

The Stage 8 development manifest must contain admitted `train` and `validation` records only plus `sealed_test_manifest_sha256`, an opaque commitment to the separately controlled test manifest. A `test` record appearing in the development manifest is a hard veto. `select_stage8_development_records(...)` also fails closed immediately on `test`.

The existing Stage 6 synthetic test split also remains sealed. Stage 8-0 does not open either test partition, enumerate test contents, or use test records for data selection, hyperparameters, architecture choices, threshold choices, or experiment comparison. Stage 8-1 must define the future one-time test-sealing procedure before any real test material exists; Stage 8-0 does not implement that procedure.

## Frozen experiment candidates

Stage 8 must compare two declared candidates before Stage 9:

### Candidate A — Stage 7-C checkpoint fine-tuning

- exact same frozen Stage 7-B CNN/GRU architecture, tokenizer, vocabulary, and preprocessing surface;
- initialization from the exact Stage 7-C checkpoint identified above;
- checkpoint bytes must pass exact SHA-256 and strict state/model compatibility validation before use;
- no fallback to a different checkpoint, external pretrained model, teacher model, or silently recreated artifact;
- if the exact checkpoint is unavailable, Candidate A is **blocked** until a separately approved artifact/recreation decision exists.

### Candidate B — same architecture from scratch

- exact same architecture family and supported-V1 token surface;
- initialized from the frozen Stage 7 deterministic initialization path rather than any checkpoint;
- no external pretrained weights or teacher outputs.

For the primary comparison, both candidates must use the same admitted real-data manifest version, same train/validation families, same preprocessing/tokenizer/target contract, same metric definitions, and a pre-frozen Stage 8 run profile. Any intentional optimizer/budget difference must be declared before execution and versioned as a separate experiment; test data may not be used to choose it.

## Metrics and quality interpretation

Stage 8 validation evidence must retain the Stage 7 metric family at minimum: validation loss, token error rate, exact sequence accuracy, detokenization success, semantic validity, and MusicXML regeneration validity. Real-data failure analysis may add metrics, but Stage 9 owns production-candidate thresholds and the first sealed-test quality decision.

Neither a lower validation loss nor valid grammar-constrained MusicXML is sufficient to declare the model production quality.

## Repository and artifact boundary

This repository may contain source code, contracts, tests using synthetic/hash-only fixtures, and small provenance metadata. It must not contain real/user documents, real score images, real MusicXML corpora, raw permission/license documents containing private information, large datasets, or model checkpoints.

Stage 8-0 does not create storage credentials, a model registry, a cloud bucket, an upload endpoint, a ScoreMosaic connector, or a data-ingestion service.

## Required gate before Stage 8-0 closure

```text
exact verified Stage 7-C main baseline
        ↓
real-data rights/provenance/pairing contract
        ↓
quarantine + admission metadata model
        ↓
family/hash/semantic leakage validator
        ↓
Stage 8 test-access veto
        ↓
Candidate A / Candidate B experiment contract
        ↓
ScoreMosaic/teacher-correction isolation contract
        ↓
focused positive + negative tests
        ↓
full repository regression + pip check + compileall
        ↓
exact PR-head GitHub CI
        ↓
separate explicit merge approval
        ↓
post-merge exact-main CI
        ↓
Stage 8-0 CLOSED
```

## Explicitly out of scope

Stage 8-0 does not:

- ingest any real, user, or teacher-correction data;
- download or scrape scores;
- create a real dataset;
- inspect or open either sealed test split;
- load the Stage 7-C checkpoint bytes;
- move, publish, copy, upload, or preserve the Stage 7-C artifact;
- run training or fine-tuning;
- change model architecture or tokenizer vocabulary;
- start Stage 9 or Stage 10;
- integrate with ScoreMosaic;
- enable online/automatic learning.
