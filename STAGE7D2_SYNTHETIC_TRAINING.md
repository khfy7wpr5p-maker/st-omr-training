# Stage 7-D2 — Synthetic V1 Train / Validation Execution

Stage 7-D2 trains the existing bounded ST-OMR baseline architecture on the accepted Synthetic Curriculum v1. It does not change the model architecture, tokenizer, preprocessing policy, real-data lane, or production integration boundary.

## Frozen input

D2 is bound to the D1-accepted corpus:

```text
source commit              adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90
build id                   d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352
config fingerprint         154bf1c3e6dfe4e6db096f8b668f29df0623cfd38352b89a04d295764c7458cb
manifest SHA-256           44a963cd7dbc612fa29c2953ea8b2c8776d89ce470074e8f8b3fe25c6e165f34
transport SHA-256          4a9f3bb337ef99386081dff29c4c1fc3047dc3ada4db13c93b6254e680918e2b
archive                    st-omr-synthetic-curriculum-v1-d9320e362f162cd2.tar.gz
archive bytes              494006801
D1 artifact binding        e603b945c6dc60cf7e618ae28a7734dee97cf0e05a81891479107b18a87af540
```

The D1 byte gate is re-run before D2 admits any development sample. The accepted D1 target/image byte totals are also frozen.

## Development split boundary

```text
TRAIN       410 families / 1230 images  -> parameter updates only
VALIDATION   51 families /  153 images  -> checkpoint selection + development metrics
TEST         51 families /  153 images  -> sealed until Stage 9
```

After D1 returns, D2 skips every TEST manifest row before deriving any test artifact path or reading any test artifact byte. D1's own TEST reads remain restricted to storage-integrity hashing and return no test sample data.

## Frozen training profile

D2 deliberately holds the Stage 7-C model, trainer, tokenizer, and preprocessing policy constant so the larger curriculum is the isolated experimental variable.

- epochs: 40
- batch size: 4
- training samples: all 1230
- validation samples: all 153
- max decode tokens: 1536
- decode measure count: 8
- retained checkpoint count: 1
- device/runtime contract: CPU with the repository's exact pinned runtime

Expected optimizer steps: `ceil(1230 / 4) * 40 = 12320`.

## Acceptance evidence

A completed authoritative run must emit:

1. hash-addressed best checkpoint;
2. canonical D2 metrics JSON;
3. strict checkpoint reload with finite model state;
4. canonical authoritative verification JSON;
5. exact source commit/repository SHA and runtime identity;
6. exact D1 corpus identities and artifact binding;
7. untrained and best validation loss;
8. validation token error rate, exact sequence accuracy, detokenization success, semantic validity, and MusicXML regeneration validity;
9. explicit zero TEST samples exposed to model development.

The run fails closed if validation loss does not improve on the deterministic untrained model or if no validation prediction crosses the semantic gate.

## Colab/local execution

Run only from the exact clean PR checkout with the accepted corpus already extracted:

```bash
python -m st_omr_training.stage7d2_execution \
  --corpus-root /content/d1/st-omr-synthetic-curriculum-v1 \
  --archive /content/drive/MyDrive/ST-OMR-SYNTHETIC/d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352/st-omr-synthetic-curriculum-v1-d9320e362f162cd2.tar.gz \
  --run-root /content/st-omr-stage7d2-runs \
  --repository-root /content/st-omr-training
```

Run output/checkpoint bytes remain outside normal Git. Only small durable hashes and accepted metrics may be committed after independent verification.

## Non-goals

D2 does not open real-data Stage 8 work, TEST evaluation, production candidacy, ScoreMosaic integration, online learning, automatic teacher-correction learning, new notation surfaces, or a larger model architecture.
